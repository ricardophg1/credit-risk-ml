"""
FastAPI application for real-time credit risk scoring
Serves ML model with <100ms latency at scale
Author: Ricardo Ferreira dos Santos
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import redis
import json
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest
import uvicorn

from src.models.ensemble import CreditRiskEnsemble
from src.features.feature_engineering import FeatureEngineer
from src.monitoring.drift_detection import DriftDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk ML API",
    description="Real-time credit risk assessment with 94% accuracy",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Prometheus metrics
prediction_counter = Counter('credit_risk_predictions_total', 'Total number of predictions')
prediction_latency = Histogram('credit_risk_prediction_duration_seconds', 'Prediction latency')
prediction_errors = Counter('credit_risk_prediction_errors_total', 'Total prediction errors')

# Global model and feature engineer
model: Optional[CreditRiskEnsemble] = None
feature_engineer: Optional[FeatureEngineer] = None
drift_detector: Optional[DriftDetector] = None


class TransactionRequest(BaseModel):
    """Schema for transaction scoring request."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    merchant_category: Optional[str] = Field(None, description="Merchant category code")
    
    # Additional optional fields for better predictions
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    
    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TRX123456",
                "customer_id": "CUST789",
                "merchant_id": "MERCH456",
                "amount": 1500.00,
                "timestamp": "2024-01-15T10:30:00Z",
                "merchant_category": "5411"
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    transaction_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Risk probability (0-1)")
    risk_category: str = Field(..., description="Risk category: low, medium, high")
    should_block: bool = Field(..., description="Whether to block transaction")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_version: str = Field(..., description="Model version used")
    
    # Explainability
    top_risk_factors: Optional[List[Dict[str, float]]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TRX123456",
                "risk_score": 0.12,
                "risk_category": "low",
                "should_block": False,
                "inference_time_ms": 87.3,
                "model_version": "v1.0.0",
                "top_risk_factors": [
                    {"factor": "transaction_hour", "contribution": 0.25},
                    {"factor": "amount_vs_avg", "contribution": 0.18}
                ]
            }
        }


class BatchRequest(BaseModel):
    """Schema for batch prediction request."""
    transactions: List[TransactionRequest]
    

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    predictions_served: int


@app.on_event("startup")
async def startup_event():
    """Load model and initialize services on startup."""
    global model, feature_engineer, drift_detector
    
    logger.info("Starting Credit Risk ML API...")
    
    try:
        # Load model
        model = CreditRiskEnsemble.load("models/credit_risk_ensemble.pkl")
        logger.info("Model loaded successfully")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        logger.info("Feature engineer initialized")
        
        # Initialize drift detector
        drift_detector = DriftDetector(baseline_path="models/baseline_stats.pkl")
        logger.info("Drift detector initialized")
        
        # Warm up model with dummy prediction
        dummy_data = pd.DataFrame({
            'transaction_id': ['DUMMY'],
            'customer_id': ['DUMMY'],
            'merchant_id': ['DUMMY'],
            'amount': [100.0],
            'timestamp': [datetime.now()]
        })
        _ = await predict_transaction(dummy_data.iloc[0].to_dict())
        logger.info("Model warmed up")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Credit Risk ML API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version="v1.0.0",
        uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        predictions_served=int(prediction_counter._value.get())
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TransactionRequest, background_tasks: BackgroundTasks):
    """
    Single transaction risk scoring endpoint.
    Returns risk score in <100ms.
    """
    
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = f"prediction:{request.transaction_id}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for transaction {request.transaction_id}")
            return JSONResponse(content=json.loads(cached_result))
        
        # Convert request to DataFrame
        transaction_df = pd.DataFrame([request.dict()])
        
        # Engineer features
        features_df = feature_engineer.create_features(transaction_df)
        
        # Get model features
        model_features = model.feature_names
        X = features_df[model_features]
        
        # Make prediction
        with prediction_latency.time():
            risk_score = model.predict_proba(X)[0, 1]
            
        # Determine risk category and action
        if risk_score < 0.3:
            risk_category = "low"
            should_block = False
        elif risk_score < 0.7:
            risk_category = "medium"
            should_block = False
        else:
            risk_category = "high"
            should_block = True
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000
        
        # Get top risk factors (simplified for example)
        feature_importance = model.get_feature_importance()
        top_factors = []
        for _, row in feature_importance.head(3).iterrows():
            factor_value = float(X[row['feature']].iloc[0])
            contribution = row['importance'] * factor_value
            top_factors.append({
                "factor": row['feature'],
                "contribution": round(contribution, 4)
            })
        
        response = PredictionResponse(
            transaction_id=request.transaction_id,
            risk_score=round(float(risk_score), 4),
            risk_category=risk_category,
            should_block=should_block,
            inference_time_ms=round(inference_time, 1),
            model_version="v1.0.0",
            top_risk_factors=top_factors
        )
        
        # Cache result
        redis_client.setex(
            cache_key,
            300,  # 5 minutes TTL
            json.dumps(response.dict())
        )
        
        # Update metrics
        prediction_counter.inc()
        
        # Background tasks
        background_tasks.add_task(
            log_prediction,
            request.transaction_id,
            risk_score,
            inference_time
        )
        
        # Check for drift
        background_tasks.add_task(
            check_drift,
            features_df
        )
        
        return response
        
    except Exception as e:
        prediction_errors.inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchRequest):
    """
    Batch prediction endpoint for multiple transactions.
    Optimized for throughput.
    """
    
    # Process transactions in parallel
    tasks = [predict(transaction, BackgroundTasks()) for transaction in request.transactions]
    results = await asyncio.gather(*tasks)
    
    return results


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.get("/model/info")
async def model_info():
    """Get model information and performance stats."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "ensemble",
        "algorithms": model.models,
        "weights": model.weights,
        "threshold": model.threshold,
        "n_features": len(model.feature_names),
        "top_features": model.get_feature_importance().head(10).to_dict('records')
    }


async def predict_transaction(transaction_data: Dict) -> float:
    """Internal prediction function."""
    transaction_df = pd.DataFrame([transaction_data])
    features_df = feature_engineer.create_features(transaction_df)
    X = features_df[model.feature_names]
    return model.predict_proba(X)[0, 1]


async def log_prediction(transaction_id: str, risk_score: float, inference_time: float):
    """Log prediction for monitoring and analysis."""
    # Implement your logging logic here
    # Could write to database, Kafka, etc.
    logger.info(
        f"Prediction logged - ID: {transaction_id}, "
        f"Score: {risk_score:.4f}, Time: {inference_time:.1f}ms"
    )


async def check_drift(features_df: pd.DataFrame):
    """Check for feature drift in production data."""
    if drift_detector:
        drift_results = drift_detector.check_drift(features_df)
        if drift_results.get('drift_detected', False):
            logger.warning(f"Drift detected: {drift_results}")
            # Trigger alerts here


if __name__ == "__main__":
    app.state.start_time = time.time()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )