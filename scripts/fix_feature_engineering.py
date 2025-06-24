"""
Script para corrigir o erro no notebook 02_feature_engineering.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import pathlib
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de plotagem
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Obter o caminho da raiz do projeto
current_path = pathlib.Path().absolute()
if current_path.name == 'scripts':
    project_root = current_path.parent
else:
    project_root = current_path

# Definir caminhos para as pastas de dados
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')

# Garantir que as pastas existam
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Carregar Dados e Insights da EDA
print("Carregando dados de transações...")
transactions_file = os.path.join(RAW_DATA_DIR, 'transactions_2024.parquet')
df = pd.read_parquet(transactions_file)
print(f"Carregados {len(df):,} transações")

# Carregar insights da EDA
insights_file = os.path.join(PROCESSED_DATA_DIR, 'eda_insights.json')
with open(insights_file, 'r') as f:
    insights = json.load(f)
print("Insights da EDA carregados com sucesso!")

# Função corrigida para contar transações nas últimas 24 horas
def count_txns_last_24h(group):
    """
    Conta quantas transações foram realizadas nas últimas 24 horas para cada transação
    """
    # Garantir que timestamp seja datetime
    if not pd.api.types.is_datetime64_any_dtype(group['timestamp']):
        group['timestamp'] = pd.to_datetime(group['timestamp'])
    
    # Ordenar transações por timestamp
    group = group.sort_values('timestamp')
    timestamps = group['timestamp'].tolist()
    counts = []
    
    for i, ts in enumerate(timestamps):
        # Correção: Calcular a diferença de tempo para cada transação anterior
        count = 0
        for prev_ts in timestamps[:i]:
            # Verificar se a transação anterior ocorreu nas últimas 24 horas
            if isinstance(ts, datetime) and isinstance(prev_ts, datetime):
                if (ts - prev_ts) <= timedelta(hours=24):
                    count += 1
        counts.append(count)
    
    return counts

# 3. Feature Engineering
print("\n=== Criando Features Temporais ===")

# Converter timestamp para datetime se necessário
if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extrair features temporais
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Visualizar distribuição de transações por hora do dia
plt.figure(figsize=(10, 6))
sns.countplot(x='hour_of_day', data=df)
plt.title('Distribuição de Transações por Hora do Dia')
plt.xlabel('Hora do Dia')
plt.ylabel('Número de Transações')
plt.tight_layout()
plt.show()

# Features de Velocidade
print("\n=== Criando Features de Velocidade ===")
print("Calculando transações nas últimas 24 horas por cliente...")

# Aplicar a função por cliente
df['txn_count_24h'] = df.groupby('customer_id').apply(
    lambda x: pd.Series(count_txns_last_24h(x), index=x.index)
).reset_index(level=0, drop=True)

# Feature de alta velocidade
df['high_velocity'] = (df['txn_count_24h'] > insights['velocity_threshold']).astype(int)

# 4. Features de Valor
print("\n=== Criando Features de Valor ===")

# Calcular estatísticas de valor por cliente
customer_stats = df.groupby('customer_id')['amount'].agg(['mean', 'std', 'max']).reset_index()
customer_stats.columns = ['customer_id', 'avg_amount', 'std_amount', 'max_amount']

# Mesclar de volta ao dataframe principal
df = pd.merge(df, customer_stats, on='customer_id', how='left')

# Criar feature de valor anômalo
df['amount_zscore'] = df.groupby('customer_id')['amount'].transform(
    lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
)
df['is_amount_anomaly'] = (abs(df['amount_zscore']) > insights['amount_zscore_threshold']).astype(int)

# 5. Features de Localização
print("\n=== Criando Features de Localização ===")

# Calcular distância entre transações consecutivas (simplificado)
def calculate_distance_features(group):
    group = group.sort_values('timestamp')
    
    # Inicializar colunas
    group['distance_from_prev'] = 0.0
    group['time_since_prev'] = pd.Timedelta(seconds=0)
    group['speed'] = 0.0
    
    # Calcular para cada transação após a primeira
    for i in range(1, len(group)):
        # Distância em km (simplificada usando coordenadas cartesianas)
        lat1, lon1 = group.iloc[i-1]['latitude'], group.iloc[i-1]['longitude']
        lat2, lon2 = group.iloc[i]['latitude'], group.iloc[i]['longitude']
        
        # Distância euclidiana aproximada (para simplificar)
        dist = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # 1 grau ≈ 111 km
        group.iloc[i, group.columns.get_loc('distance_from_prev')] = dist
        
        # Tempo desde a transação anterior
        time_diff = group.iloc[i]['timestamp'] - group.iloc[i-1]['timestamp']
        group.iloc[i, group.columns.get_loc('time_since_prev')] = time_diff
        
        # Velocidade em km/h
        hours = time_diff.total_seconds() / 3600
        if hours > 0:
            speed = dist / hours
            group.iloc[i, group.columns.get_loc('speed')] = speed
    
    return group

# Aplicar função por cliente
print("Calculando features de distância e velocidade...")
df = df.groupby('customer_id').apply(calculate_distance_features).reset_index(drop=True)

# Identificar transações com velocidade impossível
df['impossible_travel'] = (df['speed'] > insights['max_possible_speed']).astype(int)

# 6. Salvar Dataset Processado
print("\n=== Salvando Dataset com Features ===")
output_file = os.path.join(PROCESSED_DATA_DIR, 'transactions_features.parquet')
df.to_parquet(output_file)
print(f"Dataset salvo em {output_file}")
print(f"Total de features criadas: {df.shape[1] - 7}")  # Subtrair as colunas originais

# Mostrar as primeiras linhas do dataset com as novas features
print("\nPrimeiras linhas do dataset com as novas features:")
print(df.head())

# Resumo das features criadas
print("\nResumo das features criadas:")
print("1. Features Temporais: hour_of_day, day_of_week, is_weekend")
print("2. Features de Velocidade: txn_count_24h, high_velocity")
print("3. Features de Valor: avg_amount, std_amount, max_amount, amount_zscore, is_amount_anomaly")
print("4. Features de Localização: distance_from_prev, time_since_prev, speed, impossible_travel")
