from setuptools import setup, find_packages

setup(
    name="credit_risk_ml",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=[
        # Dependências serão instaladas a partir do requirements.txt
    ],
    author="Ricardo",
    author_email="ricardo@example.com",
    description="Sistema de análise de risco de crédito usando ML",
    keywords="machine learning, credit risk, finance",
    python_requires=">=3.8",
)