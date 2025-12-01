from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from allocation.optimizer import AllocationResult, PoolCandidate, PortfolioOptimizer

app = FastAPI(
    title="Lending Pool Risk Scoring API",
    description="API for predicting risk scores of lending pools using a trained Random Forest model.",
    version="1.0.0",
)

try:
    model = joblib.load("random_forest_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def get_chain_score(chain_name: str) -> float:
    chain_name = str(chain_name).lower()
    if chain_name in ['ethereum']: 
        return 0.0
    if chain_name in ['arbitrum', 'optimism', 'base', 'polygon']: 
        return 10.0
    if chain_name in ['solana', 'bsc', 'avalanche']: 
        return 20.0
    return 40.0

def calculate_pool_features(history: list, chain: str):
    
    data_dicts = [r.model_dump() for r in history]
    df = pd.DataFrame(data_dicts)

    tvl_current = df["tvlUsd"].iloc[-1]
    tvl_mean = df["tvlUsd"].mean()
    tvl_std = df["tvlUsd"].std()
    
    tvl_volatility = tvl_std / tvl_mean if tvl_mean > 0 else 0

    
    rolling_max = df["tvlUsd"].cummax()
    drawdown = (df['tvlUsd'] - rolling_max) / rolling_max
    drawdown = drawdown.fillna(0)
    max_drawdown = drawdown.min()

    apy_mean = df["apy"].mean()
    apy_std = df["apy"].std()

    chain_score = get_chain_score(chain)

    features_df = pd.DataFrame([{
        'tvl_current': tvl_current,
        'tvl_mean': tvl_mean,
        'tvl_volatility': tvl_volatility,
        'max_drawdown': max_drawdown,
        'apy_mean': apy_mean,
        'apy_std': apy_std,
        'chain_score': chain_score
    }])
    
    # Metrics for Response
    metrics = {
        "tvl_current": tvl_current,
        "tvl_mean": tvl_mean,
        "tvl_volatility": tvl_volatility,
        "max_drawdown": max_drawdown,
        "apy_mean": apy_mean,
        "apy_std": apy_std
    }

    return features_df, metrics


class DailyRecord(BaseModel):
    tvlUsd: float
    apy: float

class PoolHistoryInput(BaseModel):
    pool_name: str
    chain: str
    history: list[DailyRecord] = Field(..., description="List of daily records with TVL and APY.")

    @field_validator('history')
    @classmethod
    def check_history_length(cls, v):
        if len(v) < 30:
            raise ValueError("History must contain at least thirty records.")
        return v

class MetricsAnalyzed(BaseModel):
    tvl_current: float
    avg_apy: float
    max_drawdown: float

class PredictionResponse(BaseModel):
    pool_name: str
    risk_evaluation: str
    confidence_score: float 
    analyzed_metrics: MetricsAnalyzed 

class VaultRequest(BaseModel):
    vault_type: Literal["Conservative", "Balanced", "Aggressive"]
    total_capital: float
    pools_history: list[PoolHistoryInput]

class AllocationResponse(BaseModel):
    vault_type: str
    total_capital: float
    estimated_vault_apy: float
    allocation: list[AllocationResult]



@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "API is running"}


@app.post("/predict", tags=["Risk Prediction"], response_model=PredictionResponse)
def predict_risk_score(data: PoolHistoryInput):
    r"""Predict the risk score of a lending pool based on its historical data."""

    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available."
        )

    try:
        features_df, metrics = calculate_pool_features(data.history, data.chain)

        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence = np.max(probabilities)

        return PredictionResponse(
            pool_name=data.pool_name,
            risk_evaluation=prediction,
            confidence_score=round(float(confidence), 4),
            analyzed_metrics=MetricsAnalyzed(
                tvl_current=round(metrics["tvl_current"], 2),
                avg_apy=round(metrics["apy_mean"], 2),
                max_drawdown=round(metrics["max_drawdown"], 4)
            )
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {e}"
        )


@app.post("/vault/allocate", tags=["Vault Allocation"], response_model=AllocationResponse)
def allocate_vault(req: VaultRequest):
    r"""Allocate funds across lending pools based on vault type and pool risk scores."""
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available."
        )

    candidates = []

    try:
        for pool_input in req.pools_history:
            features_df, metrics = calculate_pool_features(pool_input.history, pool_input.chain)
            
            risk_label = model.predict(features_df)[0]
            confidence = np.max(model.predict_proba(features_df)[0])
            
            candidate = PoolCandidate(
                pool_name=pool_input.pool_name,
                risk_label=risk_label,
                confidence=float(confidence),
                apy=metrics["apy_mean"],      
                volatility=metrics["tvl_volatility"]
            )
            candidates.append(candidate)

        optimizer = PortfolioOptimizer(total_capital=req.total_capital)
        
        portfolio = optimizer.optimize(req.vault_type, candidates)

        weighted_apy = 0.0
        if portfolio:
            total_allocated = sum(p.amount for p in portfolio)
            
            if total_allocated > 0:
                weighted_apy = sum(p.amount * p.expected_apy for p in portfolio) / total_allocated

        return AllocationResponse(
            vault_type=req.vault_type,
            total_capital=req.total_capital,
            estimated_vault_apy=round(weighted_apy, 2),
            allocation=portfolio
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during allocation: {e}"
        )