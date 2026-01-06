from math import log10
from typing import List, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from allocation.optimizer import AllocationResult, PoolCandidate, PortfolioOptimizer

app = FastAPI(
    title="Risk Scoring & Allocation API",
    description="API cung cấp 4 endpoints: 2 cho Classification (V0) và 2 cho Regression (V2).",
    version="2.0.0",
)

# ---  LOAD MODELS ---

# Load Classification Model (V0)
try:
    clf_model = joblib.load("random_forest_model.pkl")
    print("Classification Model (V0) loaded.")
except Exception as e:
    print(f"Error loading classification model: {e}")
    clf_model = None

# Load Regression Model (V2)
try:
    reg_model = joblib.load("training/risk_score_model_v2.pkl")
    print("Regression Model (V2) loaded.")
except Exception as e:
    print(f"Error loading regression model: {e}")
    reg_model = None


# ---  HELPER FUNCTIONS ---

def get_chain_score(chain_name: str) -> float:
    chain_name = str(chain_name).lower().strip()
    if chain_name in ['ethereum']: return 0.0 # Hoặc 1.0 tùy hệ quy chiếu model cũ
    if chain_name in ['arbitrum', 'optimism', 'base', 'polygon']: return 10.0
    if chain_name in ['solana', 'bsc', 'avalanche']: return 20.0
    return 40.0

def get_token_risk_score(symbol: str) -> float:
    symbol = str(symbol).upper().strip()
    if symbol in ['USDC', 'USDT', 'PYUSD', 'GUSD', 'DAI']: return 1.0
    if symbol in ['LUSD', 'FRAX', 'RETH', 'WETH', 'WBTC', 'CBETH']: return 3.0
    if symbol in ['SUSDS', 'USDE', 'CRVUSD', 'AUSD', 'GHO']: return 5.0
    return 8.0

def calculate_features_v1(history: list, chain: str):
    """Feature Engineering cho Model Classification cũ"""
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
    
    metrics = {
        "tvl_current": tvl_current,
        "apy_mean": apy_mean,
        "max_drawdown": max_drawdown,
        "tvl_volatility": tvl_volatility
    }
    return features_df, metrics

def calculate_features_v2(history: list, chain: str, symbol: str):
    """Feature Engineering cho Model Regression V2"""
    df = pd.DataFrame([h.model_dump() for h in history])
    df["tvlUsd"] = df["tvlUsd"].fillna(0)
    df["apy"] = df["apy"].fillna(0)

    # 1. Log TVL
    tvl_current = df["tvlUsd"].iloc[-1]
    log_tvl = log10(tvl_current) if tvl_current > 1 else 0.0

    # 2. TVL Change 7D
    idx_7d = -7 if len(df) >= 7 else 0
    tvl_7d = df["tvlUsd"].iloc[idx_7d]
    tvl_change_7d = (tvl_current - tvl_7d) / tvl_7d if tvl_7d > 0 else 0.0

    # 3. Volatility & Drawdown
    tvl_mean = df["tvlUsd"].mean()
    tvl_std = df["tvlUsd"].std()
    tvl_volatility = tvl_std / tvl_mean if tvl_mean > 0 else 0.0

    rolling_max = df["tvlUsd"].cummax()
    drawdown = (df["tvlUsd"] - rolling_max) / rolling_max
    drawdown = drawdown.replace([np.inf, -np.inf], 0).fillna(0)
    max_drawdown = drawdown.min()

    # 4. APY
    apy_mean = df["apy"].mean()
    apy_std = df["apy"].std()

    # 5. Scores
    chain_s = get_chain_score(chain)
    token_s = get_token_risk_score(symbol)

    # DataFrame đúng thứ tự features lúc train
    features_df = pd.DataFrame([{
        'log_tvl': log_tvl,
        'tvl_change_7d': tvl_change_7d,
        'max_drawdown': max_drawdown,
        'tvl_volatility': tvl_volatility,
        'apy_mean': apy_mean,
        'apy_std': apy_std,
        'chain_score': chain_s,
        'token_score': token_s
    }])

    metrics = {
        "tvl_current": tvl_current,
        "log_tvl": log_tvl,
        "tvl_change_7d": tvl_change_7d,
        "max_drawdown": max_drawdown,
        "apy_mean": apy_mean,
        "apy_std": apy_std,
        "tvl_volatility": tvl_volatility,
        "token_score": token_s
    }
    return features_df, metrics




class DailyRecord(BaseModel):
    timestamp: str | None = None
    tvlUsd: float
    apy: float

class PoolHistoryInput(BaseModel):
    pool_name: str
    chain: str
    symbol: str = "UNKNOWN" 
    project: str = "unknown"
    history: List[DailyRecord] = Field(..., description="List of daily records.")

    @field_validator('history')
    @classmethod
    def check_history_length(cls, v):
        if len(v) < 7:
            raise ValueError("History must contain at least 7 records.")
        return v

class MetricsAnalyzedV1(BaseModel):
    tvl_current: float
    avg_apy: float
    max_drawdown: float

class PredictionResponseV1(BaseModel):
    pool_name: str
    risk_evaluation: str
    confidence_score: float 
    analyzed_metrics: MetricsAnalyzedV1 

class MetricsAnalyzedV2(BaseModel):
    tvl_current: float
    log_tvl: float
    tvl_change_7d: float
    max_drawdown: float
    apy_mean: float
    apy_std: float
    token_score: float

class PredictionResponseV2(BaseModel):
    pool_name: str
    risk_score: float       
    risk_label: str     
    metrics: MetricsAnalyzedV2

class VaultRequest(BaseModel):
    vault_type: Literal["Conservative", "Balanced", "Aggressive"]
    total_capital: float
    pools_history: list[PoolHistoryInput]

class AllocationResponse(BaseModel):
    vault_type: str
    total_capital: float
    estimated_vault_apy: float
    allocation: list[AllocationResult]


# --- API ENDPOINTS ---

@app.get("/", tags=["Health Check"])
def health_check():
    return {
        "status": "API is running",
        "models": {
            "classification_v0": clf_model is not None,
            "regression_v2": reg_model is not None
        }
    }

# =========== V0: CLASSIFICATION ENDPOINTS ===========

@app.post("/predict/classification", tags=["V0: Classification"], response_model=PredictionResponseV1)
def predict_risk_score_v1(data: PoolHistoryInput):
    """Dự đoán nhãn rủi ro (Conservative/Balanced/Aggressive) dùng model cũ."""
    if not clf_model:
        raise HTTPException(status_code=503, detail="Classification model not available.")

    try:
        features_df, metrics = calculate_features_v1(data.history, data.chain)
        prediction = clf_model.predict(features_df)[0]
        probabilities = clf_model.predict_proba(features_df)[0]
        confidence = np.max(probabilities)

        return PredictionResponseV1(
            pool_name=data.pool_name,
            risk_evaluation=prediction,
            confidence_score=round(float(confidence), 4),
            analyzed_metrics=MetricsAnalyzedV1(
                tvl_current=round(metrics["tvl_current"], 2),
                avg_apy=round(metrics["apy_mean"], 2),
                max_drawdown=round(metrics["max_drawdown"], 4)
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/vault/allocate/v0", tags=["V0: Classification"], response_model=AllocationResponse)
def allocate_vault_v0(req: VaultRequest):
    """Phân bổ vốn dùng model phân loại (cũ)."""
    if not clf_model:
        raise HTTPException(status_code=503, detail="Classification model not available.")

    candidates = []
    try:
        for pool in req.pools_history:
            features_df, metrics = calculate_features_v1(pool.history, pool.chain)
            risk_label = clf_model.predict(features_df)[0]
            confidence = np.max(clf_model.predict_proba(features_df)[0])
            
            candidates.append(PoolCandidate(
                pool_name=pool.pool_name,
                risk_label=risk_label,
                confidence=float(confidence),
                apy=metrics["apy_mean"],      
                volatility=metrics["tvl_volatility"]
            ))

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
        raise HTTPException(status_code=500, detail=f"Allocation error: {e}")


# =========== V2: REGRESSION ENDPOINTS ===========

@app.post("/predict/regression", tags=["V2: Regression"], response_model=PredictionResponseV2)
def predict_risk_score_v2(data: PoolHistoryInput):
    """Dự đoán điểm rủi ro (0-10) dùng model hồi quy mới."""
    if not reg_model:
        raise HTTPException(status_code=503, detail="Regression model not available.")

    try:
        features_df, metrics = calculate_features_v2(data.history, data.chain, data.symbol)
        predicted_score = reg_model.predict(features_df)[0]
        final_score = max(0.0, min(10.0, float(predicted_score)))

        # Mapping Score -> Label
        label = "Balanced"
        if final_score <= 2.5: label = "Conservative"
        elif final_score >= 6.5: label = "Aggressive"

        return PredictionResponseV2(
            pool_name=data.pool_name,
            risk_score=round(final_score, 2),
            risk_label=label,
            metrics=MetricsAnalyzedV2(
                tvl_current=round(metrics["tvl_current"], 2),
                log_tvl=round(metrics["log_tvl"], 2),
                tvl_change_7d=round(metrics["tvl_change_7d"], 4),
                max_drawdown=round(metrics["max_drawdown"], 4),
                apy_mean=round(metrics["apy_mean"], 2),
                apy_std=round(metrics["apy_std"], 2),
                token_score=metrics["token_score"]
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/vault/allocate/v2", tags=["V2: Regression"], response_model=AllocationResponse)
def allocate_vault_v2(req: VaultRequest):
    """Phân bổ vốn dùng model hồi quy (mới) với features nâng cao."""
    if not reg_model:
        raise HTTPException(status_code=503, detail="Regression model not available.")

    candidates = []
    try:
        for pool in req.pools_history:
            # 1. Tính features V2
            features_df, metrics = calculate_features_v2(pool.history, pool.chain, pool.symbol)
            
            # 2. Dự đoán điểm số
            predicted_score = reg_model.predict(features_df)[0]
            final_score = max(0.0, min(10.0, float(predicted_score)))
            
            # 3. Chuyển đổi điểm số thành nhãn để lọc
            risk_label = "Balanced"
            if final_score <= 2.5: risk_label = "Conservative"
            elif final_score >= 6.5: risk_label = "Aggressive"
            
            # 4. Tạo ứng viên (Với Regression, ta giả định confidence = 1.0 hoặc dựa trên điểm số)
            candidates.append(PoolCandidate(
                pool_name=pool.pool_name,
                risk_label=risk_label,
                confidence=1.0, # Regression model không trả về xác suất, mặc định tin cậy cao
                apy=metrics["apy_mean"],
                volatility=metrics["tvl_volatility"]
            ))

        # 5. Chạy tối ưu hóa
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
        raise HTTPException(status_code=500, detail=f"Allocation V2 error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)