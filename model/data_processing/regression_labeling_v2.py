import asyncio
import math
from datetime import datetime
from math import log10
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from beanie import Document
from pydantic import Field

from clients import Clients
from configs import get_logger
from mongo.schemas import (
    PoolsSnapshot30dV1,
    PoolsSnapshotAllTime,
    PoolsSnapshotTestRegressionV2,
    PoolsSnapshotTrainRegressionV2,
)

mongo_client = Clients.get_mongo_client()
logger = get_logger(__name__)

# --- 1. HÀM TÍNH ĐIỂM THÀNH PHẦN (SCORING HELPERS) ---

def get_chain_score(chain_name: str) -> float:
    chain = str(chain_name).lower().strip()
    if chain == "ethereum": return 1.0
    if chain in ["arbitrum", "optimism", "base", "polygon"]: return 3.0
    if chain in ["solana", "bsc", "avalanche"]: return 5.0
    return 8.0

def get_token_risk_score(symbol: str) -> float:
    """
    Chấm điểm uy tín của đồng Coin (Asset Quality).
    Điểm càng cao -> Rủi ro càng cao.
    """
    symbol = str(symbol).upper().strip()
    
    # Tier 1: Blue-chip Fiat-backed (An toàn nhất)
    if symbol in ['USDC', 'USDT', 'PYUSD', 'GUSD']: 
        return 1.0
    
    # Tier 2: Blue-chip Decentralized / Over-collateralized
    if symbol in ['DAI', 'LUSD', 'FRAX', 'WBTC', 'WETH', 'CBETH']: 
        return 2.5
    
    # Tier 3: New / Yield-bearing / Complex Mechanisms
    if symbol in ['SUSDS', 'USDE', 'CRVUSD', 'GHO', 'AUSD']: 
        return 5.0
    
    # Tier 4: Volatile / Low cap / Algo-stables (Mặc định)
    return 8.0

# --- 2. HÀM TÍNH TOÁN FEATURES VÀ NHÃN (CORE LOGIC) ---

def calculate_v2_features_and_label(doc: PoolsSnapshot30dV1):
    charts = doc.pool_charts_30d
    # Yêu cầu tối thiểu 15 điểm dữ liệu để tính toán thống kê
    if not charts or len(charts) < 15: 
        return None

    # Chuyển đổi dữ liệu sang Pandas DataFrame
    data = [c.model_dump() for c in charts]
    df = pd.DataFrame(data)
    
    # Fill NaN bằng 0 để tránh lỗi tính toán
    df["tvlUsd"] = df["tvlUsd"].fillna(0)
    df["apy"] = df["apy"].fillna(0)

    # --- A. FEATURE ENGINEERING ---
    
    # 1. Logarit TVL: Giúp model xử lý tốt dải giá trị rộng của TVL
    tvl_current = df["tvlUsd"].iloc[-1]
    # Tránh log(0) hoặc log số âm
    log_tvl = log10(tvl_current) if tvl_current > 1 else 0.0
    
    # 2. TVL Trend (7 ngày): Phát hiện dòng tiền rút ra/vào (Bank run signal)
    # Lấy index của 7 ngày trước, nếu không đủ thì lấy ngày đầu tiên
    idx_7d = -7 if len(df) >= 7 else 0
    tvl_7d = df["tvlUsd"].iloc[idx_7d]
    
    tvl_change_7d = 0.0
    if tvl_7d > 0:
        tvl_change_7d = (tvl_current - tvl_7d) / tvl_7d
    
    # 3. Volatility & Drawdown
    tvl_mean = df["tvlUsd"].mean()
    tvl_std = df["tvlUsd"].std()
    # Tránh chia cho 0
    tvl_volatility = tvl_std / tvl_mean if tvl_mean > 0 else 0.0
    
    rolling_max = df["tvlUsd"].cummax()
    drawdown = (df["tvlUsd"] - rolling_max) / rolling_max
    # Xử lý NaN/Inf
    drawdown = drawdown.replace([np.inf, -np.inf], 0).fillna(0)
    max_drawdown = drawdown.min() # Giá trị âm (vd: -0.2)

    # 4. APY Metrics
    apy_mean = df["apy"].mean()
    apy_std = df["apy"].std()
    
    # 5. External Scores
    chain_s = get_chain_score(doc.chain)
    token_s = get_token_risk_score(doc.symbol) 

    # --- B. LABELING (WEIGHTED FORMULA 0-10) ---
    
    # TVL Score: TVL càng cao (log lớn) -> điểm rủi ro càng thấp
    # Mốc chuẩn: $1B (log=9) -> 0 điểm. $10k (log=4) -> 10 điểm.
    tvl_risk_component = max(0, min(10, (9.0 - log_tvl) * 2.0))
    
    # Drawdown Risk: Giảm 40% (-0.4) là Max rủi ro (10đ)
    dd_risk_component = min(10, abs(max_drawdown) * 25)
    
    # Trend Risk: Nếu đang bị rút tiền mạnh (-20%) -> Phạt thêm điểm nặng
    trend_penalty = 0
    if tvl_change_7d < -0.1: trend_penalty = 1.5
    if tvl_change_7d < -0.3: trend_penalty = 4.0

    # TỔNG HỢP (Trọng số điều chỉnh)
    final_score = (
        (tvl_risk_component * 0.25) +
        (token_s * 0.20) +       # Uy tín Token chiếm 20%
        (dd_risk_component * 0.30) + # Sụt giảm quá khứ chiếm 30%
        (chain_s * 0.10) +
        (tvl_volatility * 0.10) +
        (apy_std * 0.05)
    ) + trend_penalty
    
    # Clip kết quả trong khoảng [0, 10] và làm tròn
    final_score = round(max(0.0, min(10.0, final_score)), 2)

    return {
        "chain": doc.chain,
        "project": doc.project,
        "symbol": doc.symbol,
        "pool_name": doc.pool_name,
        "window_start_time": doc.window_start_time,
        "window_end_time": doc.window_end_time,
        
        # Features mới cho model V2
        "log_tvl": round(log_tvl, 2),
        "tvl_change_7d": round(tvl_change_7d, 4),
        "tvl_volatility": round(tvl_volatility, 4),
        "max_drawdown": round(max_drawdown, 4),
        "apy_mean": round(apy_mean, 2),
        "apy_std": round(apy_std, 2),
        "chain_score": chain_s,
        "token_score": token_s,
        
        # Target Label
        "risk_score": final_score
    }

# --- 3. MAIN EXECUTION ---

async def process_data_v2():
    await mongo_client.initialize()
    
    logger.info("Cleaning V2 collections...")
    # Xóa dữ liệu cũ trước khi chạy lại
    await PoolsSnapshotTrainRegressionV2.delete_all()
    await PoolsSnapshotTestRegressionV2.delete_all()

    # --- SPLIT TRAIN/TEST POOLS ---
    logger.info("Fetching unique pools for split...")
    
    # Lấy danh sách tất cả các pool từ collection AllTime để đảm bảo đầy đủ
    all_pool_docs = await PoolsSnapshotAllTime.find_all().to_list()
    
    # Lấy danh sách tên pool duy nhất
    unique_pool_names = list(set([p.pool_name for p in all_pool_docs]))
    
    total_pools = len(unique_pool_names)
    if total_pools == 0:
        logger.warning("No pools found in PoolsSnapshotAllTime. Exiting.")
        return

    import random
    # Shuffle để chia ngẫu nhiên
    random.shuffle(unique_pool_names)
    
    # Chia 80% Train - 20% Test
    split_index = int(total_pools * 0.8)
    train_pools = set(unique_pool_names[:split_index])
    test_pools = set(unique_pool_names[split_index:])
    
    logger.info(f"Total Unique Pools: {total_pools}")
    logger.info(f"Train Pools: {len(train_pools)}")
    logger.info(f"Test Pools: {len(test_pools)}")

    # --- PROCESS DATA ---
    
    batch_train = []
    batch_test = []
    BATCH_SIZE = 1000 # Kích thước lô để insert bulk
    count = 0

    logger.info("Starting processing windows...")
    
    # Dùng cursor (find_many) để duyệt qua từng document 30d mà không load hết vào RAM
    async for doc in PoolsSnapshot30dV1.find_many():
        processed_data = calculate_v2_features_and_label(doc)
        
        if not processed_data:
            continue
            
        if doc.pool_name in train_pools:
            batch_train.append(PoolsSnapshotTrainRegressionV2(**processed_data))
        elif doc.pool_name in test_pools:
            batch_test.append(PoolsSnapshotTestRegressionV2(**processed_data))
        else:
            continue
            
        count += 1
        if count % 5000 == 0:
            logger.info(f"Processed {count} windows...")
            
        # Batch insert để tối ưu tốc độ ghi DB
        if len(batch_train) >= BATCH_SIZE:
            await PoolsSnapshotTrainRegressionV2.insert_many(batch_train)
            batch_train = []
            
        if len(batch_test) >= BATCH_SIZE:
            await PoolsSnapshotTestRegressionV2.insert_many(batch_test)
            batch_test = []

    if batch_train: 
        await PoolsSnapshotTrainRegressionV2.insert_many(batch_train)
    if batch_test: 
        await PoolsSnapshotTestRegressionV2.insert_many(batch_test)

    logger.info(f"MIGRATION V2 COMPLETED. Total processed windows: {count}")

if __name__ == "__main__":
    asyncio.run(process_data_v2())