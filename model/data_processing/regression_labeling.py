import asyncio
import math
from math import log10
from typing import Literal

import numpy as np
import pandas as pd

from clients import Clients
from configs import get_logger
from mongo.schemas import (
    PoolsSnapshot30dV1,
    PoolsSnapshotAllTime,
    PoolsSnapshotTestRegression,
    PoolsSnapshotTrainRegression,
)

mongo_client = Clients.get_mongo_client()
logger = get_logger(__name__)


def score_chain_risk(chain_name: str) -> float:
    """
    Chấm điểm rủi ro hạ tầng Blockchain (0-10).
    """
    chain = chain_name.lower()
    if chain == "ethereum": return 1.0
    if chain in ["arbitrum", "optimism", "base", "polygon"]: return 3.0
    if chain in ["solana", "bsc", "avalanche"]: return 5.0
    if chain in ["fantom", "cronos", "moonriver"]: return 7.0
    return 9.0 

def score_tvl_risk(tvl_current: float) -> float:
    """
    Chấm điểm quy mô TVL bằng Logarit.
    - TVL >= 500M -> 0.5 điểm (Rất an toàn)
    - TVL <= 100k -> 10.0 điểm (Rất rủi ro)
    """
    if tvl_current <= 0: return 10.0
    
    cap = 500_000_000 
    floor = 100_000  
    
    if tvl_current >= cap: return 0.5
    if tvl_current <= floor: return 10.0
    
    log_tvl = log10(tvl_current)
    log_cap = log10(cap)
    log_floor = log10(floor)
    score = 10.0 * (log_cap - log_tvl) / (log_cap - log_floor)
    return round(score, 2)

def score_drawdown_risk(max_drawdown: float) -> float:
    """
    Chấm điểm sụt giảm.
    - Drawdown 0% -> 0 điểm
    - Drawdown -50% (mất nửa TVL) -> 10 điểm
    """
    drawdown_abs = abs(max_drawdown)
    
    if drawdown_abs <= 0.01: return 0.0 # < 1% coi như không sụt giảm
    if drawdown_abs >= 0.5: return 10.0 # > 50% coi như rủi ro max
    
    score = (drawdown_abs / 0.5) * 10.0
    return round(score, 2)

def score_volatility_risk(volatility: float) -> float:
    """
    Chấm điểm biến động (Std/Mean).
    - Vol < 1% -> 0.5 điểm
    - Vol > 20% -> 10 điểm
    """
    if volatility <= 0.01: return 0.5
    if volatility >= 0.2: return 10.0
    
    score = (volatility / 0.2) * 10.0
    return round(score, 2)



async def calculating_and_labeling_data(pool_snapshot: PoolsSnapshot30dV1, tag: Literal["Train", "Test"]):
    charts = pool_snapshot.pool_charts_30d

    if not charts or len(charts) < 15:
        return

    data_dicts = [chart.model_dump() for chart in charts]
    df = pd.DataFrame(data_dicts)
    
    df["tvlUsd"] = df["tvlUsd"].fillna(0)
    tvl_current = df["tvlUsd"].iloc[-1]
    tvl_mean = df["tvlUsd"].mean()
    tvl_std = df["tvlUsd"].std()
    
    tvl_volatility = tvl_std / tvl_mean if tvl_mean > 100 else 0


    rolling_max = df["tvlUsd"].cummax()
    
    drawdown = np.where(rolling_max > 0, (df['tvlUsd'] - rolling_max) / rolling_max, 0)
    max_drawdown = drawdown.min()


    df["apy"] = df["apy"].fillna(0)
    apy_mean = df["apy"].mean()
    apy_std = df["apy"].std()


    s_chain = score_chain_risk(pool_snapshot.chain)
    s_tvl = score_tvl_risk(tvl_current)
    s_drawdown = score_drawdown_risk(max_drawdown)
    s_vol = score_volatility_risk(tvl_volatility)


    W_TVL = 0.35      
    W_DD = 0.30      
    W_VOL = 0.20    
    W_CHAIN = 0.15   

    risk_score = (s_chain * W_CHAIN) + \
                 (s_tvl * W_TVL) + \
                 (s_drawdown * W_DD) + \
                 (s_vol * W_VOL)

    
    final_risk_score = round(risk_score, 2)

    statistics_data = {
        "chain": pool_snapshot.chain,
        "project": pool_snapshot.project,
        "symbol": pool_snapshot.symbol,
        "pool_name": pool_snapshot.pool_name,
        "window_start_time": pool_snapshot.window_start_time,
        "window_end_time": pool_snapshot.window_end_time,
        
        "tvl_current": tvl_current,
        "tvl_mean": tvl_mean,
        "tvl_volatility": tvl_volatility,
        "max_drawdown": max_drawdown,
        "apy_mean": apy_mean,
        "apy_std": apy_std,
        "chain_score": s_chain, 
        
        "risk_score": final_risk_score
    }

    try:
        if tag == "Train":
            new_doc = PoolsSnapshotTrainRegression(**statistics_data)
            await new_doc.insert()
        else:
            new_doc = PoolsSnapshotTestRegression(**statistics_data)
            await new_doc.insert()
    except Exception as e:
        logger.error(f"Error inserting {pool_snapshot.pool_name}: {e}")


async def process_all_data():
    await mongo_client.initialize()


    all_pool = await PoolsSnapshotAllTime.find_many().to_list()
    total_pools = len(all_pool)
    logger.info(f"Total pools in source: {total_pools}")
    
    if total_pools == 0:
        return
        
    np.random.shuffle(all_pool)

    training_percentage = 0.8
    split_index = int(total_pools * training_percentage)
    train_list = all_pool[:split_index]
    test_list = all_pool[split_index:]

    logger.info(f"Train size: {len(train_list)}, Test size: {len(test_list)}")

    BATCH_SIZE = 50

    async def run_batch(pool_list, tag):
        tasks = []
        for pool in pool_list:
            snapshots = await PoolsSnapshot30dV1.find(
                PoolsSnapshot30dV1.pool_name == pool.pool_name
            ).to_list()
            
            for snap in snapshots:
                tasks.append(calculating_and_labeling_data(snap, tag))
                
            if len(tasks) >= BATCH_SIZE:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)

    logger.info("Processing Train data...")
    await run_batch(train_list, "Train")
    
    logger.info("Processing Test data...")
    await run_batch(test_list, "Test")
    
    logger.info("Done processing all data!")

if __name__ == "__main__":
    asyncio.run(process_all_data())