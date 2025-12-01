import asyncio
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from clients import Clients
from configs import get_logger
from mongo.schemas import (
    PoolsSnapshot30d,
    PoolsSnapshot120d,
    PoolsSnapshotTest,
    PoolsSnapshotTrain,
)

mongo_client = Clients.get_mongo_client()
logger = get_logger(__name__)



def get_chain_score(chain_name: str) -> float:
    chain_name = chain_name.lower()
    if chain_name == "ethereum":
        return 0.0
    if chain_name in ["arbitrum", "optimism", "base", "polygon"]:
        return 10.0
    if chain_name in ["solana", "bsc", "avalanche"]:
        return 20.0
    return 40.0






async def calculating_and_labeling_data(pool_snapshot: PoolsSnapshot30d, tag: Literal["Train", "Test"]):
    charts = pool_snapshot.pool_charts_30d

    if not charts or len(charts) < 2:
        logger.warning(f"Insufficient data in pool {pool_snapshot.pool_name}. Skipping.")
        return
    data_dicts = [chart.model_dump() for chart in charts]
    df = pd.DataFrame(data_dicts)
    df["tvlUsd"] = df["tvlUsd"].fillna(0)
    tvl_current = df["tvlUsd"].iloc[-1]
    tvl_mean = df["tvlUsd"].mean()
    tvl_std = df["tvlUsd"].std()
    tvl_volatility = tvl_std / tvl_mean if tvl_mean > 0 else 0

    rolling_max = df["tvlUsd"].cummax()
    drawdown = np.where(rolling_max > 0, (df['tvlUsd'] - rolling_max) / rolling_max, 0)
    max_drawdown = drawdown.min()


    df["apy"] = df["apy"].fillna(0)
    apy_mean = df["apy"].mean()
    apy_std = df["apy"].std()

    chain_score = get_chain_score(pool_snapshot.chain)

    risk_score = 0.0
    risk_score += chain_score

    if tvl_current < 1_000_000:
        risk_score += 50
    elif tvl_current < 5_000_000:
        risk_score += 20

    if max_drawdown < -0.3:
        risk_score += 50
    elif max_drawdown < -0.1:
        risk_score += 20


    if apy_std > 5.0:
        risk_score += 30

    label = "Balanced"
    if risk_score <= 25.0:
        label = "Conservative"
    elif risk_score >= 65.0:
        label = "Aggressive"

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
        "chain_score": chain_score,
        "label": label,
    }
    try:
        if tag == "Train":
            new_doc = PoolsSnapshotTrain(**statistics_data)
            await new_doc.insert()
        else:
            new_doc = PoolsSnapshotTest(**statistics_data)
            await new_doc.insert()
        logger.info(f"Inserted statistics for pool {pool_snapshot.pool_name} into {tag} dataset.")
    except Exception as e:
        logger.error(f"Error inserting statistics for pool {pool_snapshot.pool_name}: {e}")




async def process_all_data():
    await mongo_client.initialize()

    all_pool = await PoolsSnapshot120d.find_many().to_list()
    total_pools = len(all_pool)
    logger.info(f"Total pools in PoolsSnapshot120d: {total_pools}")
    if total_pools == 0:
        logger.warning("No data found in PoolsSnapshot120d. Exiting processing.")
        return
    np.random.shuffle(all_pool)

    training_percentage = 0.8
    split_index = int(total_pools * training_percentage)
    train_list = all_pool[:split_index]
    test_list = all_pool[split_index:]

    logger.info(f"Training set size: {len(train_list)}, Test set size: {len(test_list)}")

    tasks = []
    for pool in train_list:
        pool_30d_snapshot = await PoolsSnapshot30d.find(
            PoolsSnapshot30d.pool_name == pool.pool_name
        ).to_list()

        tasks.extend(
            calculating_and_labeling_data(pool_snapshot=snapshot, tag="Train")
            for snapshot in pool_30d_snapshot
        )

    for pool in test_list:
        pool_30d_snapshot = await PoolsSnapshot30d.find(
            PoolsSnapshot30d.pool_name == pool.pool_name
        ).to_list()
        tasks.extend(
            calculating_and_labeling_data(pool_snapshot=snapshot, tag="Test")
            for snapshot in pool_30d_snapshot
        )
    _ = await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(process_all_data())