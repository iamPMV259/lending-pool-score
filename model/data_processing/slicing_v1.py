import asyncio

import numpy as np
import pandas as pd

from clients import Clients
from configs import get_logger
from mongo.schemas import PoolCharts, PoolsSnapshot30dV1, PoolsSnapshotAllTime

mongo_client = Clients.get_mongo_client()
logger = get_logger(__name__)




async def process_and_store_data(raw_data: PoolsSnapshotAllTime):
    original_pool_charts_all_time = raw_data.pool_charts_all_time

    WINDOW_SIZE = 30
    STEP = 7

    if len(original_pool_charts_all_time) < WINDOW_SIZE:
        return

    snapshots_to_insert = []

    for i in range(0, len(original_pool_charts_all_time) - WINDOW_SIZE + 1, STEP):
        chunk = original_pool_charts_all_time[i : i + WINDOW_SIZE]

        slice_end_time = chunk[-1].timestamp
        slice_start_time = chunk[0].timestamp

        new_snapshot = PoolsSnapshot30dV1(
            chain=raw_data.chain,
            project=raw_data.project,
            symbol=raw_data.symbol,
            pool_name=raw_data.pool_name,
            window_start_time=slice_start_time,
            window_end_time=slice_end_time,
            pool_charts_30d=chunk,
        )
        snapshots_to_insert.append(new_snapshot)

    if snapshots_to_insert:
        _ = await PoolsSnapshot30dV1.insert_many(snapshots_to_insert)
        logger.info(
            f"Inserted {len(snapshots_to_insert)} snapshots for pool {raw_data.pool_name}."
        )




async def process_all_data():
    await mongo_client.initialize()
    raw_data_list = await PoolsSnapshotAllTime.find_many().to_list()
    logger.info(f"Processing {len(raw_data_list)} records from PoolsSnapshotAllTime.")

    tasks = [process_and_store_data(raw_data) for raw_data in raw_data_list]
    _ = await asyncio.gather(*tasks)

    logger.info("Data processing and storage completed.")


if __name__ == "__main__":
    asyncio.run(process_all_data())