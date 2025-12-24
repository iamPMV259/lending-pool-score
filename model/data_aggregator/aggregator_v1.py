import asyncio
import json
from datetime import datetime, timedelta, timezone

from clients import Clients
from configs import get_logger
from hooks.error import FailedExternalAPI, GenericServiceError
from mongo.schemas import (
    APYStatistics,
    PoolCharts,
    PoolsSnapshotAllTime,
    Predictions,
)
from utils import hasher
from utils.endpoints import (
    DEFILLAMA_POOL_CHARTS_ENDPOINT,
    DEFILLAMA_POOLS_ENDPOINT,
)
from utils.enums import SupportedChain, SupportedToken

defillama = Clients.get_service_client().get_defillama_client()
mongo_client = Clients.get_mongo_client()
logger = get_logger("defillama_stable_pools_v1")


def parse_iso_datetime_naive(date_str: str) -> datetime:
    if date_str.endswith("Z"):
        date_str = date_str.replace("Z", "+00:00")
    return datetime.fromisoformat(date_str).replace(tzinfo=None)


async def get_pool_charts_all_time(pool_address: str) -> list[PoolCharts]:
    url = DEFILLAMA_POOL_CHARTS_ENDPOINT.format(pool_address=pool_address)
    try:
        response = await defillama.async_get_request(url=url)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        if isinstance(response, dict) and "data" in response:
            charts_data = response["data"]  # pyright: ignore[reportUnknownVariableType]
        else:
            charts_data = response  # pyright: ignore[reportUnknownVariableType]
        # Filter data for the all times
        pool_charts_all_time: list[PoolCharts] = []
        if isinstance(charts_data, list):
            for item in charts_data:
                if isinstance(item, dict):
                    pool_charts_all_time.append(PoolCharts.model_validate(item))
        return pool_charts_all_time
    except (GenericServiceError, FailedExternalAPI) as e:
        logger.error(f"Error fetching pool charts for {pool_address}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching pool charts for {pool_address}: {e}")
        return []


async def aggregate_pools():
    await mongo_client.initialize()
    try:
        response = await defillama.async_get_request(url=DEFILLAMA_POOLS_ENDPOINT)  # pyright: ignore[reportUnknownMemberType]
        pools = response.get("data", [])
        logger.info(f"Fetched {len(pools)} pools from DefiLlama.")

     
        chain_pools = [
            pool
            for pool in pools
            if pool.get("stablecoin") is True
        ]

        logger.info(f"Filtered {len(chain_pools)} stable pools.")

        # json.dump(chain_pools, open("stable_pools.json", "w"), default=str, indent=4)
        # logger.info(f"Saved {len(chain_pools)} stable pools to stable_pools.json.")

        pool_snapshot_db = await PoolsSnapshotAllTime.find_all().to_list()

        print(f"Loaded {len(pool_snapshot_db)} existing pool snapshots from DB.")
        existing_pool_names = {pool.pool_name for pool in pool_snapshot_db}

        chain_pools_to_process = [pool for pool in chain_pools if pool["pool"] not in existing_pool_names]
        logger.info(f"{len(chain_pools_to_process)} new pools to process.")
        
        cnt = 0
        tasks = []
        for pool in chain_pools_to_process:
            tasks.append(process_pool(pool))
            cnt += 1
            if cnt == 20:
                _ = await asyncio.gather(*tasks)
                await asyncio.sleep(60)  # Throttle to avoid rate limits
                tasks = []
                cnt = 0
        if len(tasks) > 0:
            _ = await asyncio.gather(*tasks)
    except (GenericServiceError, FailedExternalAPI) as e:
        logger.error(f"Error fetching pools data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def process_pool(pool: dict):
    try:
        pool_charts_all_time = await get_pool_charts_all_time(pool["pool"])
        update_time = datetime.now(timezone.utc).isoformat()
        pool_snapshot = PoolsSnapshotAllTime(
            id=hasher.get_hash(f"{pool['symbol']}-{pool['project']}-{update_time}"),
            chain=pool["chain"],
            update_at=update_time,
            project=pool["project"],
            symbol=pool["symbol"],
            pool_name=pool["pool"],
            pool_charts_all_time=pool_charts_all_time,
        )
        await pool_snapshot.save()
        logger.info(f"Saved snapshot for pool {pool['pool']}.")
    except Exception as e:
        logger.error(f"Error processing pool {pool['pool']}: {e}")


async def main():
    _ = await aggregate_pools()


if __name__ == "__main__":
    asyncio.run(main())
