import asyncio
from datetime import datetime, timedelta, timezone

from clients import Clients
from configs import get_logger
from hooks.error import FailedExternalAPI, GenericServiceError
from mongo.schemas import (
    APYStatistics,
    PoolCharts,
    PoolsSnapshot120d,
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
logger = get_logger("defillama_stable_pools")


def parse_iso_datetime_naive(date_str: str) -> datetime:
    if date_str.endswith("Z"):
        date_str = date_str.replace("Z", "+00:00")
    return datetime.fromisoformat(date_str).replace(tzinfo=None)


async def get_pool_charts_120d(pool_address: str) -> list[PoolCharts]:
    url = DEFILLAMA_POOL_CHARTS_ENDPOINT.format(pool_address=pool_address)
    try:
        response = await defillama.async_get_request(url=url)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        if isinstance(response, dict) and "data" in response:
            charts_data = response["data"]  # pyright: ignore[reportUnknownVariableType]
        else:
            charts_data = response  # pyright: ignore[reportUnknownVariableType]
        # Filter data for the last 120 days
        last_120d = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=120)
        pool_charts_120d: list[PoolCharts] = []
        if isinstance(charts_data, list):
            for item in charts_data:
                if isinstance(item, dict) and "timestamp" in item:
                    timestamp_str = str(item["timestamp"])  # pyright: ignore[reportUnknownArgumentType]
                    ts = parse_iso_datetime_naive(timestamp_str)
                    item["timestamp"] = ts
                    if ts >= last_120d:
                        pool_charts_120d.append(PoolCharts.model_validate(item))
        return pool_charts_120d
    except (GenericServiceError, FailedExternalAPI) as e:
        logger.error(f"Error fetching pool charts for {pool_address}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching pool charts for {pool_address}: {e}")
        return []


async def aggregate_pools(
    # chain_id: SupportedChain | None,
    tokens: list[SupportedToken] | None,
):
    await mongo_client.initialize()
    try:
        response = await defillama.async_get_request(url=DEFILLAMA_POOLS_ENDPOINT)  # pyright: ignore[reportUnknownMemberType]
        pools = response.get("data", [])
        logger.info(f"Fetched {len(pools)} pools from DefiLlama.")
        if tokens is None:
            tokens = [SupportedToken.USDT.value, SupportedToken.USDC.value]  # pyright: ignore[reportAssignmentType]

        # chain_pools = [
        #     pool
        #     for pool in pools
        #     if pool.get("chain", "").lower() == chain_id
        #     and pool.get("stablecoin") is True
        #     and pool.get("symbol") in tokens
        # ]
        chain_pools = [
            pool
            for pool in pools
            if pool.get("stablecoin") is True
            and pool.get("symbol") in tokens
        ]
        # logger.info(f"Found {len(chain_pools)} stable pools for chain {chain_id}.")

        # tasks = [process_pool(pool) for pool in chain_pools]
        # _ = await asyncio.gather(*tasks)
        cnt = 0
        tasks = []
        for pool in chain_pools:
            tasks.append(process_pool(pool))
            cnt += 1
            if cnt == 20:
                _ = await asyncio.gather(*tasks)
                await asyncio.sleep(90)  # Throttle to avoid rate limits
                tasks = []
                cnt = 0
    except (GenericServiceError, FailedExternalAPI) as e:
        logger.error(f"Error fetching pools data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def process_pool(pool: dict):
    try:
        pool_charts_120d = await get_pool_charts_120d(pool["pool"])
        pool_predictions = Predictions.model_validate(pool.get("predictions", {}))
        pool_apy_statistics = APYStatistics(
            mu=pool.get("mu", 0), sigma=pool.get("sigma", 0), count=pool.get("count", 0)
        )
        update_time = datetime.now(timezone.utc).isoformat()
        pool_metadata = await PoolsMetadata.find_one(
            PoolsMetadata.defillama_id == pool["pool"]
        )
        pool_snapshot = PoolsSnapshot120d(
            id=hasher.get_hash(f"{pool['symbol']}-{pool['project']}-{update_time}"),
            chain=pool["chain"],
            update_at=update_time,
            project=pool["project"],
            symbol=pool["symbol"],
            pool_name=pool_metadata.final_name if pool_metadata else pool["pool"],
            predictions=pool_predictions,
            apy_statistics=pool_apy_statistics,
            pool_charts_120d=pool_charts_120d,
        )
        await pool_snapshot.save()
        logger.info(f"Saved snapshot for pool {pool['pool']}.")
    except Exception as e:
        logger.error(f"Error processing pool {pool['pool']}: {e}")


async def main():
    # chains = SupportedChain.get_all_chains()
    tokens = SupportedToken.get_all_tokens()

    # logger.info(f"Supported chains: {chains}")
    logger.info(f"Supported tokens: {tokens}")

    # tasks = [
    #     aggregate_pools(chain_id=chain, tokens=tokens)  # pyright: ignore[reportArgumentType]
    #     for chain in chains
    # ]

    # _ = await asyncio.gather(*tasks)

    _ = await aggregate_pools(
        tokens=tokens,
    )


if __name__ == "__main__":
    asyncio.run(main())
