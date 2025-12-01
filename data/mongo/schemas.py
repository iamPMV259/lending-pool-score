from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from beanie import Document, Link
from pydantic import BaseModel, Field


class Predictions(BaseModel):
    predictedClass: str | None
    predictedProbability: float | None
    binnedConfidence: float | None


class APYStatistics(BaseModel):
    mu: float | None
    sigma: float | None
    count: int | None


class PoolCharts(BaseModel):
    timestamp: datetime
    tvlUsd: float | None
    apy: float | None







class PoolsSnapshot120d(Document):
    id: UUID
    chain: str
    update_at: datetime
    project: str
    symbol: str
    pool_name: str
    predictions: Predictions
    apy_statistics: APYStatistics
    pool_charts_120d: list[PoolCharts]

    class Settings:
        name = "pools_snapshot_v0"
        validate_on_save = True


class PoolsSnapshot30d(Document):
    id: UUID = Field(default_factory=uuid4)
    chain: str
    window_start_time: datetime
    window_end_time: datetime | None = None
    project: str
    symbol: str
    pool_name: str
    pool_charts_30d: list[PoolCharts]
    
    class Settings:
        name = "pools_snapshot_v1"
        validate_on_save = True


class PoolsSnapshotTrain(Document):
    id: UUID = Field(default_factory=uuid4)
    chain: str
    project: str
    symbol: str
    pool_name: str
    window_start_time: datetime
    window_end_time: datetime | None = None
    tvl_current: float | None = None
    tvl_mean: float | None = None
    tvl_volatility: float | None = None
    max_drawdown: float | None = None

    apy_mean: float | None = None
    apy_std: float | None = None
    chain_score: float | None = None

    label: Literal["Conservative", "Balanced", "Aggressive"] | None = None

    class Settings:
        name = "pools_snapshot_train_v0"
        validate_on_save = True


class PoolsSnapshotTest(Document):
    id: UUID = Field(default_factory=uuid4)
    chain: str
    project: str
    symbol: str
    pool_name: str
    window_start_time: datetime
    window_end_time: datetime | None = None
    tvl_current: float | None = None
    tvl_mean: float | None = None
    tvl_volatility: float | None = None
    max_drawdown: float | None = None

    apy_mean: float | None = None
    apy_std: float | None = None
    chain_score: float | None = None

    label: Literal["Conservative", "Balanced", "Aggressive"] | None = None

    class Settings:
        name = "pools_snapshot_test_v0"
        validate_on_save = True


DocumentModels = [
    PoolsSnapshot120d,
    PoolsSnapshot30d,
    PoolsSnapshotTrain,
    PoolsSnapshotTest,
]
