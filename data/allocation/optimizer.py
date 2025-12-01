from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel


class PoolCandidate(BaseModel):
    pool_name: str
    risk_label: str
    confidence: float
    apy: float
    volatility: float


class AllocationResult(BaseModel):
    pool_name: str
    weight: float
    amount: float
    expected_apy: float


class PortfolioOptimizer:
    def __init__(self, total_capital: float = 1000.0, max_weight_per_pool: float = 0.3):
        self.total_capital = total_capital
        self.max_weight = max_weight_per_pool

    def optimize(self, vault_type: Literal["Conservative", "Balanced", "Aggressive"], pools: list[PoolCandidate]) -> list[AllocationResult]:
        
        candidates = self._filter_pools(vault_type, pools)


        if not candidates:
            return []

        scored_pools = []
        total_score = 0.0

        for p in candidates:
            profile_multiplier = 1.0
            if vault_type == "Balanced" and p.risk_label == "Balanced":
                profile_multiplier = 1.2

            score = (p.apy * p.confidence * profile_multiplier) / max(p.volatility, 0.05)

            scored_pools.append({
                "candidate": p,
                "score": score
            })
            total_score += score

        allocations = []
        current_total_weight = 0.0

        scored_pools.sort(key=lambda x: x["score"], reverse=True)

        for item in scored_pools:
            p = item["candidate"]

            raw_weight = item["score"] / total_score
            final_weight = min(raw_weight, self.max_weight)

            if current_total_weight + final_weight > 1.0:
                final_weight = 1.0 - current_total_weight

            if final_weight <= 0.01:
                continue

            current_total_weight += final_weight

            allocations.append(
                AllocationResult(
                    pool_name=p.pool_name,
                    weight=round(final_weight, 4),
                    amount=round(final_weight * self.total_capital, 2),
                    expected_apy=p.apy
                )
            )
            if current_total_weight >= 0.99:
                break
        return allocations

    def _filter_pools(self, vault_type: Literal["Conservative", "Balanced", "Aggressive"], pools: list[PoolCandidate]) -> list[PoolCandidate]:
        valid_pools = []
        for p in pools:
            if vault_type == "Conservative":
                if p.risk_label == "Conservative":
                    valid_pools.append(p)
            elif vault_type == "Balanced":
                if p.risk_label in ["Conservative", "Balanced"]:
                    valid_pools.append(p)
            else:  # Aggressive
                if p.apy > 5.0:
                    valid_pools.append(p)
        return valid_pools