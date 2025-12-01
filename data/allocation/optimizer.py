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
    def __init__(self, total_capital: float = 1000.0, max_weight_per_pool: float = 0.4):
        self.total_capital = total_capital
        self.default_max_weight = max_weight_per_pool

    def optimize(self, vault_type: Literal["Conservative", "Balanced", "Aggressive"], pools: list[PoolCandidate]) -> list[AllocationResult]:
        
        candidates = self._filter_pools(vault_type, pools)

        if not candidates:
            return []

        scored_pools = []
        
        for p in candidates:
            profile_multiplier = 1.0
            if vault_type == "Balanced" and p.risk_label == "Balanced":
                profile_multiplier = 1.2

            score = (p.apy * p.confidence * profile_multiplier) / max(p.volatility, 0.05)

            scored_pools.append({
                "candidate": p,
                "score": score,
                "weight": 0.0  
            })

 
        min_needed_weight = 1.0 / len(scored_pools)
        effective_cap = max(self.default_max_weight, min_needed_weight)
        
        total_score = sum(item["score"] for item in scored_pools)
        if total_score == 0:
            return []
            
        for item in scored_pools:
            item["weight"] = item["score"] / total_score

        for _ in range(3):
            excess_weight = 0.0
            
            for item in scored_pools:
                if item["weight"] > effective_cap:
                    excess_weight += item["weight"] - effective_cap
                    item["weight"] = effective_cap
            
            if excess_weight < 0.0001:
                break
            
            uncapped_items = [i for i in scored_pools if i["weight"] < effective_cap]
            
            if not uncapped_items:
                break
            
            total_uncapped_score = sum(i["score"] for i in uncapped_items)
            
            if total_uncapped_score == 0:
                split_val = excess_weight / len(uncapped_items)
                for item in uncapped_items:
                    item["weight"] += split_val
            else:
                for item in uncapped_items:
                    item["weight"] += (item["score"] / total_uncapped_score) * excess_weight

        allocations = []
        
        scored_pools.sort(key=lambda x: x["weight"], reverse=True)

        for item in scored_pools:
            weight = round(item["weight"], 4)
            
            if weight <= 0.001:
                continue

            p = item["candidate"]
            allocations.append(
                AllocationResult(
                    pool_name=p.pool_name,
                    weight=weight,
                    amount=round(weight * self.total_capital, 2),
                    expected_apy=p.apy
                )
            )

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