from enum import Enum


class SupportedChain(Enum):
    SOL = "solana"
    APT = "aptos"
    ETH = "ethereum"
    BSC = "bsc"
    BASE = "base"
    ARB = "arbitrum"
    SUI = "sui"
    TON = "ton"
    OPT = "optimism"
    POLY = "polygon"
    AVAX = "avalanche"

    @staticmethod
    def get_all_chains():
        return list(map(lambda c: c.value, SupportedChain))


class SupportedToken(Enum):
    USDT = "USDT"
    USDC = "USDC"

    @staticmethod
    def get_all_tokens():
        return list(map(lambda t: t.value, SupportedToken))


class PoolsAlert(Enum):
    SEVERE_TVL_DROP = "severe_tvl_drop"
    HIGH_TVL_VOLATILITY = "high_tvl_volatility"
    HIGH_APY_VOLATILITY = "high_apy_volatility"
    HIGH_UTILIZATION_RATE = "high_utilization_rate"

    @staticmethod
    def get_all_alerts():
        return list(map(lambda a: a.value, PoolsAlert))
