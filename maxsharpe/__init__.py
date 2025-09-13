"""
Max Sharpe Portfolio Optimizer

一个用于构建最大夏普比率投资组合的Python工具包，支持中国A股和美股市场。
"""

__version__ = "1.0.0"
__author__ = "Henry Wen"
__email__ = "henrywen98@example.com"

from .optimizer import MaxSharpeOptimizer
from .data import DataFetcher
from .utils import get_valid_trade_range

# 为了向后兼容，保留原有的函数接口
from .core import compute_max_sharpe, PortfolioOptimizer

__all__ = [
    "MaxSharpeOptimizer",
    "DataFetcher",
    "get_valid_trade_range",
    "compute_max_sharpe",
    "PortfolioOptimizer",
]
