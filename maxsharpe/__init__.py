"""
Max Sharpe Portfolio Optimizer

一个用于构建投资组合的Python工具包（仅支持中国A股）。

支持多种优化策略:
- 最大夏普比率 (Max Sharpe)
- 最小方差 (Minimum Variance)
- 风险平价 (Risk Parity)
- 最大分散化 (Maximum Diversification)
- 等权重 (Equal Weight)

功能特性:
- 多策略优化
- 行业约束
- 交易成本计算
- 回测分析
"""

__version__ = "2.0.0"
__author__ = "Henry Wen"
__email__ = "henrywen98@example.com"

# 核心优化器
from .optimizer import (
    MaxSharpeOptimizer,
    MinVarianceOptimizer,
    RiskParityOptimizer,
    MaxDiversificationOptimizer,
    EqualWeightOptimizer,
    BaseOptimizer,
    OptimizationStrategy,
    PortfolioOptimizerFactory,
)

# 数据获取
from .data import DataFetcher, get_default_tickers

# 工具函数
from .utils import get_valid_trade_range, calculate_returns, validate_price_data

# 向后兼容接口
from .core import compute_max_sharpe, fetch_prices, PortfolioOptimizer

# 约束模块
from .constraints import (
    SectorConstraint,
    TransactionCost,
    ConstrainedOptimizer,
    Sector,
    calculate_portfolio_concentration,
    suggest_rebalance,
)

# 回测模块
from .backtest import (
    Backtester,
    BacktestConfig,
    BacktestResult,
    generate_backtest_report,
)

__all__ = [
    # 版本信息
    "__version__",
    "__author__",

    # 核心优化器
    "MaxSharpeOptimizer",
    "MinVarianceOptimizer",
    "RiskParityOptimizer",
    "MaxDiversificationOptimizer",
    "EqualWeightOptimizer",
    "BaseOptimizer",
    "OptimizationStrategy",
    "PortfolioOptimizerFactory",

    # 数据
    "DataFetcher",
    "get_default_tickers",

    # 工具
    "get_valid_trade_range",
    "calculate_returns",
    "validate_price_data",

    # 高级接口
    "compute_max_sharpe",
    "fetch_prices",
    "PortfolioOptimizer",

    # 约束
    "SectorConstraint",
    "TransactionCost",
    "ConstrainedOptimizer",
    "Sector",
    "calculate_portfolio_concentration",
    "suggest_rebalance",

    # 回测
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "generate_backtest_report",
]
