"""
核心功能模块 - Core Module

提供向后兼容的接口和主要功能
"""

import logging
from typing import Dict, Tuple, Any, Iterable, Optional
import pandas as pd

from .data import DataFetcher
from .optimizer import (
    MaxSharpeOptimizer,
    MinVarianceOptimizer,
    RiskParityOptimizer,
    MaxDiversificationOptimizer,
    EqualWeightOptimizer,
    OptimizationStrategy,
    PortfolioOptimizerFactory,
)
from .utils import (
    get_valid_trade_range,
    format_performance_output,
    validate_price_data,
)
from .constraints import (
    SectorConstraint,
    TransactionCost,
    ConstrainedOptimizer,
    calculate_portfolio_concentration,
)


logger = logging.getLogger(__name__)


def compute_max_sharpe(prices: pd.DataFrame, rf: float = 0.02,
                      max_weight: float = 1.0) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    计算最大夏普比率投资组合 (向后兼容接口)

    Args:
        prices: 价格数据DataFrame
        rf: 无风险利率
        max_weight: 最大单一资产权重

    Returns:
        (weights, performance): 权重字典和性能指标字典
    """
    validate_price_data(prices)
    optimizer = MaxSharpeOptimizer(risk_free_rate=rf, max_weight=max_weight)
    weights, performance = optimizer.optimize(prices)
    return weights, format_performance_output(weights, performance)


def fetch_prices(tickers: Iterable[str], start_date: str, end_date: str,
                market: str = "CN", adjust: str = "hfq") -> pd.DataFrame:
    """
    获取股票价格数据 (向后兼容接口)

    Args:
        tickers: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        market: 市场类型
        adjust: 复权方式

    Returns:
        价格数据DataFrame
    """
    fetcher = DataFetcher(market=market)
    return fetcher.fetch_prices(tickers, start_date, end_date, adjust)


class PortfolioOptimizer:
    """
    投资组合优化器主类

    提供完整的投资组合优化工作流程，支持多种优化策略
    """

    def __init__(self, market: str = "CN", risk_free_rate: float = 0.02,
                 max_weight: float = 1.0, min_weight: float = 0.0,
                 strategy: str = "max_sharpe"):
        """
        初始化投资组合优化器

        Args:
            market: 市场类型（仅支持 "CN"）
            risk_free_rate: 无风险利率
            max_weight: 最大单一资产权重
            min_weight: 最小单一资产权重
            strategy: 优化策略 ("max_sharpe", "min_variance", "risk_parity",
                      "max_diversification", "equal_weight")
        """
        self.market = market
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.strategy_name = strategy

        self.data_fetcher = DataFetcher(market=market)

        # 根据策略名称创建优化器
        strategy_enum = self._get_strategy_enum(strategy)
        self.optimizer = PortfolioOptimizerFactory.create(
            strategy=strategy_enum,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            min_weight=min_weight
        )

        # 可选：行业约束
        self.sector_constraint: Optional[SectorConstraint] = None
        self.transaction_cost: Optional[TransactionCost] = None

    def _get_strategy_enum(self, strategy: str) -> OptimizationStrategy:
        """将策略名称转换为枚举"""
        strategy_map = {
            "max_sharpe": OptimizationStrategy.MAX_SHARPE,
            "min_variance": OptimizationStrategy.MIN_VARIANCE,
            "risk_parity": OptimizationStrategy.RISK_PARITY,
            "max_diversification": OptimizationStrategy.MAX_DIVERSIFICATION,
            "equal_weight": OptimizationStrategy.EQUAL_WEIGHT,
        }
        if strategy.lower() not in strategy_map:
            raise ValueError(
                f"不支持的策略: {strategy}. "
                f"可用策略: {list(strategy_map.keys())}"
            )
        return strategy_map[strategy.lower()]

    def set_sector_constraint(self, max_sector_weight: float = 0.3,
                             min_sectors: int = 3,
                             sector_mapping: Optional[Dict[str, str]] = None) -> None:
        """
        设置行业约束

        Args:
            max_sector_weight: 单一行业最大权重
            min_sectors: 最少行业数量
            sector_mapping: 自定义行业映射
        """
        self.sector_constraint = SectorConstraint(
            max_sector_weight=max_sector_weight,
            min_sectors=min_sectors,
        )
        if sector_mapping:
            for ticker, sector in sector_mapping.items():
                self.sector_constraint.add_sector_mapping(ticker, sector)

    def set_transaction_cost(self, commission_rate: float = 0.0003,
                            stamp_duty: float = 0.001,
                            slippage: float = 0.001) -> None:
        """
        设置交易成本

        Args:
            commission_rate: 佣金率
            stamp_duty: 印花税
            slippage: 滑点
        """
        self.transaction_cost = TransactionCost(
            commission_rate=commission_rate,
            stamp_duty=stamp_duty,
            slippage=slippage,
        )

    def optimize_portfolio(self, tickers: Optional[Iterable[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          years: Optional[int] = None,
                          prices: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        执行完整的投资组合优化流程

        Args:
            tickers: 股票代码列表，如果为None则使用默认股票池
            start_date: 开始日期
            end_date: 结束日期
            years: 回溯年数（与start_date/end_date互斥）
            prices: 直接提供价格数据（可选）

        Returns:
            (weights, performance): 权重字典和性能指标字典
        """
        # 如果直接提供了价格数据
        if prices is not None:
            return self._optimize_with_prices(prices)

        # 处理日期参数
        if years is not None and (start_date is not None or end_date is not None):
            raise ValueError("years参数与start_date/end_date参数不能同时使用")

        # 设置默认日期
        if years is not None:
            import datetime
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=years*365)).strftime("%Y-%m-%d")
        elif start_date is None or end_date is None:
            import datetime
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")

        # 使用默认股票池
        if tickers is None:
            from .data import get_default_tickers
            tickers = get_default_tickers(self.market)

        # 调整日期到有效交易日
        from .utils import get_exchange_for_market
        exchange = get_exchange_for_market(self.market)
        start_date, end_date = get_valid_trade_range(start_date, end_date, exchange)

        logger.info(f"获取 {self.market} 市场数据，时间范围: {start_date} 到 {end_date}")
        logger.info(f"优化策略: {self.strategy_name}")
        logger.info(f"股票代码: {list(tickers)[:5]}...")

        # 获取价格数据并验证
        fetched_prices = self.data_fetcher.fetch_prices(tickers, start_date, end_date)

        return self._optimize_with_prices(fetched_prices)

    def _optimize_with_prices(self, prices: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """使用价格数据执行优化"""
        validate_price_data(prices)
        logger.info(f"成功获取 {len(prices.columns)} 只股票的价格数据")

        # 执行优化
        if self.sector_constraint:
            # 使用带约束的优化器
            constrained_optimizer = ConstrainedOptimizer(
                base_optimizer=self.optimizer,
                sector_constraint=self.sector_constraint,
                transaction_cost=self.transaction_cost,
            )
            weights, performance = constrained_optimizer.optimize(prices)
        else:
            weights, performance = self.optimizer.optimize(prices)

        # 添加集中度指标
        concentration = calculate_portfolio_concentration(weights)
        performance['concentration_metrics'] = concentration

        logger.info(f"投资组合优化完成，夏普比率: {performance.get('sharpe_ratio', 0):.3f}")

        return weights, performance

    def compare_strategies(self, tickers: Optional[Iterable[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          years: Optional[int] = None) -> Dict[str, Tuple[Dict[str, float], Dict[str, Any]]]:
        """
        对比所有策略的结果

        Returns:
            各策略的优化结果字典
        """
        # 获取价格数据
        if years is not None:
            import datetime
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=years*365)).strftime("%Y-%m-%d")
        elif start_date is None or end_date is None:
            import datetime
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")

        if tickers is None:
            from .data import get_default_tickers
            tickers = get_default_tickers(self.market)

        from .utils import get_exchange_for_market
        exchange = get_exchange_for_market(self.market)
        start_date, end_date = get_valid_trade_range(start_date, end_date, exchange)

        prices = self.data_fetcher.fetch_prices(tickers, start_date, end_date)

        results = {}
        strategies = ["max_sharpe", "min_variance", "risk_parity", "max_diversification", "equal_weight"]

        for strategy in strategies:
            try:
                optimizer = PortfolioOptimizer(
                    market=self.market,
                    risk_free_rate=self.risk_free_rate,
                    max_weight=self.max_weight,
                    min_weight=self.min_weight,
                    strategy=strategy,
                )
                weights, performance = optimizer.optimize_portfolio(prices=prices)
                results[strategy] = (weights, performance)
                logger.info(f"策略 {strategy}: 夏普比率 {performance.get('sharpe_ratio', 0):.3f}")
            except Exception as e:
                logger.error(f"策略 {strategy} 失败: {e}")

        return results

    def save_results(self, weights: Dict[str, float], performance: Dict[str, Any],
                    prices: pd.DataFrame, output_dir: str,
                    start_date: str, end_date: str) -> None:
        """
        保存优化结果

        Args:
            weights: 投资组合权重
            performance: 性能指标
            prices: 价格数据
            output_dir: 输出目录
            start_date: 开始日期
            end_date: 结束日期
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # 保存权重
        weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        weights_file = f"{output_dir}/weights_{start_date}_{end_date}.csv"
        weights_df.to_csv(weights_file)
        logger.info(f"权重已保存至: {weights_file}")

        # 保存价格数据
        price_file = f"{output_dir}/stock_data_{start_date}_{end_date}.csv"
        prices.to_csv(price_file)
        logger.info(f"价格数据已保存至: {price_file}")

        # 保存性能指标 (需要处理不可序列化的对象)
        serializable_performance = {}
        for k, v in performance.items():
            if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                serializable_performance[k] = v
            elif hasattr(v, 'tolist'):  # numpy arrays
                serializable_performance[k] = v.tolist()
            else:
                serializable_performance[k] = str(v)

        performance_file = f"{output_dir}/performance_{start_date}_{end_date}.json"
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_performance, f, indent=2, ensure_ascii=False)
        logger.info(f"性能指标已保存至: {performance_file}")

    def update_settings(self, risk_free_rate: Optional[float] = None,
                       max_weight: Optional[float] = None,
                       min_weight: Optional[float] = None,
                       strategy: Optional[str] = None) -> None:
        """
        更新优化器设置

        Args:
            risk_free_rate: 新的无风险利率
            max_weight: 新的最大权重限制
            min_weight: 新的最小权重限制
            strategy: 新的优化策略
        """
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate

        if max_weight is not None:
            self.max_weight = max_weight

        if min_weight is not None:
            self.min_weight = min_weight

        if strategy is not None:
            self.strategy_name = strategy

        # 重新创建优化器
        strategy_enum = self._get_strategy_enum(self.strategy_name)
        self.optimizer = PortfolioOptimizerFactory.create(
            strategy=strategy_enum,
            risk_free_rate=self.risk_free_rate,
            max_weight=self.max_weight,
            min_weight=self.min_weight
        )

    @staticmethod
    def available_strategies() -> list:
        """返回可用的策略列表"""
        return ["max_sharpe", "min_variance", "risk_parity", "max_diversification", "equal_weight"]
