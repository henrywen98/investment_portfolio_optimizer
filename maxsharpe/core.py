"""
核心功能模块 - Core Module

提供向后兼容的接口和主要功能
"""

import logging
from typing import Dict, Tuple, Any, Iterable, Optional
import pandas as pd

from .data import DataFetcher
from .optimizer import MaxSharpeOptimizer
from .utils import (
    get_valid_trade_range,
    format_performance_output,
    validate_price_data,
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
    
    提供完整的投资组合优化工作流程
    """
    
    def __init__(self, market: str = "CN", risk_free_rate: float = 0.02, 
                 max_weight: float = 1.0):
        """
        初始化投资组合优化器
        
        Args:
            market: 市场类型（仅支持 "CN"）
            risk_free_rate: 无风险利率
            max_weight: 最大单一资产权重
        """
        self.market = market
        self.data_fetcher = DataFetcher(market=market)
        self.optimizer = MaxSharpeOptimizer(
            risk_free_rate=risk_free_rate, 
            max_weight=max_weight
        )
        
    def optimize_portfolio(self, tickers: Optional[Iterable[str]] = None,
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None,
                          years: Optional[int] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        执行完整的投资组合优化流程
        
        Args:
            tickers: 股票代码列表，如果为None则使用默认股票池
            start_date: 开始日期
            end_date: 结束日期  
            years: 回溯年数（与start_date/end_date互斥）
            
        Returns:
            (weights, performance): 权重字典和性能指标字典
        """
        # 处理日期参数
        if years is not None and (start_date is not None or end_date is not None):
            raise ValueError("years参数与start_date/end_date参数不能同时使用")
        
        # 设置默认日期
        if years is not None:
            import datetime
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=years*365)).strftime("%Y-%m-%d")
        elif start_date is None or end_date is None:
            # 默认使用最近5年
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
        logger.info(f"股票代码: {list(tickers)}")
        
        # 获取价格数据并验证
        prices = self.data_fetcher.fetch_prices(tickers, start_date, end_date)
        validate_price_data(prices)

        logger.info(f"成功获取 {len(prices.columns)} 只股票的价格数据")

        # 执行优化
        weights, performance = self.optimizer.optimize(prices)
        
        logger.info(f"投资组合优化完成，夏普比率: {performance.get('sharpe_ratio', 0):.3f}")
        
        return weights, performance
    
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
        
        # 保存性能指标
        performance_file = f"{output_dir}/performance_{start_date}_{end_date}.json"
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(performance, f, indent=2, ensure_ascii=False)
        logger.info(f"性能指标已保存至: {performance_file}")
    
    def update_settings(self, risk_free_rate: Optional[float] = None,
                       max_weight: Optional[float] = None) -> None:
        """
        更新优化器设置
        
        Args:
            risk_free_rate: 新的无风险利率
            max_weight: 新的最大权重限制
        """
        if risk_free_rate is not None:
            self.optimizer.set_risk_free_rate(risk_free_rate)
        
        if max_weight is not None:
            self.optimizer.set_max_weight(max_weight)
