"""
投资组合优化器模块 - Portfolio Optimizer Module

包含投资组合优化的核心算法
"""

import logging
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np

from .utils import validate_price_data, calculate_returns, format_performance_output


logger = logging.getLogger(__name__)


class MaxSharpeOptimizer:
    """最大夏普比率投资组合优化器"""
    
    def __init__(self, risk_free_rate: float = 0.02, max_weight: float = 1.0):
        """
        初始化优化器
        
        Args:
            risk_free_rate: 无风险利率 (年化)
            max_weight: 单一资产最大权重限制
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        
        if not 0 <= max_weight <= 1:
            raise ValueError("最大权重应在0和1之间")
        if risk_free_rate < 0:
            logger.warning("无风险利率为负值，这在某些经济环境下是可能的")
    
    def optimize(self, prices: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        执行投资组合优化
        
        Args:
            prices: 价格数据DataFrame
            
        Returns:
            (weights, performance): 权重字典和性能指标字典
        """
        # 验证输入数据
        validate_price_data(prices)
        
        # 计算收益率
        returns = calculate_returns(prices)
        
        if returns.empty:
            raise ValueError("无法计算收益率")
        
        # 执行优化
        weights, performance = self._compute_max_sharpe(returns)
        
        return weights, performance
    
    def _compute_max_sharpe(self, returns: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        计算最大夏普比率投资组合
        
        Args:
            returns: 收益率DataFrame
            
        Returns:
            (weights, performance): 权重字典和性能指标字典
        """
        try:
            from pypfopt.efficient_frontier import EfficientFrontier
        except ImportError as e:
            raise ImportError("需要安装 PyPortfolioOpt：pip install PyPortfolioOpt") from e
        
        # 额外验证返回数据
        if returns.isnull().any().any():
            raise ValueError("收益率数据中包含NaN值")
        
        if (returns == float('inf')).any().any() or (returns == -float('inf')).any().any():
            raise ValueError("收益率数据中包含无穷大值")
            
        if len(returns) < 30:
            raise ValueError("收益率数据点不足，至少需要30个数据点")
        
        # 使用自定义的方法计算预期收益率，避免PyPortfolioOpt的NaN问题
        import numpy as np
        
        # 计算年化预期收益率 (使用简单均值方法)
        daily_returns = returns.mean()
        mu = daily_returns * 252  # 年化
        
        # 再次验证预期收益率
        if mu.isnull().any():
            # 如果仍有NaN，用0替换
            logger.warning("预期收益率计算包含NaN值，使用0替换")
            mu = mu.fillna(0)
        
        # 计算协方差矩阵 (使用更稳健的方法)
        try:
            # 使用pandas的cov方法，更稳定
            S = returns.cov() * 252  # 年化协方差矩阵
            
            # 验证协方差矩阵
            if pd.DataFrame(S).isnull().any().any():
                raise ValueError("协方差矩阵包含NaN值")
                
            # 检查协方差矩阵是否为正定
            eigenvals = np.linalg.eigvals(S.values)
            if (eigenvals <= 0).any():
                logger.warning("协方差矩阵不是正定的，进行正则化")
                # 添加一个小的正则化项到对角线
                S = S + np.eye(len(S)) * 1e-6
                
        except Exception as e:
            logger.error(f"协方差矩阵计算失败: {e}")
            # 使用对角协方差矩阵作为备选
            variances = returns.var() * 252
            S = pd.DataFrame(np.diag(variances), index=returns.columns, columns=returns.columns)
        
        # 创建有效前沿
        ef = EfficientFrontier(mu, S)
        
        # 添加权重约束
        if self.max_weight < 1.0:
            ef.add_constraint(lambda w: w <= self.max_weight)
        
        # 优化最大夏普比率
        try:
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned_weights = ef.clean_weights()
        except Exception as e:
            logger.error(f"优化失败: {e}")
            # 如果优化失败，使用等权重作为后备方案
            logger.warning("使用等权重作为后备方案")
            n_assets = len(returns.columns)
            equal_weight = 1.0 / n_assets
            cleaned_weights = {asset: equal_weight for asset in returns.columns}
        
        # 计算投资组合性能
        performance = self._calculate_performance(returns, cleaned_weights)
        
        return cleaned_weights, performance
    
    def _calculate_performance(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        计算投资组合性能指标
        
        Args:
            returns: 收益率DataFrame
            weights: 权重字典
            
        Returns:
            性能指标字典
        """
        # 转换权重为Series
        weight_series = pd.Series(weights)
        
        # 对齐权重和收益率
        aligned_weights = weight_series.reindex(returns.columns, fill_value=0)
        
        # 计算投资组合收益率
        portfolio_returns = (returns * aligned_weights).sum(axis=1)
        
        # 计算年化指标
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 计算夏普比率
        if annual_volatility > 0:
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        else:
            sharpe_ratio = 0.0
        
        # 计算最大回撤
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 计算VaR (5%分位数)
        var_5 = np.percentile(portfolio_returns, 5)
        
        return {
            'expected_annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_5_percent': var_5,
            'total_return': cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0,
        }
    
    def set_risk_free_rate(self, rate: float) -> None:
        """设置无风险利率"""
        self.risk_free_rate = rate
    
    def set_max_weight(self, weight: float) -> None:
        """设置最大权重限制"""
        if not 0 <= weight <= 1:
            raise ValueError("最大权重应在0和1之间")
        self.max_weight = weight
