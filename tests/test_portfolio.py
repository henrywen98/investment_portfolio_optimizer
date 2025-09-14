import math
import pandas as pd
import numpy as np
import pytest
import tempfile
import os
import warnings
from unittest.mock import patch, MagicMock

from portfolio import compute_max_sharpe, get_valid_trade_range, save_outputs

# 尝试导入新模块进行测试
try:
    from maxsharpe import MaxSharpeOptimizer, DataFetcher
    from maxsharpe.utils import validate_price_data, calculate_returns
    HAS_NEW_MODULES = True
except ImportError:
    HAS_NEW_MODULES = False


def make_prices(days=200, assets=3, seed=42):
    """生成测试用的价格数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=days)
    rets = rng.normal(loc=0.0005, scale=0.01, size=(days, assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"STOCK_{i+1:02d}" for i in range(assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def make_correlated_prices(days=200, assets=3, seed=42, correlation=0.3):
    """生成具有相关性的价格数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=days)
    
    # 创建相关性矩阵
    corr_matrix = np.full((assets, assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # 生成相关收益率
    rets = rng.multivariate_normal(
        mean=[0.0005] * assets,
        cov=corr_matrix * 0.01**2,
        size=days
    )
    
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"STOCK_{i+1:02d}" for i in range(assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)


class TestBasicFunctionality:
    """测试基本功能"""
    
    def test_compute_max_sharpe_basic(self):
        """测试基本的最大夏普比率计算"""
        prices = make_prices()
        weights, perf = compute_max_sharpe(prices, rf=0.01, max_weight=0.8)

        # 权重验证
        w_vals = list(weights.values())
        assert all(0 <= w <= 0.80001 for w in w_vals)
        assert math.isclose(sum(w_vals), 1.0, rel_tol=1e-4, abs_tol=1e-4)

        # 性能指标验证
        assert isinstance(perf, tuple) and len(perf) == 3
        assert all(np.isfinite(x) for x in perf)

    def test_compute_max_sharpe_single_asset(self):
        """测试单一资产的情况"""
        prices = make_prices(assets=1)
        with pytest.raises(ValueError, match="至少需要2只股票"):
            compute_max_sharpe(prices, rf=0.01, max_weight=0.8)

    def test_compute_max_sharpe_empty_data(self):
        """测试空数据"""
        with pytest.raises(ValueError):
            compute_max_sharpe(pd.DataFrame(), rf=0.01, max_weight=0.8)

    def test_compute_max_sharpe_invalid_prices(self):
        """测试无效价格数据"""
        prices = make_prices()
        prices.iloc[0, 0] = -10  # 添加负价格
        with pytest.raises(ValueError, match="非正数值"):
            compute_max_sharpe(prices, rf=0.01, max_weight=0.8)

    def test_different_max_weights(self):
        """测试不同的最大权重约束"""
        prices = make_prices(assets=5)
        
        # 测试不同的权重限制
        for max_weight in [0.2, 0.5, 1.0]:
            weights, _ = compute_max_sharpe(prices, rf=0.02, max_weight=max_weight)
            max_actual_weight = max(weights.values())
            assert max_actual_weight <= max_weight + 1e-6

    def test_different_risk_free_rates(self):
        """测试不同的无风险利率"""
        prices = make_prices()
        
        results = []
        for rf in [0.0, 0.02, 0.05]:
            weights, perf = compute_max_sharpe(prices, rf=rf, max_weight=0.5)
            results.append((rf, perf[2]))  # 保存无风险利率和夏普比率
        
        # 验证所有计算都成功
        assert all(np.isfinite(sharpe) for _, sharpe in results)


class TestTradeDateRange:
    """测试交易日期相关功能"""
    
    def test_get_valid_trade_range_cn(self):
        """测试中国市场交易日获取"""
        start_date, end_date = get_valid_trade_range("2023-01-01", "2023-01-10", "XSHG")
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)
        assert start_date <= end_date

    # 移除美股相关测试

    def test_get_valid_trade_range_invalid_exchange(self):
        """测试无效的交易所"""
        with pytest.raises(Exception):  # 可能是ValueError或其他异常
            get_valid_trade_range("2023-01-01", "2023-01-10", "INVALID")


class TestFileOperations:
    """测试文件操作"""
    
    def test_save_outputs(self):
        """测试保存输出文件"""
        prices = make_prices()
        weights = {"STOCK_01": 0.4, "STOCK_02": 0.6, "STOCK_03": 0.0}
        performance = (0.12, 0.18, 0.67)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            prices_file, weights_file, perf_file = save_outputs(
                prices, weights, performance, tmp_dir, "2022-01-01", "2022-12-31"
            )
            
            # 验证文件存在
            assert os.path.exists(prices_file)
            assert os.path.exists(weights_file)
            assert os.path.exists(perf_file)
            
            # 验证文件内容
            saved_prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
            assert saved_prices.shape == prices.shape
            
            saved_weights = pd.read_csv(weights_file)
            assert len(saved_weights) == 2  # 只保存非零权重
            assert "STOCK_03" not in saved_weights["Ticker"].values


@pytest.mark.skipif(not HAS_NEW_MODULES, reason="新模块不可用")
class TestNewModules:
    """测试新的模块化代码"""
    
    def test_max_sharpe_optimizer_init(self):
        """测试MaxSharpeOptimizer初始化"""
        optimizer = MaxSharpeOptimizer(risk_free_rate=0.02, max_weight=0.3)
        assert optimizer.risk_free_rate == 0.02
        assert optimizer.max_weight == 0.3

    def test_max_sharpe_optimizer_invalid_params(self):
        """测试无效参数"""
        with pytest.raises(ValueError):
            MaxSharpeOptimizer(max_weight=1.5)  # 超过1.0
        
        with pytest.raises(ValueError):
            MaxSharpeOptimizer(max_weight=-0.1)  # 负数

    def test_max_sharpe_optimizer_optimize(self):
        """测试优化功能"""
        prices = make_prices(assets=4)
        optimizer = MaxSharpeOptimizer(risk_free_rate=0.02, max_weight=0.4)
        
        weights, performance = optimizer.optimize(prices)
        
        # 验证权重
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(w <= 0.4 + 1e-6 for w in weights.values())
        
        # 验证性能指标
        assert 'expected_annual_return' in performance
        assert 'annual_volatility' in performance
        assert 'sharpe_ratio' in performance

    def test_data_fetcher_init(self):
        """测试DataFetcher初始化"""
        fetcher = DataFetcher(market="CN")
        assert fetcher.market == "CN"
        
        with pytest.raises(ValueError):
            DataFetcher(market="INVALID")

    def test_validate_price_data(self):
        """测试价格数据验证"""
        # 正常数据
        prices = make_prices()
        validate_price_data(prices)  # 应该不抛出异常
        
        # 空数据
        with pytest.raises(ValueError, match="价格数据为空"):
            validate_price_data(pd.DataFrame())
        
        # 单一资产
        with pytest.raises(ValueError, match="至少需要2只股票"):
            validate_price_data(make_prices(assets=1))

    def test_calculate_returns(self):
        """测试收益率计算"""
        prices = make_prices()
        returns = calculate_returns(prices)
        
        assert len(returns) == len(prices) - 1
        assert returns.columns.equals(prices.columns)
        assert not returns.isnull().all().any()


class TestEdgeCases:
    """测试边界情况"""
    
    def test_high_correlation_assets(self):
        """测试高相关性资产"""
        prices = make_correlated_prices(correlation=0.9)
        weights, perf = compute_max_sharpe(prices, rf=0.02, max_weight=0.8)
        
        # 高相关性下，应该集中投资
        max_weight = max(weights.values())
        assert max_weight > 0.5  # 预期会有较高的集中度

    def test_zero_volatility_asset(self):
        """测试零波动率资产"""
        prices = make_prices(assets=3)
        # 将一个资产设为常数价格（零波动率）
        prices.iloc[:, 0] = 100
        
        # 应该能处理这种情况而不崩溃
        weights, perf = compute_max_sharpe(prices, rf=0.02, max_weight=0.5)
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-4)

    def test_very_short_time_series(self):
        """测试很短的时间序列"""
        prices = make_prices(days=10)  # 只有10天数据

        # 应该给出警告但仍能计算
        with warnings.catch_warnings(record=True) as warning_list:
            weights, perf = compute_max_sharpe(prices, rf=0.02, max_weight=0.5)

        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-4)

    def test_negative_risk_free_rate(self):
        """测试负无风险利率"""
        prices = make_prices()
        
        # 负利率应该能够处理
        weights, perf = compute_max_sharpe(prices, rf=-0.01, max_weight=0.5)
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-4)


class TestPerformanceMetrics:
    """测试性能指标"""
    
    def test_performance_calculation_consistency(self):
        """测试性能指标计算的一致性"""
        prices = make_prices(days=252, assets=3)  # 一年的数据
        weights, perf = compute_max_sharpe(prices, rf=0.02, max_weight=0.5)
        
        annual_return, annual_vol, sharpe = perf
        
        # 验证夏普比率计算
        expected_sharpe = (annual_return - 0.02) / annual_vol
        assert abs(sharpe - expected_sharpe) < 1e-6

    def test_portfolio_return_calculation(self):
        """测试投资组合收益率计算"""
        prices = make_prices(assets=3)
        weights = {"STOCK_01": 0.5, "STOCK_02": 0.3, "STOCK_03": 0.2}
        
        # 手动计算投资组合收益率
        returns = prices.pct_change().dropna()
        portfolio_returns = returns @ pd.Series(weights)
        
        # 验证计算正确性
        assert len(portfolio_returns) == len(returns)
        assert not portfolio_returns.isnull().any()


if __name__ == "__main__":
    # 可以直接运行此文件进行测试
    pytest.main([__file__, "-v"])
