"""
测试新模块的功能 - 仅当新模块可用时运行
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# 尝试导入新模块
try:
    from maxsharpe import MaxSharpeOptimizer, DataFetcher
    from maxsharpe.utils import validate_price_data, calculate_returns, get_exchange_for_market
    from maxsharpe.data import get_default_tickers
    from maxsharpe.core import PortfolioOptimizer
    HAS_NEW_MODULES = True
except ImportError:
    HAS_NEW_MODULES = False

pytestmark = pytest.mark.skipif(not HAS_NEW_MODULES, reason="新模块不可用")


def make_test_prices(days=100, assets=3, seed=42):
    """生成测试用价格数据"""
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-01", periods=days)
    prices = np.random.lognormal(mean=0.001, sigma=0.02, size=(days, assets))
    prices = pd.DataFrame(
        prices.cumprod(axis=0) * 100,
        index=dates,
        columns=[f"STOCK_{i+1}" for i in range(assets)]
    )
    return prices


class TestDataFetcher:
    """测试数据获取器"""
    
    def test_init_cn_market(self):
        """测试中国市场初始化"""
        fetcher = DataFetcher(market="CN")
        assert fetcher.market == "CN"
    
    def test_init_us_market(self):
        """测试美国市场初始化"""
        fetcher = DataFetcher(market="US")
        assert fetcher.market == "US"
    
    def test_invalid_market(self):
        """测试无效市场"""
        with pytest.raises(ValueError, match="不支持的市场类型"):
            DataFetcher(market="INVALID")
    
    @patch('maxsharpe.data.ak')
    def test_fetch_cn_prices_success(self, mock_ak):
        """测试成功获取A股数据"""
        # 模拟akshare返回的数据
        mock_data = pd.DataFrame({
            '日期': pd.date_range('2022-01-01', periods=10),
            '收盘': np.random.uniform(10, 20, 10)
        })
        mock_ak.stock_zh_a_hist.return_value = mock_data
        
        fetcher = DataFetcher(market="CN")
        result = fetcher.fetch_prices(['600519'], '2022-01-01', '2022-01-10')
        
        assert not result.empty
        assert '600519' in result.columns
        mock_ak.stock_zh_a_hist.assert_called_once()
    
    @patch('maxsharpe.data.yf')
    def test_fetch_us_prices_success(self, mock_yf):
        """测试成功获取美股数据"""
        # 模拟yfinance返回的数据
        mock_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 10)
        }, index=pd.date_range('2022-01-01', periods=10))
        
        mock_yf.download.return_value = mock_data
        
        fetcher = DataFetcher(market="US")
        result = fetcher.fetch_prices(['AAPL'], '2022-01-01', '2022-01-10')
        
        assert not result.empty
        assert 'AAPL' in result.columns
        mock_yf.download.assert_called_once()
    
    def test_get_default_tickers(self):
        """测试获取默认股票池"""
        cn_tickers = get_default_tickers("CN")
        us_tickers = get_default_tickers("US")
        
        assert isinstance(cn_tickers, list)
        assert isinstance(us_tickers, list)
        assert len(cn_tickers) > 0
        assert len(us_tickers) > 0
        assert cn_tickers != us_tickers


class TestMaxSharpeOptimizer:
    """测试最大夏普比率优化器"""
    
    def test_init_default(self):
        """测试默认初始化"""
        optimizer = MaxSharpeOptimizer()
        assert optimizer.risk_free_rate == 0.02
        assert optimizer.max_weight == 1.0
    
    def test_init_custom(self):
        """测试自定义参数初始化"""
        optimizer = MaxSharpeOptimizer(risk_free_rate=0.03, max_weight=0.25)
        assert optimizer.risk_free_rate == 0.03
        assert optimizer.max_weight == 0.25
    
    def test_invalid_max_weight(self):
        """测试无效的最大权重"""
        with pytest.raises(ValueError, match="最大权重应在0和1之间"):
            MaxSharpeOptimizer(max_weight=1.5)
        
        with pytest.raises(ValueError, match="最大权重应在0和1之间"):
            MaxSharpeOptimizer(max_weight=-0.1)
    
    def test_optimize_success(self):
        """测试成功优化"""
        prices = make_test_prices(assets=4)
        optimizer = MaxSharpeOptimizer(risk_free_rate=0.02, max_weight=0.3)
        
        weights, performance = optimizer.optimize(prices)
        
        # 验证权重
        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(0 <= w <= 0.3 + 1e-6 for w in weights.values())
        
        # 验证性能指标
        required_keys = ['expected_annual_return', 'annual_volatility', 'sharpe_ratio']
        assert all(key in performance for key in required_keys)
        assert all(np.isfinite(performance[key]) for key in required_keys)
    
    def test_set_risk_free_rate(self):
        """测试设置无风险利率"""
        optimizer = MaxSharpeOptimizer()
        optimizer.set_risk_free_rate(0.05)
        assert optimizer.risk_free_rate == 0.05
    
    def test_set_max_weight(self):
        """测试设置最大权重"""
        optimizer = MaxSharpeOptimizer()
        optimizer.set_max_weight(0.4)
        assert optimizer.max_weight == 0.4
        
        with pytest.raises(ValueError):
            optimizer.set_max_weight(1.2)


class TestUtils:
    """测试工具函数"""
    
    def test_validate_price_data_success(self):
        """测试有效价格数据验证"""
        prices = make_test_prices()
        validate_price_data(prices)  # 应该不抛出异常
    
    def test_validate_price_data_empty(self):
        """测试空价格数据"""
        with pytest.raises(ValueError, match="价格数据为空"):
            validate_price_data(pd.DataFrame())
    
    def test_validate_price_data_single_asset(self):
        """测试单一资产"""
        prices = make_test_prices(assets=1)
        with pytest.raises(ValueError, match="至少需要2只股票"):
            validate_price_data(prices)
    
    def test_validate_price_data_negative_prices(self):
        """测试负价格"""
        prices = make_test_prices()
        prices.iloc[0, 0] = -10
        with pytest.raises(ValueError, match="非正数值"):
            validate_price_data(prices)
    
    def test_calculate_returns(self):
        """测试收益率计算"""
        prices = make_test_prices()
        returns = calculate_returns(prices)
        
        assert len(returns) == len(prices) - 1
        assert returns.columns.equals(prices.columns)
        assert not returns.isnull().all().any()
    
    def test_get_exchange_for_market(self):
        """测试获取市场对应的交易所"""
        assert get_exchange_for_market("CN") == "XSHG"
        assert get_exchange_for_market("US") == "NYSE"
        assert get_exchange_for_market("cn") == "XSHG"
        assert get_exchange_for_market("us") == "NYSE"
        assert get_exchange_for_market("UNKNOWN") == "XSHG"  # 默认值


class TestPortfolioOptimizer:
    """测试投资组合优化器主类"""
    
    def test_init_default(self):
        """测试默认初始化"""
        optimizer = PortfolioOptimizer()
        assert optimizer.market == "CN"
    
    def test_init_us_market(self):
        """测试美股市场初始化"""
        optimizer = PortfolioOptimizer(market="US", risk_free_rate=0.03, max_weight=0.2)
        assert optimizer.market == "US"
    
    @patch.object(DataFetcher, 'fetch_prices')
    def test_optimize_portfolio_success(self, mock_fetch):
        """测试投资组合优化成功"""
        # 模拟数据获取
        mock_prices = make_test_prices(assets=5)
        mock_fetch.return_value = mock_prices
        
        optimizer = PortfolioOptimizer(market="CN")
        weights, performance = optimizer.optimize_portfolio(
            tickers=['600519', '000858', '601318', '600036', '000063'],
            start_date='2022-01-01',
            end_date='2022-12-31'
        )
        
        # 验证结果
        assert isinstance(weights, dict)
        assert isinstance(performance, dict)
        assert len(weights) == 5
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        mock_fetch.assert_called_once()
    
    def test_update_settings(self):
        """测试更新设置"""
        optimizer = PortfolioOptimizer()
        
        optimizer.update_settings(risk_free_rate=0.04, max_weight=0.3)
        assert optimizer.optimizer.risk_free_rate == 0.04
        assert optimizer.optimizer.max_weight == 0.3
        
        optimizer.update_settings(risk_free_rate=0.05)
        assert optimizer.optimizer.risk_free_rate == 0.05
        assert optimizer.optimizer.max_weight == 0.3  # 不变
    
    def test_years_vs_dates_conflict(self):
        """测试years和dates参数冲突"""
        optimizer = PortfolioOptimizer()
        
        with pytest.raises(ValueError, match="years参数与start_date/end_date参数不能同时使用"):
            optimizer.optimize_portfolio(
                years=5,
                start_date='2022-01-01',
                end_date='2022-12-31'
            )


class TestIntegration:
    """集成测试"""
    
    @patch.object(DataFetcher, 'fetch_prices')
    def test_full_workflow(self, mock_fetch):
        """测试完整工作流程"""
        # 模拟数据
        mock_prices = make_test_prices(days=252, assets=3)  # 一年数据
        mock_fetch.return_value = mock_prices
        
        # 创建优化器并运行
        optimizer = PortfolioOptimizer(
            market="US",
            risk_free_rate=0.02,
            max_weight=0.4
        )
        
        weights, performance = optimizer.optimize_portfolio(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            years=1
        )
        
        # 验证结果的合理性
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(0 <= w <= 0.4 + 1e-6 for w in weights.values())
        
        # 验证性能指标
        assert performance['sharpe_ratio'] > -5  # 合理的夏普比率范围
        assert performance['sharpe_ratio'] < 5
        assert performance['annual_volatility'] > 0
        assert performance['annual_volatility'] < 1  # 年化波动率不应超过100%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
