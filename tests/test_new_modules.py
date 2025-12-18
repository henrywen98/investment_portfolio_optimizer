"""
测试新模块的功能 - 综合测试套件

覆盖:
- 多种优化策略
- 回测模块
- 约束模块
- 核心功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# 尝试导入新模块
try:
    from maxsharpe import (
        MaxSharpeOptimizer,
        MinVarianceOptimizer,
        RiskParityOptimizer,
        MaxDiversificationOptimizer,
        EqualWeightOptimizer,
        DataFetcher,
        PortfolioOptimizerFactory,
        OptimizationStrategy,
    )
    from maxsharpe.utils import (
        validate_price_data,
        calculate_returns,
        get_exchange_for_market,
    )
    from maxsharpe.data import get_default_tickers
    from maxsharpe.core import PortfolioOptimizer
    from maxsharpe.constraints import (
        SectorConstraint,
        TransactionCost,
        ConstrainedOptimizer,
        Sector,
        calculate_portfolio_concentration,
        suggest_rebalance,
    )
    from maxsharpe.backtest import (
        Backtester,
        BacktestConfig,
        BacktestResult,
        generate_backtest_report,
    )
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

    def test_invalid_market(self):
        """测试无效市场"""
        with pytest.raises(ValueError, match="不支持的市场类型"):
            DataFetcher(market="INVALID")

    @patch('maxsharpe.data.ak')
    def test_fetch_cn_prices_success(self, mock_ak):
        """测试成功获取A股数据"""
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

    def test_get_default_tickers(self):
        """测试获取默认股票池"""
        cn_tickers = get_default_tickers("CN")
        assert isinstance(cn_tickers, list)
        assert len(cn_tickers) > 0


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

    def test_invalid_weight_constraint(self):
        """测试无效的权重约束"""
        with pytest.raises(ValueError, match="权重约束无效"):
            MaxSharpeOptimizer(min_weight=0.5, max_weight=0.3)

    def test_optimize_success(self):
        """测试成功优化"""
        prices = make_test_prices(assets=4)
        optimizer = MaxSharpeOptimizer(risk_free_rate=0.02, max_weight=0.3)

        weights, performance = optimizer.optimize(prices)

        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(0 <= w <= 0.3 + 1e-6 for w in weights.values())

        required_keys = ['expected_annual_return', 'annual_volatility', 'sharpe_ratio',
                        'sortino_ratio', 'calmar_ratio', 'max_drawdown']
        assert all(key in performance for key in required_keys)

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


class TestMinVarianceOptimizer:
    """测试最小方差优化器"""

    def test_optimize_success(self):
        """测试最小方差优化"""
        prices = make_test_prices(assets=4)
        optimizer = MinVarianceOptimizer(max_weight=0.4)

        weights, performance = optimizer.optimize(prices)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert 'annual_volatility' in performance


class TestRiskParityOptimizer:
    """测试风险平价优化器"""

    def test_optimize_success(self):
        """测试风险平价优化"""
        prices = make_test_prices(assets=4)
        optimizer = RiskParityOptimizer()

        weights, performance = optimizer.optimize(prices)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6


class TestMaxDiversificationOptimizer:
    """测试最大分散化优化器"""

    def test_optimize_success(self):
        """测试最大分散化优化"""
        prices = make_test_prices(assets=4)
        optimizer = MaxDiversificationOptimizer()

        weights, performance = optimizer.optimize(prices)

        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert 'diversification_ratio' in performance
        assert performance['diversification_ratio'] >= 1.0


class TestEqualWeightOptimizer:
    """测试等权重优化器"""

    def test_optimize_success(self):
        """测试等权重优化"""
        prices = make_test_prices(assets=4)
        optimizer = EqualWeightOptimizer()

        weights, performance = optimizer.optimize(prices)

        assert isinstance(weights, dict)
        assert all(abs(w - 0.25) < 1e-6 for w in weights.values())


class TestPortfolioOptimizerFactory:
    """测试优化器工厂"""

    def test_create_all_strategies(self):
        """测试创建所有策略优化器"""
        for strategy in OptimizationStrategy:
            optimizer = PortfolioOptimizerFactory.create(strategy)
            assert optimizer is not None

    def test_available_strategies(self):
        """测试获取可用策略列表"""
        strategies = PortfolioOptimizerFactory.available_strategies()
        assert len(strategies) == 5
        assert 'max_sharpe' in strategies


class TestConstraints:
    """测试约束模块"""

    def test_transaction_cost_buy(self):
        """测试买入成本计算"""
        cost = TransactionCost(commission_rate=0.0003, slippage=0.001)
        buy_cost = cost.calculate_buy_cost(100000)
        assert buy_cost > 0

    def test_transaction_cost_sell(self):
        """测试卖出成本计算"""
        cost = TransactionCost(commission_rate=0.0003, stamp_duty=0.001, slippage=0.001)
        sell_cost = cost.calculate_sell_cost(100000)
        assert sell_cost > cost.calculate_buy_cost(100000)  # 卖出有印花税

    def test_sector_constraint_validation(self):
        """测试行业约束验证"""
        constraint = SectorConstraint(max_sector_weight=0.3, min_sectors=2)

        # 测试违规情况
        weights = {'600519': 0.5, '000858': 0.3, '600036': 0.2}  # 食品饮料超过30%
        is_valid, violations = constraint.validate_weights(weights)
        assert not is_valid
        assert len(violations) > 0

    def test_sector_weights_calculation(self):
        """测试行业权重计算"""
        constraint = SectorConstraint()
        weights = {'600519': 0.3, '000858': 0.3, '600036': 0.4}
        sector_weights = constraint.get_sector_weights(weights)
        assert isinstance(sector_weights, dict)

    def test_portfolio_concentration(self):
        """测试投资组合集中度计算"""
        weights = {'A': 0.5, 'B': 0.3, 'C': 0.2}
        metrics = calculate_portfolio_concentration(weights)

        assert 'hhi' in metrics
        assert 'effective_n' in metrics
        assert 'top5_weight' in metrics
        assert metrics['hhi'] > 0
        assert metrics['effective_n'] > 0

    def test_suggest_rebalance(self):
        """测试再平衡建议"""
        current = {'A': 0.5, 'B': 0.3, 'C': 0.2}
        target = {'A': 0.3, 'B': 0.4, 'C': 0.3}

        suggestions = suggest_rebalance(current, target, threshold=0.05)

        assert 'increase' in suggestions
        assert 'decrease' in suggestions
        assert 'A' in suggestions['decrease']
        assert 'B' in suggestions['increase']


class TestConstrainedOptimizer:
    """测试带约束的优化器"""

    def test_with_sector_constraint(self):
        """测试带行业约束的优化"""
        prices = make_test_prices(assets=4)
        base_optimizer = MaxSharpeOptimizer(max_weight=0.4)
        sector_constraint = SectorConstraint(max_sector_weight=0.5)

        constrained = ConstrainedOptimizer(
            base_optimizer=base_optimizer,
            sector_constraint=sector_constraint
        )

        weights, performance = constrained.optimize(prices)
        assert isinstance(weights, dict)


class TestBacktest:
    """测试回测模块"""

    def test_backtest_config_default(self):
        """测试回测配置默认值"""
        config = BacktestConfig()
        assert config.lookback_days == 252
        assert config.rebalance_frequency == 63
        assert config.initial_capital == 1_000_000

    def test_backtester_init(self):
        """测试回测器初始化"""
        config = BacktestConfig(
            lookback_days=126,
            rebalance_frequency=21,
            strategy=OptimizationStrategy.MIN_VARIANCE
        )
        backtester = Backtester(config)
        assert backtester.config.lookback_days == 126

    def test_backtest_run(self):
        """测试运行回测"""
        prices = make_test_prices(days=400, assets=4)

        config = BacktestConfig(
            lookback_days=100,
            rebalance_frequency=20,
            strategy=OptimizationStrategy.EQUAL_WEIGHT
        )
        backtester = Backtester(config)

        result = backtester.run(prices)

        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_values) > 0
        assert 'annual_return' in result.metrics
        assert 'sharpe_ratio' in result.metrics
        assert 'total_trades' in result.metrics

    def test_generate_backtest_report(self):
        """测试生成回测报告"""
        prices = make_test_prices(days=400, assets=4)

        config = BacktestConfig(
            lookback_days=100,
            rebalance_frequency=20
        )

        # 运行两种策略
        results = {}
        for strategy in [OptimizationStrategy.MAX_SHARPE, OptimizationStrategy.EQUAL_WEIGHT]:
            config.strategy = strategy
            backtester = Backtester(config)
            results[strategy.value] = backtester.run(prices)

        report = generate_backtest_report(results)

        assert isinstance(report, pd.DataFrame)
        assert len(report) == 2


class TestUtils:
    """测试工具函数"""

    def test_validate_price_data_success(self):
        """测试有效价格数据验证"""
        prices = make_test_prices()
        validate_price_data(prices)

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

    def test_get_exchange_for_market(self):
        """测试获取交易所"""
        assert get_exchange_for_market("CN") == "XSHG"


class TestPortfolioOptimizer:
    """测试投资组合优化器主类"""

    def test_init_with_strategy(self):
        """测试带策略的初始化"""
        for strategy in ["max_sharpe", "min_variance", "risk_parity"]:
            optimizer = PortfolioOptimizer(strategy=strategy)
            assert optimizer.strategy_name == strategy

    def test_invalid_strategy(self):
        """测试无效策略"""
        with pytest.raises(ValueError, match="不支持的策略"):
            PortfolioOptimizer(strategy="invalid_strategy")

    @patch.object(DataFetcher, 'fetch_prices')
    def test_optimize_with_sector_constraint(self, mock_fetch):
        """测试带行业约束的优化"""
        mock_prices = make_test_prices(assets=5)
        mock_fetch.return_value = mock_prices

        optimizer = PortfolioOptimizer()
        optimizer.set_sector_constraint(max_sector_weight=0.4, min_sectors=2)

        weights, performance = optimizer.optimize_portfolio(
            tickers=['600519', '000858', '601318', '600036', '000063'],
            years=1
        )

        assert isinstance(weights, dict)
        assert 'sector_weights' in performance or 'concentration_metrics' in performance

    def test_available_strategies(self):
        """测试获取可用策略"""
        strategies = PortfolioOptimizer.available_strategies()
        assert len(strategies) == 5

    @patch.object(DataFetcher, 'fetch_prices')
    def test_compare_strategies(self, mock_fetch):
        """测试策略对比"""
        mock_prices = make_test_prices(assets=4, days=252)
        mock_fetch.return_value = mock_prices

        optimizer = PortfolioOptimizer()
        results = optimizer.compare_strategies(
            tickers=['A', 'B', 'C', 'D'],
            years=1
        )

        assert len(results) > 0
        for strategy, (weights, perf) in results.items():
            assert isinstance(weights, dict)
            assert 'sharpe_ratio' in perf

    def test_update_settings(self):
        """测试更新设置"""
        optimizer = PortfolioOptimizer()

        optimizer.update_settings(
            risk_free_rate=0.04,
            max_weight=0.3,
            strategy="min_variance"
        )

        assert optimizer.risk_free_rate == 0.04
        assert optimizer.max_weight == 0.3
        assert optimizer.strategy_name == "min_variance"


class TestIntegration:
    """集成测试"""

    @patch.object(DataFetcher, 'fetch_prices')
    def test_full_workflow_all_strategies(self, mock_fetch):
        """测试所有策略的完整工作流"""
        mock_prices = make_test_prices(days=252, assets=5)
        mock_fetch.return_value = mock_prices

        for strategy in ["max_sharpe", "min_variance", "risk_parity",
                        "max_diversification", "equal_weight"]:
            optimizer = PortfolioOptimizer(
                market="CN",
                risk_free_rate=0.02,
                max_weight=0.4,
                strategy=strategy
            )

            weights, performance = optimizer.optimize_portfolio(
                tickers=['600519', '000858', '601318', '600036', '000063'],
                years=1
            )

            assert len(weights) == 5
            assert abs(sum(weights.values()) - 1.0) < 1e-6
            assert all(0 <= w <= 0.4 + 1e-6 for w in weights.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
