"""
pytest配置文件 - 定义测试相关的配置和fixtures
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock


@pytest.fixture
def sample_price_data():
    """提供示例价格数据的fixture"""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=252)  # 一年交易日
    n_assets = 5
    
    # 生成相对真实的价格走势
    returns = np.random.multivariate_normal(
        mean=[0.0008] * n_assets,
        cov=np.eye(n_assets) * 0.0004 + np.ones((n_assets, n_assets)) * 0.0001,
        size=len(dates)
    )
    
    # 转换为价格
    initial_prices = [100, 150, 80, 200, 120]
    prices = np.zeros((len(dates), n_assets))
    prices[0] = initial_prices
    
    for i in range(1, len(dates)):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    return pd.DataFrame(
        prices,
        index=dates,
        columns=['600519', '000858', '601318', '600036', '000063']
    )


@pytest.fixture
def sample_cn_price_data():
    """提供A股示例价格数据的fixture"""
    np.random.seed(123)
    dates = pd.bdate_range("2022-01-01", periods=200)
    n_assets = 4
    
    returns = np.random.multivariate_normal(
        mean=[0.0005] * n_assets,
        cov=np.eye(n_assets) * 0.0003 + np.ones((n_assets, n_assets)) * 0.00008,
        size=len(dates)
    )
    
    initial_prices = [50, 25, 80, 35]
    prices = np.zeros((len(dates), n_assets))
    prices[0] = initial_prices
    
    for i in range(1, len(dates)):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    return pd.DataFrame(
        prices,
        index=dates,
        columns=['600519', '000858', '601318', '600036']
    )


@pytest.fixture
def temp_output_dir():
    """提供临时输出目录的fixture"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_akshare_data():
    """提供模拟akshare数据的fixture"""
    dates = pd.date_range("2022-01-01", periods=100)
    return pd.DataFrame({
        '日期': dates,
        '收盘': np.random.uniform(20, 30, 100),
        '开盘': np.random.uniform(19, 31, 100),
        '最高': np.random.uniform(25, 35, 100),
        '最低': np.random.uniform(15, 25, 100),
        '成交量': np.random.randint(1000000, 10000000, 100)
    })


# 已移除：yfinance 模拟数据 fixture（不再支持美股）


@pytest.fixture
def sample_weights():
    """提供示例权重数据的fixture"""
    return {
        '600519': 0.35,
        '000858': 0.25,
        '601318': 0.20,
        '600036': 0.15,
        '000063': 0.05
    }


@pytest.fixture
def sample_performance():
    """提供示例性能指标的fixture"""
    return (0.12, 0.18, 0.67)  # (年化收益率, 年化波动率, 夏普比率)


# 自定义标记
def pytest_configure(config):
    """配置自定义pytest标记"""
    config.addinivalue_line(
        "markers", "slow: 标记测试为慢速测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记为集成测试"
    )
    config.addinivalue_line(
        "markers", "network: 标记需要网络连接的测试"
    )


# 跳过网络测试的条件
def pytest_collection_modifyitems(config, items):
    """修改测试收集，添加跳过条件"""
    if config.getoption("--skip-network"):
        skip_network = pytest.mark.skip(reason="跳过网络测试")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--skip-network",
        action="store_true",
        default=False,
        help="跳过需要网络连接的测试"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="运行慢速测试"
    )
