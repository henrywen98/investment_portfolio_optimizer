"""
工具函数模块（Utilities）

提供与数据校验、收益率与年化指标计算、交易日区间校准、结果输出格式化等相关的实用函数。

主要功能：
- 日志初始化：`setup_logger`
- 交易日区间校准：`get_valid_trade_range`
- 收益率与年化指标：`calculate_returns`、`calculate_annual_metrics`
- 数据有效性校验与清洗：`validate_price_data`
- 结果格式化：`format_performance_output`
- 市场到交易所映射：`get_exchange_for_market`

注意：本模块仅包含“纯工具”逻辑，不涉及网络请求或优化算法，便于单元测试与复用。
"""

import logging
from typing import Tuple
import pandas as pd


logger = logging.getLogger(__name__)


def setup_logger(verbose: bool = True) -> None:
    """
    设置简单的全局日志格式与等级。

    Args:
        verbose: 若为 True，则使用 INFO 等级；否则使用 WARNING。

    示例：
        >>> setup_logger(verbose=True)
        >>> logger.info("开始计算……")
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_valid_trade_range(start_date: str, end_date: str, exchange: str = "XSHG") -> Tuple[str, str]:
    """
    将输入的日期范围对齐到指定交易所的实际交易日。

    Args:
        start_date: 开始日期，格式 YYYY-MM-DD。
        end_date: 结束日期，格式 YYYY-MM-DD。
        exchange: 交易所代码，例如：
            - XSHG: 上海证券交易所（中国 A 股）
            - NYSE: 纽约证券交易所（美股）
            - NASDAQ: 纳斯达克（美股）

    Returns:
        二元组 (start_date, end_date)：均为对齐后的首个/最后一个交易日字符串。

    Raises:
        ImportError: 未安装 pandas-market-calendars。
        ValueError: 在给定区间内找不到有效交易日。
        Exception: 底层交易日历获取异常将直接抛出。

    说明：
        本函数依赖 `pandas-market-calendars` 的交易所日历以保证日期有效性，
        常用于拉取历史价格数据前的预处理步骤。
    """
    try:
        import pandas_market_calendars as mcal
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "需要安装 pandas-market-calendars：pip install pandas-market-calendars"
        ) from e

    try:
        cal = mcal.get_calendar(exchange)
        schedule = cal.schedule(start_date=start_date, end_date=end_date)
    except Exception as e:
        logger.warning(f"获取交易日历失败: {e}")
        raise

    if schedule.empty:
        raise ValueError(
            f"未找到有效的交易日，请检查时间范围或交易所日历！交易所: {exchange}"
        )

    start = schedule.index[0].strftime("%Y-%m-%d")
    end = schedule.index[-1].strftime("%Y-%m-%d")

    return start, end


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    计算按列（资产）逐日简单收益率。

    输入以价格矩阵为准（行索引为日期，列为资产），输出与输入列对齐。

    Args:
        prices: 价格数据 DataFrame，要求为正数且按时间升序排列。

    Returns:
        DataFrame: 与 `prices` 同列的日频简单收益率（使用 `pct_change()`），首行缺失被丢弃。

    示例：
        >>> returns = calculate_returns(prices)
        >>> assert returns.shape[0] == prices.shape[0] - 1
    """
    return prices.pct_change().dropna()


def calculate_annual_metrics(returns: pd.DataFrame, trading_days: int = 252) -> dict:
    """
    基于日频收益率计算年化收益与年化波动率（逐列）。

    Args:
        returns: 日频收益率 DataFrame。
        trading_days: 年化所用的交易日数量，常用 252。

    Returns:
        dict: 包含两项 pandas.Series：
            - 'annual_returns': 年化收益率 = 日均收益 × trading_days
            - 'annual_volatility': 年化波动率 = 日波动 × sqrt(trading_days)
    """
    annual_returns = returns.mean() * trading_days
    annual_volatility = returns.std() * (trading_days ** 0.5)
    
    return {
        'annual_returns': annual_returns,
        'annual_volatility': annual_volatility,
    }


def validate_price_data(prices: pd.DataFrame) -> None:
    """
    验证并就地清洗价格数据的基本质量问题。

    校验与处理流程：
    - 非空性：DataFrame 不能为空，且至少包含 2 个资产列。
    - 类型转换：尝试将所有列转换为数值；不可解析的值将设为 NaN。
    - 缺失值：若存在 NaN，使用前向填充并再次丢弃残余缺失；若仍有缺失则报错。
    - 合法性：所有价格必须为正数；否则报错。
    - 时长提示：若样本长度 < 30，给出警告（不报错）。

    Args:
        prices: 价格数据 DataFrame，将在必要时进行原位修改（ffill/dropna/类型转换）。

    Raises:
        ValueError: 当数据为空、资产数不足、存在非正数或清洗后仍有缺失值等情况。
    """
    if prices.empty:
        raise ValueError("价格数据为空")

    if len(prices.columns) < 2:
        raise ValueError("至少需要2只股票才能进行组合优化")

    # 转换为数值类型
    prices[:] = prices.apply(pd.to_numeric, errors='coerce')

    # 检查并处理缺失值
    if prices.isnull().any().any():
        nan_info = prices.isnull().sum()
        logger.warning(
            f"价格数据中存在缺失值，将进行前向填充: {nan_info[nan_info > 0].to_dict()}"
        )
        
        prices.ffill(inplace=True)
        prices.dropna(inplace=True)

        # 清理后再次检查
        if prices.isnull().any().any():
            nan_info = prices.isnull().sum()
            raise ValueError(
                f"价格数据中仍存在缺失值: {nan_info[nan_info > 0].to_dict()}"
            )

    if prices.empty:
        raise ValueError("清理后价格数据为空")
    
    # 检查是否有负数或零
    if (prices <= 0).any().any():
        raise ValueError("价格数据中包含非正数值")
    
    # 检查数据长度
    if len(prices) < 30:
        logger.warning("价格数据长度较短，可能影响优化结果")


def format_performance_output(weights: dict, performance: dict) -> dict:
    """
    将权重与性能指标统一转换为可序列化的纯 Python 基本类型。

    设计目的：确保写入 JSON/CSV 时不包含 numpy.float64/pandas 对象。

    Args:
        weights: 资产权重字典，如 {ticker: weight}。
        performance: 含期望年化收益、年化波动、夏普比率等指标的字典。

    Returns:
        dict: 包含浮点化后的权重与关键指标，键包括：
            - 'weights'
            - 'expected_annual_return'
            - 'annual_volatility'
            - 'sharpe_ratio'
    """
    return {
        'weights': {k: float(v) for k, v in weights.items()},
        'expected_annual_return': float(performance.get('expected_annual_return', 0)),
        'annual_volatility': float(performance.get('annual_volatility', 0)),
        'sharpe_ratio': float(performance.get('sharpe_ratio', 0)),
    }


def get_exchange_for_market(market: str) -> str:
    """
    根据市场代码返回默认交易所代码。

    当前映射：
        - 'CN' -> 'XSHG'（上交所）
        - 'US' -> 'NYSE'（纽交所）

    若传入未知市场，默认返回 'XSHG'。

    Args:
        market: 市场代码（大小写不敏感），如 'CN' 或 'US'。

    Returns:
        交易所代码字符串。
    """
    market_exchange_map = {
        'CN': 'XSHG',  # 上海证券交易所
        'US': 'NYSE',  # 纽约证券交易所
    }
    return market_exchange_map.get(market.upper(), 'XSHG')
