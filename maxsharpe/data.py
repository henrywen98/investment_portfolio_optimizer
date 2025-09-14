"""
数据获取模块 - Data Fetcher Module

负责从各种数据源获取股票价格数据
"""

import logging
from typing import Iterable, Dict, Any
import pandas as pd

try:  # pragma: no cover - optional dependencies
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None

# 移除美股依赖（yfinance 等），仅保留 A 股数据源


logger = logging.getLogger(__name__)


class DataFetcher:
    """数据获取器类"""
    
    def __init__(self, market: str = "CN"):
        """
        初始化数据获取器
        
        Args:
            market: 市场类型，仅支持 "CN"（中国A股）
        """
        self.market = market.upper()
        if self.market not in ["CN"]:
            raise ValueError(f"不支持的市场类型: {market}. 仅支持 'CN'")
    
    def fetch_prices(self, tickers: Iterable[str], start_date: str, end_date: str, 
                    adjust: str = "hfq") -> pd.DataFrame:
        """
        获取股票价格数据
        
        Args:
            tickers: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            adjust: 复权方式 (仅对A股有效: "hfq"前复权, "qfq"后复权, ""不复权)
            
        Returns:
            DataFrame with:
            - Index: DatetimeIndex
            - Columns: 股票代码
            - Values: 收盘价
        """
        if self.market == "CN":
            return self._fetch_cn_prices(tickers, start_date, end_date, adjust)
        else:
            raise ValueError(f"不支持的市场: {self.market}")
    
    def _fetch_cn_prices(self, tickers: Iterable[str], start_date, 
                        end_date, adjust: str = "hfq") -> pd.DataFrame:
        """获取A股价格数据"""
        if ak is None:
            raise ImportError("需要安装 akshare 才能下载A股数据：pip install akshare")

        # 确保日期格式正确
        if hasattr(start_date, 'strftime'):
            start_date_str = start_date.strftime("%Y%m%d")
        else:
            start_date_str = str(start_date).replace("-", "")
            
        if hasattr(end_date, 'strftime'):
            end_date_str = end_date.strftime("%Y%m%d")
        else:
            end_date_str = str(end_date).replace("-", "")

        data = pd.DataFrame()

        for ticker in tickers:
            try:
                logger.info(f"正在下载 {ticker} 的价格数据...")
                df = ak.stock_zh_a_hist(
                    symbol=ticker,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    adjust=adjust
                )

                if df.empty:
                    logger.warning(f"股票 {ticker} 没有数据")
                    continue

                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                df.sort_index(inplace=True)

                # 确保价格为数值类型
                close = pd.to_numeric(df['收盘'], errors='coerce')
                data[ticker] = close

            except Exception as e:
                logger.error(f"下载股票 {ticker} 数据失败: {e}")
                continue

        if data.empty:
            raise ValueError("未能获取任何价格数据")

        # 转换为数值并清理缺失值
        data = data.apply(pd.to_numeric, errors='coerce')
        nan_info = data.isna().sum()
        if nan_info.any():
            logger.debug(f"A股数据缺失值统计: {nan_info[nan_info > 0].to_dict()}")
        data = data.ffill().dropna()
        
        return data
    
    # 已移除：美股数据获取与清洗相关方法


# 默认股票池（仅中国A股）
DEFAULT_TICKERS_CN = [
    "600519", "000858", "600887", "002594", "000333",
    "601888", "000063", "002230", "600941", "600036",
    "601318", "600028", "601012", "600438", "600031",
    "600585", "600019", "600276", "601899", "002352",
    "601766", "600030", "600406", "601668", "002714",
]

DEFAULT_TICKERS = {
    "CN": DEFAULT_TICKERS_CN,
}


def get_default_tickers(market: str) -> list:
    """获取默认股票池（仅支持 CN）"""
    return DEFAULT_TICKERS.get("CN", DEFAULT_TICKERS_CN)
