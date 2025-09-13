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

try:  # pragma: no cover - optional dependencies
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


logger = logging.getLogger(__name__)


class DataFetcher:
    """数据获取器类"""
    
    def __init__(self, market: str = "CN"):
        """
        初始化数据获取器
        
        Args:
            market: 市场类型，"CN" 表示中国A股，"US" 表示美股
        """
        self.market = market.upper()
        if self.market not in ["CN", "US"]:
            raise ValueError(f"不支持的市场类型: {market}. 请使用 'CN' 或 'US'")
    
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
        elif self.market == "US":
            return self._fetch_us_prices(tickers, start_date, end_date)
        else:
            raise ValueError(f"不支持的市场: {self.market}")
    
    def _fetch_cn_prices(self, tickers: Iterable[str], start_date: str, 
                        end_date: str, adjust: str = "hfq") -> pd.DataFrame:
        """获取A股价格数据"""
        if ak is None:
            raise ImportError("需要安装 akshare 才能下载A股数据：pip install akshare")

        data = pd.DataFrame()
        
        for ticker in tickers:
            try:
                logger.info(f"正在下载 {ticker} 的价格数据...")
                df = ak.stock_zh_a_hist(
                    symbol=ticker,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust=adjust
                )
                
                if df.empty:
                    logger.warning(f"股票 {ticker} 没有数据")
                    continue
                
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                df.sort_index(inplace=True)
                
                data[ticker] = df['收盘']
                
            except Exception as e:
                logger.error(f"下载股票 {ticker} 数据失败: {e}")
                continue
        
        if data.empty:
            raise ValueError("未能获取任何价格数据")
        
        # 前向填充缺失值
        data = data.fillna(method='ffill').dropna()
        
        return data
    
    def _fetch_us_prices(self, tickers: Iterable[str], start_date: str,
                        end_date: str) -> pd.DataFrame:
        """获取美股价格数据"""
        if yf is None:
            raise ImportError("需要安装 yfinance 才能下载美股数据：pip install yfinance")

        try:
            logger.info(f"正在下载美股数据: {list(tickers)}")
            data = yf.download(
                tickers=list(tickers),
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                raise ValueError("未能获取任何价格数据")
            
            # 如果只有一只股票，yfinance返回的格式不同
            if len(list(tickers)) == 1:
                if 'Close' in data.columns:
                    result = pd.DataFrame({list(tickers)[0]: data['Close']})
                else:
                    result = data.to_frame(list(tickers)[0])
            else:
                # 多只股票时，提取Close价格
                if 'Close' in data.columns:
                    result = data['Close']
                else:
                    result = data
            
            # 前向填充缺失值
            result = result.fillna(method='ffill').dropna()
            
            return result
            
        except Exception as e:
            logger.error(f"下载美股数据失败: {e}")
            raise


# 默认股票池
DEFAULT_TICKERS_CN = [
    "600519", "000858", "600887", "002594", "000333",
    "601888", "000063", "002230", "600941", "600036",
    "601318", "600028", "601012", "600438", "600031",
    "600585", "600019", "600276", "601899", "002352",
    "601766", "600030", "600406", "601668", "002714",
]

DEFAULT_TICKERS_US = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "BRK-B", "V", "JNJ",
    "UNH", "XOM", "WMT", "LLY", "PG",
    "MA", "JPM", "HD", "CVX", "MRK",
    "ABBV", "KO", "AVGO", "PEP", "PFE",
]

DEFAULT_TICKERS = {
    "CN": DEFAULT_TICKERS_CN,
    "US": DEFAULT_TICKERS_US,
}


def get_default_tickers(market: str) -> list:
    """获取默认股票池"""
    return DEFAULT_TICKERS.get(market.upper(), DEFAULT_TICKERS_CN)
