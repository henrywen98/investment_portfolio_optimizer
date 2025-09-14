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
    
    def _fetch_us_prices(self, tickers: Iterable[str], start_date,
                        end_date) -> pd.DataFrame:
        """获取美股价格数据。

        首选 yfinance（Yahoo Finance）。若遭遇 429/空数据等问题，自动回退至
        pandas-datareader 的 Stooq 源，保证在 Yahoo 受限时仍可用。
        """
        if yf is None:
            raise ImportError("需要安装 yfinance 才能下载美股数据：pip install yfinance")

        # 确保日期格式正确
        if hasattr(start_date, 'strftime'):
            start_date_str = start_date.strftime("%Y-%m-%d")
        else:
            start_date_str = str(start_date)
            
        if hasattr(end_date, 'strftime'):
            end_date_str = end_date.strftime("%Y-%m-%d")
        else:
            end_date_str = str(end_date)

        import time
        import numpy as np

        def simple_download_with_long_delay(tickers_list, max_retries=2):
            """简化的下载函数，使用长延迟避免速率限制"""
            for attempt in range(max_retries):
                try:
                    logger.info(f"正在下载美股数据: {tickers_list} (尝试 {attempt + 1}/{max_retries})")
                    
                    # 逐个下载股票以减少失败概率
                    all_data = pd.DataFrame()
                    successful_tickers = []
                    
                    for i, ticker in enumerate(tickers_list):
                        try:
                            # 简短延迟，避免过于频繁，但不阻塞回退
                            if i > 0:
                                wait_time = 0.5
                                logger.info(f"等待 {wait_time} 秒后下载下一个股票...")
                                time.sleep(wait_time)
                            
                            logger.info(f"下载 {ticker} (第 {i+1}/{len(tickers_list)} 个)")
                            
                            # 使用最简单的方式下载单个股票
                            ticker_data = yf.download(
                                ticker,
                                start=start_date_str,
                                end=end_date_str,
                                progress=False,
                                auto_adjust=True,
                                prepost=False,
                                threads=False
                            )
                            
                            if not ticker_data.empty:
                                # 提取收盘价
                                if 'Close' in ticker_data.columns:
                                    close_prices = ticker_data['Close']
                                elif len(ticker_data.columns) == 1:
                                    close_prices = ticker_data.iloc[:, 0]
                                else:
                                    # 如果有多列，尝试找到最后一列（通常是收盘价）
                                    close_prices = ticker_data.iloc[:, -1]
                                
                                # 确保数据有效
                                close_prices = close_prices.dropna()
                                if len(close_prices) > 10:  # 至少需要10个数据点
                                    all_data[ticker] = close_prices
                                    successful_tickers.append(ticker)
                                    logger.info(f"✅ {ticker} 下载成功 ({len(close_prices)} 个数据点)")
                                else:
                                    logger.warning(f"⚠️ {ticker} 数据点不足")
                            else:
                                logger.warning(f"⚠️ {ticker} 下载失败：数据为空")
                                
                        except Exception as e:
                            logger.warning(f"⚠️ {ticker} 下载失败: {e}")
                            continue
                    
                    if not all_data.empty:
                        logger.info(f"成功下载 {len(successful_tickers)} 只股票: {successful_tickers}")
                        return all_data
                    else:
                        raise ValueError("所有股票下载失败")
                        
                except Exception as e:
                    logger.warning(f"下载尝试 {attempt + 1} 失败: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)  # 30秒, 60秒
                        logger.info(f"等待 {wait_time} 秒后重试整个流程...")
                        time.sleep(wait_time)
                    else:
                        raise e

        # 先尝试通过 yfinance 下载
        try:
            # 仅尝试一次，失败则快速回退至 Stooq
            data = simple_download_with_long_delay(list(tickers), max_retries=1)

            if data.empty:
                raise ValueError("未能获取任何有效价格数据")

            # 清理与标准化
            result = self._clean_us_prices_df(data)
            logger.info(f"美股数据最终结果（Yahoo）：{result.shape[1]} 只股票，{result.shape[0]} 个交易日")
            return result

        except Exception as e:
            logger.warning(f"Yahoo 下载失败，尝试使用 Stooq 作为后备数据源: {e}")
            # 若 yfinance 失败，尝试使用 Stooq 作为后备
            try:
                result = self._fetch_us_prices_stooq(tickers, start_date_str, end_date_str)
                if result is not None and not result.empty:
                    logger.info(
                        f"美股数据最终结果（Stooq）：{result.shape[1]} 只股票，{result.shape[0]} 个交易日"
                    )
                    return result
                else:
                    raise ValueError("Stooq 返回空数据")
            except Exception as e2:
                logger.error(f"Stooq 备选方案也失败: {e2}")
                # 提供更详细的错误信息
                msg = str(e) + " | fallback: " + str(e2)
                if "429" in msg or "too many" in msg.lower():
                    raise ValueError("Yahoo Finance API 速率限制，且 Stooq 备选失败；请稍后再试或减少股票数量")
                raise ValueError(f"美股数据获取失败（Yahoo 与 Stooq 均失败）: {msg}")

    def _clean_us_prices_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """统一清理美股价格数据表（各源共用）。"""
        import numpy as np  # 局部导入以避免未使用时的依赖

        result = data.copy()
        # 删除完全缺失的列
        result = result.dropna(axis=1, how='all')
        # 数值化
        result = result.apply(pd.to_numeric, errors='coerce')
        # 缺失值处理
        result = result.ffill().bfill()
        result = result.dropna()
        # 非正数处理
        result = result[(result > 0).all(axis=1)]
        # 无穷值处理
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.dropna()
        if result.empty:
            raise ValueError("数据清理后没有有效数据")
        return result

    def _fetch_us_prices_stooq(self, tickers: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
        """从 Stooq 获取美股价格作为后备数据源。

        依赖 pandas-datareader（多数环境已自带）。
        """
        try:  # 延迟导入，避免未使用时报错
            from pandas_datareader import data as pdr
        except Exception as e:  # pragma: no cover
            raise ImportError("需要安装 pandas-datareader 才能从 Stooq 下载数据：pip install pandas-datareader") from e

        all_data = pd.DataFrame()
        successful = []
        for ticker in tickers:
            try:
                df = pdr.DataReader(ticker, 'stooq', start=start_date, end=end_date)
                if df is None or df.empty:
                    logger.warning(f"Stooq 源 {ticker}: 无数据，跳过")
                    continue
                # Stooq 返回为降序日期，先升序
                df = df.sort_index()
                close = pd.to_numeric(df['Close'], errors='coerce').dropna()
                if len(close) > 10:
                    all_data[ticker] = close
                    successful.append(ticker)
                    logger.info(f"✅ Stooq {ticker} 下载成功 ({len(close)} 个数据点)")
                else:
                    logger.warning(f"⚠️ Stooq {ticker} 数据点不足")
            except Exception as e:
                logger.warning(f"⚠️ Stooq {ticker} 下载失败: {e}")
                continue

        if all_data.empty:
            raise ValueError("Stooq 未能获取到任何价格数据")

        # 清理/标准化
        return self._clean_us_prices_df(all_data)


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
