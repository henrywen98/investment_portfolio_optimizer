#!/usr/bin/env python3
"""
Max Sharpe Portfolio Optimizer - 主入口文件

保持向后兼容的同时使用新的模块化结构
"""

import argparse
import datetime
import json
import logging
import os
from typing import Dict, Iterable, Tuple

import pandas as pd

# 从新的模块化结构导入
try:
    from maxsharpe import PortfolioOptimizer, get_valid_trade_range
    from maxsharpe.data import get_default_tickers
    from maxsharpe.utils import setup_logger

    # 向后兼容的导入
    from maxsharpe.core import compute_max_sharpe as _compute_max_sharpe, fetch_prices

    def compute_max_sharpe(prices: pd.DataFrame, rf: float = 0.02, max_weight: float = 1.0):
        weights, perf = _compute_max_sharpe(prices, rf=rf, max_weight=max_weight)
        return weights, (
            perf["expected_annual_return"],
            perf["annual_volatility"],
            perf["sharpe_ratio"],
        )

    USE_NEW_MODULES = True
    
except ImportError:
    # 如果新模块不可用，使用原有实现作为后备
    logging.warning("使用旧版本实现，建议使用模块化版本")
    
    # 向后兼容的实现
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage

    USE_NEW_MODULES = False

    # 保留原有的常量定义
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

    DEFAULT_TICKERS = DEFAULT_TICKERS_CN
    
    def setup_logger(verbose: bool = True) -> None:
        level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def get_valid_trade_range(start_date: str, end_date: str, exchange: str = "XSHG") -> Tuple[str, str]:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar(exchange)
        schedule = cal.schedule(start_date=start_date, end_date=end_date)
        if schedule.empty:
            raise ValueError(f"未找到有效的交易日，请检查时间范围或交易所日历！Exchange: {exchange}")
        start = schedule.index[0].strftime("%Y-%m-%d")
        end = schedule.index[-1].strftime("%Y-%m-%d")
        return start, end

    def fetch_prices(tickers: Iterable[str], start_date: str, end_date: str, market: str = "CN", adjust: str = "hfq") -> pd.DataFrame:
        if market.upper() == "CN":
            return _fetch_cn_prices(tickers, start_date, end_date, adjust)
        elif market.upper() == "US":
            return _fetch_us_prices(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unsupported market: {market}. Use 'CN' or 'US'.")

    def _fetch_cn_prices(tickers: Iterable[str], start_date: str, end_date: str, adjust: str = "hfq") -> pd.DataFrame:
        data = pd.DataFrame()
        try:
            import akshare as ak
        except Exception as e:
            raise ImportError("需要安装 akshare 才能下载行情数据：pip install akshare") from e

        for ticker in tickers:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=ticker,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust=adjust,
                )
                if df is None or df.empty:
                    logging.warning(f"{ticker}: 无数据返回，已跳过")
                    continue
                df["日期"] = pd.to_datetime(df["日期"])
                df.set_index("日期", inplace=True)
                close_series = df["收盘"].astype(float)
                close_series.name = ticker
                data = pd.concat([data, close_series], axis=1)
                logging.info(f"已下载 {ticker} 的收盘价，共 {close_series.shape[0]} 行")
            except Exception as e:
                logging.error(f"下载 {ticker} 失败: {e}")
        return data.sort_index()

    def _fetch_us_prices(tickers: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
        data = pd.DataFrame()
        try:
            import yfinance as yf
        except Exception as e:
            raise ImportError("需要安装 yfinance 才能下载美股数据：pip install yfinance") from e

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date, auto_adjust=True)
                if df is None or df.empty:
                    logging.warning(f"{ticker}: 无数据返回，已跳过")
                    continue
                close_series = df["Close"].astype(float)
                close_series.name = ticker
                data = pd.concat([data, close_series], axis=1)
                logging.info(f"已下载 {ticker} 的收盘价，共 {close_series.shape[0]} 行")
            except Exception as e:
                logging.error(f"下载 {ticker} 失败: {e}")
        return data.sort_index()

    def compute_max_sharpe(prices: pd.DataFrame, rf: float = 0.01696, max_weight: float = 0.25) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
        if prices.empty:
            raise ValueError("价格数据为空，无法计算投资组合！")
        if prices.isnull().mean().max() > 0.5:
            raise ValueError("数据存在重大缺失（超过50%缺失），请检查股票或时间范围！")

        mu = mean_historical_return(prices)
        S = CovarianceShrinkage(prices).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w <= max_weight)
        ef.max_sharpe(risk_free_rate=rf)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        return weights, performance
    
    def get_default_tickers(market: str) -> list:
        if market.upper() == "US":
            return DEFAULT_TICKERS_US
        return DEFAULT_TICKERS_CN


def save_outputs(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    performance: Tuple[float, float, float],
    output_dir: str,
    start_date: str,
    end_date: str,
) -> Tuple[str, str, str]:
    """Save raw prices, weights and performance to files under output_dir.

    Returns (prices_csv, weights_csv, performance_json)
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{start_date}_{end_date}"

    prices_csv = os.path.join(output_dir, f"stock_data_{tag}.csv")
    prices.to_csv(prices_csv)

    weights_df = (
        pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
        .query("Weight > 0")
        .sort_values("Weight", ascending=False)
    )
    weights_csv = os.path.join(output_dir, f"weights_{tag}.csv")
    weights_df.to_csv(weights_csv, index=False)

    ann_ret, ann_vol, sharpe = performance
    monthly_return = (1 + ann_ret) ** (1 / 12) - 1
    monthly_volatility = ann_vol / (12 ** 0.5)
    perf_payload = {
        "annual": {
            "expected_return": ann_ret,
            "volatility": ann_vol,
            "sharpe": sharpe,
        },
        "monthly": {
            "expected_return": monthly_return,
            "volatility": monthly_volatility,
            "sharpe": sharpe,
        },
        "window": {"start": start_date, "end": end_date},
    }
    performance_json = os.path.join(output_dir, f"performance_{tag}.json")
    with open(performance_json, "w", encoding="utf-8") as f:
        json.dump(perf_payload, f, ensure_ascii=False, indent=2)

    return prices_csv, weights_csv, performance_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Max Sharpe Portfolio for China A-shares and US stocks")
    parser.add_argument(
        "--market",
        type=str,
        choices=["CN", "US"],
        default="CN",
        help="市场选择：CN（中国A股）或 US（美股），默认CN",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="股票代码（逗号分隔），不指定则使用对应市场的默认股票池",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="回溯年数（与 --start-date 互斥，若同时提供则以 start/end 为准）",
    )
    parser.add_argument("--start-date", type=str, default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--rf", type=float, default=0.01696, help="无风险利率（年化）")
    parser.add_argument("--max-weight", type=float, default=0.25, help="单一资产最大权重上限")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), help="输出目录")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(verbose=not args.quiet)

    # 市场和交易所设置
    market = args.market.upper()
    if market == "CN":
        exchange = "XSHG"
        default_tickers = get_default_tickers("CN")
    else:  # US
        exchange = "NYSE"
        default_tickers = get_default_tickers("US")

    # 选择股票
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = default_tickers

    # 日期设置
    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
    else:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=args.years * 365)).strftime("%Y-%m-%d")

    # 调整到有效交易日
    start_date, end_date = get_valid_trade_range(start_date, end_date, exchange)

    logging.info(f"市场: {market} | 交易所: {exchange}")
    logging.info(f"日期范围: {start_date} ~ {end_date}")
    logging.info(f"股票池: {tickers[:5]}{'...' if len(tickers) > 5 else ''} (共{len(tickers)}只)")
    logging.info(f"无风险利率: {args.rf:.3%} | 最大权重: {args.max_weight:.1%}")

    # 使用新的模块化接口或传统接口
    if USE_NEW_MODULES:
        try:
            # 使用新的PortfolioOptimizer类
            optimizer = PortfolioOptimizer(
                market=market,
                risk_free_rate=args.rf,
                max_weight=args.max_weight
            )
            
            weights, performance = optimizer.optimize_portfolio(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            
            # 获取价格数据用于保存
            prices = fetch_prices(tickers, start_date, end_date, market)
            
            # 转换性能指标格式以保持兼容性
            perf_tuple = (
                performance['expected_annual_return'],
                performance['annual_volatility'],
                performance['sharpe_ratio']
            )
            
        except Exception as e:
            logging.error(f"使用新模块失败，回退到传统方法: {e}")
            # 回退到传统方法
            prices = fetch_prices(tickers, start_date, end_date, market)
            weights, perf_tuple = compute_max_sharpe(prices, rf=args.rf, max_weight=args.max_weight)
    else:
        # 使用传统方法
        prices = fetch_prices(tickers, start_date, end_date, market)
        weights, perf_tuple = compute_max_sharpe(prices, rf=args.rf, max_weight=args.max_weight)

    # 显示结果
    logging.info("=" * 60)
    logging.info("投资组合优化结果")
    logging.info("=" * 60)

    weights_filtered = {k: v for k, v in weights.items() if v > 0.001}
    for ticker, weight in sorted(weights_filtered.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"{ticker:>8s}: {weight:>7.2%}")

    ann_ret, ann_vol, sharpe = perf_tuple
    logging.info("-" * 40)
    logging.info(f"预期年化收益: {ann_ret:>7.2%}")
    logging.info(f"年化波动率:   {ann_vol:>7.2%}")
    logging.info(f"夏普比率:     {sharpe:>7.3f}")

    # 保存结果
    prices_file, weights_file, perf_file = save_outputs(
        prices, weights, perf_tuple, args.output, start_date, end_date
    )

    logging.info("=" * 60)
    logging.info("文件保存完成")
    logging.info(f"价格数据: {prices_file}")
    logging.info(f"权重配置: {weights_file}")
    logging.info(f"性能指标: {perf_file}")


if __name__ == "__main__":
    main()
