import argparse
import datetime
import json
import logging
import os
from typing import Dict, Iterable, Tuple

import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage


DEFAULT_TICKERS = [
    "600519", "000858", "600887", "002594", "000333",
    "601888", "000063", "002230", "600941", "600036",
    "601318", "600028", "601012", "600438", "600031",
    "600585", "600019", "600276", "601899", "002352",
    "601766", "600030", "600406", "601668", "002714",
]


def _setup_logger(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_valid_trade_range(start_date: str, end_date: str, exchange: str = "XSHG") -> Tuple[str, str]:
    """Clamp the input date range to actual trading days of the given exchange.

    Returns (start_date, end_date) in YYYY-MM-DD.
    """
    # Lazy import to avoid unnecessary dependency at import time
    import pandas_market_calendars as mcal  # type: ignore
    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    if schedule.empty:
        raise ValueError("未找到有效的交易日，请检查时间范围或交易所日历！")
    start = schedule.index[0].strftime("%Y-%m-%d")
    end = schedule.index[-1].strftime("%Y-%m-%d")
    return start, end


def fetch_prices(tickers: Iterable[str], start_date: str, end_date: str, adjust: str = "hfq") -> pd.DataFrame:
    """Download A-share price data via akshare and return close price DataFrame.

    Index: DatetimeIndex
    Columns: tickers
    Values: close prices (float)
    """
    data = pd.DataFrame()
    # Lazy import to avoid heavy dependency at import time
    try:
        import akshare as ak  # type: ignore
    except Exception as e:  # noqa: BLE001
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
            # akshare 中文列名：日期、收盘
            df["日期"] = pd.to_datetime(df["日期"])  # type: ignore[index]
            df.set_index("日期", inplace=True)
            close_series = df["收盘"].astype(float)
            close_series.name = ticker
            data = pd.concat([data, close_series], axis=1)
            logging.info(f"已下载 {ticker} 的收盘价，共 {close_series.shape[0]} 行")
        except Exception as e:  # noqa: BLE001
            logging.error(f"下载 {ticker} 失败: {e}")
    return data.sort_index()


def compute_max_sharpe(prices: pd.DataFrame, rf: float = 0.01696, max_weight: float = 0.25) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """Compute Max Sharpe portfolio.

    Returns (weights_dict, (annual_return, annual_vol, sharpe)).
    """
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
    parser = argparse.ArgumentParser(description="Max Sharpe Portfolio for China A-shares")
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="股票代码（逗号分隔），默认内置一组样例",
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
    _setup_logger(verbose=not args.quiet)

    # 日期处理
    if args.start_date and args.end_date:
        start_raw, end_raw = args.start_date, args.end_date
    else:
        end_raw = datetime.datetime.today().strftime("%Y-%m-%d")
        start_raw = (datetime.datetime.today() - datetime.timedelta(days=365 * args.years)).strftime(
            "%Y-%m-%d"
        )
    start_date, end_date = get_valid_trade_range(start_raw, end_raw)
    logging.info(f"计算区间：{start_date} -> {end_date}")

    # 股票列表
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    logging.info(f"股票数量：{len(tickers)} — {tickers}")

    # 下载数据并计算
    prices = fetch_prices(tickers, start_date, end_date)
    if prices.empty:
        raise ValueError("未能获取任何股票价格数据，请检查网络、股票代码或时间范围！")

    weights, performance = compute_max_sharpe(prices, rf=args.rf, max_weight=args.max_weight)
    prices_csv, weights_csv, perf_json = save_outputs(
        prices, weights, performance, args.output, start_date, end_date
    )

    # 控制台友好输出
    weights_df = (
        pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
        .query("Weight > 0")
        .sort_values("Weight", ascending=False)
    )
    print("\nPortfolio Weights (%):")
    print((weights_df.assign(Weight=lambda df: df["Weight"] * 100)).to_string(index=False))

    ann_ret, ann_vol, sharpe = performance
    monthly_return = (1 + ann_ret) ** (1 / 12) - 1
    monthly_volatility = ann_vol / (12 ** 0.5)
    perf_df = pd.DataFrame(
        {
            "Metric": ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio"],
            "Value Annually": [ann_ret * 100, ann_vol * 100, sharpe],
            "Value Monthly": [monthly_return * 100, monthly_volatility * 100, sharpe],
        }
    )
    print("\nPortfolio Performance:")
    print(perf_df.to_string(index=False))

    print("\nArtifacts saved:")
    print(f"- Prices CSV: {prices_csv}")
    print(f"- Weights CSV: {weights_csv}")
    print(f"- Performance JSON: {perf_json}")


if __name__ == "__main__":
    main()
