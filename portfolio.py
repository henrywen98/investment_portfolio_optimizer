#!/usr/bin/env python3
"""
Max Sharpe Portfolio Optimizer - 主入口文件 (CLI)

支持多种优化策略:
- max_sharpe: 最大夏普比率
- min_variance: 最小方差
- risk_parity: 风险平价
- max_diversification: 最大分散化
- equal_weight: 等权重
"""

import argparse
import datetime
import json
import logging
import os
from typing import Dict, Tuple

import pandas as pd

from maxsharpe import PortfolioOptimizer, get_valid_trade_range
from maxsharpe.data import get_default_tickers
from maxsharpe.utils import setup_logger
from maxsharpe.core import fetch_prices


def save_outputs(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    performance: Dict[str, float],
    output_dir: str,
    start_date: str,
    end_date: str,
) -> Tuple[str, str, str]:
    """
    保存价格数据、权重和性能指标到文件

    Returns:
        (prices_csv, weights_csv, performance_json) 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{start_date}_{end_date}"

    # 保存价格数据
    prices_csv = os.path.join(output_dir, f"stock_data_{tag}.csv")
    prices.to_csv(prices_csv)

    # 保存权重
    weights_df = (
        pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
        .query("Weight > 0")
        .sort_values("Weight", ascending=False)
    )
    weights_csv = os.path.join(output_dir, f"weights_{tag}.csv")
    weights_df.to_csv(weights_csv, index=False)

    # 保存性能指标
    ann_ret = performance.get('expected_annual_return', 0)
    ann_vol = performance.get('annual_volatility', 0)
    sharpe = performance.get('sharpe_ratio', 0)

    monthly_return = (1 + ann_ret) ** (1 / 12) - 1
    monthly_volatility = ann_vol / (12 ** 0.5)

    perf_payload = {
        "annual": {
            "expected_return": ann_ret,
            "volatility": ann_vol,
            "sharpe": sharpe,
            "sortino": performance.get('sortino_ratio', 0),
            "calmar": performance.get('calmar_ratio', 0),
            "max_drawdown": performance.get('max_drawdown', 0),
        },
        "monthly": {
            "expected_return": monthly_return,
            "volatility": monthly_volatility,
            "sharpe": sharpe,
        },
        "risk": {
            "var_5_percent": performance.get('var_5_percent', 0),
            "var_1_percent": performance.get('var_1_percent', 0),
            "cvar_5_percent": performance.get('cvar_5_percent', 0),
        },
        "window": {"start": start_date, "end": end_date},
        "strategy": performance.get('strategy', 'max_sharpe'),
    }

    performance_json = os.path.join(output_dir, f"performance_{tag}.json")
    with open(performance_json, "w", encoding="utf-8") as f:
        json.dump(perf_payload, f, ensure_ascii=False, indent=2)

    return prices_csv, weights_csv, performance_json


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="投资组合优化器 - 支持多种优化策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数（最大夏普比率）
  python portfolio.py

  # 使用最小方差策略
  python portfolio.py --strategy min_variance

  # 自定义股票和参数
  python portfolio.py --tickers 600519,000858,000333 --years 3 --max-weight 0.3

  # 策略对比
  python portfolio.py --compare

可用策略:
  max_sharpe          最大夏普比率（默认）
  min_variance        最小方差
  risk_parity         风险平价
  max_diversification 最大分散化
  equal_weight        等权重
        """
    )

    parser.add_argument(
        "--market",
        type=str,
        choices=["CN"],
        default="CN",
        help="市场选择：仅支持 CN（中国A股）",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["max_sharpe", "min_variance", "risk_parity", "max_diversification", "equal_weight"],
        default="max_sharpe",
        help="优化策略（默认: max_sharpe）",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="股票代码（逗号分隔），不指定则使用默认股票池",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="回溯年数（默认: 5）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="开始日期 YYYY-MM-DD（与 --years 互斥）"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="结束日期 YYYY-MM-DD"
    )
    parser.add_argument(
        "--rf",
        type=float,
        default=0.02,
        help="无风险利率（年化，默认: 0.02）"
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.25,
        help="单一资产最大权重上限（默认: 0.25）"
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="单一资产最小权重下限（默认: 0.0）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="输出目录"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比所有策略"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出"
    )

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    setup_logger(verbose=not args.quiet)

    # 市场和交易所设置
    market = args.market.upper()
    exchange = "XSHG"
    default_tickers = get_default_tickers("CN")

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
    logging.info(f"优化策略: {args.strategy}")

    # 策略对比模式
    if args.compare:
        logging.info("=" * 60)
        logging.info("策略对比模式")
        logging.info("=" * 60)

        optimizer = PortfolioOptimizer(
            market=market,
            risk_free_rate=args.rf,
            max_weight=args.max_weight,
            min_weight=args.min_weight,
        )

        results = optimizer.compare_strategies(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )

        # 打印对比结果
        logging.info("-" * 60)
        logging.info(f"{'策略':<20} {'年化收益':>10} {'波动率':>10} {'夏普比率':>10}")
        logging.info("-" * 60)

        for strategy, (weights, perf) in results.items():
            ann_ret = perf.get('expected_annual_return', 0)
            ann_vol = perf.get('annual_volatility', 0)
            sharpe = perf.get('sharpe_ratio', 0)
            logging.info(f"{strategy:<20} {ann_ret:>10.2%} {ann_vol:>10.2%} {sharpe:>10.3f}")

        return

    # 单策略优化
    optimizer = PortfolioOptimizer(
        market=market,
        risk_free_rate=args.rf,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        strategy=args.strategy,
    )

    # 执行优化
    weights, performance = optimizer.optimize_portfolio(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    # 获取价格数据用于保存
    prices = fetch_prices(tickers, start_date, end_date, market)

    # 显示结果
    logging.info("=" * 60)
    logging.info(f"投资组合优化结果 [{args.strategy}]")
    logging.info("=" * 60)

    weights_filtered = {k: v for k, v in weights.items() if v > 0.001}
    for ticker, weight in sorted(weights_filtered.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"{ticker:>8s}: {weight:>7.2%}")

    ann_ret = performance.get('expected_annual_return', 0)
    ann_vol = performance.get('annual_volatility', 0)
    sharpe = performance.get('sharpe_ratio', 0)
    max_dd = performance.get('max_drawdown', 0)
    sortino = performance.get('sortino_ratio', 0)

    logging.info("-" * 40)
    logging.info(f"预期年化收益: {ann_ret:>7.2%}")
    logging.info(f"年化波动率:   {ann_vol:>7.2%}")
    logging.info(f"夏普比率:     {sharpe:>7.3f}")
    logging.info(f"Sortino比率:  {sortino:>7.3f}")
    logging.info(f"最大回撤:     {max_dd:>7.2%}")

    # 集中度指标
    concentration = performance.get('concentration_metrics', {})
    if concentration:
        logging.info("-" * 40)
        logging.info("集中度指标:")
        logging.info(f"  HHI指数:      {concentration.get('hhi', 0):.4f}")
        logging.info(f"  有效持仓数:   {concentration.get('effective_n', 0):.1f}")
        logging.info(f"  前5大权重:    {concentration.get('top5_weight', 0):.2%}")

    # 添加策略信息到性能字典
    performance['strategy'] = args.strategy

    # 保存结果
    prices_file, weights_file, perf_file = save_outputs(
        prices, weights, performance, args.output, start_date, end_date
    )

    logging.info("=" * 60)
    logging.info("文件保存完成")
    logging.info(f"价格数据: {prices_file}")
    logging.info(f"权重配置: {weights_file}")
    logging.info(f"性能指标: {perf_file}")


if __name__ == "__main__":
    main()
