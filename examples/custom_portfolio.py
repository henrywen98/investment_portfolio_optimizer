"""
Customize portfolio parameters via CLI.

Run:
  python examples/custom_portfolio.py --tickers 600519,000858,601318 --years 3 --rf 0.02 --max-weight 0.25
"""

import argparse
from maxsharpe.core import PortfolioOptimizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, default="600519,000858,601318")
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--rf", type=float, default=0.02)
    p.add_argument("--max-weight", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    optimizer = PortfolioOptimizer(market="CN", risk_free_rate=args.rf, max_weight=args.max_weight)
    weights, performance = optimizer.optimize_portfolio(tickers=tickers, years=args.years)

    print("Tickers:", tickers)
    print("Weights:")
    for k, v in weights.items():
        print(f"  {k}: {v:.2%}")

    print("\nPerformance:")
    for k, v in performance.items():
        try:
            print(f"  {k}: {float(v):.6f}")
        except Exception:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

