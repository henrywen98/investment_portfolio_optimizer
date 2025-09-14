"""
Visualize optimized weights (bar chart). Requires matplotlib.

Run:
  python examples/visualization.py --tickers 600519,000858,601318 --years 3
"""

import argparse
from maxsharpe.core import PortfolioOptimizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, default="600519,000858,601318")
    p.add_argument("--years", type=int, default=3)
    return p.parse_args()


def main() -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        raise SystemExit("matplotlib is required for visualization: pip install matplotlib")

    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    optimizer = PortfolioOptimizer(market="CN", risk_free_rate=0.02, max_weight=0.25)
    weights, performance = optimizer.optimize_portfolio(tickers=tickers, years=args.years)

    # Plot weights
    names = list(weights.keys())
    vals = [weights[k] for k in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, vals, color="#3b82f6")
    plt.ylabel("Weight")
    plt.title("Max Sharpe Weights")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

