"""
Basic example: optimize a CN portfolio using default tickers and print results.

Run:
  python examples/basic_usage.py
"""

from maxsharpe.core import PortfolioOptimizer
from maxsharpe.data import get_default_tickers


def main() -> None:
    tickers = get_default_tickers("CN")[:5]
    optimizer = PortfolioOptimizer(market="CN", risk_free_rate=0.02, max_weight=0.25)
    weights, performance = optimizer.optimize_portfolio(tickers=tickers, years=3)

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

