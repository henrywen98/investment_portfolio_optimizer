# Examples

This folder contains runnable examples demonstrating CLI and Python API usage.

- basic_usage.py — Run optimizer with default CN tickers and print results
- custom_portfolio.py — Customize risk-free rate and max weight
- visualization.py — Plot the optimized weights (requires matplotlib)

Run examples from repo root with an activated virtual environment:

```bash
python examples/basic_usage.py
python examples/custom_portfolio.py --tickers 600519,000858,601318 --years 3 --rf 0.02 --max-weight 0.25
python examples/visualization.py --tickers 600519,000858,601318 --years 3
```

