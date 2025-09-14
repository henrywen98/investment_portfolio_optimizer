import streamlit as st
import pandas as pd
from maxsharpe.core import PortfolioOptimizer
from maxsharpe.data import get_default_tickers

st.set_page_config(page_title="Max Sharpe Optimizer")

st.title("📈 Max Sharpe Portfolio Optimizer")

market = st.selectbox("Market", ["CN"], index=0)

default_tickers = ",".join(get_default_tickers(market))
user_tickers = st.text_input("Tickers (comma separated)", default_tickers)
tickers = [t.strip() for t in user_tickers.split(",") if t.strip()]

years = st.number_input("Years of history", min_value=1, max_value=10, value=5)
rf = st.number_input("Risk-free rate", value=0.02, step=0.001, format="%.3f")
max_weight = st.slider("Max weight per asset", 0.0, 1.0, 0.25)

@st.cache_data(show_spinner=False)
def run_optimization(market: str, tickers: list, years: int, rf: float, max_weight: float):
    optimizer = PortfolioOptimizer(market=market, risk_free_rate=rf, max_weight=max_weight)
    weights, performance = optimizer.optimize_portfolio(tickers=tickers, years=years)
    return weights, performance

if st.button("Optimize"):
    with st.spinner("Optimizing portfolio..."):
        try:
            weights, performance = run_optimization(
                market, tickers, int(years), rf, max_weight
            )
            st.subheader("Weights")
            st.dataframe(
                pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
            )
            st.subheader("Performance")
            st.json(performance)
        except Exception as e:  # pragma: no cover - UI feedback
            st.error(f"Optimization failed: {e}")
