"""
投资组合优化器 Streamlit Web应用

功能:
- 多策略优化对比
- 交互式权重配置
- 性能指标可视化
- 行业分布分析
- 回测结果展示
"""

import streamlit as st
import pandas as pd
import numpy as np

# 页面配置
st.set_page_config(
    page_title="投资组合优化器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入模块
try:
    from maxsharpe.core import PortfolioOptimizer
    from maxsharpe.data import get_default_tickers
    from maxsharpe.constraints import calculate_portfolio_concentration, SectorConstraint
    from maxsharpe.optimizer import OptimizationStrategy
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"模块导入失败: {e}")


def create_pie_chart(weights: dict, title: str = "投资组合权重分布"):
    """创建权重饼图"""
    # 过滤小权重
    filtered = {k: v for k, v in weights.items() if v > 0.01}

    if not filtered:
        st.warning("没有有效的权重数据")
        return

    labels = list(filtered.keys())
    values = list(filtered.values())

    # 使用Streamlit原生图表
    df = pd.DataFrame({'股票': labels, '权重': values})
    df = df.sort_values('权重', ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(title)
        # 条形图
        st.bar_chart(df.set_index('股票')['权重'])

    with col2:
        st.subheader("权重明细")
        df['权重%'] = df['权重'].apply(lambda x: f"{x:.2%}")
        st.dataframe(df[['股票', '权重%']], hide_index=True, use_container_width=True)


def display_performance_metrics(performance: dict):
    """显示性能指标"""
    st.subheader("📈 性能指标")

    # 主要指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        annual_return = performance.get('expected_annual_return', 0)
        st.metric(
            "预期年化收益",
            f"{annual_return:.2%}",
            delta=f"{annual_return - 0.05:.2%} vs 5%基准"
        )

    with col2:
        volatility = performance.get('annual_volatility', 0)
        st.metric(
            "年化波动率",
            f"{volatility:.2%}"
        )

    with col3:
        sharpe = performance.get('sharpe_ratio', 0)
        color = "green" if sharpe > 1 else ("orange" if sharpe > 0.5 else "red")
        st.metric(
            "夏普比率",
            f"{sharpe:.3f}",
            delta="优秀" if sharpe > 1 else ("良好" if sharpe > 0.5 else "一般")
        )

    with col4:
        max_dd = performance.get('max_drawdown', 0)
        st.metric(
            "最大回撤",
            f"{max_dd:.2%}"
        )

    # 附加指标
    st.subheader("📊 详细指标")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**风险指标**")
        st.write(f"- Sortino比率: {performance.get('sortino_ratio', 0):.3f}")
        st.write(f"- Calmar比率: {performance.get('calmar_ratio', 0):.3f}")
        st.write(f"- VaR (5%): {performance.get('var_5_percent', 0):.4f}")
        st.write(f"- CVaR (5%): {performance.get('cvar_5_percent', 0):.4f}")

    with col2:
        st.write("**收益指标**")
        st.write(f"- 总收益: {performance.get('total_return', 0):.2%}")
        st.write(f"- 交易天数: {performance.get('trading_days', 0)}")

    with col3:
        # 集中度指标
        concentration = performance.get('concentration_metrics', {})
        if concentration:
            st.write("**集中度指标**")
            st.write(f"- HHI指数: {concentration.get('hhi', 0):.4f}")
            st.write(f"- 有效持仓数: {concentration.get('effective_n', 0):.1f}")
            st.write(f"- 前5大权重: {concentration.get('top5_weight', 0):.2%}")
            st.write(f"- 实际持仓数: {concentration.get('num_positions', 0)}")


def display_sector_analysis(weights: dict):
    """显示行业分析"""
    st.subheader("🏭 行业分布")

    try:
        sector_constraint = SectorConstraint()
        sector_weights = sector_constraint.get_sector_weights(weights)

        if sector_weights:
            # 过滤零权重行业
            filtered_sectors = {k: v for k, v in sector_weights.items() if v > 0.001}

            df = pd.DataFrame({
                '行业': list(filtered_sectors.keys()),
                '权重': list(filtered_sectors.values())
            }).sort_values('权重', ascending=False)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.bar_chart(df.set_index('行业')['权重'])

            with col2:
                df['权重%'] = df['权重'].apply(lambda x: f"{x:.2%}")
                st.dataframe(df[['行业', '权重%']], hide_index=True, use_container_width=True)
        else:
            st.info("暂无行业分布数据")

    except Exception as e:
        st.warning(f"行业分析不可用: {e}")


def compare_strategies_view(optimizer: PortfolioOptimizer, tickers: list, years: int):
    """策略对比视图"""
    st.subheader("🔄 策略对比")

    with st.spinner("正在对比各策略..."):
        try:
            results = optimizer.compare_strategies(tickers=tickers, years=years)

            # 创建对比表格
            comparison_data = []
            for strategy, (weights, perf) in results.items():
                comparison_data.append({
                    '策略': strategy,
                    '年化收益': f"{perf.get('expected_annual_return', 0):.2%}",
                    '波动率': f"{perf.get('annual_volatility', 0):.2%}",
                    '夏普比率': f"{perf.get('sharpe_ratio', 0):.3f}",
                    '最大回撤': f"{perf.get('max_drawdown', 0):.2%}",
                    'Sortino': f"{perf.get('sortino_ratio', 0):.3f}",
                    '持仓数': len([w for w in weights.values() if w > 0.01])
                })

            df = pd.DataFrame(comparison_data)
            st.dataframe(df, hide_index=True, use_container_width=True)

            # 返回结果供后续使用
            return results

        except Exception as e:
            st.error(f"策略对比失败: {e}")
            return None


def main():
    # 标题
    st.title("📊 投资组合优化器")
    st.markdown("---")

    if not MODULES_AVAILABLE:
        st.error("请确保已正确安装 maxsharpe 模块")
        return

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 参数配置")

        # 市场选择
        market = st.selectbox("市场", ["CN"], index=0, help="目前仅支持中国A股")

        # 策略选择
        strategy = st.selectbox(
            "优化策略",
            ["max_sharpe", "min_variance", "risk_parity", "max_diversification", "equal_weight"],
            index=0,
            format_func=lambda x: {
                "max_sharpe": "最大夏普比率",
                "min_variance": "最小方差",
                "risk_parity": "风险平价",
                "max_diversification": "最大分散化",
                "equal_weight": "等权重"
            }.get(x, x)
        )

        st.markdown("---")

        # 股票选择
        st.subheader("📋 股票选择")
        default_tickers = get_default_tickers(market)
        use_default = st.checkbox("使用默认股票池", value=True)

        if use_default:
            tickers = default_tickers
            st.info(f"默认股票池: {len(tickers)} 只股票")
        else:
            user_input = st.text_area(
                "输入股票代码 (逗号分隔)",
                ",".join(default_tickers[:10]),
                height=100
            )
            tickers = [t.strip() for t in user_input.split(",") if t.strip()]

        st.markdown("---")

        # 时间参数
        st.subheader("📅 时间设置")
        years = st.slider("历史数据年数", 1, 10, 3)

        st.markdown("---")

        # 优化参数
        st.subheader("🎯 优化参数")
        rf = st.number_input(
            "无风险利率",
            value=0.02,
            min_value=0.0,
            max_value=0.1,
            step=0.001,
            format="%.3f",
            help="通常使用国债收益率"
        )

        max_weight = st.slider(
            "单一资产最大权重",
            0.05, 1.0, 0.25,
            help="限制单一股票的最大配置比例"
        )

        min_weight = st.slider(
            "单一资产最小权重",
            0.0, 0.1, 0.0,
            help="强制每只股票的最小配置比例"
        )

        st.markdown("---")

        # 行业约束
        st.subheader("🏭 行业约束")
        enable_sector_constraint = st.checkbox("启用行业约束", value=False)

        if enable_sector_constraint:
            max_sector_weight = st.slider(
                "单一行业最大权重",
                0.1, 0.5, 0.3
            )
            min_sectors = st.slider(
                "最少行业数量",
                1, 10, 3
            )

    # 主区域
    col1, col2 = st.columns([3, 1])

    with col2:
        optimize_btn = st.button("🚀 开始优化", type="primary", use_container_width=True)
        compare_btn = st.button("📊 策略对比", use_container_width=True)

    if optimize_btn:
        with st.spinner("正在优化投资组合..."):
            try:
                # 创建优化器
                optimizer = PortfolioOptimizer(
                    market=market,
                    risk_free_rate=rf,
                    max_weight=max_weight,
                    min_weight=min_weight,
                    strategy=strategy
                )

                # 设置行业约束
                if enable_sector_constraint:
                    optimizer.set_sector_constraint(
                        max_sector_weight=max_sector_weight,
                        min_sectors=min_sectors
                    )

                # 执行优化
                weights, performance = optimizer.optimize_portfolio(
                    tickers=tickers,
                    years=years
                )

                # 保存到session state
                st.session_state['weights'] = weights
                st.session_state['performance'] = performance
                st.session_state['strategy'] = strategy

                st.success("✅ 优化完成!")

            except Exception as e:
                st.error(f"❌ 优化失败: {e}")
                import traceback
                st.code(traceback.format_exc())

    # 显示结果
    if 'weights' in st.session_state and 'performance' in st.session_state:
        weights = st.session_state['weights']
        performance = st.session_state['performance']

        # 策略信息
        st.info(f"当前策略: **{st.session_state.get('strategy', 'max_sharpe')}**")

        # 性能指标
        display_performance_metrics(performance)

        st.markdown("---")

        # 权重分布
        create_pie_chart(weights, "投资组合权重分布")

        st.markdown("---")

        # 行业分析
        display_sector_analysis(weights)

        st.markdown("---")

        # 详细数据
        with st.expander("📋 查看完整数据"):
            tab1, tab2 = st.tabs(["权重详情", "性能详情"])

            with tab1:
                weights_df = pd.DataFrame.from_dict(
                    weights, orient='index', columns=['权重']
                ).sort_values('权重', ascending=False)
                weights_df['权重'] = weights_df['权重'].apply(lambda x: f"{x:.4%}")
                st.dataframe(weights_df, use_container_width=True)

            with tab2:
                # 格式化性能数据
                formatted_perf = {}
                for k, v in performance.items():
                    if isinstance(v, float):
                        if 'return' in k.lower() or 'volatility' in k.lower() or 'drawdown' in k.lower():
                            formatted_perf[k] = f"{v:.4%}"
                        else:
                            formatted_perf[k] = f"{v:.4f}"
                    elif isinstance(v, dict):
                        formatted_perf[k] = str(v)
                    else:
                        formatted_perf[k] = v

                st.json(formatted_perf)

    # 策略对比
    if compare_btn:
        try:
            optimizer = PortfolioOptimizer(
                market=market,
                risk_free_rate=rf,
                max_weight=max_weight,
                min_weight=min_weight,
            )
            compare_strategies_view(optimizer, tickers, years)
        except Exception as e:
            st.error(f"策略对比失败: {e}")

    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        投资组合优化器 v2.0 | 仅供研究参考，不构成投资建议
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
