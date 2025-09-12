#!/usr/bin/env python3
"""
结果可视化示例 - Visualization Example

演示如何可视化投资组合优化结果
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio import compute_max_sharpe


def create_sample_data_for_visualization():
    """为可视化创建示例数据"""
    np.random.seed(42)
    
    # 创建8只股票的数据，有不同的特征
    stock_info = {
        'AAPL': {'sector': 'Technology', 'annual_return': 0.15, 'volatility': 0.25},
        'GOOGL': {'sector': 'Technology', 'annual_return': 0.12, 'volatility': 0.23},
        'JPM': {'sector': 'Finance', 'annual_return': 0.08, 'volatility': 0.20},
        'JNJ': {'sector': 'Healthcare', 'annual_return': 0.10, 'volatility': 0.16},
        'XOM': {'sector': 'Energy', 'annual_return': 0.05, 'volatility': 0.30},
        'WMT': {'sector': 'Consumer', 'annual_return': 0.07, 'volatility': 0.18},
        'PG': {'sector': 'Consumer', 'annual_return': 0.06, 'volatility': 0.15},
        'KO': {'sector': 'Consumer', 'annual_return': 0.05, 'volatility': 0.17},
    }
    
    n_periods = 504  # 2年数据
    dates = pd.bdate_range("2022-01-01", periods=n_periods)
    
    price_data = {}
    
    for ticker, info in stock_info.items():
        daily_return = info['annual_return'] / 252
        daily_vol = info['volatility'] / np.sqrt(252)
        
        returns = np.random.normal(daily_return, daily_vol, n_periods)
        
        # 添加市场相关性
        market_returns = np.random.normal(0.0003, 0.012, n_periods)
        returns += 0.4 * market_returns
        
        # 生成价格序列
        initial_price = 100
        prices = [initial_price]
        for i in range(1, n_periods):
            prices.append(prices[-1] * (1 + returns[i]))
        
        price_data[ticker] = prices
    
    return pd.DataFrame(price_data, index=dates), stock_info


def plot_price_trends(price_data, output_dir):
    """绘制价格走势图"""
    plt.figure(figsize=(12, 8))
    
    # 计算标准化价格（以初始价格为100）
    normalized_prices = price_data / price_data.iloc[0] * 100
    
    for column in normalized_prices.columns:
        plt.plot(normalized_prices.index, normalized_prices[column], 
                label=column, linewidth=2, alpha=0.8)
    
    plt.title('股票价格走势对比 (标准化至100)', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('标准化价格', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    price_trend_file = f"{output_dir}/price_trends.png"
    plt.savefig(price_trend_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return price_trend_file


def plot_portfolio_weights(weights, stock_info, output_dir):
    """绘制投资组合权重饼图和柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
    wedges, texts, autotexts = ax1.pie(
        weights.values(), 
        labels=[f"{k}\n({v:.1%})" for k, v in weights.items()],
        autopct='',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    ax1.set_title('投资组合权重分布', fontsize=14, fontweight='bold')
    
    # 柱状图（按行业分组）
    sector_data = {}
    for ticker, weight in weights.items():
        sector = stock_info.get(ticker, {}).get('sector', 'Unknown')
        if sector not in sector_data:
            sector_data[sector] = []
        sector_data[sector].append((ticker, weight))
    
    x_pos = 0
    bar_width = 0.6
    colors_sector = plt.cm.Set2(np.linspace(0, 1, len(sector_data)))
    
    for i, (sector, stocks) in enumerate(sector_data.items()):
        sector_weight = sum(weight for _, weight in stocks)
        bars = ax2.bar(x_pos, sector_weight, bar_width, 
                      color=colors_sector[i], alpha=0.8, label=sector)
        
        # 在柱子上添加具体股票信息
        y_offset = 0
        for ticker, weight in stocks:
            if weight > 0.02:  # 只显示权重>2%的股票
                ax2.text(x_pos, y_offset + weight/2, f'{ticker}\n{weight:.1%}', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            y_offset += weight
        
        x_pos += 1
    
    ax2.set_title('按行业分组的权重分布', fontsize=14, fontweight='bold')
    ax2.set_xlabel('行业', fontsize=12)
    ax2.set_ylabel('权重', fontsize=12)
    ax2.set_xticks(range(len(sector_data)))
    ax2.set_xticklabels(sector_data.keys(), rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    weights_file = f"{output_dir}/portfolio_weights.png"
    plt.savefig(weights_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return weights_file


def plot_risk_return_analysis(price_data, weights, performance, output_dir):
    """绘制风险收益分析图"""
    # 计算个股收益率和波动率
    returns = price_data.pct_change().dropna()
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 散点图：风险vs收益
    colors = plt.cm.viridis(np.linspace(0, 1, len(price_data.columns)))
    
    for i, ticker in enumerate(price_data.columns):
        weight = weights.get(ticker, 0)
        size = 200 + weight * 2000  # 根据权重调整点的大小
        ax1.scatter(annual_volatility[ticker], annual_returns[ticker], 
                   s=size, alpha=0.7, color=colors[i], label=ticker)
        
        # 添加标签
        ax1.annotate(ticker, 
                    (annual_volatility[ticker], annual_returns[ticker]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 标记投资组合位置
    portfolio_return = performance['expected_annual_return']
    portfolio_vol = performance['annual_volatility']
    ax1.scatter(portfolio_vol, portfolio_return, 
               s=500, marker='*', color='red', 
               label=f'投资组合 (夏普比率: {performance["sharpe_ratio"]:.3f})', 
               edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('年化波动率', fontsize=12)
    ax1.set_ylabel('年化收益率', fontsize=12)
    ax1.set_title('风险-收益散点图', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 相关性热力图
    correlation_matrix = returns.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('股票收益率相关性矩阵', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    risk_return_file = f"{output_dir}/risk_return_analysis.png"
    plt.savefig(risk_return_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return risk_return_file


def plot_cumulative_returns(price_data, weights, output_dir):
    """绘制累计收益率对比"""
    # 计算收益率
    returns = price_data.pct_change().dropna()
    
    # 计算投资组合收益率
    portfolio_returns = returns @ pd.Series(weights)
    
    # 计算累计收益率
    individual_cumret = (1 + returns).cumprod()
    portfolio_cumret = (1 + portfolio_returns).cumprod()
    
    plt.figure(figsize=(14, 8))
    
    # 绘制个股累计收益率
    for column in individual_cumret.columns:
        weight = weights.get(column, 0)
        alpha = 0.3 + weight * 2  # 根据权重调整透明度
        linewidth = 1 + weight * 3  # 根据权重调整线宽
        plt.plot(individual_cumret.index, individual_cumret[column], 
                label=f'{column} (权重: {weight:.1%})', 
                alpha=alpha, linewidth=linewidth)
    
    # 绘制投资组合累计收益率
    plt.plot(portfolio_cumret.index, portfolio_cumret, 
            color='red', linewidth=3, label='投资组合', alpha=0.9)
    
    plt.title('累计收益率对比', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('累计收益率', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    cumret_file = f"{output_dir}/cumulative_returns.png"
    plt.savefig(cumret_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cumret_file


def main():
    """主函数 - 可视化示例"""
    print("📊 Max Sharpe Portfolio Optimizer - 结果可视化示例")
    print("=" * 60)
    
    # 检查matplotlib后端
    try:
        import matplotlib
        print(f"Matplotlib 后端: {matplotlib.get_backend()}")
    except:
        print("⚠️  Matplotlib 可能需要配置")
    
    # 创建输出目录
    output_dir = "examples/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 创建示例数据
    print("\n📊 创建示例数据...")
    price_data, stock_info = create_sample_data_for_visualization()
    print(f"数据形状: {price_data.shape}")
    print(f"股票池: {list(price_data.columns)}")
    
    # 2. 优化投资组合
    print("\n🎯 优化投资组合...")
    try:
        weights, performance = compute_max_sharpe(
            prices=price_data,
            rf=0.02,
            max_weight=0.30
        )
        
        print("✅ 优化完成!")
        print(f"夏普比率: {performance['sharpe_ratio']:.3f}")
        
    except Exception as e:
        print(f"❌ 优化失败: {e}")
        return
    
    # 3. 生成可视化图表
    print("\n🎨 生成可视化图表...")
    
    try:
        # 价格走势图
        print("  📈 绘制价格走势图...")
        price_trend_file = plot_price_trends(price_data, output_dir)
        print(f"     已保存: {price_trend_file}")
        
        # 投资组合权重图
        print("  🥧 绘制权重分布图...")
        weights_file = plot_portfolio_weights(weights, stock_info, output_dir)
        print(f"     已保存: {weights_file}")
        
        # 风险收益分析图
        print("  📊 绘制风险收益分析图...")
        risk_return_file = plot_risk_return_analysis(price_data, weights, performance, output_dir)
        print(f"     已保存: {risk_return_file}")
        
        # 累计收益率图
        print("  📈 绘制累计收益率图...")
        cumret_file = plot_cumulative_returns(price_data, weights, output_dir)
        print(f"     已保存: {cumret_file}")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        print("💡 提示: 请确保安装了 matplotlib 和 seaborn:")
        print("   pip install matplotlib seaborn")
        return
    
    # 4. 生成投资组合报告
    print("\n📋 生成投资组合分析报告...")
    
    # 计算一些额外的统计指标
    returns = price_data.pct_change().dropna()
    portfolio_returns = returns @ pd.Series(weights)
    
    # 计算最大回撤
    cumret = (1 + portfolio_returns).cumprod()
    rolling_max = cumret.expanding().max()
    drawdown = (cumret - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 计算VaR (95%置信度)
    var_95 = np.percentile(portfolio_returns, 5)
    
    # 生成报告文本
    report = f"""
投资组合分析报告
================

基本信息:
- 分析期间: {price_data.index[0].strftime('%Y-%m-%d')} 至 {price_data.index[-1].strftime('%Y-%m-%d')}
- 股票数量: {len(weights)}
- 优化目标: 最大夏普比率

投资组合权重:
"""
    
    for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        sector = stock_info.get(ticker, {}).get('sector', 'Unknown')
        report += f"- {ticker} ({sector}): {weight:.2%}\n"
    
    report += f"""
风险收益指标:
- 预期年化收益率: {performance['expected_annual_return']:.2%}
- 年化波动率: {performance['annual_volatility']:.2%}
- 夏普比率: {performance['sharpe_ratio']:.3f}
- 最大回撤: {max_drawdown:.2%}
- VaR(95%): {var_95:.2%}

行业分布:
"""
    
    # 行业汇总
    sector_weights = {}
    for ticker, weight in weights.items():
        sector = stock_info.get(ticker, {}).get('sector', 'Unknown')
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    for sector, weight in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True):
        report += f"- {sector}: {weight:.2%}\n"
    
    report += f"""
生成的图表文件:
- 价格走势图: price_trends.png
- 权重分布图: portfolio_weights.png  
- 风险收益分析: risk_return_analysis.png
- 累计收益率: cumulative_returns.png

投资建议:
- 投资组合实现了较好的风险分散
- 建议定期重新评估和再平衡
- 关注市场环境变化对投资组合的影响
"""
    
    # 保存报告
    report_file = f"{output_dir}/portfolio_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 分析报告已保存: {report_file}")
    
    print(f"\n🎉 可视化示例运行完成!")
    print(f"📁 所有输出文件位于: {output_dir}/")
    print(f"🔍 请查看生成的图表和报告文件")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 请安装必要的依赖包:")
        print("   pip install matplotlib seaborn")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
