#!/usr/bin/env python3
"""
自定义股票组合示例 - Custom Portfolio Example

演示如何使用自定义股票列表进行投资组合优化
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio import compute_max_sharpe


def create_realistic_stock_data():
    """创建更真实的股票价格数据，模拟不同行业特征"""
    
    # 定义不同类型的股票及其特征
    stock_profiles = {
        "TECH_01": {"sector": "Technology", "volatility": 0.25, "trend": 0.12},
        "FINANCE_01": {"sector": "Finance", "volatility": 0.20, "trend": 0.08},
        "HEALTHCARE_01": {"sector": "Healthcare", "volatility": 0.18, "trend": 0.10},
        "ENERGY_01": {"sector": "Energy", "volatility": 0.30, "trend": 0.05},
        "CONSUMER_01": {"sector": "Consumer", "volatility": 0.15, "trend": 0.07},
        "UTILITIES_01": {"sector": "Utilities", "volatility": 0.12, "trend": 0.06},
    }
    
    n_periods = 504  # 约2年的交易日
    dates = pd.bdate_range("2022-01-01", periods=n_periods)
    
    np.random.seed(42)
    price_data = {}
    
    for ticker, profile in stock_profiles.items():
        # 基于行业特征生成价格序列
        volatility = profile["volatility"]
        annual_return = profile["trend"]
        daily_return = annual_return / 252
        daily_vol = volatility / np.sqrt(252)
        
        # 生成收益率序列
        returns = np.random.normal(daily_return, daily_vol, n_periods)
        
        # 添加一些市场相关性
        market_factor = np.random.normal(0, 0.01, n_periods)
        returns += 0.3 * market_factor  # 30%的市场β
        
        # 转换为价格
        initial_price = np.random.uniform(50, 200)
        prices = [initial_price]
        
        for i in range(1, n_periods):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 1))  # 防止价格变为负数或零
        
        price_data[ticker] = prices
    
    return pd.DataFrame(price_data, index=dates)


def analyze_portfolio_composition(weights, stock_profiles):
    """分析投资组合的行业构成"""
    
    # 按行业汇总权重
    sector_weights = {}
    for ticker, weight in weights.items():
        if ticker in stock_profiles:
            sector = stock_profiles[ticker]["sector"]
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    return sector_weights


def main():
    """主函数 - 自定义股票组合示例"""
    print("🎯 Max Sharpe Portfolio Optimizer - 自定义股票组合示例")
    print("=" * 65)
    
    # 股票资料
    stock_profiles = {
        "TECH_01": {"sector": "Technology", "volatility": 0.25, "trend": 0.12},
        "FINANCE_01": {"sector": "Finance", "volatility": 0.20, "trend": 0.08},
        "HEALTHCARE_01": {"sector": "Healthcare", "volatility": 0.18, "trend": 0.10},
        "ENERGY_01": {"sector": "Energy", "volatility": 0.30, "trend": 0.05},
        "CONSUMER_01": {"sector": "Consumer", "volatility": 0.15, "trend": 0.07},
        "UTILITIES_01": {"sector": "Utilities", "volatility": 0.12, "trend": 0.06},
    }
    
    # 1. 创建真实感的股票数据
    print("\n📊 创建多行业股票数据...")
    price_data = create_realistic_stock_data()
    
    print(f"数据形状: {price_data.shape}")
    print(f"日期范围: {price_data.index[0]} 到 {price_data.index[-1]}")
    
    print("\n🏢 股票行业分布:")
    for ticker, profile in stock_profiles.items():
        print(f"  {ticker}: {profile['sector']} (预期年化: {profile['trend']:.1%}, 波动率: {profile['volatility']:.1%})")
    
    # 显示价格统计
    print("\n📈 价格统计信息:")
    price_stats = price_data.describe()
    print(price_stats.round(2))
    
    # 2. 多种优化场景
    scenarios = [
        {"name": "保守型", "rf": 0.03, "max_weight": 0.25},
        {"name": "平衡型", "rf": 0.02, "max_weight": 0.35},
        {"name": "激进型", "rf": 0.015, "max_weight": 0.50},
    ]
    
    output_dir = "examples/output"
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n🎯 场景 {i+1}: {scenario['name']}")
        print("-" * 40)
        
        try:
            weights, performance = compute_max_sharpe(
                prices=price_data,
                rf=scenario["rf"],
                max_weight=scenario["max_weight"]
            )
            
            print(f"✅ 优化完成!")
            
            # 投资组合权重
            print(f"\n📋 投资组合权重 ({scenario['name']}):")
            for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                sector = stock_profiles.get(ticker, {}).get("sector", "Unknown")
                print(f"  {ticker} ({sector}): {weight:.2%}")
            
            # 行业分析
            sector_weights = analyze_portfolio_composition(weights, stock_profiles)
            print(f"\n🏭 行业权重分布:")
            for sector, weight in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True):
                print(f"  {sector}: {weight:.2%}")
            
            # 表现指标
            print(f"\n📊 投资组合表现:")
            print(f"  预期年化收益率: {performance['expected_annual_return']:.2%}")
            print(f"  年化波动率: {performance['annual_volatility']:.2%}")
            print(f"  夏普比率: {performance['sharpe_ratio']:.3f}")
            print(f"  无风险利率: {scenario['rf']:.1%}")
            print(f"  最大单一权重: {scenario['max_weight']:.1%}")
            
            # 保存结果
            scenario_name = scenario['name'].replace('型', '')
            
            # 权重文件
            weights_df = pd.DataFrame([
                {
                    'Ticker': ticker,
                    'Weight': weight,
                    'Sector': stock_profiles.get(ticker, {}).get('sector', 'Unknown')
                }
                for ticker, weight in weights.items()
            ])
            weights_file = f"{output_dir}/custom_portfolio_{scenario_name}_weights.csv"
            weights_df.to_csv(weights_file, index=False)
            
            # 记录汇总信息
            results_summary.append({
                'Scenario': scenario['name'],
                'Risk_Free_Rate': scenario['rf'],
                'Max_Weight': scenario['max_weight'],
                'Expected_Return': performance['expected_annual_return'],
                'Volatility': performance['annual_volatility'],
                'Sharpe_Ratio': performance['sharpe_ratio'],
                'Top_Holding': max(weights.items(), key=lambda x: x[1])[0],
                'Top_Weight': max(weights.values()),
                'Diversification': len([w for w in weights.values() if w > 0.05])  # 权重>5%的股票数
            })
            
        except Exception as e:
            print(f"❌ 优化失败: {e}")
            continue
    
    # 3. 场景对比分析
    if results_summary:
        print(f"\n📊 场景对比分析")
        print("=" * 65)
        
        summary_df = pd.DataFrame(results_summary)
        
        print(f"\n风险收益特征:")
        for _, row in summary_df.iterrows():
            print(f"  {row['Scenario']:8s}: 收益 {row['Expected_Return']:6.2%} | "
                  f"风险 {row['Volatility']:6.2%} | 夏普 {row['Sharpe_Ratio']:6.3f}")
        
        print(f"\n集中度分析:")
        for _, row in summary_df.iterrows():
            print(f"  {row['Scenario']:8s}: 最大持仓 {row['Top_Weight']:6.2%} | "
                  f"主要持仓数 {row['Diversification']:2d}")
        
        # 保存对比结果
        summary_file = f"{output_dir}/custom_portfolio_comparison.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 对比分析已保存至: {summary_file}")
        
        # 保存价格数据
        price_file = f"{output_dir}/custom_portfolio_prices.csv"
        price_data.to_csv(price_file)
        print(f"💾 价格数据已保存至: {price_file}")
    
    # 4. 投资建议
    print(f"\n💡 投资建议:")
    print("  - 保守型适合风险厌恶的投资者，更均衡的权重分配")
    print("  - 平衡型在风险和收益之间取得较好平衡")
    print("  - 激进型追求更高收益，但承担更大风险")
    print("  - 建议定期重新评估和再平衡投资组合")
    print("  - 考虑交易成本和税务影响")
    
    print(f"\n🎉 自定义股票组合示例运行完成!")


if __name__ == "__main__":
    main()
