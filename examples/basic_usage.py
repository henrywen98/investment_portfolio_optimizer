#!/usr/bin/env python3
"""
基础使用示例 - Basic Usage Example

演示如何使用 Max Sharpe Portfolio Optimizer 的基本功能
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加父目录到路径，以便导入主模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio import compute_max_sharpe


def create_sample_data(n_assets=5, n_periods=252, seed=42):
    """创建示例股票价格数据"""
    np.random.seed(seed)
    
    # 创建股票代码
    tickers = [f"STOCK_{i+1:02d}" for i in range(n_assets)]
    
    # 创建日期索引
    dates = pd.bdate_range("2023-01-01", periods=n_periods)
    
    # 模拟价格走势
    returns = np.random.multivariate_normal(
        mean=[0.0008] * n_assets,  # 日均收益率
        cov=np.random.uniform(0.0001, 0.0004, size=(n_assets, n_assets)),
        size=n_periods
    )
    
    # 确保协方差矩阵是正定的
    cov_matrix = np.cov(returns.T)
    eigenvals, eigenvects = np.linalg.eigh(cov_matrix)
    eigenvals = np.maximum(eigenvals, 0.0001)  # 确保所有特征值为正
    cov_matrix = eigenvects @ np.diag(eigenvals) @ eigenvects.T
    
    returns = np.random.multivariate_normal(
        mean=[0.0008] * n_assets,
        cov=cov_matrix,
        size=n_periods
    )
    
    # 转换为价格（假设初始价格为100）
    initial_prices = [100] * n_assets
    prices = np.zeros((n_periods, n_assets))
    prices[0] = initial_prices
    
    for i in range(1, n_periods):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    return pd.DataFrame(prices, index=dates, columns=tickers)


def main():
    """主函数 - 演示基础使用"""
    print("🚀 Max Sharpe Portfolio Optimizer - 基础使用示例")
    print("=" * 60)
    
    # 1. 创建示例数据
    print("\n📊 创建示例数据...")
    price_data = create_sample_data(n_assets=5, n_periods=252)
    print(f"数据形状: {price_data.shape}")
    print(f"日期范围: {price_data.index[0]} 到 {price_data.index[-1]}")
    print(f"股票代码: {list(price_data.columns)}")
    
    # 显示前几行数据
    print("\n前5行价格数据:")
    print(price_data.head())
    
    # 2. 计算最优投资组合
    print("\n🎯 计算最大夏普比率投资组合...")
    
    # 基础参数
    risk_free_rate = 0.02  # 2% 无风险利率
    max_weight = 0.4       # 单一资产最大权重 40%
    
    try:
        weights, performance = compute_max_sharpe(
            prices=price_data,
            rf=risk_free_rate,
            max_weight=max_weight
        )
        
        print("\n✅ 优化成功完成!")
        
        # 3. 显示结果
        print("\n📋 投资组合权重:")
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight:.2%}")
        
        # 检查性能数据格式
        if isinstance(performance, tuple):
            # 旧格式：(annual_return, annual_vol, sharpe)
            ann_ret, ann_vol, sharpe = performance
            performance_dict = {
                'expected_annual_return': ann_ret,
                'annual_volatility': ann_vol,
                'sharpe_ratio': sharpe
            }
        else:
            # 新格式：字典
            performance_dict = performance
        
        print(f"\n📈 投资组合表现:")
        print(f"  预期年化收益率: {performance_dict['expected_annual_return']:.2%}")
        print(f"  年化波动率: {performance_dict['annual_volatility']:.2%}")
        print(f"  夏普比率: {performance_dict['sharpe_ratio']:.3f}")
        
        # 4. 保存结果
        output_dir = "examples/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存权重
        weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        weights_file = f"{output_dir}/basic_example_weights.csv"
        weights_df.to_csv(weights_file)
        print(f"\n💾 权重已保存至: {weights_file}")
        
        # 保存价格数据
        price_file = f"{output_dir}/basic_example_prices.csv"
        price_data.to_csv(price_file)
        print(f"💾 价格数据已保存至: {price_file}")
        
        # 保存表现指标
        import json
        performance_file = f"{output_dir}/basic_example_performance.json"
        
        # 确保是字典格式
        if isinstance(performance, tuple):
            perf_dict = {
                'expected_annual_return': performance_dict['expected_annual_return'],
                'annual_volatility': performance_dict['annual_volatility'],
                'sharpe_ratio': performance_dict['sharpe_ratio']
            }
        else:
            perf_dict = performance_dict
            
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(perf_dict, f, indent=2, ensure_ascii=False)
        print(f"💾 表现指标已保存至: {performance_file}")
        
        # 5. 风险分析
        print(f"\n🔍 风险分析:")
        total_weight = sum(weights.values())
        print(f"  权重总和: {total_weight:.1%}")
        
        max_single_weight = max(weights.values())
        min_single_weight = min(weights.values())
        print(f"  最大单一权重: {max_single_weight:.1%}")
        print(f"  最小单一权重: {min_single_weight:.1%}")
        
        # 计算集中度（HHI指数）
        hhi = sum(w**2 for w in weights.values())
        print(f"  集中度指数(HHI): {hhi:.3f}")
        
        if hhi > 0.25:
            print("  ⚠️  投资组合集中度较高")
        else:
            print("  ✅ 投资组合分散化良好")
        
        print(f"\n🎉 基础示例运行完成! 结果文件保存在 {output_dir}/ 目录中")
        
    except Exception as e:
        print(f"\n❌ 优化过程中出现错误: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ 示例运行成功!")
    else:
        print("\n💥 示例运行失败!")
        sys.exit(1)
