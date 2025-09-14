#!/usr/bin/env python3
"""
Debug script to identify where NaN values are introduced in the pipeline
"""

import logging
import pandas as pd
import numpy as np
from maxsharpe.core import PortfolioOptimizer
from maxsharpe.data import get_default_tickers
from maxsharpe.utils import calculate_returns, validate_price_data
from datetime import datetime, timedelta

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def debug_data_step_by_step():
    """Debug the data processing pipeline step by step"""
    
    print("=== NaN调试分析 ===")
    
    try:
        # Step 1: Create optimizer
        print("\n1. 创建优化器...")
        optimizer = PortfolioOptimizer(market="CN", risk_free_rate=0.03, max_weight=0.3)
        
        # Step 2: Get tickers
        print("\n2. 获取股票代码...")
        tickers = get_default_tickers("CN")[:3]  # Only use 3 stocks for debugging
        print(f"使用股票: {tickers}")
        
        # Step 3: Set date range
        print("\n3. 设置日期范围...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years
        print(f"日期范围: {start_date.date()} to {end_date.date()}")
        
        # Step 4: Fetch raw data
        print("\n4. 获取原始价格数据...")
        prices = optimizer.data_fetcher.fetch_prices(tickers, start_date, end_date)
        
        print(f"原始数据形状: {prices.shape}")
        print(f"原始数据列: {prices.columns.tolist()}")
        print(f"原始数据NaN数量: {prices.isnull().sum().sum()}")
        print(f"原始数据inf数量: {np.isinf(prices.values).sum()}")
        print("原始数据前5行:")
        print(prices.head())
        print("原始数据后5行:")
        print(prices.tail())
        
        # Check for any zero or negative values
        zero_or_neg = (prices <= 0).sum().sum()
        print(f"零值或负值数量: {zero_or_neg}")
        
        # Step 5: Validate price data
        print("\n5. 验证价格数据...")
        try:
            validate_price_data(prices)
            print("价格数据验证通过")
        except Exception as e:
            print(f"价格数据验证失败: {e}")
            return
        
        print(f"验证后数据形状: {prices.shape}")
        print(f"验证后数据NaN数量: {prices.isnull().sum().sum()}")
        print(f"验证后数据inf数量: {np.isinf(prices.values).sum()}")
        
        # Step 6: Calculate returns
        print("\n6. 计算收益率...")
        try:
            returns = calculate_returns(prices)
            print(f"收益率数据形状: {returns.shape}")
            print(f"收益率数据NaN数量: {returns.isnull().sum().sum()}")
            print(f"收益率数据inf数量: {np.isinf(returns.values).sum()}")
            print(f"收益率数据描述性统计:")
            print(returns.describe())
            
            # Check for extreme values
            max_return = returns.max().max()
            min_return = returns.min().min()
            print(f"最大收益率: {max_return}")
            print(f"最小收益率: {min_return}")
            
            if abs(max_return) > 1.0 or abs(min_return) > 1.0:
                print("警告: 发现极端收益率值（>100%）")
                
        except Exception as e:
            print(f"计算收益率失败: {e}")
            return
        
        # Step 7: Test our custom expected returns calculation
        print("\n7. 测试自定义预期收益率计算...")
        try:
            # Calculate using our method
            daily_returns = returns.mean()
            mu = daily_returns * 252  # Annualized
            
            print(f"预期收益率形状: {mu.shape}")
            print(f"预期收益率NaN数量: {mu.isnull().sum()}")
            print(f"预期收益率值:")
            print(mu)
            
            if mu.isnull().any():
                print("错误: 预期收益率包含NaN值!")
            else:
                print("成功: 预期收益率计算正常")
                
        except Exception as e:
            print(f"计算预期收益率失败: {e}")
            return
        
        # Step 8: Test our custom covariance calculation
        print("\n8. 测试自定义协方差矩阵计算...")
        try:
            # Use pandas cov method instead of PyPortfolioOpt
            S = returns.cov() * 252  # Annualized covariance matrix
            
            print(f"协方差矩阵形状: {S.shape}")
            print(f"协方差矩阵NaN数量: {pd.DataFrame(S).isnull().sum().sum()}")
            
            # Check if positive definite
            eigenvals = np.linalg.eigvals(S.values)
            print(f"最小特征值: {eigenvals.min()}")
            
            if (eigenvals <= 0).any():
                print("警告: 协方差矩阵不是正定的")
            else:
                print("成功: 协方差矩阵计算正常且为正定")
            
        except Exception as e:
            print(f"计算协方差矩阵失败: {e}")
            return
        
        # Step 9: Test the complete optimization
        print("\n9. 测试完整优化...")
        try:
            weights, performance = optimizer.optimize_portfolio(tickers, years=2)
            print("优化成功!")
            print(f"权重: {weights}")
            print(f"夏普比率: {performance.get('sharpe_ratio', 'N/A')}")
            
        except Exception as e:
            print(f"完整优化失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n=== 调试完成 ===")
        
    except Exception as e:
        print(f"调试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_step_by_step()