#!/usr/bin/env python3
"""
快速测试简化后的美股数据获取
"""

import logging
from maxsharpe.data import DataFetcher
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def quick_us_test():
    """快速测试美股数据获取"""
    
    print("=== 快速测试美股数据获取 ===")
    
    # 创建数据获取器
    fetcher = DataFetcher(market="US")
    
    # 测试单个股票
    print("\n--- 测试单个股票 (AAPL) ---")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 2个月的数据
        
        data = fetcher._fetch_us_prices(['AAPL'], start_date, end_date)
        
        if not data.empty:
            print(f"✅ 单个股票测试成功!")
            print(f"数据形状: {data.shape}")
            print(f"日期范围: {data.index[0]} 到 {data.index[-1]}")
            print(f"最新价格: {data.iloc[-1, 0]:.2f}")
            print()
            
            # 如果第一个成功，尝试两个股票
            print("--- 测试两个股票 (AAPL, MSFT) ---")
            try:
                data2 = fetcher._fetch_us_prices(['AAPL', 'MSFT'], start_date, end_date)
                print(f"✅ 两个股票测试成功!")
                print(f"数据形状: {data2.shape}")
                print(f"股票列: {data2.columns.tolist()}")
            except Exception as e:
                print(f"❌ 两个股票测试失败: {e}")
        else:
            print(f"❌ 单个股票测试失败: 数据为空")
            
    except Exception as e:
        print(f"❌ 单个股票测试失败: {e}")

if __name__ == "__main__":
    quick_us_test()