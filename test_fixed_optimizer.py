#!/usr/bin/env python3
"""
测试修复后的优化器，使用默认设置
"""

import logging
from maxsharpe.core import PortfolioOptimizer
from maxsharpe.data import get_default_tickers

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_fixed_optimizer():
    """测试修复后的投资组合优化器"""
    
    print("=== 测试修复后的优化器 ===")
    
    # 测试中国市场（使用更多股票和较低的max_weight约束）
    print("\n=== 测试中国市场 ===")
    try:
        optimizer_cn = PortfolioOptimizer(market="CN", risk_free_rate=0.03, max_weight=0.15)
        tickers_cn = get_default_tickers("CN")[:8]  # 使用8只股票
        
        print(f"使用股票: {tickers_cn}")
        
        weights_cn, performance_cn = optimizer_cn.optimize_portfolio(
            tickers=tickers_cn, 
            years=3
        )
        
        print("中国市场优化成功!")
        print("权重分配:")
        for ticker, weight in weights_cn.items():
            print(f"  {ticker}: {weight:.2%}")
        
        print("\n性能指标:")
        for key, value in performance_cn.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()
        
    except Exception as e:
        print(f"中国市场优化失败: {e}")
        import traceback
        traceback.print_exc()
        print()

    # 已移除：美国市场测试

if __name__ == "__main__":
    test_fixed_optimizer()
