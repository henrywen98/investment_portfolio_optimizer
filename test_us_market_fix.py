#!/usr/bin/env python3
"""
测试修复后的美股数据获取功能
"""

import logging
from maxsharpe.core import PortfolioOptimizer
from maxsharpe.data import get_default_tickers

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_us_market_fix():
    """测试修复后的美股市场数据获取"""
    
    print("=== 测试修复后的美股数据获取 ===")
    
    # 使用较少的股票来降低失败风险
    test_scenarios = [
        {
            'name': '单个股票测试',
            'tickers': ['AAPL'],
            'max_weight': 1.0
        },
        {
            'name': '两个股票测试',
            'tickers': ['AAPL', 'MSFT'],
            'max_weight': 0.8
        },
        {
            'name': '三个股票测试',
            'tickers': ['AAPL', 'MSFT', 'GOOGL'],
            'max_weight': 0.6
        },
        {
            'name': '五个股票测试',
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            'max_weight': 0.4
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n=== {scenario['name']} ===")
        try:
            optimizer = PortfolioOptimizer(
                market="US", 
                risk_free_rate=0.02, 
                max_weight=scenario['max_weight']
            )
            
            print(f"使用股票: {scenario['tickers']}")
            
            weights, performance = optimizer.optimize_portfolio(
                tickers=scenario['tickers'], 
                years=1  # 使用较短的时间窗口以提高成功率
            )
            
            print(f"✅ {scenario['name']} 成功!")
            print("权重分配:")
            for ticker, weight in weights.items():
                print(f"  {ticker}: {weight:.2%}")
            
            print(f"夏普比率: {performance.get('sharpe_ratio', 'N/A'):.3f}")
            print(f"年化收益率: {performance.get('expected_annual_return', 'N/A'):.3f}")
            print(f"年化波动率: {performance.get('annual_volatility', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"❌ {scenario['name']} 失败: {e}")
            continue
    
    print("\n=== 测试完成 ===")

def test_mixed_markets():
    """测试中美两个市场的完整流程"""
    
    print("\n=== 测试中美市场对比 ===")
    
    markets = [
        {
            'name': '中国A股',
            'market': 'CN',
            'tickers': get_default_tickers("CN")[:6],
            'rf': 0.03,
            'max_weight': 0.2
        },
        {
            'name': '美股',
            'market': 'US',
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'rf': 0.02,
            'max_weight': 0.3
        }
    ]
    
    results = {}
    
    for market_config in markets:
        print(f"\n--- {market_config['name']} ---")
        try:
            optimizer = PortfolioOptimizer(
                market=market_config['market'],
                risk_free_rate=market_config['rf'],
                max_weight=market_config['max_weight']
            )
            
            weights, performance = optimizer.optimize_portfolio(
                tickers=market_config['tickers'],
                years=2
            )
            
            results[market_config['name']] = {
                'success': True,
                'weights': weights,
                'performance': performance
            }
            
            print(f"✅ {market_config['name']} 优化成功!")
            print(f"夏普比率: {performance.get('sharpe_ratio', 0):.3f}")
            print(f"年化收益率: {performance.get('expected_annual_return', 0):.3f}")
            print(f"年化波动率: {performance.get('annual_volatility', 0):.3f}")
            
        except Exception as e:
            print(f"❌ {market_config['name']} 失败: {e}")
            results[market_config['name']] = {'success': False, 'error': str(e)}
    
    # 总结结果
    print(f"\n=== 总结 ===")
    for market, result in results.items():
        if result['success']:
            print(f"✅ {market}: 成功 (夏普比率: {result['performance'].get('sharpe_ratio', 0):.3f})")
        else:
            print(f"❌ {market}: 失败 ({result.get('error', '未知错误')})")

if __name__ == "__main__":
    test_us_market_fix()
    test_mixed_markets()