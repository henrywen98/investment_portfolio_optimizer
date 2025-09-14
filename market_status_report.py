#!/usr/bin/env python3
"""
美股数据获取状态报告和用户指导
"""

import logging
from maxsharpe.core import PortfolioOptimizer
from maxsharpe.data import get_default_tickers

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def comprehensive_market_test():
    """综合市场测试和用户指导"""
    
    print("=" * 60)
    print("📊 投资组合优化器 - 市场数据获取报告")
    print("=" * 60)
    
    # 测试中国市场
    print("\n🇨🇳 中国A股市场测试")
    print("-" * 30)
    try:
        optimizer_cn = PortfolioOptimizer(market="CN", risk_free_rate=0.03, max_weight=0.2)
        tickers_cn = get_default_tickers("CN")[:5]  # 使用5只股票
        
        weights_cn, performance_cn = optimizer_cn.optimize_portfolio(
            tickers=tickers_cn, 
            years=2
        )
        
        print("✅ 中国A股市场：正常工作")
        print(f"   - 成功优化 {len(weights_cn)} 只股票")
        print(f"   - 夏普比率: {performance_cn.get('sharpe_ratio', 0):.3f}")
        print(f"   - 年化收益率: {performance_cn.get('expected_annual_return', 0):.1%}")
        print(f"   - 年化波动率: {performance_cn.get('annual_volatility', 0):.1%}")
        
        cn_status = "✅ 正常"
        
    except Exception as e:
        print(f"❌ 中国A股市场：存在问题")
        print(f"   错误: {e}")
        cn_status = "❌ 异常"
    
    # 测试美国市场
    print(f"\n🇺🇸 美国股市场测试")
    print("-" * 30)
    try:
        optimizer_us = PortfolioOptimizer(market="US", risk_free_rate=0.02, max_weight=0.3)
        tickers_us = ['AAPL', 'MSFT']  # 只测试2只股票
        
        weights_us, performance_us = optimizer_us.optimize_portfolio(
            tickers=tickers_us, 
            years=1
        )
        
        print("✅ 美国股市：正常工作")
        print(f"   - 成功优化 {len(weights_us)} 只股票")
        print(f"   - 夏普比率: {performance_us.get('sharpe_ratio', 0):.3f}")
        print(f"   - 年化收益率: {performance_us.get('expected_annual_return', 0):.1%}")
        print(f"   - 年化波动率: {performance_us.get('annual_volatility', 0):.1%}")
        
        us_status = "✅ 正常"
        
    except Exception as e:
        print(f"❌ 美国股市：当前不可用")
        error_msg = str(e)
        if "429" in error_msg or "速率限制" in error_msg or "too many" in error_msg.lower():
            print("   原因: Yahoo Finance API速率限制")
            print("   这是临时性问题，通常在几小时后恢复")
        elif "timezone" in error_msg or "delisted" in error_msg:
            print("   原因: Yahoo Finance API访问限制")
            print("   可能需要更换数据源或使用VPN")
        else:
            print(f"   原因: {error_msg}")
        
        us_status = "❌ 暂时不可用"
    
    # 总结报告
    print(f"\n📋 状态总结")
    print("=" * 30)
    print(f"🇨🇳 中国A股市场: {cn_status}")
    print(f"🇺🇸 美国股市场: {us_status}")
    
    # 用户建议
    print(f"\n💡 使用建议")
    print("=" * 30)
    
    if cn_status.startswith("✅"):
        print("✅ 中国A股功能完全可用，建议优先使用")
        print("   - 支持25只默认A股标的")
        print("   - 数据来源稳定 (akshare)")
        print("   - 适合中国投资者")
    
    if us_status.startswith("❌"):
        print("⚠️  美股功能暂时受限，但系统已优化处理:")
        print("   - 实现了重试机制和速率限制处理")
        print("   - 添加了长延迟避免API限制")
        print("   - 系统会在数据可用时自动恢复")
        print()
        print("🔧 解决方案:")
        print("   1. 等待几小时后重试 (API限制通常是临时的)")
        print("   2. 减少同时请求的股票数量 (≤3只)")
        print("   3. 使用VPN切换IP地址")
        print("   4. 考虑使用付费的数据API (Alpha Vantage, IEX Cloud等)")
    else:
        print("✅ 美股功能正常，可以正常使用")
    
    print(f"\n🚀 下一步操作")
    print("=" * 30)
    print("1. 使用Streamlit Web界面进行交互式优化")
    print("   命令: streamlit run streamlit_app.py")
    print()
    print("2. 使用命令行进行批量优化")
    print("   命令: python portfolio.py --market CN --years 3")
    print()
    print("3. 查看使用文档")
    print("   文件: README.md")

if __name__ == "__main__":
    comprehensive_market_test()