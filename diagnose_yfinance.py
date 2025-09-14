#!/usr/bin/env python3
"""
诊断yfinance网络问题
"""

import yfinance as yf
import requests
import time
from datetime import datetime, timedelta

def test_basic_connectivity():
    """测试基本网络连接"""
    print("=== 测试基本网络连接 ===")
    
    # 测试Yahoo Finance主站
    try:
        response = requests.get("https://finance.yahoo.com", timeout=10)
        print(f"Yahoo Finance主站: {response.status_code}")
    except Exception as e:
        print(f"Yahoo Finance主站连接失败: {e}")
    
    # 测试API端点
    try:
        response = requests.get("https://query1.finance.yahoo.com", timeout=10)
        print(f"Yahoo Finance API: {response.status_code}")
    except Exception as e:
        print(f"Yahoo Finance API连接失败: {e}")

def test_single_ticker():
    """测试单个股票数据获取"""
    print("\n=== 测试单个股票数据获取 ===")
    
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    for ticker in test_tickers:
        print(f"\n测试 {ticker}:")
        try:
            stock = yf.Ticker(ticker)
            
            # 测试基本信息
            info = stock.info
            print(f"  基本信息获取: {'成功' if info and 'symbol' in info else '失败'}")
            
            # 测试历史数据
            hist = stock.history(period="5d")
            print(f"  历史数据获取: {'成功' if not hist.empty else '失败'}")
            
            if not hist.empty:
                print(f"  数据点数量: {len(hist)}")
                print(f"  最新价格: {hist['Close'].iloc[-1]:.2f}")
            
        except Exception as e:
            print(f"  {ticker} 获取失败: {e}")

def test_batch_download():
    """测试批量下载"""
    print("\n=== 测试批量下载 ===")
    
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        print(f"批量下载 {tickers}...")
        data = yf.download(
            tickers=tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            group_by='ticker',
            auto_adjust=True,
            prepost=True,
            threads=True,
            proxy=None
        )
        
        if not data.empty:
            print(f"批量下载成功!")
            print(f"数据形状: {data.shape}")
            print(f"可用股票: {data.columns.levels[0].tolist() if hasattr(data.columns, 'levels') else 'Single ticker'}")
        else:
            print("批量下载失败: 数据为空")
            
    except Exception as e:
        print(f"批量下载失败: {e}")

def test_different_methods():
    """测试不同的下载方法"""
    print("\n=== 测试不同的下载方法 ===")
    
    ticker = 'AAPL'
    
    # 方法1: 使用session
    print("方法1: 使用自定义session")
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period="5d")
        print(f"  结果: {'成功' if not hist.empty else '失败'}")
        
    except Exception as e:
        print(f"  失败: {e}")
    
    # 方法2: 设置不同的参数
    print("方法2: 设置不同参数")
    try:
        data = yf.download(
            ticker,
            period="5d",
            interval="1d",
            auto_adjust=True,
            prepost=False,
            threads=False,
            proxy=None,
            progress=False
        )
        print(f"  结果: {'成功' if not data.empty else '失败'}")
        
    except Exception as e:
        print(f"  失败: {e}")

if __name__ == "__main__":
    print("开始诊断yfinance网络问题...")
    
    test_basic_connectivity()
    test_single_ticker()
    test_batch_download()
    test_different_methods()
    
    print("\n诊断完成!")