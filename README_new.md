# Max Sharpe Portfolio (Multi-Market)

一个用于下载股票收盘价并构建"最大夏普比率"投资组合的轻量脚本，支持中国A股和美股市场。

## 功能
- 支持中国A股和美股两个市场
- 自动对齐对应交易所的交易日区间
- 使用 akshare 抓取A股后复权收盘价，使用 yfinance 抓取美股数据
- 用 PyPortfolioOpt 计算最大夏普组合，支持单资产权重上限约束
- 输出三类制品：
  - 原始价格 CSV
  - 组合权重 CSV
  - 组合表现 JSON（年化/按月折算）

## 快速开始

### 1) 安装依赖
建议使用 Conda 或 venv 创建环境，然后安装 `requirements.txt`：

```bash
pip install -r requirements.txt
```

### 2) 运行

#### 中国A股市场（默认）：
```bash
python portfolio.py \
  --market CN \
  --years 5 \
  --rf 0.01696 \
  --max-weight 0.25 \
  --output ./data
```

#### 美股市场：
```bash
python portfolio.py \
  --market US \
  --years 5 \
  --rf 0.02 \
  --max-weight 0.25 \
  --output ./data
```

#### 自定义股票代码：
```bash
# A股
python portfolio.py --market CN --tickers "600519,000858,601318"

# 美股
python portfolio.py --market US --tickers "AAPL,MSFT,GOOGL"
```

### 3) 参数说明

- `--market` 市场选择：CN（中国A股）或 US（美股），默认CN
- `--tickers` 指定股票列表（逗号分隔）；不指定则使用对应市场的默认股票池
- `--years` 回溯年数（与 `--start-date/--end-date` 互斥）
- `--start-date/--end-date` 指定区间（YYYY-MM-DD）
- `--rf` 无风险利率（年化）
- `--max-weight` 单一资产最大权重上限
- `--output` 输出目录
- `--quiet` 降低日志输出量

### 4) 输出文件

运行完成后，会在 `data/` 目录看到：
- `stock_data_<start>_<end>.csv` - 股票价格数据
- `weights_<start>_<end>.csv` - 投资组合权重
- `performance_<start>_<end>.json` - 投资组合表现指标

## 默认股票池

### 中国A股（CN）
包含25只主流A股，如茅台(600519)、平安银行(000001)等。

### 美股（US）
包含25只主流美股，如苹果(AAPL)、微软(MSFT)、谷歌(GOOGL)等。

## 技术细节

### 数据源
- **中国A股**: 使用 `akshare` 获取数据，支持前复权、后复权等调整
- **美股**: 使用 `yfinance` 获取数据，自动调整股价

### 交易日历
- **中国A股**: 使用上海证券交易所 (XSHG) 日历
- **美股**: 使用纽约证券交易所 (NYSE) 日历

## 说明与限制
- 数据来源于第三方接口，受网络与数据接口变动影响
- 回测仅演示组合构建，不包含交易成本、滑点和再平衡逻辑
- 本仓库以学习与研究为目的，不构成投资建议

## 许可证
本项目使用 MIT 许可证，详见 `LICENSE`。
