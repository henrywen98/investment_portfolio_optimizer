# Max Sharpe Portfolio (A-shares)

一个用于下载中国A股收盘价并构建“最大夏普比率”投资组合的轻量脚本。

## 功能
- 自动对齐上交所交易日区间。
- 使用 akshare 抓取后复权收盘价。
- 用 PyPortfolioOpt 计算最大夏普组合，支持单资产权重上限约束。
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

```bash
python portfolio.py \
  --years 5 \
  --rf 0.01696 \
  --max-weight 0.25 \
  --output ./data
```

可选参数：
- `--tickers` 指定股票列表（逗号分隔）；默认内置一组样例。
- `--start-date/--end-date` 指定区间（YYYY-MM-DD），与 `--years` 互斥。
- `--quiet` 降低日志输出量。

运行完成后，会在 `data/` 目录看到：
- `stock_data_<start>_<end>.csv`
- `weights_<start>_<end>.csv`
- `performance_<start>_<end>.json`

## 说明与限制
- 数据来源于 akshare，受网络与数据接口变动影响；如遇列名变化（例如“收盘”），请在 `fetch_prices` 中调整。
- 回测仅演示组合构建，不包含交易成本、滑点和再平衡逻辑。
- 本仓库以学习与研究为目的，不构成投资建议。

## 许可证
本项目使用 MIT 许可证，详见 `LICENSE`。
