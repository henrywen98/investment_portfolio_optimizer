# Max Sharpe Portfolio Optimizer 📈

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

一个用于下载股票收盘价并构建"最大夏普比率"投资组合的Python工具，支持中国A股和美股市场。

> ⚠️ **免责声明**: 本项目仅用于教育和研究目的，不构成投资建议。投资有风险，请谨慎决策。

## ✨ 特性

## ✨ 特性

- 🌏 **多市场支持**: 支持中国A股和美股两个市场
- 📅 **智能日期对齐**: 自动对齐对应交易所的交易日区间
- 📊 **数据来源多样**: 使用 akshare 获取A股后复权收盘价，使用 yfinance 获取美股数据
- 🎯 **优化算法**: 基于 PyPortfolioOpt 计算最大夏普比率组合，支持权重约束
- 📋 **丰富输出**: 生成价格数据、组合权重、表现指标三类文件
- 🔧 **命令行友好**: 提供完整的CLI接口，易于使用和集成

## 🚀 快速开始

## 🚀 快速开始

### 📋 前置要求

- Python 3.8+
- pip 或 conda

### 🔧 安装

#### 方法一：克隆仓库
```bash
git clone https://github.com/henrywen98/investment_portfolio_optimizer.git
cd investment_portfolio_optimizer
pip install -r requirements.txt
```

#### 方法二：直接安装（推荐使用虚拟环境）
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 💡 使用示例

### 💡 使用示例

#### 🇨🇳 中国A股市场（默认）
```bash
python portfolio.py \
  --market CN \
  --years 5 \
  --rf 0.01696 \
  --max-weight 0.25 \
  --output ./data
```

#### 🇺🇸 美股市场
```bash
python portfolio.py \
  --market US \
  --years 5 \
  --rf 0.02 \
  --max-weight 0.25 \
  --output ./data
```

#### 🎯 自定义股票代码
```bash
# A股示例
python portfolio.py --market CN --tickers "600519,000858,601318"

# 美股示例
python portfolio.py --market US --tickers "AAPL,MSFT,GOOGL"
```

## ⚙️ 参数说明

## ⚙️ 参数说明

| 参数 | 描述 | 默认值 | 示例 |
|------|------|--------|------|
| `--market` | 市场选择：CN（中国A股）或 US（美股） | CN | `--market US` |
| `--tickers` | 自定义股票列表（逗号分隔） | 使用默认股票池 | `--tickers "AAPL,MSFT"` |
| `--years` | 回溯年数 | 5 | `--years 3` |
| `--start-date` | 开始日期（YYYY-MM-DD） | 自动计算 | `--start-date 2020-01-01` |
| `--end-date` | 结束日期（YYYY-MM-DD） | 今天 | `--end-date 2023-12-31` |
| `--rf` | 无风险利率（年化） | 0.02 | `--rf 0.015` |
| `--max-weight` | 单一资产最大权重上限 | 1.0 | `--max-weight 0.3` |
| `--output` | 输出目录 | ./data | `--output /path/to/output` |
| `--quiet` | 减少日志输出 | False | `--quiet` |

## 📁 输出文件

## 📁 输出文件

运行完成后，会在指定的输出目录生成以下文件：

| 文件类型 | 文件名格式 | 描述 |
|----------|------------|------|
| 📊 价格数据 | `stock_data_<start>_<end>.csv` | 包含所有股票的历史价格数据 |
| 🎯 权重配置 | `weights_<start>_<end>.csv` | 最优投资组合的权重分配 |
| 📈 表现指标 | `performance_<start>_<end>.json` | 投资组合的详细表现指标 |

### 表现指标说明

`performance.json` 文件包含以下关键指标：

- **年化收益率**: 投资组合的预期年化收益
- **年化波动率**: 投资组合的风险水平
- **夏普比率**: 风险调整后的收益指标
- **最大回撤**: 历史最大损失幅度

## 📈 默认股票池

## 📈 默认股票池

### 🇨🇳 中国A股（CN）
精选25只主流A股，包括：
- **消费**: 贵州茅台(600519)、五粮液(000858)
- **金融**: 中国平安(601318)、招商银行(600036)  
- **科技**: 中兴通讯(000063)、科大讯飞(002230)
- **能源**: 中国石化(600028)、中国石油(601857)
- *...更多优质标的*

### 🇺🇸 美股（US）
精选25只主流美股，包括：
- **科技**: 苹果(AAPL)、微软(MSFT)、谷歌(GOOGL)
- **消费**: 亚马逊(AMZN)、特斯拉(TSLA)
- **金融**: 摩根大通(JPM)、Visa(V)
- **医疗**: 强生(JNJ)、辉瑞(PFE)
- *...更多蓝筹股*

## 🛠️ 技术架构

## 🛠️ 技术架构

### 📊 数据源
- **🇨🇳 中国A股**: 使用 [akshare](https://akshare.akfamily.xyz/) 获取数据，支持前复权、后复权调整
- **🇺🇸 美股**: 使用 [yfinance](https://github.com/ranaroussi/yfinance) 获取数据，自动价格调整

### 📅 交易日历
- **🇨🇳 中国A股**: 基于上海证券交易所 (XSHG) 交易日历
- **🇺🇸 美股**: 基于纽约证券交易所 (NYSE) 交易日历

### 🎯 优化算法
- 使用 [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) 进行组合优化
- 支持最大夏普比率优化
- 可设置单资产权重上限约束
- 基于历史协方差矩阵和预期收益率

## ⚠️ 注意事项

- 📡 **数据依赖**: 依赖第三方数据接口，可能受网络状况影响
- 💰 **交易成本**: 回测未考虑实际交易成本、滑点和税费
- 🔄 **再平衡**: 未包含动态再平衡策略
- 📚 **仅供学习**: 本项目主要用于教育和研究，不构成投资建议

## 🤝 贡献指南

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 开发流程
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 💡 问题和建议: [Issues](https://github.com/henrywen98/investment_portfolio_optimizer/issues)
- 🌟 如果这个项目对你有帮助，请给个星标！

## 🙏 致谢

感谢以下开源项目：
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) - 投资组合优化
- [akshare](https://github.com/akfamily/akshare) - 中国金融数据
- [yfinance](https://github.com/ranaroussi/yfinance) - 美股数据
- [pandas-market-calendars](https://github.com/rsheftel/pandas_market_calendars) - 市场交易日历
