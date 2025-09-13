# 更新日志 / CHANGELOG

本文档记录了项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增
- 模块化代码结构，提高代码组织性和可维护性
- 新的 `MaxSharpeOptimizer` 类，提供更灵活的优化接口
- `DataFetcher` 类用于统一数据获取
- 完整的示例代码和可视化演示
- 全面的测试套件，包括单元测试和集成测试
- GitHub Actions CI/CD 流水线
- 代码质量检查（black, isort, flake8）
- 详细的贡献指南和项目文档
- Streamlit 前端界面，提供交互式使用体验

### 改进
- 更好的错误处理和日志记录
- 向后兼容的接口设计
- 增强的性能指标计算（包括最大回撤、VaR等）
- 更完善的权重约束验证
- 改进的README文档，包含徽章和详细使用说明
- 投资组合优化前增加价格数据验证

### 修复
- 修复了空数据和无效数据的处理
- 改进了相关性矩阵的数值稳定性
- 修复了权重和为0的边界情况

## [1.0.0] - 2025-01-XX

### 新增
- 初始发布版本
- 支持中国A股和美股市场
- 最大夏普比率投资组合优化
- 自动交易日对齐功能
- 命令行接口
- 基础测试套件

### 支持的功能
- 使用 akshare 获取A股数据
- 使用 yfinance 获取美股数据
- PyPortfolioOpt 投资组合优化
- 权重约束支持
- 多种输出格式（CSV, JSON）
- 性能指标计算

---

## 贡献者

- [@henrywen98](https://github.com/henrywen98) - 项目创始人和主要开发者

## 致谢

感谢以下开源项目的支持：
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)
- [akshare](https://github.com/akfamily/akshare)  
- [yfinance](https://github.com/ranaroussi/yfinance)
- [pandas-market-calendars](https://github.com/rsheftel/pandas_market_calendars)
