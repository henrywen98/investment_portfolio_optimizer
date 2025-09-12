# 贡献指南 / Contributing Guide

首先，感谢您考虑为 Max Sharpe Portfolio Optimizer 项目做出贡献！🎉

## 💡 如何贡献

### 报告问题
- 使用 [GitHub Issues](https://github.com/henrywen98/investment_portfolio_optimizer/issues) 报告 bug
- 在提交新 issue 前，请先搜索是否已有相似问题
- 请提供详细的问题描述、重现步骤和环境信息

### 建议新功能
- 使用 GitHub Issues 提出新功能建议
- 清楚描述功能的用途和价值
- 如果可能，提供实现思路

### 提交代码
1. **Fork 仓库**
   ```bash
   git clone https://github.com/YOUR_USERNAME/investment_portfolio_optimizer.git
   cd investment_portfolio_optimizer
   ```

2. **创建开发环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **创建特性分支**
   ```bash
   git checkout -b feature/amazing-feature
   ```

4. **进行开发**
   - 编写代码
   - 添加测试
   - 更新文档

5. **代码检查**
   ```bash
   # 代码格式化
   black .
   isort .
   
   # 代码检查
   flake8 .
   
   # 运行测试
   pytest tests/ -v
   ```

6. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   git push origin feature/amazing-feature
   ```

7. **创建 Pull Request**

## 📋 代码规范

### Python 代码风格
- 使用 [Black](https://github.com/psf/black) 进行代码格式化
- 使用 [isort](https://github.com/PyCQA/isort) 进行导入排序
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 编码规范
- 使用 [flake8](https://flake8.pycqa.org/) 进行代码检查

### 提交信息规范
使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### 提交类型
- `feat`: 新功能
- `fix`: bug 修复
- `docs`: 文档更新
- `style`: 代码风格调整（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

#### 示例
```
feat(portfolio): add support for cryptocurrency markets

Add basic support for fetching cryptocurrency data from Binance API.
This includes price data fetching and portfolio optimization.

Closes #123
```

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_portfolio.py

# 运行测试并生成覆盖率报告
pytest --cov=. --cov-report=html
```

### 编写测试
- 为新功能编写单元测试
- 确保测试覆盖率不低于现有水平
- 测试文件命名：`test_*.py`
- 测试函数命名：`test_*`

### 测试数据
- 使用模拟数据进行测试
- 不要依赖外部 API（除非是集成测试）
- 保持测试的独立性和可重复性

## 📚 文档

### 代码文档
- 为函数和类添加清晰的文档字符串
- 使用 Google 风格的文档字符串

```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate the Sharpe ratio for a given return series.
    
    Args:
        returns: Time series of returns
        risk_free_rate: Annual risk-free rate (default: 0.02)
        
    Returns:
        The Sharpe ratio
        
    Raises:
        ValueError: If returns series is empty
    """
    pass
```

### README 更新
- 如果添加新功能，请更新 README.md
- 包含使用示例
- 更新参数说明

## 🔒 安全

- 不要在代码中硬编码敏感信息（API 密钥、密码等）
- 使用环境变量或配置文件管理敏感数据
- 报告安全问题请发送邮件至 henrywen98@example.com

## 💬 社区

### 行为准则
- 保持友善和尊重
- 欢迎新贡献者
- 提供建设性的反馈
- 尊重不同的观点和经验

### 沟通渠道
- GitHub Issues：技术讨论和问题报告
- GitHub Discussions：一般讨论和问答
- Pull Request：代码审查和讨论

## 🎯 开发优先级

当前项目的主要发展方向：

1. **数据源扩展**
   - 支持更多股票市场
   - 增加债券、商品等资产类型
   - 实时数据支持

2. **算法优化**
   - 更多优化目标（最小方差、风险平价等）
   - 动态再平衡策略
   - 回测功能增强

3. **用户体验**
   - Web 界面
   - 更好的可视化
   - 配置文件支持

4. **性能优化**
   - 并行计算
   - 缓存机制
   - 大数据集支持

## 📝 发布流程

1. **版本号管理**
   - 遵循 [Semantic Versioning](https://semver.org/)
   - 格式：`MAJOR.MINOR.PATCH`

2. **发布准备**
   - 更新版本号
   - 更新 CHANGELOG.md
   - 确保所有测试通过

3. **创建 Release**
   - 在 GitHub 上创建 release
   - 自动触发构建和发布流程

## 🙏 致谢

感谢所有贡献者的努力！您的贡献使这个项目变得更好。

## 📞 联系方式

如果您有任何问题，请通过以下方式联系：

- GitHub Issues: [项目 Issues](https://github.com/henrywen98/investment_portfolio_optimizer/issues)
- Email: henrywen98@example.com

再次感谢您的贡献！🚀
