# Dockerfile for Max Sharpe Portfolio Optimizer

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建非root用户
RUN useradd -m -u 1000 portfolio && \
    chown -R portfolio:portfolio /app
USER portfolio

# 创建输出目录
RUN mkdir -p /app/data

# 暴露端口（如果将来添加web界面）
EXPOSE 8000

# 默认命令
CMD ["python", "portfolio.py", "--help"]

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import maxsharpe; print('OK')" || exit 1

# 添加标签
LABEL maintainer="henrywen98@example.com"
LABEL version="1.0.0"
LABEL description="Max Sharpe Portfolio Optimizer"
LABEL org.opencontainers.image.source="https://github.com/henrywen98/investment_portfolio_optimizer"
