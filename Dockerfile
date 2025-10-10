# HarborAI Docker 镜像构建文件
# 多阶段构建，优化镜像大小和安全性

# 第一阶段：构建阶段
FROM python:3.10-slim as builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml README.md ./
COPY requirements.txt ./

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级 pip 并安装构建工具和依赖
RUN pip install --upgrade pip && \
    pip install --no-cache-dir build wheel && \
    pip install --no-cache-dir -r requirements.txt

# 第二阶段：运行阶段
FROM python:3.10-slim as runtime

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HARBORAI_ENV=production

# 创建非 root 用户
RUN groupadd -r harborai && useradd -r -g harborai harborai

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY harborai/ ./harborai/
COPY README.md LICENSE ./

# 安装应用本身（以开发模式）
RUN pip install -e .

# 创建必要的目录
RUN mkdir -p /app/logs /app/data && \
    chown -R harborai:harborai /app

# 切换到非 root 用户
USER harborai

# 健康检查 - 使用curl检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "harborai.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]