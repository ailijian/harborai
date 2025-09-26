# HarborAI

<div align="center">

![HarborAI Logo](https://via.placeholder.com/200x100/2563eb/ffffff?text=HarborAI)

**高性能AI API代理和管理平台**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/harborai/harborai/workflows/Tests/badge.svg)](https://github.com/harborai/harborai/actions)
[![Coverage](https://codecov.io/gh/harborai/harborai/branch/main/graph/badge.svg)](https://codecov.io/gh/harborai/harborai)
[![PyPI Version](https://img.shields.io/pypi/v/harborai.svg)](https://pypi.org/project/harborai/)
[![Docker](https://img.shields.io/docker/v/harborai/harborai?label=docker)](https://hub.docker.com/r/harborai/harborai)

[文档](https://harborai.github.io/harborai/) | [快速开始](#快速开始) | [API文档](#api文档) | [贡献指南](#贡献指南)

</div>

## 🚀 特性

- **🔄 多提供商支持**: 统一接口支持 OpenAI、Anthropic、Google Gemini 等主流AI服务
- **⚡ 高性能**: 基于 FastAPI 和异步架构，支持高并发请求处理
- **🛡️ 安全可靠**: 内置认证、授权、限流和安全防护机制
- **📊 监控告警**: 完整的监控指标、分布式追踪和性能分析
- **🔧 易于扩展**: 模块化设计，支持自定义插件和中间件
- **📈 智能缓存**: 多层缓存策略，显著提升响应速度
- **🔀 负载均衡**: 智能路由和故障转移，确保服务高可用
- **📝 完整日志**: 结构化日志记录，便于问题排查和审计

## 📋 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [配置](#配置)
- [API文档](#api文档)
- [架构设计](#架构设计)
- [测试](#测试)
- [部署](#部署)
- [监控](#监控)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 🛠️ 安装

### 使用 pip 安装

```bash
pip install harborai
```

### 使用 Docker 安装

```bash
docker pull harborai/harborai:latest
docker run -p 8000:8000 harborai/harborai:latest
```

### 从源码安装

```bash
git clone https://github.com/harborai/harborai.git
cd harborai
pip install -e .
```

## 🚀 快速开始

### 1. 基本配置

复制环境配置文件并填入你的API密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# AI服务提供商API密钥
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/harborai
REDIS_URL=redis://localhost:6379/0
```

### 2. 启动服务

```bash
# 开发模式
harborai dev

# 或者使用 uvicorn
uvicorn harborai.main:app --reload
```

### 3. 测试API

```bash
# 健康检查
curl http://localhost:8000/health

# 聊天完成
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ]
  }'
```

## ⚙️ 配置

### 环境变量

HarborAI 支持通过环境变量进行配置。主要配置项包括：

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `ENVIRONMENT` | 运行环境 | `development` |
| `DEBUG` | 调试模式 | `false` |
| `HOST` | 服务器地址 | `0.0.0.0` |
| `PORT` | 服务器端口 | `8000` |
| `DATABASE_URL` | 数据库连接URL | - |
| `REDIS_URL` | Redis连接URL | - |
| `OPENAI_API_KEY` | OpenAI API密钥 | - |
| `ANTHROPIC_API_KEY` | Anthropic API密钥 | - |
| `GOOGLE_API_KEY` | Google API密钥 | - |

完整的配置选项请参考 [.env.example](.env.example) 文件。

### 配置文件

你也可以使用 YAML 或 JSON 配置文件：

```yaml
# config.yaml
app:
  name: HarborAI
  version: 1.0.0
  environment: production

server:
  host: 0.0.0.0
  port: 8000
  workers: 4

database:
  url: postgresql://user:password@localhost:5432/harborai
  pool_size: 10

redis:
  url: redis://localhost:6379/0
  max_connections: 10

ai_providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
    timeout: 60
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    base_url: https://api.anthropic.com
    timeout: 60
```

## 📚 API文档

### 聊天完成 API

**POST** `/v1/chat/completions`

与 OpenAI Chat Completions API 完全兼容的接口。

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "stream": false
}
```

### 流式响应

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Tell me a story"}
  ],
  "stream": true
}
```

### 结构化输出

```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Extract person info from: John Doe, 30 years old"}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person_info",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name", "age"]
      }
    }
  }
}
```

### 推理模型支持

```json
{
  "model": "o1-preview",
  "messages": [
    {"role": "user", "content": "Solve this math problem step by step: 2x + 5 = 13"}
  ]
}
```

## 🏗️ 架构设计

```mermaid
graph TB
    Client[客户端] --> LB[负载均衡器]
    LB --> API[API网关]
    API --> Auth[认证中间件]
    Auth --> RateLimit[限流中间件]
    RateLimit --> Cache[缓存层]
    Cache --> Router[智能路由]
    Router --> OpenAI[OpenAI]
    Router --> Anthropic[Anthropic]
    Router --> Google[Google Gemini]
    API --> Monitor[监控系统]
    API --> DB[(PostgreSQL)]
    API --> Redis[(Redis)]
```

### 核心组件

- **API网关**: 统一入口，处理请求路由和协议转换
- **认证授权**: 支持API Key、JWT等多种认证方式
- **智能路由**: 基于模型、负载、成本等因素的智能路由
- **缓存系统**: 多层缓存，包括响应缓存和模型缓存
- **监控系统**: 实时监控、告警和性能分析
- **数据存储**: PostgreSQL + Redis 的混合存储架构

## 🧪 测试

### 运行测试

```bash
# 安装测试依赖
pip install -r requirements-test.txt

# 运行所有测试
pytest

# 运行特定类型的测试
pytest tests/unit/          # 单元测试
pytest tests/functional/    # 功能测试
pytest tests/integration/   # 集成测试
pytest tests/performance/   # 性能测试

# 生成覆盖率报告
pytest --cov=harborai --cov-report=html
```

### 测试配置

```bash
# 设置测试环境
cp .env.example .env.test

# 运行测试数据库
docker run -d --name harborai-test-db \
  -e POSTGRES_DB=harborai_test \
  -e POSTGRES_USER=testuser \
  -e POSTGRES_PASSWORD=testpass \
  -p 5433:5432 postgres:15

# 运行测试Redis
docker run -d --name harborai-test-redis \
  -p 6380:6379 redis:7
```

### 性能测试

```bash
# 运行性能基准测试
pytest tests/performance/ -m benchmark

# 运行负载测试
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

## 🚀 部署

### Docker 部署

```bash
# 构建镜像
docker build -t harborai:latest .

# 使用 Docker Compose
docker-compose up -d
```

### Kubernetes 部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: harborai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: harborai
  template:
    metadata:
      labels:
        app: harborai
    spec:
      containers:
      - name: harborai
        image: harborai/harborai:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: harborai-secrets
              key: database-url
```

### 生产环境配置

```bash
# 使用 Gunicorn 部署
gunicorn harborai.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## 📊 监控

### Prometheus 指标

HarborAI 提供丰富的 Prometheus 指标：

- `harborai_requests_total`: 请求总数
- `harborai_request_duration_seconds`: 请求延迟
- `harborai_active_connections`: 活跃连接数
- `harborai_cache_hits_total`: 缓存命中数
- `harborai_ai_provider_requests_total`: AI提供商请求数
- `harborai_ai_provider_errors_total`: AI提供商错误数

### Grafana 仪表板

我们提供了预配置的 Grafana 仪表板模板，包括：

- 系统概览
- API性能监控
- AI提供商状态
- 错误率和延迟分析
- 资源使用情况

### 日志聚合

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "harborai.api",
  "message": "Chat completion request processed",
  "request_id": "req_123456",
  "user_id": "user_789",
  "model": "gpt-3.5-turbo",
  "tokens": 150,
  "duration_ms": 1200,
  "provider": "openai"
}
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/harborai/harborai.git
cd harborai

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装开发依赖
pip install -e ".[dev,test]"

# 安装 pre-commit 钩子
pre-commit install

# 运行测试
pytest
```

### 代码规范

- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 使用 flake8 进行代码检查
- 使用 mypy 进行类型检查
- 测试覆盖率不低于 80%

### 提交规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
feat: 添加新功能
fix: 修复bug
docs: 更新文档
style: 代码格式调整
refactor: 代码重构
test: 添加测试
chore: 构建过程或辅助工具的变动
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢以下开源项目：

- [FastAPI](https://fastapi.tiangolo.com/) - 现代、快速的 Web 框架
- [SQLAlchemy](https://www.sqlalchemy.org/) - Python SQL 工具包
- [Redis](https://redis.io/) - 内存数据结构存储
- [Prometheus](https://prometheus.io/) - 监控和告警工具
- [OpenTelemetry](https://opentelemetry.io/) - 可观测性框架

## 📞 联系我们

- 📧 邮箱: team@harborai.com
- 💬 讨论: [GitHub Discussions](https://github.com/harborai/harborai/discussions)
- 🐛 问题反馈: [GitHub Issues](https://github.com/harborai/harborai/issues)
- 📖 文档: [https://harborai.github.io/harborai/](https://harborai.github.io/harborai/)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个星标！**

</div>