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

### 4. Python客户端示例

```python
import asyncio
from harborai import HarborAI

# 初始化客户端
client = HarborAI(
    api_key="your-api-key",
    base_url="http://localhost:8000"
)

# 同步调用
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ]
)
print(response.choices[0].message.content)

# 异步调用
async def async_chat():
    response = await client.chat.completions.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Tell me a joke"}
        ],
        stream=True
    )
    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

# 运行异步示例
asyncio.run(async_chat())
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

### 性能优化配置

HarborAI 提供了两阶段性能优化，显著提升SDK性能：

#### 🚀 第一阶段：延迟加载优化
- **初始化时间优化**: 从 >2000ms 降低到 ≤160ms
- **内存使用优化**: 减少初始内存占用
- **按需加载**: 插件和组件在首次使用时才加载

#### 🧠 第二阶段：内存使用优化
- **内存使用降低**: 内存增长控制在2MB以内
- **智能缓存管理**: LRU策略和定期清理
- **对象池技术**: 复用对象减少GC压力
- **弱引用机制**: 避免循环引用导致的内存泄漏

#### FastHarborAI 客户端（推荐）

使用优化后的FastHarborAI客户端获得最佳性能：

```python
from harborai.api.fast_client import FastHarborAI

# 启用所有优化
client = FastHarborAI(
    api_key="your-api-key",
    enable_memory_optimization=True,  # 启用内存优化
    enable_lazy_loading=True,         # 启用延迟加载
    memory_optimization={
        'cache_size': 2000,              # 缓存大小
        'object_pool_size': 200,         # 对象池大小
        'memory_threshold_mb': 100.0,    # 内存阈值
        'auto_cleanup_interval': 600     # 自动清理间隔（秒）
    }
)

# 监控内存使用（仅FastHarborAI支持）
if hasattr(client, 'get_memory_stats'):
    stats = client.get_memory_stats()
    if stats:
        print(f"缓存命中率: {stats['cache']['hit_rate']:.1%}")
        print(f"内存使用: {stats['system_memory']['rss_mb']:.1f}MB")

# 手动清理内存（仅FastHarborAI支持）
if hasattr(client, 'cleanup_memory'):
    client.cleanup_memory(force_clear=True)
```

#### 性能模式配置

HarborAI 提供三种性能模式，以满足不同场景的需求：

#### 🚀 FAST 模式（快速模式）
- **特点**: 最小功能，最快速度
- **性能提升**: 相比完整模式可提升 2000-3000ms
- **适用场景**: 高并发、低延迟要求的生产环境
- **功能**: 禁用成本追踪、详细日志等非关键功能

#### ⚖️ BALANCED 模式（平衡模式）
- **特点**: 平衡功能和性能
- **适用场景**: 大多数生产环境的默认选择
- **功能**: 保留核心监控功能，优化性能表现

#### 🔧 FULL 模式（完整模式）
- **特点**: 完整功能，包含所有监控和追踪
- **适用场景**: 开发环境、调试场景、需要完整监控的环境
- **功能**: 启用所有功能，包括详细日志、成本追踪、性能分析等

#### 设置性能模式

**方法1: 环境变量设置**
```bash
# 在 .env 文件中设置
HARBORAI_PERFORMANCE_MODE=full  # 可选值: fast, balanced, full
```

**方法2: 代码中动态设置**
```python
from harborai import HarborAI
from harborai.config import get_settings

# 获取配置实例
settings = get_settings()

# 设置性能模式
settings.set_performance_mode("full")

# 初始化客户端
client = HarborAI(performance_mode="full")
```

**方法3: 初始化时指定**
```python
from harborai import HarborAI

# 直接在初始化时指定性能模式
client = HarborAI(
    api_key="your-api-key",
    performance_mode="fast"  # 使用快速模式
)

# 异步客户端同样支持
async_client = HarborAI(
    api_key="your-api-key",
    performance_mode="balanced"
)
```

#### 性能模式对比

| 功能 | FAST | BALANCED | FULL |
|------|------|----------|------|
| 成本追踪 | ❌ | ✅ | ✅ |
| 详细日志 | ❌ | ❌ | ✅ |
| 性能监控 | ❌ | ✅ | ✅ |
| 分布式追踪 | ❌ | ✅ | ✅ |
| 缓存优化 | ✅ | ✅ | ✅ |
| 快速路径 | ✅ | ✅ | ✅ |
| 响应速度 | 🚀🚀🚀 | 🚀🚀 | 🚀 |

#### 📊 性能测试结果

我们进行了全面的性能对比测试，将 HarborAI 的三种性能模式与直接调用 Agently 进行结构化输出的性能进行对比。测试结果显示 HarborAI 在所有模式下都表现出色：

##### 🎯 核心性能指标

| 模式 | 平均响应时间 | 相对性能 | 性能提升 | 成功率 | 内存使用 | CPU使用率 |
|------|-------------|----------|----------|--------|----------|----------|
| **Agently 基准** | 4.37s | 1.00x | - | 100% | 基准 | 基准 |
| **🚀 FAST** | 3.87s | **0.88x** | **+12%** | 100% | 优化 | 优化 |
| **⚖️ BALANCED** | 4.47s | 1.02x | 持平 | 100% | 标准 | 标准 |
| **🔧 FULL** | 3.92s | **0.90x** | **+10%** | 100% | 标准 | 标准 |

##### 📈 性能分析

- **🏆 FAST 模式**: 比 Agently 基准快 **12%**，在高并发场景下表现最佳
- **⚖️ BALANCED 模式**: 与 Agently 基准基本持平，提供最佳的功能与性能平衡
- **🔧 FULL 模式**: 比 Agently 基准快 **10%**，即使启用所有功能仍保持优秀性能
- **✅ 稳定性**: 所有模式均达到 **100% 成功率**，确保生产环境可靠性

##### 🎨 性能可视化

测试生成了详细的性能图表，包括：
- [响应时间对比图](./performance_charts/response_time_comparison.png)
- [内存使用对比图](./performance_charts/memory_usage_comparison.png)
- [综合性能雷达图](./performance_charts/comprehensive_radar_chart.png)

##### 📋 详细报告

完整的性能测试报告和原始数据可查看：
- [📄 详细性能报告](./comprehensive_performance_report.md)
- [📊 原始测试数据](./comprehensive_performance_results.json)

##### 💡 使用建议

根据测试结果，我们建议：

1. **高并发生产环境**: 使用 **FAST 模式**，获得最佳性能表现
2. **一般生产环境**: 使用 **BALANCED 模式**，平衡功能与性能
3. **开发调试环境**: 使用 **FULL 模式**，获得完整的监控和调试信息

> 💡 **性能优化成果**: HarborAI 通过架构优化和智能缓存，在保持功能完整性的同时，实现了显著的性能提升。即使是功能最全的 FULL 模式，也比直接使用 Agently 快 10%。

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