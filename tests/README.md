# HarborAI 测试项目

## 项目概述

HarborAI 测试项目是一个全面的测试框架，用于验证 HarborAI 系统的功能、性能、安全性和可靠性。本测试项目采用分层测试架构，包含单元测试、集成测试、功能测试、性能测试和安全测试。

## 目录结构

```
tests/
├── conftest.py              # pytest 全局配置
├── pytest.ini               # pytest 配置文件
├── requirements-test.txt     # 测试依赖包
├── test_config.py           # 测试配置模块
├── docker-compose.yml       # Docker 环境配置
├── .env.test                # 测试环境变量
├── README.md                # 本文档
├── data/                    # 测试数据
│   ├── fixtures/            # 测试夹具数据
│   ├── sql/                 # 数据库初始化脚本
│   └── uploads/             # 测试上传文件
├── fixtures/                # pytest fixtures
├── functional/              # 功能测试
├── integration/             # 集成测试
├── performance/             # 性能测试
├── security/                # 安全测试
├── utils/                   # 测试工具
├── reports/                 # 测试报告
├── scripts/                 # 测试脚本
└── logs/                    # 测试日志
```

## 测试框架

### 核心技术栈

- **pytest**: 主要测试框架
- **pytest-asyncio**: 异步测试支持
- **pytest-cov**: 代码覆盖率
- **pytest-html**: HTML 测试报告
- **pytest-xdist**: 并行测试执行
- **requests**: HTTP 客户端测试
- **aiohttp**: 异步 HTTP 测试
- **sqlalchemy**: 数据库测试
- **redis**: 缓存测试
- **locust**: 性能测试
- **bandit**: 安全扫描

### 测试分类

#### 1. 单元测试 (Unit Tests)
- 位置: 各模块目录下的 `test_*.py` 文件
- 目标: 测试单个函数或类的功能
- 覆盖率要求: ≥ 80%

#### 2. 集成测试 (Integration Tests)
- 位置: `integration/` 目录
- 目标: 测试模块间的交互
- 包含: API 集成、数据库集成、第三方服务集成

#### 3. 功能测试 (Functional Tests)
- 位置: `functional/` 目录
- 目标: 测试完整的业务流程
- 包含: 端到端测试、用户场景测试

#### 4. 性能测试 (Performance Tests)
- 位置: `performance/` 目录
- 目标: 测试系统性能指标
- 包含: 负载测试、压力测试、并发测试

#### 5. 安全测试 (Security Tests)
- 位置: `security/` 目录
- 目标: 测试系统安全性
- 包含: 漏洞扫描、渗透测试、权限测试

## 环境配置

### 1. 安装依赖

```bash
# 安装测试依赖
pip install -r requirements-test.txt
```

### 2. 环境变量配置

```bash
# 复制环境变量模板
cp .env.test .env

# 根据实际情况修改环境变量
vim .env
```

### 3. Docker 环境启动

```bash
# 启动测试环境
docker-compose up -d

# 检查服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f
```

### 4. 数据库初始化

```bash
# 等待数据库启动完成
docker-compose exec postgres_test pg_isready -U test_user -d harborai_test

# 运行数据库迁移（如果需要）
python -m alembic upgrade head
```

## 测试执行

### 基本命令

```bash
# 运行所有测试
pytest

# 运行特定目录的测试
pytest tests/integration/

# 运行特定文件的测试
pytest tests/integration/test_api.py

# 运行特定测试函数
pytest tests/integration/test_api.py::test_user_login
```

### 高级选项

```bash
# 并行执行测试（使用 4 个进程）
pytest -n 4

# 生成覆盖率报告
pytest --cov=src --cov-report=html

# 生成 HTML 测试报告
pytest --html=reports/report.html --self-contained-html

# 详细输出
pytest -v

# 只运行失败的测试
pytest --lf

# 遇到第一个失败就停止
pytest -x

# 显示最慢的 10 个测试
pytest --durations=10
```

### 按标记运行测试

```bash
# 运行单元测试
pytest -m unit

# 运行集成测试
pytest -m integration

# 运行功能测试
pytest -m functional

# 运行性能测试
pytest -m performance

# 运行安全测试
pytest -m security

# 跳过慢速测试
pytest -m "not slow"
```

## 性能测试

### Locust 性能测试

```bash
# 启动 Locust Web UI
locust -f performance/locustfile.py --host=http://localhost:8001

# 命令行模式运行性能测试
locust -f performance/locustfile.py --host=http://localhost:8001 \
       --users 100 --spawn-rate 10 --run-time 60s --headless
```

### 性能监控

- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3001 (admin/test_admin)

## 安全测试

### 代码安全扫描

```bash
# 使用 bandit 进行安全扫描
bandit -r src/ -f json -o reports/security_scan.json

# 使用 safety 检查依赖漏洞
safety check --json --output reports/safety_report.json
```

### 渗透测试

```bash
# 运行渗透测试脚本
python security/penetration_test.py
```

## 测试数据管理

### Fixtures 数据

- 位置: `fixtures/` 目录
- 格式: JSON, YAML, SQL
- 用途: 提供测试所需的静态数据

### 动态数据生成

```python
# 使用 Faker 生成测试数据
from faker import Faker
fake = Faker('zh_CN')

# 生成用户数据
user_data = {
    'name': fake.name(),
    'email': fake.email(),
    'phone': fake.phone_number()
}
```

## 测试报告

### 报告类型

1. **HTML 报告**: `reports/report.html`
2. **覆盖率报告**: `reports/htmlcov/index.html`
3. **性能报告**: `reports/performance/`
4. **安全报告**: `reports/security/`

### CI/CD 集成

```yaml
# GitHub Actions 示例
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements-test.txt
      - name: Start test services
        run: docker-compose up -d
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 最佳实践

### 1. 测试编写规范

- 测试函数命名: `test_功能描述`
- 测试类命名: `Test功能模块`
- 使用描述性的断言消息
- 遵循 AAA 模式 (Arrange, Act, Assert)

### 2. 测试数据隔离

- 每个测试使用独立的数据
- 测试结束后清理数据
- 使用事务回滚机制

### 3. 异步测试

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### 4. 参数化测试

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6)
])
def test_multiply_by_two(input, expected):
    assert multiply_by_two(input) == expected
```

## 故障排查

### 常见问题

1. **数据库连接失败**
   - 检查 Docker 服务是否启动
   - 验证环境变量配置
   - 查看数据库日志

2. **测试超时**
   - 增加超时时间配置
   - 检查网络连接
   - 优化测试逻辑

3. **内存不足**
   - 减少并行进程数
   - 优化测试数据大小
   - 增加系统内存

### 调试技巧

```bash
# 启用详细日志
pytest --log-cli-level=DEBUG

# 进入调试模式
pytest --pdb

# 捕获输出
pytest -s
```

## 贡献指南

1. 新增测试时，请确保遵循现有的目录结构和命名规范
2. 提交前运行完整的测试套件
3. 确保代码覆盖率不低于 80%
4. 添加必要的文档和注释
5. 遵循 VIBE Coding 规范

## 联系方式

如有问题或建议，请联系测试团队或提交 Issue。

---

**注意**: 本文档会随着项目发展持续更新，请定期查看最新版本。