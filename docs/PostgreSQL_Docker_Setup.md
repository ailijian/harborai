# HarborAI PostgreSQL Docker 环境配置指南

## 概述

本文档提供了为 HarborAI 项目配置 PostgreSQL Docker 环境的完整指南，包括安装、配置、测试和故障排查。

## 环境要求

- Docker Desktop for Windows
- PowerShell 5.0+
- Python 3.8+
- 项目依赖：psycopg2-binary

## 快速开始

### 1. 启动 PostgreSQL 服务

```powershell
# 在项目根目录下执行
docker-compose up -d
```

### 2. 验证服务状态

```powershell
# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs postgres
```

### 3. 运行测试

```powershell
# 运行完整测试套件
python test_postgres_docker.py
```

## 详细配置

### Docker Compose 配置

项目使用 `docker-compose.yml` 配置 PostgreSQL 服务：

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: harborai_postgres
    environment:
      POSTGRES_DB: harborai
      POSTGRES_USER: harborai
      POSTGRES_PASSWORD: harborai_password_2024
    ports:
      - "5433:5432"  # 避免与本地 PostgreSQL 冲突
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init:/docker-entrypoint-initdb.d
    networks:
      - harborai_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U harborai -d harborai"]
      interval: 10s
      timeout: 5s
      retries: 5
    command: [
      "postgres",
      "-c", "log_statement=all",
      "-c", "log_destination=stderr",
      "-c", "logging_collector=off",
      "-c", "log_min_duration_statement=1000"
    ]

volumes:
  postgres_data:

networks:
  harborai_network:
    driver: bridge
```

### 数据库初始化

初始化脚本位于 `docker/postgres/init/01-init-database.sql`，包含：

- 核心表结构（api_logs, trace_logs, model_usage, harborai_logs）
- 索引优化
- 视图定义
- 权限设置

### 环境变量配置

`.env` 文件中的 PostgreSQL 配置：

```env
# PostgreSQL 配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=harborai
POSTGRES_USER=harborai
POSTGRES_PASSWORD=harborai_password_2024
```

## 功能测试

### 测试脚本功能

`test_postgres_docker.py` 提供了全面的测试套件：

1. **基础连接测试** - 验证数据库连接
2. **表结构测试** - 检查所有必需表是否存在
3. **PostgreSQL 客户端测试** - 测试查询功能
4. **日志记录器测试** - 验证日志写入功能
5. **异步日志测试** - 测试异步日志记录
6. **日志查看测试** - 验证日志检索功能
7. **性能测试** - 测试批量操作性能

### 运行特定测试

```powershell
# 运行完整测试
python test_postgres_docker.py

# 查看详细日志
docker-compose logs -f postgres
```

## 日志功能

### 日志记录器使用

```python
from harborai.storage import PostgreSQLLogger

# 初始化日志记录器
logger = PostgreSQLLogger(
    connection_string="postgresql://harborai:harborai_password_2024@localhost:5433/harborai",
    table_name="harborai_logs",
    batch_size=100,
    flush_interval=5.0
)

# 启动日志记录器
logger.start()

# 记录请求日志
logger.log_request(
    trace_id="unique_trace_id",
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    provider="openai"
)

# 记录响应日志
logger.log_response(
    trace_id="unique_trace_id",
    response=response_object,
    latency=1.5,
    success=True
)

# 停止日志记录器
logger.stop()
```

### 日志查询

```python
from harborai.database import PostgreSQLClient

# 初始化客户端
client = PostgreSQLClient(
    host="localhost",
    port=5433,
    database="harborai",
    user="harborai",
    password="harborai_password_2024"
)

# 查询最近日志
logs = client.get_recent_logs(limit=100)

# 按模型查询
model_logs = client.get_logs_by_model("gpt-4", limit=50)

# 按时间范围查询
from datetime import datetime, timedelta
start_time = datetime.now() - timedelta(hours=1)
recent_logs = client.get_logs_by_time_range(start_time, datetime.now())
```

## 常用命令

### Docker 管理

```powershell
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart postgres

# 查看日志
docker-compose logs -f postgres

# 清理数据（谨慎使用）
docker-compose down -v
```

### 数据库管理

```powershell
# 连接到数据库
docker exec -it harborai_postgres psql -U harborai -d harborai

# 执行 SQL 文件
Get-Content docker/postgres/init/01-init-database.sql | docker exec -i harborai_postgres psql -U harborai -d harborai

# 备份数据库
docker exec harborai_postgres pg_dump -U harborai harborai > backup.sql

# 恢复数据库
Get-Content backup.sql | docker exec -i harborai_postgres psql -U harborai -d harborai
```

## 故障排查

### 常见问题

#### 1. 容器启动失败

**症状**: `docker-compose up -d` 失败或容器不断重启

**解决方案**:
```powershell
# 查看详细错误日志
docker-compose logs postgres

# 检查端口占用
netstat -an | findstr :5433

# 清理并重新创建
docker-compose down -v
docker-compose up -d
```

#### 2. 数据库连接失败

**症状**: 测试脚本报告连接错误

**解决方案**:
```powershell
# 检查服务状态
docker-compose ps

# 验证健康检查
docker exec harborai_postgres pg_isready -U harborai -d harborai

# 检查网络连接
telnet localhost 5433
```

#### 3. 初始化脚本未执行

**症状**: 表不存在或结构不正确

**解决方案**:
```powershell
# 手动执行初始化脚本
Get-Content docker/postgres/init/01-init-database.sql | docker exec -i harborai_postgres psql -U harborai -d harborai

# 或者清理数据重新初始化
docker-compose down -v
docker-compose up -d
```

#### 4. 日志记录失败

**症状**: 日志无法写入数据库

**解决方案**:
```powershell
# 检查表权限
docker exec -it harborai_postgres psql -U harborai -d harborai -c "\dt"

# 检查日志记录器配置
python -c "from harborai.storage import PostgreSQLLogger; print('Logger available')"

# 运行测试验证
python test_postgres_docker.py
```

### 性能优化

#### 1. 批量写入优化

```python
# 调整批量大小和刷新间隔
logger = PostgreSQLLogger(
    connection_string=connection_string,
    batch_size=500,  # 增加批量大小
    flush_interval=10.0  # 增加刷新间隔
)
```

#### 2. 索引优化

```sql
-- 为常用查询添加索引
CREATE INDEX CONCURRENTLY idx_harborai_logs_trace_id ON harborai_logs(trace_id);
CREATE INDEX CONCURRENTLY idx_harborai_logs_timestamp ON harborai_logs(timestamp);
CREATE INDEX CONCURRENTLY idx_harborai_logs_model ON harborai_logs(model);
```

#### 3. 连接池配置

```python
# 使用连接池提高性能
import psycopg2.pool

pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=20,
    dsn=connection_string
)
```

## 监控和维护

### 健康检查

```powershell
# 检查服务健康状态
docker-compose ps

# 检查数据库连接
python -c "import psycopg2; conn = psycopg2.connect('postgresql://harborai:harborai_password_2024@localhost:5433/harborai'); print('Connection OK')"
```

### 日志轮转

```sql
-- 清理旧日志（保留最近30天）
DELETE FROM harborai_logs WHERE timestamp < NOW() - INTERVAL '30 days';

-- 分析表以优化性能
ANALYZE harborai_logs;
```

### 备份策略

```powershell
# 定期备份脚本
$date = Get-Date -Format "yyyyMMdd_HHmmss"
docker exec harborai_postgres pg_dump -U harborai harborai > "backup_$date.sql"

# 压缩备份
Compress-Archive -Path "backup_$date.sql" -DestinationPath "backup_$date.zip"
```

## 安全注意事项

1. **密码安全**: 生产环境中使用强密码
2. **网络隔离**: 限制数据库访问端口
3. **权限控制**: 使用最小权限原则
4. **数据加密**: 考虑启用 SSL 连接
5. **定期更新**: 保持 PostgreSQL 版本更新

## 扩展配置

### 高可用配置

```yaml
# docker-compose.yml 中添加副本
postgres-replica:
  image: postgres:15-alpine
  environment:
    POSTGRES_DB: harborai
    POSTGRES_USER: harborai
    POSTGRES_PASSWORD: harborai_password_2024
    POSTGRES_MASTER_SERVICE: postgres
  depends_on:
    - postgres
```

### 监控集成

```yaml
# 添加 pgAdmin 管理界面
pgadmin:
  image: dpage/pgadmin4:latest
  environment:
    PGADMIN_DEFAULT_EMAIL: admin@harborai.com
    PGADMIN_DEFAULT_PASSWORD: admin_password
  ports:
    - "8080:80"
  depends_on:
    - postgres
```

## 联系支持

如果遇到问题，请：

1. 查看本文档的故障排查部分
2. 运行 `python test_postgres_docker.py` 获取详细诊断信息
3. 检查 Docker 和 PostgreSQL 日志
4. 联系技术支持团队

---

**最后更新**: 2025-10-15
**版本**: 1.0.0
**维护者**: HarborAI 团队