# HarborAI PostgreSQL Docker 使用说明

## 🚀 快速开始

### 1. 启动服务
```powershell
docker-compose up -d
```

### 2. 验证安装
```powershell
python test_postgres_docker.py
```

### 3. 查看服务状态
```powershell
docker-compose ps
```

## 📊 测试结果

✅ **所有测试通过！** 

测试覆盖：
- ✅ 基础连接测试
- ✅ 表结构验证
- ✅ PostgreSQL 客户端功能
- ✅ 日志记录器功能
- ✅ 异步日志记录
- ✅ 日志查看功能
- ✅ 性能测试

## 🔧 核心功能

### 日志记录
```python
from harborai.storage import PostgreSQLLogger

logger = PostgreSQLLogger(
    connection_string="postgresql://harborai:harborai_password_2024@localhost:5433/harborai"
)
logger.start()

# 记录请求
logger.log_request(
    trace_id="unique_id",
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    provider="openai"
)

# 记录响应
logger.log_response(
    trace_id="unique_id",
    response=response_obj,
    latency=1.5,
    success=True
)

logger.stop()
```

### 日志查询
```python
from harborai.database import PostgreSQLClient

client = PostgreSQLClient(
    host="localhost", port=5433, 
    database="harborai", user="harborai", 
    password="harborai_password_2024"
)

# 查询最近日志
logs = client.get_recent_logs(limit=100)
```

## 🛠️ 常用命令

```powershell
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 查看日志
docker-compose logs postgres

# 重启服务
docker-compose restart postgres

# 连接数据库
docker exec -it harborai_postgres psql -U harborai -d harborai

# 备份数据
docker exec harborai_postgres pg_dump -U harborai harborai > backup.sql
```

## 🔍 故障排查

### 容器启动失败
```powershell
docker-compose logs postgres
docker-compose down -v
docker-compose up -d
```

### 连接失败
```powershell
docker-compose ps
docker exec harborai_postgres pg_isready -U harborai -d harborai
```

### 表不存在
```powershell
Get-Content docker/postgres/init/01-init-database.sql | docker exec -i harborai_postgres psql -U harborai -d harborai
```

## 📈 性能优化

- 批量大小：100-500 条记录
- 刷新间隔：5-10 秒
- 连接池：1-20 个连接

## 🔒 安全配置

- 端口：5433（避免冲突）
- 用户：harborai
- 密码：harborai_password_2024
- 网络：隔离的 Docker 网络

## 📋 配置文件

### docker-compose.yml
- PostgreSQL 15-alpine
- 健康检查
- 数据持久化
- 网络隔离

### 初始化脚本
- 表结构创建
- 索引优化
- 权限设置
- 视图定义

---

**状态**: ✅ 已完成配置和测试  
**版本**: PostgreSQL 15  
**更新**: 2025-10-15