# HarborAI SDK 部署运维指南

## 概述

本指南详细介绍了HarborAI SDK性能优化版本的部署、配置、监控和维护方法。该版本集成了延迟加载、内存优化和并发优化三大核心功能，为生产环境提供了高性能、高稳定性的解决方案。

## 系统要求

### 最低系统要求
- **Python版本**: 3.8+
- **内存**: 最低512MB，推荐2GB+
- **CPU**: 最低1核，推荐2核+
- **磁盘空间**: 最低100MB
- **网络**: 稳定的互联网连接

### 推荐系统配置
- **Python版本**: 3.11+
- **内存**: 4GB+
- **CPU**: 4核+
- **磁盘空间**: 1GB+
- **网络带宽**: 100Mbps+

### 依赖要求
```txt
aiohttp>=3.8.0
psutil>=5.9.0
typing-extensions>=4.0.0
pydantic>=1.10.0
```

## 安装部署

### 1. 基础安装

#### 从PyPI安装
```bash
# 安装最新版本
pip install harborai

# 安装指定版本
pip install harborai==3.0.0

# 安装开发版本
pip install harborai[dev]
```

#### 从源码安装
```bash
# 克隆仓库
git clone https://github.com/your-org/harborai.git
cd harborai

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 2. 环境配置

#### 环境变量配置
```bash
# 基础配置
export HARBORAI_API_KEY="your-api-key"
export HARBORAI_LOG_LEVEL="INFO"

# 性能优化配置
export HARBORAI_ENABLE_MEMORY_OPTIMIZATION="true"
export HARBORAI_ENABLE_LAZY_LOADING="true"
export HARBORAI_ENABLE_CONCURRENCY_OPTIMIZATION="true"

# 内存优化配置
export HARBORAI_CACHE_SIZE="2000"
export HARBORAI_OBJECT_POOL_SIZE="200"
export HARBORAI_MEMORY_THRESHOLD_MB="100.0"

# 并发优化配置
export HARBORAI_MAX_CONNECTIONS="100"
export HARBORAI_MAX_CONNECTIONS_PER_HOST="30"
export HARBORAI_CONNECTION_TIMEOUT="30"
```

#### 配置文件方式
```yaml
# harborai_config.yaml
api_key: "your-api-key"
log_level: "INFO"

# 性能优化开关
optimizations:
  memory_optimization: true
  lazy_loading: true
  concurrency_optimization: true

# 内存优化配置
memory:
  cache_size: 2000
  object_pool_size: 200
  memory_threshold_mb: 100.0
  auto_cleanup_interval: 600

# 并发优化配置
concurrency:
  max_connections: 100
  max_connections_per_host: 30
  connection_timeout: 30
  enable_connection_pooling: true
```

### 3. 快速开始

#### 基础使用示例
```python
from harborai.api.fast_client import FastHarborAI

# 创建优化客户端
client = FastHarborAI(
    api_key="your-api-key",
    enable_memory_optimization=True,
    enable_lazy_loading=True,
    enable_concurrency_optimization=True
)

# 同步调用
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello"}]
)

# 异步调用
import asyncio

async def async_example():
    response = await client.chat.completions.acreate(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello"}]
    )
    return response

# 运行异步示例
response = asyncio.run(async_example())
```

## 生产环境部署

### 1. 生产环境配置

#### 推荐生产配置
```python
from harborai.api.fast_client import FastHarborAI
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/harborai/app.log'),
        logging.StreamHandler()
    ]
)

# 生产环境客户端配置
client = FastHarborAI(
    api_key=os.getenv("HARBORAI_API_KEY"),
    enable_memory_optimization=True,
    enable_lazy_loading=True,
    enable_concurrency_optimization=True,
    
    # 内存优化配置
    memory_optimization={
        'cache_size': 5000,              # 大缓存提高命中率
        'object_pool_size': 500,         # 大对象池减少GC
        'memory_threshold_mb': 200.0,    # 适当提高阈值
        'auto_cleanup_interval': 900     # 延长清理间隔
    },
    
    # 并发优化配置
    concurrency_optimization={
        'max_connections': 200,          # 支持更多并发
        'max_connections_per_host': 50,  # 单主机更多连接
        'connection_timeout': 60,        # 延长超时时间
        'enable_connection_pooling': True,
        'request_queue_size': 2000       # 大请求队列
    },
    
    # 日志配置
    log_level="INFO",
    enable_debug_logging=False
)
```

#### Docker部署配置
```dockerfile
# Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建日志目录
RUN mkdir -p /var/log/harborai

# 设置环境变量
ENV PYTHONPATH=/app
ENV HARBORAI_LOG_LEVEL=INFO

# 暴露端口（如果有Web服务）
EXPOSE 8000

# 启动命令
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  harborai-app:
    build: .
    environment:
      - HARBORAI_API_KEY=${HARBORAI_API_KEY}
      - HARBORAI_ENABLE_MEMORY_OPTIMIZATION=true
      - HARBORAI_ENABLE_LAZY_LOADING=true
      - HARBORAI_ENABLE_CONCURRENCY_OPTIMIZATION=true
      - HARBORAI_CACHE_SIZE=5000
      - HARBORAI_MAX_CONNECTIONS=200
    volumes:
      - ./logs:/var/log/harborai
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import harborai; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 可选：Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### 2. Kubernetes部署

#### Deployment配置
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: harborai-app
  labels:
    app: harborai-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: harborai-app
  template:
    metadata:
      labels:
        app: harborai-app
    spec:
      containers:
      - name: harborai-app
        image: your-registry/harborai-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: HARBORAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: harborai-secret
              key: api-key
        - name: HARBORAI_ENABLE_MEMORY_OPTIMIZATION
          value: "true"
        - name: HARBORAI_CACHE_SIZE
          value: "5000"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: log-volume
          mountPath: /var/log/harborai
      volumes:
      - name: log-volume
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: harborai-service
spec:
  selector:
    app: harborai-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: harborai-secret
type: Opaque
data:
  api-key: <base64-encoded-api-key>
```

#### ConfigMap配置
```yaml
# k8s-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: harborai-config
data:
  harborai_config.yaml: |
    api_key: "${HARBORAI_API_KEY}"
    log_level: "INFO"
    
    optimizations:
      memory_optimization: true
      lazy_loading: true
      concurrency_optimization: true
    
    memory:
      cache_size: 5000
      object_pool_size: 500
      memory_threshold_mb: 200.0
      auto_cleanup_interval: 900
    
    concurrency:
      max_connections: 200
      max_connections_per_host: 50
      connection_timeout: 60
      enable_connection_pooling: true
```

## 监控和告警

### 1. 性能监控

#### 内置监控指标
```python
import asyncio
import logging
from datetime import datetime

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
    
    async def start_monitoring(self, interval: int = 60):
        """启动监控"""
        while True:
            try:
                await self.collect_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"监控异常: {e}")
                await asyncio.sleep(interval)
    
    async def collect_metrics(self):
        """收集性能指标"""
        timestamp = datetime.now()
        
        # 内存指标
        memory_stats = self.client.get_memory_stats()
        if memory_stats:
            self.log_metric("memory.cache_hit_rate", 
                          memory_stats['cache']['hit_rate'], timestamp)
            self.log_metric("memory.usage_mb", 
                          memory_stats['system_memory']['rss_mb'], timestamp)
            self.log_metric("memory.cache_size", 
                          memory_stats['cache']['size'], timestamp)
        
        # 并发指标
        concurrency_stats = self.client.get_concurrency_stats()
        if concurrency_stats:
            self.log_metric("concurrency.active_connections", 
                          concurrency_stats['connections']['active'], timestamp)
            self.log_metric("concurrency.throughput", 
                          concurrency_stats['performance']['throughput'], timestamp)
            self.log_metric("concurrency.avg_response_time", 
                          concurrency_stats['performance']['avg_response_time'], timestamp)
        
        # 检查告警条件
        await self.check_alerts(memory_stats, concurrency_stats)
    
    def log_metric(self, metric_name: str, value: float, timestamp: datetime):
        """记录指标"""
        metric = {
            'name': metric_name,
            'value': value,
            'timestamp': timestamp
        }
        self.metrics_history.append(metric)
        self.logger.info(f"Metric {metric_name}: {value}")
        
        # 保持历史记录大小
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
    
    async def check_alerts(self, memory_stats, concurrency_stats):
        """检查告警条件"""
        alerts = []
        
        # 内存告警
        if memory_stats:
            if memory_stats['cache']['hit_rate'] < 0.7:
                alerts.append(f"缓存命中率过低: {memory_stats['cache']['hit_rate']:.1%}")
            
            if memory_stats['system_memory']['rss_mb'] > 500:
                alerts.append(f"内存使用过高: {memory_stats['system_memory']['rss_mb']:.1f}MB")
        
        # 并发告警
        if concurrency_stats:
            if concurrency_stats['performance']['throughput'] < 500:
                alerts.append(f"吞吐量过低: {concurrency_stats['performance']['throughput']} ops/s")
            
            if concurrency_stats['performance']['avg_response_time'] > 2000:
                alerts.append(f"响应时间过长: {concurrency_stats['performance']['avg_response_time']}ms")
        
        # 发送告警
        for alert in alerts:
            await self.send_alert(alert)
    
    async def send_alert(self, message: str):
        """发送告警"""
        self.logger.warning(f"ALERT: {message}")
        # 这里可以集成邮件、短信、Slack等告警渠道

# 启动监控
monitor = PerformanceMonitor(client)
asyncio.create_task(monitor.start_monitoring())
```

#### Prometheus集成
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

class PrometheusMetrics:
    """Prometheus指标收集器"""
    
    def __init__(self):
        # 计数器
        self.request_count = Counter(
            'harborai_requests_total',
            'Total number of requests',
            ['model', 'status']
        )
        
        # 直方图
        self.request_duration = Histogram(
            'harborai_request_duration_seconds',
            'Request duration in seconds',
            ['model']
        )
        
        # 仪表盘
        self.memory_usage = Gauge(
            'harborai_memory_usage_mb',
            'Memory usage in MB'
        )
        
        self.cache_hit_rate = Gauge(
            'harborai_cache_hit_rate',
            'Cache hit rate'
        )
        
        self.active_connections = Gauge(
            'harborai_active_connections',
            'Number of active connections'
        )
    
    def record_request(self, model: str, duration: float, status: str):
        """记录请求指标"""
        self.request_count.labels(model=model, status=status).inc()
        self.request_duration.labels(model=model).observe(duration)
    
    def update_memory_metrics(self, memory_stats):
        """更新内存指标"""
        if memory_stats:
            self.memory_usage.set(memory_stats['system_memory']['rss_mb'])
            self.cache_hit_rate.set(memory_stats['cache']['hit_rate'])
    
    def update_concurrency_metrics(self, concurrency_stats):
        """更新并发指标"""
        if concurrency_stats:
            self.active_connections.set(concurrency_stats['connections']['active'])

# 启动Prometheus HTTP服务器
start_http_server(8001)
metrics = PrometheusMetrics()

# 在客户端中集成指标收集
class MonitoredFastHarborAI(FastHarborAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = PrometheusMetrics()
    
    async def chat_completions_acreate(self, model: str, **kwargs):
        start_time = time.time()
        try:
            result = await super().chat_completions_acreate(model=model, **kwargs)
            duration = time.time() - start_time
            self.metrics.record_request(model, duration, 'success')
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_request(model, duration, 'error')
            raise
```

### 2. 日志管理

#### 结构化日志配置
```python
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加额外字段
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'model'):
            log_entry['model'] = record.model
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        
        return json.dumps(log_entry, ensure_ascii=False)

# 配置日志
def setup_logging():
    """设置日志配置"""
    logger = logging.getLogger('harborai')
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler('/var/log/harborai/app.log')
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)
    
    # 错误日志处理器
    error_handler = logging.FileHandler('/var/log/harborai/error.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    logger.addHandler(error_handler)

setup_logging()
```

#### 日志轮转配置
```python
import logging.handlers

def setup_rotating_logs():
    """设置日志轮转"""
    logger = logging.getLogger('harborai')
    
    # 按大小轮转
    size_handler = logging.handlers.RotatingFileHandler(
        '/var/log/harborai/app.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=10
    )
    size_handler.setFormatter(StructuredFormatter())
    logger.addHandler(size_handler)
    
    # 按时间轮转
    time_handler = logging.handlers.TimedRotatingFileHandler(
        '/var/log/harborai/daily.log',
        when='midnight',
        interval=1,
        backupCount=30
    )
    time_handler.setFormatter(StructuredFormatter())
    logger.addHandler(time_handler)

setup_rotating_logs()
```

### 3. 健康检查

#### 健康检查端点
```python
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import asyncio

app = FastAPI()

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查端点"""
    try:
        # 检查客户端状态
        client_healthy = await check_client_health()
        
        # 检查内存状态
        memory_healthy = check_memory_health()
        
        # 检查并发状态
        concurrency_healthy = check_concurrency_health()
        
        # 检查外部依赖
        dependencies_healthy = await check_dependencies()
        
        overall_healthy = all([
            client_healthy,
            memory_healthy,
            concurrency_healthy,
            dependencies_healthy
        ])
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "client": "healthy" if client_healthy else "unhealthy",
                "memory": "healthy" if memory_healthy else "unhealthy",
                "concurrency": "healthy" if concurrency_healthy else "unhealthy",
                "dependencies": "healthy" if dependencies_healthy else "unhealthy"
            },
            "details": await get_health_details()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

async def check_client_health() -> bool:
    """检查客户端健康状态"""
    try:
        # 简单的API调用测试
        response = await client.chat.completions.acreate(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "health check"}],
            max_tokens=1
        )
        return True
    except Exception:
        return False

def check_memory_health() -> bool:
    """检查内存健康状态"""
    try:
        memory_stats = client.get_memory_stats()
        if not memory_stats:
            return False
        
        # 检查内存使用是否在合理范围内
        memory_usage = memory_stats['system_memory']['rss_mb']
        return memory_usage < 1000  # 1GB阈值
    
    except Exception:
        return False

def check_concurrency_health() -> bool:
    """检查并发健康状态"""
    try:
        concurrency_stats = client.get_concurrency_stats()
        if not concurrency_stats:
            return False
        
        # 检查连接池状态
        active_connections = concurrency_stats['connections']['active']
        max_connections = concurrency_stats['connections']['max_connections']
        
        # 连接使用率不超过80%
        return active_connections / max_connections < 0.8
    
    except Exception:
        return False

async def check_dependencies() -> bool:
    """检查外部依赖健康状态"""
    try:
        # 检查网络连接
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.deepseek.com', timeout=5) as response:
                return response.status < 500
    except Exception:
        return False

async def get_health_details() -> Dict[str, Any]:
    """获取详细健康信息"""
    return {
        "memory_stats": client.get_memory_stats(),
        "concurrency_stats": client.get_concurrency_stats(),
        "uptime": get_uptime(),
        "version": "3.0.0"
    }

def get_uptime() -> str:
    """获取运行时间"""
    # 实现运行时间计算
    return "24h 30m 15s"

@app.get("/ready")
async def readiness_check():
    """就绪检查端点"""
    try:
        # 检查是否准备好接收请求
        if not hasattr(client, '_initialized') or not client._initialized:
            raise HTTPException(status_code=503, detail="Client not initialized")
        
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

# 启动健康检查服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 维护和故障排除

### 1. 常见问题排查

#### 内存问题排查
```python
import psutil
import gc
import tracemalloc

class MemoryDiagnostics:
    """内存诊断工具"""
    
    def __init__(self, client):
        self.client = client
        tracemalloc.start()
    
    def diagnose_memory_issue(self):
        """诊断内存问题"""
        print("=== 内存诊断报告 ===")
        
        # 系统内存信息
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"RSS内存: {memory_info.rss / 1024 / 1024:.2f} MB")
        print(f"VMS内存: {memory_info.vms / 1024 / 1024:.2f} MB")
        
        # 客户端内存统计
        memory_stats = self.client.get_memory_stats()
        if memory_stats:
            print(f"缓存大小: {memory_stats['cache']['size']}")
            print(f"缓存命中率: {memory_stats['cache']['hit_rate']:.1%}")
            print(f"对象池使用率: {memory_stats['object_pool']['usage_rate']:.1%}")
        
        # Python对象统计
        gc.collect()
        print(f"垃圾回收统计: {gc.get_stats()}")
        
        # 内存快照
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\n=== 内存使用Top 10 ===")
        for index, stat in enumerate(top_stats[:10], 1):
            print(f"{index}. {stat}")
    
    def check_memory_leaks(self):
        """检查内存泄漏"""
        print("=== 内存泄漏检查 ===")
        
        # 强制垃圾回收
        collected = gc.collect()
        print(f"垃圾回收清理对象数: {collected}")
        
        # 检查循环引用
        if gc.garbage:
            print(f"发现循环引用对象: {len(gc.garbage)}")
            for obj in gc.garbage[:5]:  # 只显示前5个
                print(f"  - {type(obj)}: {obj}")
        else:
            print("未发现循环引用")
        
        # 检查弱引用状态
        memory_stats = self.client.get_memory_stats()
        if memory_stats and 'weak_references' in memory_stats:
            weak_refs = memory_stats['weak_references']
            print(f"弱引用数量: {weak_refs['count']}")
            print(f"已清理弱引用: {weak_refs['cleaned_count']}")

# 使用诊断工具
diagnostics = MemoryDiagnostics(client)
diagnostics.diagnose_memory_issue()
diagnostics.check_memory_leaks()
```

#### 并发问题排查
```python
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ConcurrencyDiagnostics:
    """并发诊断工具"""
    
    def __init__(self, client):
        self.client = client
    
    def diagnose_concurrency_issue(self):
        """诊断并发问题"""
        print("=== 并发诊断报告 ===")
        
        # 线程信息
        active_threads = threading.active_count()
        print(f"活跃线程数: {active_threads}")
        
        for thread in threading.enumerate():
            print(f"  - {thread.name}: {thread.is_alive()}")
        
        # 并发统计
        concurrency_stats = self.client.get_concurrency_stats()
        if concurrency_stats:
            print(f"活跃连接数: {concurrency_stats['connections']['active']}")
            print(f"连接池使用率: {concurrency_stats['connections']['pool_usage_rate']:.1%}")
            print(f"请求队列长度: {concurrency_stats['requests']['queue_length']}")
            print(f"当前吞吐量: {concurrency_stats['performance']['throughput']} ops/s")
        
        # 事件循环状态
        try:
            loop = asyncio.get_running_loop()
            print(f"事件循环运行中: {loop.is_running()}")
            print(f"事件循环任务数: {len(asyncio.all_tasks(loop))}")
        except RuntimeError:
            print("没有运行中的事件循环")
    
    async def test_concurrency_performance(self, concurrency_level: int = 10):
        """测试并发性能"""
        print(f"=== 并发性能测试 (并发度: {concurrency_level}) ===")
        
        async def single_request():
            start_time = asyncio.get_event_loop().time()
            try:
                await self.client.chat.completions.acreate(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                return asyncio.get_event_loop().time() - start_time
            except Exception as e:
                print(f"请求失败: {e}")
                return None
        
        # 并发测试
        start_time = asyncio.get_event_loop().time()
        tasks = [single_request() for _ in range(concurrency_level)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = asyncio.get_event_loop().time() - start_time
        
        # 统计结果
        successful_requests = [r for r in results if isinstance(r, float)]
        failed_requests = len(results) - len(successful_requests)
        
        if successful_requests:
            avg_response_time = sum(successful_requests) / len(successful_requests)
            throughput = len(successful_requests) / total_time
            
            print(f"成功请求数: {len(successful_requests)}")
            print(f"失败请求数: {failed_requests}")
            print(f"平均响应时间: {avg_response_time*1000:.2f} ms")
            print(f"吞吐量: {throughput:.2f} ops/s")
        else:
            print("所有请求都失败了")

# 使用诊断工具
concurrency_diagnostics = ConcurrencyDiagnostics(client)
concurrency_diagnostics.diagnose_concurrency_issue()

# 异步测试
async def run_concurrency_test():
    await concurrency_diagnostics.test_concurrency_performance(20)

asyncio.run(run_concurrency_test())
```

### 2. 性能调优

#### 自动调优脚本
```python
import asyncio
import time
from typing import Dict, Any

class AutoTuner:
    """自动性能调优器"""
    
    def __init__(self, client):
        self.client = client
        self.baseline_metrics = None
        self.tuning_history = []
    
    async def auto_tune(self):
        """自动调优"""
        print("=== 开始自动调优 ===")
        
        # 获取基线指标
        self.baseline_metrics = await self.get_performance_metrics()
        print(f"基线指标: {self.baseline_metrics}")
        
        # 调优缓存大小
        await self.tune_cache_size()
        
        # 调优连接池大小
        await self.tune_connection_pool()
        
        # 调优清理间隔
        await self.tune_cleanup_interval()
        
        print("=== 调优完成 ===")
        self.print_tuning_summary()
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        # 运行性能测试
        start_time = time.time()
        
        tasks = []
        for _ in range(50):  # 50个并发请求
            task = self.client.chat.completions.acreate(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        duration = time.time() - start_time
        throughput = 50 / duration
        
        # 获取内存和并发统计
        memory_stats = self.client.get_memory_stats()
        concurrency_stats = self.client.get_concurrency_stats()
        
        return {
            'throughput': throughput,
            'memory_usage': memory_stats['system_memory']['rss_mb'] if memory_stats else 0,
            'cache_hit_rate': memory_stats['cache']['hit_rate'] if memory_stats else 0,
            'avg_response_time': concurrency_stats['performance']['avg_response_time'] if concurrency_stats else 0
        }
    
    async def tune_cache_size(self):
        """调优缓存大小"""
        print("调优缓存大小...")
        
        cache_sizes = [500, 1000, 2000, 5000]
        best_size = 1000
        best_score = 0
        
        for size in cache_sizes:
            # 调整缓存大小
            await self.client.adjust_cache_size(size)
            await asyncio.sleep(5)  # 等待调整生效
            
            # 测试性能
            metrics = await self.get_performance_metrics()
            score = self.calculate_score(metrics)
            
            print(f"缓存大小 {size}: 得分 {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_size = size
            
            self.tuning_history.append({
                'parameter': 'cache_size',
                'value': size,
                'metrics': metrics,
                'score': score
            })
        
        # 设置最佳缓存大小
        await self.client.adjust_cache_size(best_size)
        print(f"最佳缓存大小: {best_size}")
    
    async def tune_connection_pool(self):
        """调优连接池大小"""
        print("调优连接池大小...")
        
        pool_sizes = [50, 100, 200, 300]
        best_size = 100
        best_score = 0
        
        for size in pool_sizes:
            # 调整连接池大小
            await self.client.adjust_connection_pool_size(size)
            await asyncio.sleep(5)
            
            # 测试性能
            metrics = await self.get_performance_metrics()
            score = self.calculate_score(metrics)
            
            print(f"连接池大小 {size}: 得分 {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_size = size
            
            self.tuning_history.append({
                'parameter': 'connection_pool_size',
                'value': size,
                'metrics': metrics,
                'score': score
            })
        
        # 设置最佳连接池大小
        await self.client.adjust_connection_pool_size(best_size)
        print(f"最佳连接池大小: {best_size}")
    
    async def tune_cleanup_interval(self):
        """调优清理间隔"""
        print("调优清理间隔...")
        
        intervals = [60, 300, 600, 1200]  # 秒
        best_interval = 300
        best_score = 0
        
        for interval in intervals:
            # 调整清理间隔
            await self.client.adjust_cleanup_interval(interval)
            await asyncio.sleep(10)
            
            # 测试性能
            metrics = await self.get_performance_metrics()
            score = self.calculate_score(metrics)
            
            print(f"清理间隔 {interval}s: 得分 {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_interval = interval
            
            self.tuning_history.append({
                'parameter': 'cleanup_interval',
                'value': interval,
                'metrics': metrics,
                'score': score
            })
        
        # 设置最佳清理间隔
        await self.client.adjust_cleanup_interval(best_interval)
        print(f"最佳清理间隔: {best_interval}s")
    
    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """计算性能得分"""
        # 权重配置
        weights = {
            'throughput': 0.4,      # 吞吐量权重40%
            'cache_hit_rate': 0.3,  # 缓存命中率权重30%
            'memory_usage': 0.2,    # 内存使用权重20%（越低越好）
            'avg_response_time': 0.1 # 响应时间权重10%（越低越好）
        }
        
        # 归一化指标
        normalized_metrics = {
            'throughput': min(metrics['throughput'] / 1000, 1.0),  # 1000 ops/s为满分
            'cache_hit_rate': metrics['cache_hit_rate'],
            'memory_usage': max(0, 1 - metrics['memory_usage'] / 500),  # 500MB为0分
            'avg_response_time': max(0, 1 - metrics['avg_response_time'] / 1000)  # 1000ms为0分
        }
        
        # 计算加权得分
        score = sum(normalized_metrics[key] * weights[key] for key in weights)
        return score
    
    def print_tuning_summary(self):
        """打印调优总结"""
        print("\n=== 调优总结 ===")
        
        if self.baseline_metrics:
            final_metrics = asyncio.run(self.get_performance_metrics())
            
            print("基线指标 vs 调优后指标:")
            for key in self.baseline_metrics:
                baseline = self.baseline_metrics[key]
                final = final_metrics[key]
                improvement = ((final - baseline) / baseline * 100) if baseline > 0 else 0
                print(f"  {key}: {baseline:.2f} -> {final:.2f} ({improvement:+.1f}%)")
        
        print("\n调优历史:")
        for entry in self.tuning_history:
            print(f"  {entry['parameter']} = {entry['value']}: 得分 {entry['score']:.2f}")

# 使用自动调优器
tuner = AutoTuner(client)
asyncio.run(tuner.auto_tune())
```

### 3. 备份和恢复

#### 配置备份脚本
```python
import json
import os
import shutil
from datetime import datetime
import tarfile

class BackupManager:
    """备份管理器"""
    
    def __init__(self, backup_dir: str = "/var/backups/harborai"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
    
    def backup_configuration(self):
        """备份配置"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"harborai_config_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        os.makedirs(backup_path, exist_ok=True)
        
        # 备份配置文件
        config_files = [
            "/etc/harborai/config.yaml",
            "/etc/harborai/logging.conf",
            "/etc/harborai/environment.env"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                shutil.copy2(config_file, backup_path)
        
        # 备份当前性能指标
        if hasattr(self, 'client'):
            metrics = {
                'memory_stats': self.client.get_memory_stats(),
                'concurrency_stats': self.client.get_concurrency_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(backup_path, "performance_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # 创建压缩包
        tar_path = f"{backup_path}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(backup_path, arcname=backup_name)
        
        # 清理临时目录
        shutil.rmtree(backup_path)
        
        print(f"配置备份完成: {tar_path}")
        return tar_path
    
    def restore_configuration(self, backup_file: str):
        """恢复配置"""
        if not os.path.exists(backup_file):
            raise FileNotFoundError(f"备份文件不存在: {backup_file}")
        
        # 解压备份文件
        extract_path = os.path.join(self.backup_dir, "restore_temp")
        os.makedirs(extract_path, exist_ok=True)
        
        with tarfile.open(backup_file, "r:gz") as tar:
            tar.extractall(extract_path)
        
        # 恢复配置文件
        backup_content = os.listdir(extract_path)[0]  # 获取解压后的目录
        backup_content_path = os.path.join(extract_path, backup_content)
        
        config_mappings = {
            "config.yaml": "/etc/harborai/config.yaml",
            "logging.conf": "/etc/harborai/logging.conf",
            "environment.env": "/etc/harborai/environment.env"
        }
        
        for backup_file_name, target_path in config_mappings.items():
            backup_file_path = os.path.join(backup_content_path, backup_file_name)
            if os.path.exists(backup_file_path):
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(backup_file_path, target_path)
                print(f"恢复配置文件: {target_path}")
        
        # 清理临时目录
        shutil.rmtree(extract_path)
        
        print("配置恢复完成，请重启服务")
    
    def list_backups(self):
        """列出所有备份"""
        backups = []
        for file in os.listdir(self.backup_dir):
            if file.startswith("harborai_config_") and file.endswith(".tar.gz"):
                file_path = os.path.join(self.backup_dir, file)
                stat = os.stat(file_path)
                backups.append({
                    'name': file,
                    'path': file_path,
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime)
                })
        
        backups.sort(key=lambda x: x['created'], reverse=True)
        return backups
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """清理旧备份"""
        backups = self.list_backups()
        
        if len(backups) > keep_count:
            for backup in backups[keep_count:]:
                os.remove(backup['path'])
                print(f"删除旧备份: {backup['name']}")

# 使用备份管理器
backup_manager = BackupManager()

# 创建备份
backup_file = backup_manager.backup_configuration()

# 列出备份
backups = backup_manager.list_backups()
for backup in backups:
    print(f"{backup['name']} - {backup['created']} - {backup['size']} bytes")

# 清理旧备份
backup_manager.cleanup_old_backups(keep_count=5)
```

## 安全考虑

### 1. API密钥管理

```python
import os
import keyring
from cryptography.fernet import Fernet

class SecureKeyManager:
    """安全密钥管理器"""
    
    def __init__(self):
        self.service_name = "harborai"
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """获取或创建加密密钥"""
        key = keyring.get_password(self.service_name, "encryption_key")
        if not key:
            key = Fernet.generate_key().decode()
            keyring.set_password(self.service_name, "encryption_key", key)
        return key.encode()
    
    def store_api_key(self, provider: str, api_key: str):
        """安全存储API密钥"""
        encrypted_key = self.cipher.encrypt(api_key.encode())
        keyring.set_password(self.service_name, f"api_key_{provider}", encrypted_key.decode())
    
    def get_api_key(self, provider: str) -> str:
        """安全获取API密钥"""
        encrypted_key = keyring.get_password(self.service_name, f"api_key_{provider}")
        if not encrypted_key:
            raise ValueError(f"API密钥未找到: {provider}")
        
        decrypted_key = self.cipher.decrypt(encrypted_key.encode())
        return decrypted_key.decode()
    
    def rotate_api_key(self, provider: str, new_api_key: str):
        """轮换API密钥"""
        # 备份旧密钥
        try:
            old_key = self.get_api_key(provider)
            keyring.set_password(self.service_name, f"api_key_{provider}_backup", old_key)
        except ValueError:
            pass  # 没有旧密钥
        
        # 存储新密钥
        self.store_api_key(provider, new_api_key)
        print(f"API密钥已轮换: {provider}")

# 使用安全密钥管理器
key_manager = SecureKeyManager()
key_manager.store_api_key("deepseek", "your-deepseek-api-key")

# 在客户端中使用
client = FastHarborAI(
    api_key=key_manager.get_api_key("deepseek"),
    # ... 其他配置
)
```

### 2. 网络安全

```python
import ssl
import aiohttp
from aiohttp import ClientTimeout

class SecureHTTPClient:
    """安全HTTP客户端"""
    
    def __init__(self):
        # 创建安全的SSL上下文
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = True
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # 配置超时
        self.timeout = ClientTimeout(
            total=30,
            connect=10,
            sock_read=10
        )
        
        # 创建连接器
        self.connector = aiohttp.TCPConnector(
            ssl=self.ssl_context,
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
    
    async def create_session(self) -> aiohttp.ClientSession:
        """创建安全会话"""
        return aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                'User-Agent': 'HarborAI-SDK/3.0.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
        )

# 在客户端中集成安全HTTP客户端
secure_client = SecureHTTPClient()
```

---

**文档版本**: v3.0  
**最后更新**: 2025年10月3日  
**维护者**: HarborAI运维团队

## 总结

本部署运维指南提供了HarborAI SDK性能优化版本的完整部署和维护方案，包括：

1. **完整的安装部署流程**：从基础安装到生产环境部署
2. **全面的监控告警体系**：内存、并发、性能等多维度监控
3. **详细的故障排查方法**：常见问题的诊断和解决方案
4. **自动化运维工具**：性能调优、备份恢复等自动化脚本
5. **安全最佳实践**：API密钥管理、网络安全等安全措施

通过遵循本指南，可以确保HarborAI SDK在生产环境中稳定、高效、安全地运行。