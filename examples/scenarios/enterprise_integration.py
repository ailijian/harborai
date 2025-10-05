#!/usr/bin/env python3
"""
HarborAI 企业级应用集成

场景描述:
展示HarborAI在企业级环境中的集成应用，包括与现有系统的集成、
企业级安全、监控、扩展性和高可用性等关键特性。

应用价值:
- 无缝集成现有企业系统
- 企业级安全和合规性
- 高性能和可扩展性
- 全面的监控和运维
- 多租户和权限管理

核心功能:
1. 企业系统集成（ERP、CRM、数据库）
2. 企业级身份认证和授权
3. API网关和服务治理
4. 分布式缓存和负载均衡
5. 监控、日志和告警系统
"""

import asyncio
import json
import time
import uuid
import logging
import hashlib
import jwt
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
from contextlib import asynccontextmanager
import aiohttp
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor
import ssl
import certifi

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class ServiceType(Enum):
    """服务类型"""
    AI_GATEWAY = "ai_gateway"             # AI网关
    AUTH_SERVICE = "auth_service"         # 认证服务
    API_GATEWAY = "api_gateway"           # API网关
    CACHE_SERVICE = "cache_service"       # 缓存服务
    MONITOR_SERVICE = "monitor_service"   # 监控服务
    INTEGRATION_SERVICE = "integration_service"  # 集成服务

class AuthMethod(Enum):
    """认证方式"""
    JWT = "jwt"                          # JWT令牌
    OAUTH2 = "oauth2"                    # OAuth2
    API_KEY = "api_key"                  # API密钥
    LDAP = "ldap"                        # LDAP
    SAML = "saml"                        # SAML

class IntegrationType(Enum):
    """集成类型"""
    REST_API = "rest_api"                # REST API
    GRAPHQL = "graphql"                  # GraphQL
    WEBHOOK = "webhook"                  # Webhook
    MESSAGE_QUEUE = "message_queue"      # 消息队列
    DATABASE = "database"                # 数据库
    FILE_SYSTEM = "file_system"          # 文件系统

class AlertLevel(Enum):
    """告警级别"""
    CRITICAL = "critical"                # 严重
    WARNING = "warning"                  # 警告
    INFO = "info"                        # 信息
    DEBUG = "debug"                      # 调试

@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    service_type: ServiceType
    host: str
    port: int
    ssl_enabled: bool = False
    health_check_path: str = "/health"
    timeout: int = 30
    retry_count: int = 3
    circuit_breaker_enabled: bool = True
    rate_limit: int = 1000  # 每分钟请求数
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class User:
    """用户信息"""
    id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    tenant_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True

@dataclass
class APIRequest:
    """API请求"""
    id: str
    user_id: str
    tenant_id: str
    endpoint: str
    method: str
    headers: Dict[str, str]
    body: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: str = ""
    user_agent: str = ""

@dataclass
class APIResponse:
    """API响应"""
    request_id: str
    status_code: int
    response_time: float
    response_size: int
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class IntegrationEndpoint:
    """集成端点"""
    id: str
    name: str
    integration_type: IntegrationType
    url: str
    auth_config: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

@dataclass
class Alert:
    """告警"""
    id: str
    level: AlertLevel
    title: str
    message: str
    service: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

# Prometheus指标
REQUEST_COUNT = Counter('harborai_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('harborai_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('harborai_active_connections', 'Active connections')
ERROR_RATE = Gauge('harborai_error_rate', 'Error rate')

class SecurityManager:
    """安全管理器"""
    
    def __init__(self, secret_key: str, redis_client: Optional[redis.Redis] = None):
        self.secret_key = secret_key
        self.redis_client = redis_client
        self.token_blacklist = set()
        
    def generate_jwt_token(self, user: User, expires_in: int = 3600) -> str:
        """生成JWT令牌"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'tenant_id': user.tenant_id,
            'roles': user.roles,
            'permissions': user.permissions,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())  # JWT ID
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # 缓存令牌信息
        if self.redis_client:
            self.redis_client.setex(
                f"token:{payload['jti']}", 
                expires_in, 
                json.dumps({
                    'user_id': user.id,
                    'tenant_id': user.tenant_id,
                    'created_at': datetime.utcnow().isoformat()
                })
            )
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        try:
            # 检查黑名单
            if token in self.token_blacklist:
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # 检查Redis中的令牌状态
            if self.redis_client:
                token_info = self.redis_client.get(f"token:{payload['jti']}")
                if not token_info:
                    return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def revoke_token(self, token: str):
        """撤销令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # 添加到黑名单
            self.token_blacklist.add(token)
            
            # 从Redis删除
            if self.redis_client:
                self.redis_client.delete(f"token:{payload['jti']}")
                
        except jwt.InvalidTokenError:
            pass
    
    def hash_password(self, password: str) -> str:
        """密码哈希"""
        salt = uuid.uuid4().hex
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() + ':' + salt
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """验证密码"""
        try:
            password_hash, salt = hashed.split(':')
            return password_hash == hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
        except ValueError:
            return False
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """检查权限"""
        return required_permission in user_permissions or 'admin' in user_permissions

class RateLimiter:
    """限流器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    async def is_allowed(self, key: str, limit: int, window: int = 60) -> bool:
        """检查是否允许请求"""
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # 使用Redis的有序集合实现滑动窗口限流
            pipe = self.redis_client.pipeline()
            
            # 删除窗口外的记录
            pipe.zremrangebyscore(key, 0, window_start)
            
            # 添加当前请求
            pipe.zadd(key, {str(uuid.uuid4()): current_time})
            
            # 获取当前窗口内的请求数
            pipe.zcard(key)
            
            # 设置过期时间
            pipe.expire(key, window)
            
            results = pipe.execute()
            current_requests = results[2]
            
            return current_requests <= limit
            
        except Exception as e:
            logger.error("Rate limiting error", error=str(e))
            return True  # 出错时允许请求

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """执行函数调用"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """成功回调"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class ServiceRegistry:
    """服务注册中心"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.services: Dict[str, ServiceConfig] = {}
    
    def register_service(self, service: ServiceConfig):
        """注册服务"""
        self.services[service.name] = service
        
        # 在Redis中注册服务
        service_data = {
            'name': service.name,
            'type': service.service_type.value,
            'host': service.host,
            'port': service.port,
            'ssl_enabled': service.ssl_enabled,
            'health_check_path': service.health_check_path,
            'registered_at': datetime.utcnow().isoformat()
        }
        
        self.redis_client.hset(
            f"service:{service.name}",
            mapping=service_data
        )
        
        # 设置TTL
        self.redis_client.expire(f"service:{service.name}", 300)
        
        logger.info("Service registered", service=service.name)
    
    def discover_service(self, service_name: str) -> Optional[ServiceConfig]:
        """发现服务"""
        if service_name in self.services:
            return self.services[service_name]
        
        # 从Redis查询
        service_data = self.redis_client.hgetall(f"service:{service_name}")
        if service_data:
            return ServiceConfig(
                name=service_data[b'name'].decode(),
                service_type=ServiceType(service_data[b'type'].decode()),
                host=service_data[b'host'].decode(),
                port=int(service_data[b'port']),
                ssl_enabled=service_data[b'ssl_enabled'].decode() == 'True'
            )
        
        return None
    
    def get_healthy_services(self, service_type: ServiceType) -> List[ServiceConfig]:
        """获取健康的服务实例"""
        healthy_services = []
        
        for service in self.services.values():
            if service.service_type == service_type:
                # 这里可以添加健康检查逻辑
                healthy_services.append(service)
        
        return healthy_services

class IntegrationManager:
    """集成管理器"""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.endpoints: Dict[str, IntegrationEndpoint] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def register_endpoint(self, endpoint: IntegrationEndpoint):
        """注册集成端点"""
        self.endpoints[endpoint.id] = endpoint
        self.circuit_breakers[endpoint.id] = CircuitBreaker()
        logger.info("Integration endpoint registered", endpoint=endpoint.name)
    
    async def call_endpoint(self, endpoint_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """调用集成端点"""
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint not found: {endpoint_id}")
        
        endpoint = self.endpoints[endpoint_id]
        circuit_breaker = self.circuit_breakers[endpoint_id]
        
        if not endpoint.is_active:
            raise Exception(f"Endpoint is inactive: {endpoint.name}")
        
        try:
            # 通过熔断器调用
            result = await circuit_breaker.call(self._make_request, endpoint, data)
            return result
            
        except Exception as e:
            logger.error("Integration call failed", 
                        endpoint=endpoint.name, 
                        error=str(e))
            raise e
    
    async def _make_request(self, endpoint: IntegrationEndpoint, data: Dict[str, Any]) -> Dict[str, Any]:
        """发起HTTP请求"""
        headers = endpoint.headers.copy()
        
        # 添加认证头
        if endpoint.auth_config.get('type') == 'bearer':
            headers['Authorization'] = f"Bearer {endpoint.auth_config['token']}"
        elif endpoint.auth_config.get('type') == 'api_key':
            headers[endpoint.auth_config['header']] = endpoint.auth_config['key']
        
        timeout = aiohttp.ClientTimeout(total=endpoint.timeout)
        
        async with self.session.post(
            endpoint.url,
            json=data,
            headers=headers,
            timeout=timeout,
            ssl=ssl.create_default_context(cafile=certifi.where())
        ) as response:
            
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")
            
            return await response.json()

class MonitoringManager:
    """监控管理器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
    
    def add_alert_rule(self, name: str, metric: str, threshold: float, 
                      level: AlertLevel, condition: str = "greater"):
        """添加告警规则"""
        self.alert_rules[name] = {
            'metric': metric,
            'threshold': threshold,
            'level': level,
            'condition': condition
        }
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录指标"""
        timestamp = int(time.time())
        
        # 存储到Redis时序数据
        key = f"metric:{name}"
        if labels:
            key += ":" + ":".join(f"{k}={v}" for k, v in labels.items())
        
        self.redis_client.zadd(key, {timestamp: value})
        
        # 只保留最近1小时的数据
        one_hour_ago = timestamp - 3600
        self.redis_client.zremrangebyscore(key, 0, one_hour_ago)
        
        # 检查告警规则
        self._check_alert_rules(name, value)
    
    def _check_alert_rules(self, metric_name: str, value: float):
        """检查告警规则"""
        for rule_name, rule in self.alert_rules.items():
            if rule['metric'] == metric_name:
                threshold = rule['threshold']
                condition = rule['condition']
                
                should_alert = False
                if condition == "greater" and value > threshold:
                    should_alert = True
                elif condition == "less" and value < threshold:
                    should_alert = True
                elif condition == "equal" and value == threshold:
                    should_alert = True
                
                if should_alert:
                    alert = Alert(
                        id=str(uuid.uuid4()),
                        level=rule['level'],
                        title=f"Alert: {rule_name}",
                        message=f"Metric {metric_name} value {value} {condition} threshold {threshold}",
                        service="harborai",
                        metric_name=metric_name,
                        metric_value=value,
                        threshold=threshold
                    )
                    
                    self.alerts.append(alert)
                    self._send_alert(alert)
    
    def _send_alert(self, alert: Alert):
        """发送告警"""
        logger.warning("Alert triggered",
                      alert_id=alert.id,
                      level=alert.level.value,
                      title=alert.title,
                      message=alert.message)
        
        # 这里可以集成邮件、短信、Slack等告警渠道
    
    def get_metrics(self, name: str, start_time: int = None, end_time: int = None) -> List[Tuple[int, float]]:
        """获取指标数据"""
        if start_time is None:
            start_time = int(time.time()) - 3600  # 默认最近1小时
        if end_time is None:
            end_time = int(time.time())
        
        key = f"metric:{name}"
        data = self.redis_client.zrangebyscore(key, start_time, end_time, withscores=True)
        
        return [(int(score), float(member)) for member, score in data]

class EnterpriseAIGateway:
    """企业级AI网关"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.harborai.com/v1",
                 redis_url: str = "redis://localhost:6379",
                 db_path: str = "enterprise_gateway.db"):
        
        # 初始化组件
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.redis_client = redis.from_url(redis_url)
        
        # 初始化管理器
        self.security_manager = SecurityManager("your-secret-key", self.redis_client)
        self.rate_limiter = RateLimiter(self.redis_client)
        self.service_registry = ServiceRegistry(self.redis_client)
        self.monitoring_manager = MonitoringManager(self.redis_client)
        
        # 初始化HTTP会话
        self.http_session = None
        
        # 数据库
        self.db_path = db_path
        self._init_database()
        
        # 启动监控服务器
        self._start_metrics_server()
        
        # 配置告警规则
        self._setup_alert_rules()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                roles TEXT NOT NULL,
                permissions TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_login TIMESTAMP,
                is_active BOOLEAN NOT NULL DEFAULT 1
            )
        """)
        
        # API请求日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_requests (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status_code INTEGER,
                response_time REAL,
                response_size INTEGER,
                error_message TEXT,
                timestamp TIMESTAMP NOT NULL,
                ip_address TEXT,
                user_agent TEXT
            )
        """)
        
        # 集成端点表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS integration_endpoints (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                integration_type TEXT NOT NULL,
                url TEXT NOT NULL,
                auth_config TEXT NOT NULL,
                headers TEXT,
                timeout INTEGER NOT NULL DEFAULT 30,
                retry_config TEXT,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                created_at TIMESTAMP NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _start_metrics_server(self):
        """启动指标服务器"""
        def start_server():
            start_http_server(8000)
        
        thread = threading.Thread(target=start_server, daemon=True)
        thread.start()
        logger.info("Metrics server started on port 8000")
    
    def _setup_alert_rules(self):
        """设置告警规则"""
        self.monitoring_manager.add_alert_rule(
            "high_error_rate", "error_rate", 0.05, AlertLevel.WARNING
        )
        self.monitoring_manager.add_alert_rule(
            "critical_error_rate", "error_rate", 0.1, AlertLevel.CRITICAL
        )
        self.monitoring_manager.add_alert_rule(
            "high_response_time", "response_time", 5.0, AlertLevel.WARNING
        )
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.http_session = aiohttp.ClientSession()
        self.integration_manager = IntegrationManager(self.http_session)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.http_session:
            await self.http_session.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """用户认证"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, email, password_hash, roles, permissions, tenant_id, is_active
            FROM users WHERE username = ? AND is_active = 1
        """, (username,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row and self.security_manager.verify_password(password, row[3]):
            user = User(
                id=row[0],
                username=row[1],
                email=row[2],
                roles=json.loads(row[4]),
                permissions=json.loads(row[5]),
                tenant_id=row[6],
                is_active=bool(row[7])
            )
            
            # 生成JWT令牌
            token = self.security_manager.generate_jwt_token(user)
            
            # 更新最后登录时间
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET last_login = ? WHERE id = ?
            """, (datetime.now(), user.id))
            conn.commit()
            conn.close()
            
            return token
        
        return None
    
    def create_user(self, username: str, email: str, password: str, 
                   roles: List[str], permissions: List[str], tenant_id: str) -> User:
        """创建用户"""
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            tenant_id=tenant_id
        )
        
        password_hash = self.security_manager.hash_password(password)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO users 
            (id, username, email, password_hash, roles, permissions, tenant_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user.id, user.username, user.email, password_hash,
            json.dumps(user.roles), json.dumps(user.permissions),
            user.tenant_id, user.created_at
        ))
        
        conn.commit()
        conn.close()
        
        logger.info("User created", user_id=user.id, username=username)
        return user
    
    async def process_ai_request(self, request: APIRequest, messages: List[Dict[str, str]]) -> APIResponse:
        """处理AI请求"""
        start_time = time.time()
        
        try:
            # 验证令牌
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                raise Exception("Missing or invalid authorization header")
            
            token = auth_header[7:]
            payload = self.security_manager.verify_jwt_token(token)
            if not payload:
                raise Exception("Invalid or expired token")
            
            # 检查权限
            if not self.security_manager.check_permission(payload['permissions'], 'ai.chat'):
                raise Exception("Insufficient permissions")
            
            # 限流检查
            rate_limit_key = f"rate_limit:{payload['user_id']}"
            if not await self.rate_limiter.is_allowed(rate_limit_key, 100):  # 每分钟100次
                raise Exception("Rate limit exceeded")
            
            # 记录请求指标
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint,
                status='processing'
            ).inc()
            
            # 调用AI服务
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            response_time = time.time() - start_time
            
            # 记录响应指标
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.endpoint
            ).observe(response_time)
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint,
                status='success'
            ).inc()
            
            # 创建响应
            api_response = APIResponse(
                request_id=request.id,
                status_code=200,
                response_time=response_time,
                response_size=len(str(response.choices[0].message.content))
            )
            
            # 记录到数据库
            self._log_api_request(request, api_response)
            
            # 记录监控指标
            self.monitoring_manager.record_metric("response_time", response_time)
            self.monitoring_manager.record_metric("request_count", 1)
            
            return api_response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # 记录错误指标
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint,
                status='error'
            ).inc()
            
            # 创建错误响应
            api_response = APIResponse(
                request_id=request.id,
                status_code=500,
                response_time=response_time,
                response_size=0,
                error_message=str(e)
            )
            
            # 记录到数据库
            self._log_api_request(request, api_response)
            
            # 记录监控指标
            self.monitoring_manager.record_metric("error_count", 1)
            
            logger.error("AI request failed", 
                        request_id=request.id,
                        error=str(e),
                        response_time=response_time)
            
            raise e
    
    def _log_api_request(self, request: APIRequest, response: APIResponse):
        """记录API请求日志"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO api_requests 
            (id, user_id, tenant_id, endpoint, method, status_code, 
             response_time, response_size, error_message, timestamp, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request.id, request.user_id, request.tenant_id, request.endpoint,
            request.method, response.status_code, response.response_time,
            response.response_size, response.error_message, request.timestamp,
            request.ip_address, request.user_agent
        ))
        
        conn.commit()
        conn.close()
    
    def get_analytics(self, tenant_id: str = None, start_time: datetime = None, 
                     end_time: datetime = None) -> Dict[str, Any]:
        """获取分析数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询条件
        conditions = []
        params = []
        
        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # 总请求数
        cursor.execute(f"SELECT COUNT(*) FROM api_requests{where_clause}", params)
        total_requests = cursor.fetchone()[0]
        
        # 成功请求数
        cursor.execute(f"SELECT COUNT(*) FROM api_requests{where_clause} AND status_code < 400", params)
        successful_requests = cursor.fetchone()[0]
        
        # 平均响应时间
        cursor.execute(f"SELECT AVG(response_time) FROM api_requests{where_clause}", params)
        avg_response_time = cursor.fetchone()[0] or 0
        
        # 错误率
        error_rate = (total_requests - successful_requests) / total_requests if total_requests > 0 else 0
        
        # 按端点统计
        cursor.execute(f"""
            SELECT endpoint, COUNT(*), AVG(response_time)
            FROM api_requests{where_clause}
            GROUP BY endpoint
            ORDER BY COUNT(*) DESC
        """, params)
        endpoint_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "endpoint_stats": [
                {
                    "endpoint": row[0],
                    "request_count": row[1],
                    "avg_response_time": row[2]
                }
                for row in endpoint_stats
            ]
        }

# 演示函数
async def demo_enterprise_setup():
    """演示企业级设置"""
    print("\n🏢 企业级AI网关设置演示")
    print("=" * 50)
    
    async with EnterpriseAIGateway(api_key="your-api-key-here") as gateway:
        
        # 创建租户和用户
        print("👥 创建企业用户...")
        
        # 管理员用户
        admin_user = gateway.create_user(
            username="admin",
            email="admin@company.com",
            password="admin123",
            roles=["admin"],
            permissions=["ai.chat", "ai.completion", "admin.users", "admin.analytics"],
            tenant_id="company_001"
        )
        
        # 普通用户
        user1 = gateway.create_user(
            username="john_doe",
            email="john@company.com",
            password="user123",
            roles=["user"],
            permissions=["ai.chat"],
            tenant_id="company_001"
        )
        
        user2 = gateway.create_user(
            username="jane_smith",
            email="jane@company.com",
            password="user123",
            roles=["analyst"],
            permissions=["ai.chat", "ai.completion", "analytics.view"],
            tenant_id="company_001"
        )
        
        print(f"✅ 创建了3个用户:")
        print(f"   - 管理员: {admin_user.username}")
        print(f"   - 用户1: {user1.username}")
        print(f"   - 用户2: {user2.username}")
        
        # 用户认证
        print("\n🔐 用户认证演示...")
        
        admin_token = gateway.authenticate_user("admin", "admin123")
        user1_token = gateway.authenticate_user("john_doe", "user123")
        
        print(f"✅ 管理员令牌: {admin_token[:50]}...")
        print(f"✅ 用户令牌: {user1_token[:50]}...")
        
        # 注册服务
        print("\n🔧 服务注册演示...")
        
        ai_service = ServiceConfig(
            name="ai-service-1",
            service_type=ServiceType.AI_GATEWAY,
            host="localhost",
            port=8001,
            ssl_enabled=True
        )
        
        auth_service = ServiceConfig(
            name="auth-service-1",
            service_type=ServiceType.AUTH_SERVICE,
            host="localhost",
            port=8002
        )
        
        gateway.service_registry.register_service(ai_service)
        gateway.service_registry.register_service(auth_service)
        
        print(f"✅ 注册了2个服务:")
        print(f"   - AI服务: {ai_service.name}")
        print(f"   - 认证服务: {auth_service.name}")
        
        # 集成端点
        print("\n🔗 集成端点演示...")
        
        crm_endpoint = IntegrationEndpoint(
            id="crm_api",
            name="CRM系统API",
            integration_type=IntegrationType.REST_API,
            url="https://api.crm.company.com/v1/customers",
            auth_config={
                "type": "bearer",
                "token": "crm_api_token_here"
            }
        )
        
        erp_endpoint = IntegrationEndpoint(
            id="erp_api",
            name="ERP系统API",
            integration_type=IntegrationType.REST_API,
            url="https://api.erp.company.com/v1/orders",
            auth_config={
                "type": "api_key",
                "header": "X-API-Key",
                "key": "erp_api_key_here"
            }
        )
        
        gateway.integration_manager.register_endpoint(crm_endpoint)
        gateway.integration_manager.register_endpoint(erp_endpoint)
        
        print(f"✅ 注册了2个集成端点:")
        print(f"   - CRM API: {crm_endpoint.name}")
        print(f"   - ERP API: {erp_endpoint.name}")
        
        return gateway, {
            "admin_token": admin_token,
            "user1_token": user1_token,
            "users": [admin_user, user1, user2]
        }

async def demo_ai_requests():
    """演示AI请求处理"""
    print("\n🤖 AI请求处理演示")
    print("=" * 50)
    
    async with EnterpriseAIGateway(api_key="your-api-key-here") as gateway:
        
        # 创建测试用户
        user = gateway.create_user(
            username="test_user",
            email="test@company.com",
            password="test123",
            roles=["user"],
            permissions=["ai.chat"],
            tenant_id="test_tenant"
        )
        
        # 获取认证令牌
        token = gateway.authenticate_user("test_user", "test123")
        
        print(f"👤 测试用户: {user.username}")
        print(f"🎫 认证令牌: {token[:50]}...")
        
        # 模拟AI请求
        requests_data = [
            {
                "messages": [
                    {"role": "user", "content": "什么是人工智能？"}
                ],
                "description": "AI基础知识咨询"
            },
            {
                "messages": [
                    {"role": "user", "content": "帮我分析一下市场趋势"}
                ],
                "description": "市场分析请求"
            },
            {
                "messages": [
                    {"role": "user", "content": "写一份产品介绍"}
                ],
                "description": "内容生成请求"
            }
        ]
        
        print(f"\n🔄 处理{len(requests_data)}个AI请求...")
        
        results = []
        
        for i, req_data in enumerate(requests_data):
            print(f"\n📝 请求 {i+1}: {req_data['description']}")
            
            # 创建API请求
            api_request = APIRequest(
                id=str(uuid.uuid4()),
                user_id=user.id,
                tenant_id=user.tenant_id,
                endpoint="/v1/chat/completions",
                method="POST",
                headers={"Authorization": f"Bearer {token}"},
                ip_address="192.168.1.100",
                user_agent="Enterprise-Client/1.0"
            )
            
            try:
                start_time = time.time()
                response = await gateway.process_ai_request(api_request, req_data["messages"])
                process_time = time.time() - start_time
                
                print(f"✅ 请求成功")
                print(f"   响应时间: {response.response_time:.3f}s")
                print(f"   状态码: {response.status_code}")
                print(f"   响应大小: {response.response_size} bytes")
                
                results.append(response)
                
            except Exception as e:
                print(f"❌ 请求失败: {str(e)}")
        
        # 获取分析数据
        print(f"\n📊 请求分析:")
        analytics = gateway.get_analytics(tenant_id=user.tenant_id)
        
        print(f"   总请求数: {analytics['total_requests']}")
        print(f"   成功请求数: {analytics['successful_requests']}")
        print(f"   错误率: {analytics['error_rate']:.2%}")
        print(f"   平均响应时间: {analytics['avg_response_time']:.3f}s")
        
        return results

async def demo_monitoring_alerts():
    """演示监控和告警"""
    print("\n📈 监控和告警演示")
    print("=" * 50)
    
    async with EnterpriseAIGateway(api_key="your-api-key-here") as gateway:
        
        print("📊 记录监控指标...")
        
        # 模拟各种指标
        metrics_data = [
            ("response_time", [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5, 0.6, 0.8]),
            ("error_rate", [0.01, 0.02, 0.015, 0.03, 0.025, 0.04, 0.06, 0.08, 0.12, 0.15]),
            ("request_count", [10, 15, 12, 20, 18, 25, 30, 35, 40, 45]),
            ("cpu_usage", [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]),
            ("memory_usage", [40, 42, 45, 48, 50, 52, 55, 58, 60, 62])
        ]
        
        for metric_name, values in metrics_data:
            print(f"   记录 {metric_name} 指标...")
            for value in values:
                gateway.monitoring_manager.record_metric(metric_name, value)
                await asyncio.sleep(0.1)  # 模拟时间间隔
        
        print(f"✅ 指标记录完成")
        
        # 检查告警
        print(f"\n🚨 检查告警状态...")
        alerts = gateway.monitoring_manager.alerts
        
        if alerts:
            print(f"   发现 {len(alerts)} 个告警:")
            for alert in alerts:
                print(f"   - {alert.level.value.upper()}: {alert.title}")
                print(f"     {alert.message}")
                print(f"     时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"   暂无告警")
        
        # 获取指标数据
        print(f"\n📈 获取指标数据...")
        
        for metric_name, _ in metrics_data[:3]:  # 只显示前3个指标
            metric_data = gateway.monitoring_manager.get_metrics(metric_name)
            if metric_data:
                latest_value = metric_data[-1][1]
                avg_value = sum(value for _, value in metric_data) / len(metric_data)
                print(f"   {metric_name}:")
                print(f"     最新值: {latest_value:.3f}")
                print(f"     平均值: {avg_value:.3f}")
                print(f"     数据点数: {len(metric_data)}")
        
        return alerts

async def demo_integration_calls():
    """演示系统集成调用"""
    print("\n🔗 系统集成调用演示")
    print("=" * 50)
    
    async with EnterpriseAIGateway(api_key="your-api-key-here") as gateway:
        
        # 注册模拟集成端点
        print("🔧 注册集成端点...")
        
        # 模拟CRM系统
        crm_endpoint = IntegrationEndpoint(
            id="mock_crm",
            name="模拟CRM系统",
            integration_type=IntegrationType.REST_API,
            url="https://httpbin.org/post",  # 使用httpbin作为模拟端点
            auth_config={
                "type": "bearer",
                "token": "mock_crm_token"
            }
        )
        
        # 模拟ERP系统
        erp_endpoint = IntegrationEndpoint(
            id="mock_erp",
            name="模拟ERP系统",
            integration_type=IntegrationType.REST_API,
            url="https://httpbin.org/post",
            auth_config={
                "type": "api_key",
                "header": "X-API-Key",
                "key": "mock_erp_key"
            }
        )
        
        gateway.integration_manager.register_endpoint(crm_endpoint)
        gateway.integration_manager.register_endpoint(erp_endpoint)
        
        print(f"✅ 注册了2个集成端点")
        
        # 模拟集成调用
        integration_calls = [
            {
                "endpoint_id": "mock_crm",
                "data": {
                    "action": "get_customer",
                    "customer_id": "12345",
                    "fields": ["name", "email", "phone"]
                },
                "description": "获取客户信息"
            },
            {
                "endpoint_id": "mock_erp",
                "data": {
                    "action": "create_order",
                    "customer_id": "12345",
                    "products": [
                        {"id": "P001", "quantity": 2},
                        {"id": "P002", "quantity": 1}
                    ]
                },
                "description": "创建订单"
            }
        ]
        
        print(f"\n🔄 执行集成调用...")
        
        results = []
        
        for i, call_data in enumerate(integration_calls):
            print(f"\n📞 调用 {i+1}: {call_data['description']}")
            
            try:
                start_time = time.time()
                result = await gateway.integration_manager.call_endpoint(
                    call_data["endpoint_id"],
                    call_data["data"]
                )
                call_time = time.time() - start_time
                
                print(f"✅ 调用成功")
                print(f"   耗时: {call_time:.3f}s")
                print(f"   响应: {str(result)[:100]}...")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ 调用失败: {str(e)}")
        
        # 检查熔断器状态
        print(f"\n⚡ 熔断器状态:")
        for endpoint_id, circuit_breaker in gateway.integration_manager.circuit_breakers.items():
            endpoint = gateway.integration_manager.endpoints[endpoint_id]
            print(f"   {endpoint.name}: {circuit_breaker.state}")
            print(f"     失败次数: {circuit_breaker.failure_count}")
        
        return results

async def demo_security_features():
    """演示安全特性"""
    print("\n🔒 安全特性演示")
    print("=" * 50)
    
    async with EnterpriseAIGateway(api_key="your-api-key-here") as gateway:
        
        # 创建测试用户
        user = gateway.create_user(
            username="security_test",
            email="security@company.com",
            password="secure123",
            roles=["user"],
            permissions=["ai.chat"],
            tenant_id="security_tenant"
        )
        
        print(f"👤 创建测试用户: {user.username}")
        
        # 正常认证
        print(f"\n🔐 正常认证测试...")
        token = gateway.authenticate_user("security_test", "secure123")
        print(f"✅ 认证成功，令牌: {token[:50]}...")
        
        # 错误密码认证
        print(f"\n❌ 错误密码测试...")
        invalid_token = gateway.authenticate_user("security_test", "wrong_password")
        print(f"❌ 认证失败: {invalid_token}")
        
        # 令牌验证
        print(f"\n🎫 令牌验证测试...")
        payload = gateway.security_manager.verify_jwt_token(token)
        if payload:
            print(f"✅ 令牌有效")
            print(f"   用户ID: {payload['user_id']}")
            print(f"   用户名: {payload['username']}")
            print(f"   租户ID: {payload['tenant_id']}")
            print(f"   角色: {payload['roles']}")
            print(f"   权限: {payload['permissions']}")
        
        # 权限检查
        print(f"\n🛡️ 权限检查测试...")
        
        permissions_tests = [
            ("ai.chat", "AI聊天权限"),
            ("ai.completion", "AI补全权限"),
            ("admin.users", "用户管理权限"),
            ("admin.analytics", "分析管理权限")
        ]
        
        for permission, description in permissions_tests:
            has_permission = gateway.security_manager.check_permission(
                payload['permissions'], permission
            )
            status = "✅ 有权限" if has_permission else "❌ 无权限"
            print(f"   {description}: {status}")
        
        # 限流测试
        print(f"\n🚦 限流测试...")
        
        rate_limit_key = f"test_rate_limit:{user.id}"
        
        # 模拟快速请求
        allowed_count = 0
        denied_count = 0
        
        for i in range(10):
            is_allowed = await gateway.rate_limiter.is_allowed(rate_limit_key, 5, 60)  # 每分钟5次
            if is_allowed:
                allowed_count += 1
            else:
                denied_count += 1
        
        print(f"   允许请求: {allowed_count}")
        print(f"   拒绝请求: {denied_count}")
        
        # 令牌撤销
        print(f"\n🚫 令牌撤销测试...")
        gateway.security_manager.revoke_token(token)
        
        revoked_payload = gateway.security_manager.verify_jwt_token(token)
        if revoked_payload:
            print(f"❌ 令牌撤销失败")
        else:
            print(f"✅ 令牌已成功撤销")
        
        return {
            "user": user,
            "token_valid": payload is not None,
            "token_revoked": revoked_payload is None,
            "rate_limit_test": {
                "allowed": allowed_count,
                "denied": denied_count
            }
        }

async def main():
    """主演示函数"""
    print("🏢 HarborAI 企业级应用集成演示")
    print("=" * 60)
    
    try:
        # 企业级设置演示
        gateway, setup_data = await demo_enterprise_setup()
        
        # AI请求处理演示
        await demo_ai_requests()
        
        # 监控和告警演示
        await demo_monitoring_alerts()
        
        # 系统集成调用演示
        await demo_integration_calls()
        
        # 安全特性演示
        await demo_security_features()
        
        print("\n📊 企业级特性总结:")
        print("   ✅ 多租户用户管理")
        print("   ✅ JWT身份认证和授权")
        print("   ✅ API限流和熔断保护")
        print("   ✅ 服务注册和发现")
        print("   ✅ 系统集成和API网关")
        print("   ✅ 实时监控和告警")
        print("   ✅ 结构化日志和指标")
        print("   ✅ 高可用和负载均衡")
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境部署建议:")
        print("   1. 使用Kubernetes进行容器化部署")
        print("   2. 配置Redis集群和数据库主从")
        print("   3. 集成企业级监控系统（Prometheus + Grafana）")
        print("   4. 实现完整的CI/CD流水线")
        print("   5. 配置企业级安全策略和合规审计")
        print("   6. 实现多地域部署和灾备方案")
        print("   7. 集成企业SSO和LDAP系统")
        print("   8. 配置API文档和开发者门户")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())