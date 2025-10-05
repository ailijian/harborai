#!/usr/bin/env python3
"""
HarborAI 日志监控示例

这个示例展示了如何在HarborAI中实现全面的日志记录和监控，
包括请求追踪、性能监控、错误分析和实时告警。

场景描述:
- 结构化日志记录
- 性能指标监控
- 错误追踪分析
- 实时告警系统

应用价值:
- 提升系统可观测性
- 快速问题定位
- 性能优化指导
- 运维监控支持
"""

import os
import json
import time
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import statistics
import threading
from collections import defaultdict, deque
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

try:
    from harborai import HarborAI
except ImportError:
    print("❌ 请先安装 HarborAI: pip install harborai")
    exit(1)


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: str
    level: LogLevel
    message: str
    module: str
    function: str
    request_id: str = None
    user_id: str = None
    session_id: str = None
    model_name: str = None
    response_time: float = None
    token_count: int = None
    cost: float = None
    error_code: str = None
    stack_trace: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MetricEntry:
    """指标条目"""
    timestamp: str
    name: str
    type: MetricType
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==
    threshold: float
    duration: int  # 持续时间（秒）
    enabled: bool = True
    description: str = ""


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str, db_path: str = "monitoring.db"):
        self.name = name
        self.db_path = db_path
        self.init_database()
        
        # 设置标准日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 创建文件处理器
        log_file = f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT NOT NULL,
                function TEXT NOT NULL,
                request_id TEXT,
                user_id TEXT,
                session_id TEXT,
                model_name TEXT,
                response_time REAL,
                token_count INTEGER,
                cost REAL,
                error_code TEXT,
                stack_trace TEXT,
                metadata TEXT
            )
        ''')
        
        # 创建指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                value REAL NOT NULL,
                tags TEXT
            )
        ''')
        
        # 创建告警表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rule_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold REAL NOT NULL,
                message TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TEXT
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_request_id ON logs(request_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)')
        
        conn.commit()
        conn.close()
    
    def log(self, entry: LogEntry):
        """记录日志条目"""
        # 记录到标准日志
        log_level = getattr(logging, entry.level.value)
        self.logger.log(log_level, entry.message)
        
        # 记录到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO logs 
            (timestamp, level, message, module, function, request_id, user_id, 
             session_id, model_name, response_time, token_count, cost, 
             error_code, stack_trace, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.timestamp, entry.level.value, entry.message, entry.module,
            entry.function, entry.request_id, entry.user_id, entry.session_id,
            entry.model_name, entry.response_time, entry.token_count, entry.cost,
            entry.error_code, entry.stack_trace,
            json.dumps(entry.metadata) if entry.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def debug(self, message: str, **kwargs):
        """记录DEBUG级别日志"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.DEBUG,
            message=message,
            module=self.name,
            function="debug",
            **kwargs
        )
        self.log(entry)
    
    def info(self, message: str, **kwargs):
        """记录INFO级别日志"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.INFO,
            message=message,
            module=self.name,
            function="info",
            **kwargs
        )
        self.log(entry)
    
    def warning(self, message: str, **kwargs):
        """记录WARNING级别日志"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.WARNING,
            message=message,
            module=self.name,
            function="warning",
            **kwargs
        )
        self.log(entry)
    
    def error(self, message: str, **kwargs):
        """记录ERROR级别日志"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.ERROR,
            message=message,
            module=self.name,
            function="error",
            **kwargs
        )
        self.log(entry)
    
    def critical(self, message: str, **kwargs):
        """记录CRITICAL级别日志"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.CRITICAL,
            message=message,
            module=self.name,
            function="critical",
            **kwargs
        )
        self.log(entry)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=1000)  # 内存缓冲区
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        
        # 启动后台线程定期刷新指标
        self.flush_thread = threading.Thread(target=self._flush_metrics_loop, daemon=True)
        self.flush_thread.start()
    
    def _flush_metrics_loop(self):
        """后台线程定期刷新指标到数据库"""
        while True:
            time.sleep(10)  # 每10秒刷新一次
            self._flush_metrics()
    
    def _flush_metrics(self):
        """刷新指标到数据库"""
        if not self.metrics_buffer:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 批量插入指标
        metrics_to_insert = []
        while self.metrics_buffer:
            metric = self.metrics_buffer.popleft()
            metrics_to_insert.append((
                metric.timestamp, metric.name, metric.type.value,
                metric.value, json.dumps(metric.tags) if metric.tags else None
            ))
        
        if metrics_to_insert:
            cursor.executemany('''
                INSERT INTO metrics (timestamp, name, type, value, tags)
                VALUES (?, ?, ?, ?, ?)
            ''', metrics_to_insert)
        
        conn.commit()
        conn.close()
    
    def record_metric(self, metric: MetricEntry):
        """记录指标"""
        self.metrics_buffer.append(metric)
        
        # 更新内存中的指标
        if metric.type == MetricType.COUNTER:
            self.counters[metric.name] += metric.value
        elif metric.type == MetricType.GAUGE:
            self.gauges[metric.name] = metric.value
        elif metric.type == MetricType.HISTOGRAM:
            self.histograms[metric.name].append(metric.value)
            # 保持最近1000个值
            if len(self.histograms[metric.name]) > 1000:
                self.histograms[metric.name] = self.histograms[metric.name][-1000:]
        elif metric.type == MetricType.TIMER:
            self.timers[metric.name].append(metric.value)
            # 保持最近1000个值
            if len(self.timers[metric.name]) > 1000:
                self.timers[metric.name] = self.timers[metric.name][-1000:]
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """增加计数器"""
        metric = MetricEntry(
            timestamp=datetime.now().isoformat(),
            name=name,
            type=MetricType.COUNTER,
            value=value,
            tags=tags
        )
        self.record_metric(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """设置仪表盘值"""
        metric = MetricEntry(
            timestamp=datetime.now().isoformat(),
            name=name,
            type=MetricType.GAUGE,
            value=value,
            tags=tags
        )
        self.record_metric(metric)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录直方图值"""
        metric = MetricEntry(
            timestamp=datetime.now().isoformat(),
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            tags=tags
        )
        self.record_metric(metric)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """记录计时器值"""
        metric = MetricEntry(
            timestamp=datetime.now().isoformat(),
            name=name,
            type=MetricType.TIMER,
            value=duration,
            tags=tags
        )
        self.record_metric(metric)
    
    def get_metric_stats(self, name: str, metric_type: MetricType = None) -> Dict[str, Any]:
        """获取指标统计"""
        if metric_type == MetricType.COUNTER or name in self.counters:
            return {"type": "counter", "value": self.counters[name]}
        
        elif metric_type == MetricType.GAUGE or name in self.gauges:
            return {"type": "gauge", "value": self.gauges[name]}
        
        elif metric_type == MetricType.HISTOGRAM or name in self.histograms:
            values = self.histograms[name]
            if not values:
                return {"type": "histogram", "count": 0}
            
            return {
                "type": "histogram",
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
            }
        
        elif metric_type == MetricType.TIMER or name in self.timers:
            values = self.timers[name]
            if not values:
                return {"type": "timer", "count": 0}
            
            return {
                "type": "timer",
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
            }
        
        return {}


class AlertManager:
    """告警管理器"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.rules = []
        self.alert_handlers = []
        
        # 默认告警规则
        self.add_rule(AlertRule(
            name="high_response_time",
            metric_name="api_response_time",
            condition=">",
            threshold=5.0,
            duration=60,
            description="API响应时间过高"
        ))
        
        self.add_rule(AlertRule(
            name="high_error_rate",
            metric_name="api_error_rate",
            condition=">",
            threshold=0.1,
            duration=300,
            description="API错误率过高"
        ))
        
        self.add_rule(AlertRule(
            name="high_cost",
            metric_name="hourly_cost",
            condition=">",
            threshold=1.0,
            duration=3600,
            description="小时成本过高"
        ))
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules.append(rule)
    
    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """检查告警条件"""
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if rule.metric_name not in metrics:
                continue
            
            current_value = metrics[rule.metric_name]
            
            # 检查条件
            triggered = False
            if rule.condition == ">" and current_value > rule.threshold:
                triggered = True
            elif rule.condition == "<" and current_value < rule.threshold:
                triggered = True
            elif rule.condition == ">=" and current_value >= rule.threshold:
                triggered = True
            elif rule.condition == "<=" and current_value <= rule.threshold:
                triggered = True
            elif rule.condition == "==" and current_value == rule.threshold:
                triggered = True
            
            if triggered:
                self._trigger_alert(rule, current_value)
    
    def _trigger_alert(self, rule: AlertRule, current_value: float):
        """触发告警"""
        alert_data = {
            "rule_name": rule.name,
            "metric_name": rule.metric_name,
            "current_value": current_value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "description": rule.description,
            "timestamp": datetime.now().isoformat()
        }
        
        # 记录到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts 
            (timestamp, rule_name, metric_name, current_value, threshold, message)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            alert_data["timestamp"], rule.name, rule.metric_name,
            current_value, rule.threshold, rule.description
        ))
        
        conn.commit()
        conn.close()
        
        # 调用告警处理器
        for handler in self.alert_handlers:
            try:
                handler(rule.name, alert_data)
            except Exception as e:
                print(f"告警处理器错误: {e}")


class HarborAIMonitor:
    """HarborAI监控器"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.logger = StructuredLogger("harborai_monitor", db_path)
        self.metrics = MetricsCollector(db_path)
        self.alerts = AlertManager(db_path)
        
        # 添加控制台告警处理器
        self.alerts.add_alert_handler(self._console_alert_handler)
        
        # 性能统计
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0
        self.total_cost = 0
    
    def _console_alert_handler(self, rule_name: str, alert_data: Dict[str, Any]):
        """控制台告警处理器"""
        print(f"\n🚨 告警触发: {rule_name}")
        print(f"   指标: {alert_data['metric_name']}")
        print(f"   当前值: {alert_data['current_value']}")
        print(f"   阈值: {alert_data['threshold']}")
        print(f"   描述: {alert_data['description']}")
        print(f"   时间: {alert_data['timestamp']}")
    
    async def monitor_api_call(self, client: HarborAI, model_name: str, 
                              prompt: str, **kwargs) -> Any:
        """
        监控API调用
        
        Args:
            client: HarborAI客户端
            model_name: 模型名称
            prompt: 提示词
            **kwargs: 其他参数
            
        Returns:
            API响应
        """
        request_id = f"req_{int(time.time() * 1000)}_{hash(prompt) % 10000}"
        start_time = time.time()
        
        # 记录请求开始
        self.logger.info(
            f"API请求开始: {model_name}",
            request_id=request_id,
            model_name=model_name,
            metadata={
                "prompt_length": len(prompt),
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
        )
        
        # 增加请求计数
        self.metrics.increment_counter("api_requests_total", tags={"model": model_name})
        self.request_count += 1
        
        try:
            response = await client.chat.completions.acreate(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            response_time = time.time() - start_time
            usage = response.usage
            
            # 计算成本（简化）
            cost = (usage.total_tokens / 1000) * 0.002
            
            # 记录成功响应
            self.logger.info(
                f"API请求成功: {model_name}",
                request_id=request_id,
                model_name=model_name,
                response_time=response_time,
                token_count=usage.total_tokens,
                cost=cost,
                metadata={
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "response_length": len(response.choices[0].message.content)
                }
            )
            
            # 记录指标
            self.metrics.record_timer("api_response_time", response_time, tags={"model": model_name})
            self.metrics.record_histogram("api_token_count", usage.total_tokens, tags={"model": model_name})
            self.metrics.record_histogram("api_cost", cost, tags={"model": model_name})
            self.metrics.increment_counter("api_requests_success", tags={"model": model_name})
            
            # 更新统计
            self.total_response_time += response_time
            self.total_cost += cost
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # 记录错误
            self.logger.error(
                f"API请求失败: {model_name}",
                request_id=request_id,
                model_name=model_name,
                response_time=response_time,
                error_code=type(e).__name__,
                stack_trace=str(e),
                metadata={
                    "prompt_length": len(prompt)
                }
            )
            
            # 记录错误指标
            self.metrics.increment_counter("api_requests_error", tags={"model": model_name, "error": type(e).__name__})
            self.error_count += 1
            
            raise e
        
        finally:
            # 更新实时指标
            if self.request_count > 0:
                error_rate = self.error_count / self.request_count
                avg_response_time = self.total_response_time / self.request_count
                
                self.metrics.set_gauge("api_error_rate", error_rate)
                self.metrics.set_gauge("api_avg_response_time", avg_response_time)
                self.metrics.set_gauge("api_total_cost", self.total_cost)
                
                # 检查告警
                self.alerts.check_alerts({
                    "api_response_time": avg_response_time,
                    "api_error_rate": error_rate,
                    "hourly_cost": self.total_cost  # 简化为总成本
                })
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        dashboard = {
            "overview": {
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "success_rate": (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 0,
                "total_cost": self.total_cost,
                "avg_response_time": self.total_response_time / self.request_count if self.request_count > 0 else 0
            },
            "metrics": {}
        }
        
        # 获取各种指标统计
        metric_names = [
            "api_response_time", "api_token_count", "api_cost",
            "api_requests_total", "api_requests_success", "api_requests_error"
        ]
        
        for name in metric_names:
            stats = self.metrics.get_metric_stats(name)
            if stats:
                dashboard["metrics"][name] = stats
        
        return dashboard
    
    def get_recent_logs(self, limit: int = 50, level: LogLevel = None) -> List[Dict[str, Any]]:
        """获取最近的日志"""
        conn = sqlite3.connect(self.logger.db_path)
        cursor = conn.cursor()
        
        where_clause = ""
        params = []
        
        if level:
            where_clause = "WHERE level = ?"
            params.append(level.value)
        
        cursor.execute(f'''
            SELECT timestamp, level, message, module, function, request_id, 
                   model_name, response_time, token_count, cost, error_code
            FROM logs
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        ''', params + [limit])
        
        logs = []
        for row in cursor.fetchall():
            logs.append({
                "timestamp": row[0],
                "level": row[1],
                "message": row[2],
                "module": row[3],
                "function": row[4],
                "request_id": row[5],
                "model_name": row[6],
                "response_time": row[7],
                "token_count": row[8],
                "cost": row[9],
                "error_code": row[10]
            })
        
        conn.close()
        return logs
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的告警"""
        conn = sqlite3.connect(self.alerts.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, rule_name, metric_name, current_value, 
                   threshold, message, resolved
            FROM alerts
            ORDER BY timestamp DESC
            LIMIT ?
        ''', [limit])
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                "timestamp": row[0],
                "rule_name": row[1],
                "metric_name": row[2],
                "current_value": row[3],
                "threshold": row[4],
                "message": row[5],
                "resolved": bool(row[6])
            })
        
        conn.close()
        return alerts


async def monitoring_demo():
    """监控演示"""
    print("="*60)
    print("📊 HarborAI 日志监控示例")
    print("="*60)
    
    # 初始化监控器
    monitor = HarborAIMonitor()
    
    print("✅ 监控系统初始化完成")
    
    # 检查API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 需要DEEPSEEK_API_KEY环境变量")
        return
    
    client = HarborAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # 测试场景
    test_scenarios = [
        {"prompt": "你好，请介绍一下你自己。", "model": "deepseek-chat"},
        {"prompt": "请写一个Python函数来计算斐波那契数列。", "model": "deepseek-chat"},
        {"prompt": "请解释什么是机器学习。", "model": "deepseek-chat"},
        {"prompt": "这是一个故意的错误测试", "model": "invalid-model"},  # 故意的错误
        {"prompt": "请分析当前AI技术的发展趋势。", "model": "deepseek-chat"}
    ]
    
    print(f"\n🔹 1. 执行监控测试")
    print(f"📋 将执行 {len(test_scenarios)} 个测试场景")
    
    # 执行测试场景
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎯 场景 {i}: {scenario['prompt'][:50]}...")
        
        try:
            response = await monitor.monitor_api_call(
                client, scenario["model"], scenario["prompt"],
                max_tokens=200, temperature=0.7
            )
            print(f"   ✅ 成功 - 响应长度: {len(response.choices[0].message.content)}")
        except Exception as e:
            print(f"   ❌ 失败: {e}")
        
        # 短暂延迟
        await asyncio.sleep(1)
    
    # 显示监控仪表板
    print(f"\n🔹 2. 监控仪表板")
    dashboard = monitor.get_monitoring_dashboard()
    
    overview = dashboard["overview"]
    print(f"📊 总体概览:")
    print(f"   总请求数: {overview['total_requests']}")
    print(f"   错误数量: {overview['error_count']}")
    print(f"   成功率: {overview['success_rate']:.1%}")
    print(f"   总成本: ${overview['total_cost']:.4f}")
    print(f"   平均响应时间: {overview['avg_response_time']:.2f}秒")
    
    print(f"\n📈 指标统计:")
    for metric_name, stats in dashboard["metrics"].items():
        print(f"   {metric_name}:")
        if stats["type"] == "counter":
            print(f"     计数: {stats['value']}")
        elif stats["type"] == "gauge":
            print(f"     当前值: {stats['value']:.4f}")
        elif stats["type"] in ["histogram", "timer"]:
            if stats.get("count", 0) > 0:
                print(f"     计数: {stats['count']}")
                print(f"     平均: {stats['mean']:.4f}")
                print(f"     P95: {stats['p95']:.4f}")
                print(f"     最大: {stats['max']:.4f}")
    
    # 显示最近日志
    print(f"\n🔹 3. 最近日志")
    recent_logs = monitor.get_recent_logs(limit=10)
    
    if recent_logs:
        print(f"📝 最近 {len(recent_logs)} 条日志:")
        for log in recent_logs:
            timestamp = log["timestamp"][:19]  # 只显示到秒
            level_emoji = {
                "DEBUG": "🔍", "INFO": "ℹ️", "WARNING": "⚠️", 
                "ERROR": "❌", "CRITICAL": "🚨"
            }
            emoji = level_emoji.get(log["level"], "📝")
            
            print(f"   {emoji} {timestamp} [{log['level']}] {log['message']}")
            if log["request_id"]:
                print(f"      请求ID: {log['request_id']}")
            if log["response_time"]:
                print(f"      响应时间: {log['response_time']:.2f}s")
            if log["cost"]:
                print(f"      成本: ${log['cost']:.4f}")
    
    # 显示告警
    print(f"\n🔹 4. 告警状态")
    recent_alerts = monitor.get_recent_alerts(limit=5)
    
    if recent_alerts:
        print(f"🚨 最近 {len(recent_alerts)} 条告警:")
        for alert in recent_alerts:
            timestamp = alert["timestamp"][:19]
            status = "✅已解决" if alert["resolved"] else "🔴活跃"
            print(f"   {timestamp} [{status}] {alert['rule_name']}")
            print(f"      指标: {alert['metric_name']}")
            print(f"      当前值: {alert['current_value']:.4f}")
            print(f"      阈值: {alert['threshold']:.4f}")
            print(f"      描述: {alert['message']}")
    else:
        print("✅ 暂无告警")
    
    # 性能分析建议
    print(f"\n🔹 5. 性能分析建议")
    
    if overview['avg_response_time'] > 3.0:
        print("⚠️  平均响应时间较高，建议:")
        print("   - 检查网络连接")
        print("   - 优化提示词长度")
        print("   - 考虑使用更快的模型")
    
    if overview['error_count'] > 0:
        print("⚠️  存在错误请求，建议:")
        print("   - 检查模型名称是否正确")
        print("   - 验证API密钥配置")
        print("   - 添加重试机制")
    
    if overview['total_cost'] > 0.1:
        print("💰 成本提醒:")
        print(f"   - 当前总成本: ${overview['total_cost']:.4f}")
        print("   - 建议设置成本预算")
        print("   - 考虑使用更经济的模型")
    
    print(f"\n🎉 监控演示完成！")
    print(f"📁 监控数据保存在: monitoring.db")
    print(f"📝 日志文件: harborai_monitor.log")


if __name__ == "__main__":
    asyncio.run(monitoring_demo())