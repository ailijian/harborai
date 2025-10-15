#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 中级日志功能演示 - 日志监控与分析

这个脚本演示了 HarborAI 日志系统的中级功能，包括：
1. 真实模型调用和日志记录
2. 结构化日志记录
3. 性能指标监控
4. 错误追踪和分析
5. 告警规则设置
6. 日志查看和分析
7. 新的布局模式演示（classic 和 enhanced）
8. 新的 trace_id 格式支持（hb_ 前缀）
9. 配对显示和统计分析

使用方法：
    python logging_monitoring.py                    # 运行完整演示
    python logging_monitoring.py --real-calls       # 进行真实模型调用
    python logging_monitoring.py --monitoring-only  # 仅演示监控功能
    python logging_monitoring.py --layout-demo      # 演示布局模式
    python logging_monitoring.py --trace-id-demo    # 演示 trace_id 功能

更新内容：
- 支持新的 hb_ 前缀 trace_id 格式
- 添加布局模式演示（classic 和 enhanced）
- 增强性能监控和告警功能
- 改进错误处理和用户体验
- 添加配对显示和统计分析

作者: HarborAI Team
版本: 2.0.0
更新时间: 2025-01-14
"""

import os
import json
import time
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import statistics
import threading
from collections import defaultdict, deque
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加本地源码路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from harborai import HarborAI
except ImportError:
    print("❌ 无法导入 HarborAI，请检查路径配置")
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
                "max_tokens": kwargs.get("max_tokens")  # 默认无限制，由模型厂商控制
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
    print(f"   总成本: ¥{overview['total_cost']:.4f}")
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
                print(f"      成本: ¥{log['cost']:.4f}")
    
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


# ============================================================================
# 监控仪表板功能
# ============================================================================

def show_monitoring_dashboard(monitor: HarborAIMonitor):
    """显示监控仪表板"""
    print("🔸 生成监控报告...")
    
    # 显示基本统计
    print(f"📊 基本统计:")
    print(f"   - 总请求数: {monitor.request_count}")
    print(f"   - 错误数: {monitor.error_count}")
    print(f"   - 总成本: ¥{monitor.total_cost:.4f}")
    
    if monitor.request_count > 0:
        avg_response_time = monitor.total_response_time / monitor.request_count
        success_rate = ((monitor.request_count - monitor.error_count) / monitor.request_count) * 100
        print(f"   - 平均响应时间: {avg_response_time:.2f}ms")
        print(f"   - 成功率: {success_rate:.1f}%")
    
    # 显示指标统计
    print(f"\n📈 指标统计:")
    print(f"   - 计数器: {len(monitor.metrics.counters)} 个")
    print(f"   - 仪表盘: {len(monitor.metrics.gauges)} 个")
    print(f"   - 直方图: {len(monitor.metrics.histograms)} 个")
    print(f"   - 计时器: {len(monitor.metrics.timers)} 个")
    
    # 显示告警规则
    print(f"\n🚨 告警规则:")
    print(f"   - 已配置规则: {len(monitor.alerts.rules)} 个")
    enabled_rules = sum(1 for rule in monitor.alerts.rules if rule.enabled)
    print(f"   - 启用规则: {enabled_rules} 个")
    
    print("✅ 监控报告生成完成")


# ============================================================================
# 真实模型调用和完整演示功能
# ============================================================================

class RealModelDemo:
    """真实模型调用演示类"""
    
    def __init__(self):
        """初始化演示类"""
        self.harborai = HarborAI()
        self.logger = StructuredLogger("RealModelDemo")
        self.monitor = HarborAIMonitor()
        
        # 可用模型列表（从项目配置获取）
        self.available_models = [
            {'vendor': 'deepseek', 'model': 'deepseek-chat', 'is_reasoning': False},
            {'vendor': 'deepseek', 'model': 'deepseek-reasoner', 'is_reasoning': True},
            {'vendor': 'ernie', 'model': 'ernie-3.5-8k', 'is_reasoning': False},
            {'vendor': 'ernie', 'model': 'ernie-4.0-turbo-8k', 'is_reasoning': False},
            {'vendor': 'doubao', 'model': 'doubao-1-5-pro-32k-character-250715', 'is_reasoning': False},
        ]
        
        # 测试消息列表
        self.test_messages = [
            "你好，请简单介绍一下你自己。",
            "请解释什么是人工智能？",
            "写一个简单的Python函数来计算斐波那契数列。",
            "请推荐3本值得阅读的技术书籍。",
            "解释一下什么是机器学习？"
        ]
    
    def print_section(self, title: str):
        """打印章节标题"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def print_step(self, step: str):
        """打印步骤"""
        print(f"\n🔸 {step}")
    
    def check_environment(self):
        """检查环境配置"""
        self.print_step("检查环境配置...")
        
        # 检查API密钥配置
        api_keys = {
            'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
            'DOUBAO_API_KEY': os.getenv('DOUBAO_API_KEY'),
            'WENXIN_API_KEY': os.getenv('WENXIN_API_KEY'),
        }
        
        configured_keys = {k: v for k, v in api_keys.items() if v}
        
        if not configured_keys:
            print("❌ 没有配置任何API密钥")
            return False
        
        print(f"✅ 已配置的API密钥: {', '.join(configured_keys.keys())}")
        
        # 记录环境检查日志
        self.logger.info(
            "环境检查完成",
            metadata={"configured_keys": list(configured_keys.keys())}
        )
        
        return True
    
    async def test_single_model(self, model_info: Dict[str, Any], message: str, request_id: str) -> Dict[str, Any]:
        """测试单个模型"""
        model_name = model_info['model']
        vendor = model_info['vendor']
        
        # 记录请求开始
        self.logger.info(
            f"开始测试模型: {model_name}",
            request_id=request_id,
            model_name=model_name,
            metadata={"vendor": vendor, "message": message[:50]}
        )
        
        try:
            start_time = time.time()
            
            # 调用模型
            response = await self.harborai.achat(
                model=model_name,
                messages=[{"role": "user", "content": message}],
                max_tokens=100  # 限制token数以节省成本
            )
            
            duration = time.time() - start_time
            
            # 检查响应
            if response and hasattr(response, 'content'):
                # 记录成功日志
                self.logger.info(
                    f"模型调用成功: {model_name}",
                    request_id=request_id,
                    model_name=model_name,
                    response_time=duration,
                    token_count=len(response.content) if response.content else 0,
                    metadata={
                        "vendor": vendor,
                        "response_length": len(response.content) if response.content else 0,
                        "success": True
                    }
                )
                
                # 记录性能指标
                self.monitor.record_metric("api_response_time", duration, {"model": model_name, "vendor": vendor})
                self.monitor.record_metric("api_success_count", 1, {"model": model_name, "vendor": vendor})
                
                print(f"✅ {model_name}: 响应成功 ({duration:.2f}s)")
                return {
                    'model': model_name,
                    'vendor': vendor,
                    'success': True,
                    'duration': duration,
                    'response_length': len(response.content) if response.content else 0,
                    'error': None
                }
            else:
                # 记录警告日志
                self.logger.warning(
                    f"模型响应为空: {model_name}",
                    request_id=request_id,
                    model_name=model_name,
                    response_time=duration,
                    metadata={"vendor": vendor, "error": "响应为空"}
                )
                
                # 记录错误指标
                self.monitor.record_metric("api_error_count", 1, {"model": model_name, "vendor": vendor, "error_type": "empty_response"})
                
                print(f"⚠️  {model_name}: 响应为空")
                return {
                    'model': model_name,
                    'vendor': vendor,
                    'success': False,
                    'duration': duration,
                    'response_length': 0,
                    'error': "响应为空"
                }
                
        except Exception as e:
            duration = time.time() - start_time
            
            # 记录错误日志
            self.logger.error(
                f"模型调用失败: {model_name}",
                request_id=request_id,
                model_name=model_name,
                response_time=duration,
                error_code="API_ERROR",
                stack_trace=str(e),
                metadata={"vendor": vendor, "error": str(e)}
            )
            
            # 记录错误指标
            self.monitor.record_metric("api_error_count", 1, {"model": model_name, "vendor": vendor, "error_type": "exception"})
            
            print(f"❌ {model_name}: {str(e)}")
            return {
                'model': model_name,
                'vendor': vendor,
                'success': False,
                'duration': duration,
                'response_length': 0,
                'error': str(e)
            }
    
    async def run_model_tests(self, target_model: Optional[str] = None, test_only: bool = False):
        """运行模型测试"""
        self.print_step("开始模型测试...")
        
        # 过滤模型
        if target_model:
            models_to_test = [m for m in self.available_models if m['model'] == target_model]
            if not models_to_test:
                print(f"❌ 未找到模型: {target_model}")
                return []
        else:
            models_to_test = self.available_models
        
        # 选择测试消息
        messages_to_test = self.test_messages[:1] if test_only else self.test_messages[:3]
        
        results = []
        
        for model_info in models_to_test:
            for i, message in enumerate(messages_to_test):
                request_id = f"test_{model_info['model']}_{i}_{int(time.time())}"
                print(f"测试 {model_info['model']} - 消息 {i+1}")
                
                result = await self.test_single_model(model_info, message, request_id)
                results.append(result)
                
                # 避免请求过于频繁
                await asyncio.sleep(1)
        
        return results
    
    def show_test_results(self, results: List[Dict[str, Any]]):
        """显示测试结果"""
        if not results:
            return
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        self.print_step("测试结果摘要")
        
        print(f"📊 总测试数: {len(results)}")
        print(f"✅ 成功: {len(successful)}")
        print(f"❌ 失败: {len(failed)}")
        print(f"📈 成功率: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            avg_duration = sum(r['duration'] for r in successful) / len(successful)
            print(f"⏱️  平均响应时间: {avg_duration:.2f}s")
        
        # 记录测试摘要日志
        self.logger.info(
            "模型测试完成",
            metadata={
                "total_tests": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful)/len(results)*100
            }
        )
    
    def demonstrate_log_viewing(self):
        """演示日志查看功能"""
        self.print_step("演示日志查看功能...")
        
        try:
            # 导入日志查看工具
            from harborai.config.settings import get_settings
            from harborai.database.file_log_parser import FileLogParser
            
            settings = get_settings()
            parser = FileLogParser(settings.file_log_directory)
            
            # 查询最近的日志
            result = parser.query_api_logs(days=1, limit=5)
            
            if result.error:
                print(f"❌ 查询失败: {result.error}")
                return
            
            print(f"✅ 查询成功: 找到 {result.total_count} 条日志记录")
            
            if result.data:
                print("\n📋 最近的日志记录:")
                for i, log in enumerate(result.data[:3], 1):
                    timestamp = log.get('timestamp', 'N/A')
                    model = log.get('model', 'unknown')
                    success = log.get('success', False)
                    tokens = log.get('tokens', {})
                    total_tokens = tokens.get('total_tokens', 0) if isinstance(tokens, dict) else 0
                    
                    status = "✅ 成功" if success else "❌ 失败"
                    print(f"  [{i}] {timestamp} | {model} | {status} | {total_tokens} tokens")
            
        except Exception as e:
            print(f"❌ 日志查看演示失败: {e}")
    
    def show_usage_commands(self):
        """显示使用命令"""
        self.print_step("日志查看命令参考...")
        
        commands = [
            ("查看所有日志", "python view_logs.py"),
            ("查看文件日志", "python view_logs.py --source file"),
            ("查看统计信息", "python view_logs.py --stats"),
            ("查看特定模型", "python view_logs.py --model deepseek-chat"),
            ("JSON格式输出", "python view_logs.py --format json"),
            ("查看请求日志", "python view_logs.py --type request"),
            ("查看响应日志", "python view_logs.py --type response"),
            ("配对显示请求-响应", "python view_logs.py --type paired")
        ]
        
        print("\n💡 常用日志查看命令:")
        for desc, cmd in commands:
            print(f"  • {desc}: {cmd}")
    
    async def run_complete_demo(self, target_model: Optional[str] = None, test_only: bool = False, monitor_only: bool = False):
        """运行完整演示"""
        self.print_section("HarborAI 完整日志监控与真实模型调用演示")
        
        print("📝 本演示将展示:")
        print("  1. 环境配置检查")
        print("  2. 真实模型调用和日志记录")
        print("  3. 性能监控和指标收集")
        print("  4. 日志查看和分析功能")
        print("  5. 监控报告和告警演示")
        
        # 1. 检查环境
        self.print_section("1. 环境配置检查")
        if not self.check_environment():
            return
        
        # 如果是仅监控模式，跳过模型测试
        if not monitor_only:
            # 2. 运行模型测试
            self.print_section("2. 真实模型调用测试")
            results = await self.run_model_tests(target_model, test_only)
            
            # 3. 显示测试结果
            self.print_section("3. 测试结果分析")
            self.show_test_results(results)
        
        # 4. 演示日志查看
        self.print_section("4. 日志查看功能")
        self.demonstrate_log_viewing()
        
        # 5. 显示监控报告
        self.print_section("5. 监控报告")
        show_monitoring_dashboard(self.monitor)
        
        # 6. 显示使用命令
        self.print_section("6. 使用命令参考")
        self.show_usage_commands()
        
        # 总结
        self.print_section("演示完成")
        print("✅ 环境配置检查完成")
        if not monitor_only:
            print("✅ 真实模型调用测试完成")
        print("✅ 日志记录和监控功能正常")
        print("✅ 日志查看和分析功能正常")
        print("\n🎉 完整日志监控演示完成！")
        print("💡 提示: 使用上述命令查看和分析日志数据")


class LayoutModeDemo:
    """布局模式演示类"""
    
    def __init__(self):
        """初始化演示类"""
        self.project_root = Path(__file__).parent.parent.parent
        self.view_logs_script = self.project_root / "view_logs.py"
    
    def run_view_logs_command(self, args: List[str]) -> Tuple[bool, str, str]:
        """运行 view_logs.py 命令"""
        if not self.view_logs_script.exists():
            return False, "", "view_logs.py 脚本不存在"
        
        cmd = ["python", str(self.view_logs_script)] + args
        
        try:
            import subprocess
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='ignore',
                timeout=30,
                cwd=str(self.project_root)
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "命令执行超时"
        except Exception as e:
            return False, "", f"执行命令时出错: {e}"
    
    def demo_layout_modes(self):
        """演示布局模式"""
        print("\n🎨 布局模式演示:")
        
        # 1. Classic 布局
        print("   1. Classic 布局（传统表格显示）:")
        success, stdout, stderr = self.run_view_logs_command([
            "--layout", "classic", "--limit", "5"
        ])
        
        if success:
            print("     ✅ Classic 布局示例:")
            lines = stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"     {line}")
        else:
            print(f"     ❌ Classic 布局演示失败: {stderr}")
        
        print("\n   2. Enhanced 布局（智能配对显示）:")
        success, stdout, stderr = self.run_view_logs_command([
            "--layout", "enhanced", "--limit", "4"
        ])
        
        if success:
            print("     ✅ Enhanced 布局示例:")
            lines = stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"     {line}")
        else:
            print(f"     ❌ Enhanced 布局演示失败: {stderr}")
    
    def demo_trace_id_features(self):
        """演示 trace_id 功能"""
        print("\n🆔 Trace ID 功能演示:")
        
        # 1. 列出最近的 trace_id
        print("   1. 列出最近的 trace_id:")
        success, stdout, stderr = self.run_view_logs_command([
            "--list-recent-trace-ids", "--limit", "10"
        ])
        
        if success:
            print("     ✅ 最近的 trace_id:")
            for line in stdout.split('\n')[:8]:
                if line.strip():
                    print(f"     {line}")
        else:
            print(f"     ❌ 获取 trace_id 列表失败: {stderr}")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HarborAI 中级日志功能演示 - 监控与分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python logging_monitoring.py                    # 运行完整演示
  python logging_monitoring.py --real-calls       # 进行真实模型调用
  python logging_monitoring.py --monitoring-only  # 仅演示监控功能
  python logging_monitoring.py --layout-demo      # 演示布局模式
  python logging_monitoring.py --trace-id-demo    # 演示 trace_id 功能
        """
    )
    
    parser.add_argument("--real-calls", action="store_true", help="进行真实的模型调用（需要 API 密钥）")
    parser.add_argument("--monitoring-only", action="store_true", help="仅演示监控功能（不创建新数据）")
    parser.add_argument("--layout-demo", action="store_true", help="仅演示布局模式")
    parser.add_argument("--trace-id-demo", action="store_true", help="仅演示 trace_id 功能")
    parser.add_argument("--model", help="仅测试指定模型")
    
    args = parser.parse_args()
    
    try:
        if args.layout_demo:
            # 仅演示布局模式
            layout_demo = LayoutModeDemo()
            layout_demo.demo_layout_modes()
            
        elif args.trace_id_demo:
            # 仅演示 trace_id 功能
            layout_demo = LayoutModeDemo()
            layout_demo.demo_trace_id_features()
            
        else:
            # 运行原有的完整演示
            demo = RealModelDemo()
            await demo.run_complete_demo(
                target_model=args.model,
                test_only=not args.real_calls,
                monitor_only=args.monitoring_only
            )
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())