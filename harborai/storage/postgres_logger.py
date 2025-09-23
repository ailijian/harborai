"""PostgreSQL日志存储模块。"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from queue import Queue
from threading import Thread

from ..utils.logger import get_logger
from ..utils.exceptions import StorageError

logger = get_logger(__name__)


class PostgreSQLLogger:
    """PostgreSQL异步日志记录器。"""
    
    def __init__(self, 
                 connection_string: str,
                 table_name: str = "harborai_logs",
                 batch_size: int = 100,
                 flush_interval: int = 30):
        """初始化PostgreSQL日志记录器。
        
        Args:
            connection_string: PostgreSQL连接字符串
            table_name: 日志表名
            batch_size: 批量写入大小
            flush_interval: 刷新间隔（秒）
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._log_queue = Queue()
        self._worker_thread = None
        self._running = False
        self._connection = None
        
    def start(self):
        """启动日志记录器。"""
        if self._running:
            return
            
        self._running = True
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info("PostgreSQL logger started")
    
    def stop(self):
        """停止日志记录器。"""
        if not self._running:
            return
            
        self._running = False
        
        # 等待工作线程结束
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        
        # 关闭数据库连接
        if self._connection:
            self._connection.close()
            
        logger.info("PostgreSQL logger stopped")
    
    def log_request(self, 
                   trace_id: str,
                   model: str,
                   messages: List[Dict[str, Any]],
                   **kwargs):
        """记录请求日志。
        
        Args:
            trace_id: 追踪ID
            model: 模型名称
            messages: 消息列表
            **kwargs: 其他参数
        """
        log_entry = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request",
            "model": model,
            "messages": self._sanitize_messages(messages),
            "parameters": self._sanitize_parameters(kwargs),
            "reasoning_content_present": False,  # 请求阶段未知
            "structured_provider": kwargs.get("structured_provider"),
            "success": None,  # 请求阶段未知
            "latency": None,  # 请求阶段未知
            "tokens": None,  # 请求阶段未知
            "cost": None  # 请求阶段未知
        }
        
        self._log_queue.put(log_entry)
    
    def log_response(self,
                    trace_id: str,
                    response: Any,
                    latency: float,
                    success: bool = True,
                    error: Optional[str] = None):
        """记录响应日志。
        
        Args:
            trace_id: 追踪ID
            response: 响应对象
            latency: 延迟时间
            success: 是否成功
            error: 错误信息
        """
        # 提取响应信息
        tokens = None
        cost = None
        reasoning_content_present = False
        
        if hasattr(response, 'usage') and response.usage:
            tokens = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        # 检查是否包含思考内容
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'reasoning_content'):
                reasoning_content_present = bool(choice.message.reasoning_content)
        
        log_entry = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "response",
            "success": success,
            "latency": latency,
            "tokens": tokens,
            "cost": cost,
            "reasoning_content_present": reasoning_content_present,
            "error": error,
            "response_summary": self._create_response_summary(response)
        }
        
        self._log_queue.put(log_entry)
    
    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """脱敏处理消息内容。"""
        sanitized = []
        for msg in messages:
            sanitized_msg = {
                "role": msg.get("role"),
                "content_length": len(str(msg.get("content", ""))),
                "has_content": bool(msg.get("content"))
            }
            # 可选：保留内容的前100个字符用于调试
            content = str(msg.get("content", ""))
            if len(content) <= 100:
                sanitized_msg["content_preview"] = content
            else:
                sanitized_msg["content_preview"] = content[:100] + "..."
            
            sanitized.append(sanitized_msg)
        
        return sanitized
    
    def _sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """脱敏处理参数。"""
        sanitized = {}
        sensitive_keys = {"api_key", "authorization", "token", "secret"}
        
        for key, value in params.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _create_response_summary(self, response: Any) -> Dict[str, Any]:
        """创建响应摘要。"""
        summary = {}
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                message = choice.message
                summary["content_length"] = len(str(message.content or ""))
                summary["has_reasoning"] = hasattr(message, 'reasoning_content') and bool(message.reasoning_content)
                summary["has_tool_calls"] = hasattr(message, 'tool_calls') and bool(message.tool_calls)
        
        if hasattr(response, 'model'):
            summary["model"] = response.model
        
        if hasattr(response, 'id'):
            summary["response_id"] = response.id
        
        return summary
    
    def _worker_loop(self):
        """工作线程主循环。"""
        batch = []
        last_flush = time.time()
        
        while self._running:
            try:
                # 尝试获取日志条目
                try:
                    log_entry = self._log_queue.get(timeout=1.0)
                    batch.append(log_entry)
                except:
                    # 超时或队列为空
                    pass
                
                # 检查是否需要刷新
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and current_time - last_flush >= self.flush_interval)
                )
                
                if should_flush:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = current_time
                    
            except Exception as e:
                logger.error(f"Error in PostgreSQL logger worker: {e}")
                time.sleep(1)
        
        # 处理剩余的日志
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, batch: List[Dict[str, Any]]):
        """批量写入日志到数据库。"""
        if not batch:
            return
        
        try:
            self._ensure_connection()
            self._ensure_table_exists()
            
            # 构建插入SQL
            placeholders = ", ".join(["%s"] * len(batch))
            sql = f"""
                INSERT INTO {self.table_name} 
                (trace_id, timestamp, type, model, messages, parameters, 
                 reasoning_content_present, structured_provider, success, 
                 latency, tokens, cost, error, response_summary, raw_data)
                VALUES {placeholders}
            """
            
            # 准备数据
            values = []
            for entry in batch:
                values.append((
                    entry.get("trace_id"),
                    entry.get("timestamp"),
                    entry.get("type"),
                    entry.get("model"),
                    json.dumps(entry.get("messages")),
                    json.dumps(entry.get("parameters")),
                    entry.get("reasoning_content_present"),
                    entry.get("structured_provider"),
                    entry.get("success"),
                    entry.get("latency"),
                    json.dumps(entry.get("tokens")),
                    entry.get("cost"),
                    entry.get("error"),
                    json.dumps(entry.get("response_summary")),
                    json.dumps(entry)
                ))
            
            # 执行插入
            with self._connection.cursor() as cursor:
                cursor.executemany(sql, values)
                self._connection.commit()
            
            logger.debug(f"Flushed {len(batch)} log entries to PostgreSQL")
            
        except Exception as e:
            logger.error(f"Failed to flush batch to PostgreSQL: {e}")
            # 可以考虑将失败的批次写入文件作为备份
    
    def _ensure_connection(self):
        """确保数据库连接可用。"""
        if self._connection is None or self._connection.closed:
            try:
                import psycopg2
                self._connection = psycopg2.connect(self.connection_string)
                logger.info("Connected to PostgreSQL")
            except ImportError:
                raise StorageError("psycopg2 not installed. Please install it to use PostgreSQL logging.")
            except Exception as e:
                raise StorageError(f"Failed to connect to PostgreSQL: {e}")
    
    def _ensure_table_exists(self):
        """确保日志表存在。"""
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                trace_id VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                type VARCHAR(50) NOT NULL,
                model VARCHAR(255),
                messages TEXT,
                parameters TEXT,
                reasoning_content_present BOOLEAN,
                structured_provider VARCHAR(50),
                success BOOLEAN,
                latency FLOAT,
                tokens TEXT,
                cost FLOAT,
                error TEXT,
                response_summary TEXT,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_trace_id ON {self.table_name} (trace_id);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON {self.table_name} (timestamp);
            CREATE INDEX IF NOT EXISTS idx_model ON {self.table_name} (model);
            CREATE INDEX IF NOT EXISTS idx_success ON {self.table_name} (success);
        """
        
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(create_table_sql)
                self._connection.commit()
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise StorageError(f"Failed to create table: {e}")


# 全局日志记录器实例
_global_logger: Optional[PostgreSQLLogger] = None


def get_postgres_logger() -> Optional[PostgreSQLLogger]:
    """获取全局PostgreSQL日志记录器。"""
    return _global_logger


def initialize_postgres_logger(connection_string: str, **kwargs) -> PostgreSQLLogger:
    """初始化全局PostgreSQL日志记录器。"""
    global _global_logger
    
    if _global_logger:
        _global_logger.stop()
    
    _global_logger = PostgreSQLLogger(connection_string, **kwargs)
    _global_logger.start()
    
    return _global_logger


def shutdown_postgres_logger():
    """关闭全局PostgreSQL日志记录器。"""
    global _global_logger
    
    if _global_logger:
        _global_logger.stop()
        _global_logger = None