"""文件系统日志存储模块。"""

import json
import os
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue
from threading import Thread, Lock
import gzip
import shutil

from ..utils.logger import get_logger
from ..utils.exceptions import StorageError

logger = get_logger(__name__)

# 敏感信息模式定义
# 敏感信息检测模式
SENSITIVE_PATTERNS = {
    'phone': [
        r'1[3-9]\d{9}',  # 中国手机号
    ],
    'id_card': [
        r'\d{6}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]',  # 中国身份证号（更精确的格式）
    ],
    'email': [
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 邮箱地址
    ],
    'credit_card': [
        r'(?<!\d)\d{16}(?!\d)',  # 16位银行卡号（避免匹配身份证号）
        r'(?<!\d)\d{19}(?!\d)',  # 19位银行卡号
    ],
    'ip_address': [
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP地址
    ],
    'api_key': [
        r'sk-[a-zA-Z0-9]{48,}',  # OpenAI API key
        r'ak-[a-zA-Z0-9]{32,}',  # 其他API key
    ],
    'generic_key': [
        r'[a-zA-Z0-9]{32,}',  # 通用长密钥
    ]
}


class FileSystemLogger:
    """文件系统异步日志记录器。
    
    提供与PostgreSQL日志记录器相同的接口，但将日志存储到本地文件系统。
    支持日志轮转、压缩和异步批量写入。
    """
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 file_prefix: str = "harborai",
                 batch_size: int = 100,
                 flush_interval: int = 30,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 max_files: int = 10,
                 compress_old_files: bool = True):
        """初始化文件系统日志记录器。
        
        Args:
            log_dir: 日志目录路径
            file_prefix: 日志文件前缀
            batch_size: 批量写入大小
            flush_interval: 刷新间隔（秒）
            max_file_size: 单个日志文件最大大小（字节）
            max_files: 保留的最大文件数量
            compress_old_files: 是否压缩旧文件
        """
        self.log_dir = Path(log_dir)
        self.file_prefix = file_prefix
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.compress_old_files = compress_old_files
        
        self._log_queue = Queue()
        self._worker_thread = None
        self._running = False
        self._current_file = None
        self._current_file_size = 0
        self._file_lock = Lock()
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def start(self):
        """启动日志记录器。"""
        if self._running:
            return
            
        self._running = True
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info(f"FileSystem logger started, log directory: {self.log_dir}")
    
    def stop(self):
        """停止日志记录器。"""
        if not self._running:
            return
            
        self._running = False
        
        # 等待工作线程结束
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        
        # 关闭当前文件
        with self._file_lock:
            if self._current_file:
                try:
                    self._current_file.close()
                    self._current_file = None
                except Exception:
                    pass  # 忽略关闭文件时的错误
        
        # 安全地记录日志
        try:
            logger.info("FileSystem logger stopped")
        except (ValueError, OSError, AttributeError, ImportError):
            try:
                import sys
                print("FileSystem logger stopped", file=sys.stderr)
            except Exception:
                pass
    
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
            content = str(msg.get("content", ""))
            
            # 对消息内容进行敏感信息脱敏
            sanitized_content, detections = self._sanitize_text(content)
            
            sanitized_msg = {
                "role": msg.get("role"),
                "content": sanitized_content,
                "content_length": len(content),
                "has_content": bool(content),
                "sensitive_data_detected": len(detections) > 0,
                "sensitive_data_types": [d['type'] for d in detections] if detections else []
            }
            
            sanitized.append(sanitized_msg)
        
        return sanitized

    def _sanitize_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """脱敏文本中的敏感信息。
        
        Args:
            text: 原始文本
            
        Returns:
            Tuple[str, List[Dict]]: (脱敏后的文本, 检测到的敏感信息列表)
        """
        if not text:
            return text, []
        
        detections = []
        sanitized_text = text
        
        # 检测所有敏感信息
        for pattern_type, patterns in SENSITIVE_PATTERNS.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    detections.append({
                        'match': match.group(),
                        'pattern': pattern,
                        'start': match.start(),
                        'end': match.end(),
                        'type': pattern_type
                    })
        
        # 按位置倒序处理，避免位置偏移
        for detection in sorted(detections, key=lambda x: x['start'], reverse=True):
            start, end = detection['start'], detection['end']
            original = detection['match']
            masked = self._mask_sensitive_data(original, detection['type'])
            sanitized_text = sanitized_text[:start] + masked + sanitized_text[end:]
        
        return sanitized_text, detections

    def _mask_sensitive_data(self, data: str, data_type: str) -> str:
        """脱敏敏感数据。
        
        Args:
            data: 原始敏感数据
            data_type: 数据类型
            
        Returns:
            str: 脱敏后的数据
        """
        mask_char = '*'
        
        if data_type == 'email':
            # 邮箱脱敏：保留首尾字符和@域名
            parts = data.split('@')
            if len(parts) == 2:
                username, domain = parts
                if len(username) > 2:
                    masked_username = username[0] + mask_char * (len(username) - 2) + username[-1]
                else:
                    masked_username = mask_char * len(username)
                return f"{masked_username}@{domain}"
        
        elif data_type == 'credit_card':
            # 银行卡脱敏：只显示后4位
            if len(data) >= 4:
                return mask_char * (len(data) - 4) + data[-4:]
        
        elif data_type in ['api_key', 'generic_key']:
            # API密钥脱敏：保留前缀和后4位
            if len(data) > 8:
                if data.startswith(('sk-', 'ak-')):
                    prefix = data[:3]
                    suffix = data[-4:]
                    middle_length = len(data) - 7
                    return prefix + mask_char * middle_length + suffix
                else:
                    return data[:2] + mask_char * (len(data) - 6) + data[-4:]
        
        elif data_type == 'phone':
            # 手机号脱敏：保留前3位和后4位
            if len(data) == 11:
                return data[:3] + mask_char * 4 + data[-4:]
        
        elif data_type == 'id_card':
            # 身份证脱敏：保留前6位和后4位
            if len(data) == 18:
                return data[:6] + mask_char * 8 + data[-4:]
        
        elif data_type == 'ip_address':
            # IP地址脱敏：保留第一段
            parts = data.split('.')
            if len(parts) == 4:
                return parts[0] + '.' + mask_char * 3 + '.' + mask_char * 3 + '.' + mask_char * 3
        
        # 默认脱敏：保留首尾，中间用*替换
        if len(data) > 4:
            return data[0] + mask_char * (len(data) - 2) + data[-1]
        else:
            return mask_char * len(data)

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
                content = str(message.content or "")
                
                # 对响应内容进行脱敏
                sanitized_content, detections = self._sanitize_text(content)
                
                summary["content"] = sanitized_content
                summary["content_length"] = len(content)
                summary["has_reasoning"] = hasattr(message, 'reasoning_content') and bool(message.reasoning_content)
                summary["has_tool_calls"] = hasattr(message, 'tool_calls') and bool(message.tool_calls)
                summary["sensitive_data_detected"] = len(detections) > 0
                summary["sensitive_data_types"] = [d['type'] for d in detections] if detections else []
        
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
                logger.error(f"Error in FileSystem logger worker: {e}")
                time.sleep(1)
        
        # 处理剩余的日志
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, batch: List[Dict[str, Any]]):
        """批量写入日志到文件。"""
        if not batch:
            return
        
        try:
            with self._file_lock:
                # 确保有可用的日志文件
                self._ensure_log_file()
                
                # 写入日志条目
                for entry in batch:
                    log_line = json.dumps(entry, ensure_ascii=False) + "\n"
                    log_bytes = log_line.encode('utf-8')
                    
                    # 检查是否需要轮转文件
                    if self._current_file_size + len(log_bytes) > self.max_file_size:
                        self._rotate_log_file()
                        self._ensure_log_file()
                    
                    # 写入日志
                    self._current_file.write(log_line)
                    self._current_file_size += len(log_bytes)
                
                # 刷新到磁盘
                self._current_file.flush()
                os.fsync(self._current_file.fileno())
            
            logger.debug(f"Flushed {len(batch)} log entries to file system")
            
        except Exception as e:
            logger.error(f"Failed to flush batch to file system: {e}")
    
    def _ensure_log_file(self):
        """确保有可用的日志文件。"""
        if self._current_file is None:
            # 生成新的日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.file_prefix}_{timestamp}.jsonl"
            filepath = self.log_dir / filename
            
            # 打开新文件
            self._current_file = open(filepath, 'a', encoding='utf-8')
            
            # 获取文件大小
            try:
                self._current_file_size = filepath.stat().st_size
            except (OSError, FileNotFoundError):
                self._current_file_size = 0
            
            logger.info(f"Opened new log file: {filepath}")
    
    def _rotate_log_file(self):
        """轮转日志文件。"""
        if self._current_file:
            # 关闭当前文件
            current_path = Path(self._current_file.name)
            self._current_file.close()
            self._current_file = None
            self._current_file_size = 0
            
            # 压缩旧文件（如果启用）
            if self.compress_old_files:
                self._compress_file(current_path)
            
            # 清理旧文件
            self._cleanup_old_files()
            
            logger.info(f"Rotated log file: {current_path}")
    
    def _compress_file(self, filepath: Path):
        """压缩日志文件。"""
        try:
            compressed_path = filepath.with_suffix(filepath.suffix + '.gz')
            
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 删除原文件
            filepath.unlink()
            
            logger.debug(f"Compressed log file: {filepath} -> {compressed_path}")
            
        except Exception as e:
            logger.error(f"Failed to compress log file {filepath}: {e}")
    
    def _cleanup_old_files(self):
        """清理旧的日志文件。"""
        try:
            # 获取所有日志文件
            pattern = f"{self.file_prefix}_*.jsonl*"
            log_files = list(self.log_dir.glob(pattern))
            
            # 按修改时间排序（最新的在前）
            log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # 删除超出限制的文件
            for old_file in log_files[self.max_files:]:
                try:
                    old_file.unlink()
                    logger.debug(f"Deleted old log file: {old_file}")
                except Exception as e:
                    logger.error(f"Failed to delete old log file {old_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old log files: {e}")
    
    def get_log_files(self) -> List[Path]:
        """获取所有日志文件列表。"""
        pattern = f"{self.file_prefix}_*.jsonl*"
        return sorted(self.log_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    
    def read_logs(self, 
                  trace_id: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """读取日志条目。
        
        Args:
            trace_id: 过滤特定的trace_id
            start_time: 开始时间
            end_time: 结束时间
            limit: 最大返回条目数
            
        Returns:
            日志条目列表
        """
        logs = []
        
        try:
            for log_file in self.get_log_files():
                if limit and len(logs) >= limit:
                    break
                
                # 处理压缩文件
                if log_file.suffix == '.gz':
                    file_opener = gzip.open
                    mode = 'rt'
                else:
                    file_opener = open
                    mode = 'r'
                
                try:
                    with file_opener(log_file, mode, encoding='utf-8') as f:
                        for line in f:
                            if limit and len(logs) >= limit:
                                break
                            
                            try:
                                entry = json.loads(line.strip())
                                
                                # 应用过滤条件
                                if trace_id and entry.get('trace_id') != trace_id:
                                    continue
                                
                                if start_time or end_time:
                                    entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                                    if start_time and entry_time < start_time:
                                        continue
                                    if end_time and entry_time > end_time:
                                        continue
                                
                                logs.append(entry)
                                
                            except json.JSONDecodeError:
                                continue  # 跳过无效的JSON行
                                
                except Exception as e:
                    logger.error(f"Failed to read log file {log_file}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to read logs: {e}")
        
        return logs


# 全局文件系统日志记录器实例
_global_file_logger: Optional[FileSystemLogger] = None


def get_file_logger() -> Optional[FileSystemLogger]:
    """获取全局文件系统日志记录器。"""
    return _global_file_logger


def initialize_file_logger(log_dir: str = "./logs", **kwargs) -> FileSystemLogger:
    """初始化全局文件系统日志记录器。"""
    global _global_file_logger
    
    if _global_file_logger:
        _global_file_logger.stop()
    
    _global_file_logger = FileSystemLogger(log_dir=log_dir, **kwargs)
    _global_file_logger.start()
    
    return _global_file_logger


def shutdown_file_logger():
    """关闭全局文件系统日志记录器。"""
    global _global_file_logger
    
    if _global_file_logger:
        _global_file_logger.stop()
        _global_file_logger = None