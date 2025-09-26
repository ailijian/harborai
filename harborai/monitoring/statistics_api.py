#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token统计API接口模块"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, Response

from .token_statistics import get_token_statistics_collector, TokenStatisticsCollector
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TokenStatisticsAPI:
    """Token统计API类"""
    
    def __init__(self, app: Optional[Flask] = None):
        """
        初始化Token统计API
        
        Args:
            app: Flask应用实例
        """
        self.collector = get_token_statistics_collector()
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app: Flask) -> None:
        """
        初始化Flask应用
        
        Args:
            app: Flask应用实例
        """
        # 注册路由
        app.add_url_rule('/api/statistics/summary', 'get_summary_stats', 
                        self.get_summary_stats, methods=['GET'])
        app.add_url_rule('/api/statistics/models', 'get_model_stats', 
                        self.get_model_stats, methods=['GET'])
        app.add_url_rule('/api/statistics/models/<model_name>', 'get_model_detail', 
                        self.get_model_detail, methods=['GET'])
        app.add_url_rule('/api/statistics/time-windows', 'get_time_window_stats', 
                        self.get_time_window_stats, methods=['GET'])
        app.add_url_rule('/api/statistics/records', 'get_recent_records', 
                        self.get_recent_records, methods=['GET'])
        app.add_url_rule('/api/statistics/export', 'export_statistics', 
                        self.export_statistics, methods=['GET'])
        app.add_url_rule('/api/statistics/health', 'health_check', 
                        self.health_check, methods=['GET'])
        app.add_url_rule('/api/statistics/cleanup', 'cleanup_old_records', 
                        self.cleanup_old_records, methods=['POST'])
    
    def get_summary_stats(self) -> Response:
        """
        获取汇总统计信息
        
        Returns:
            JSON响应包含汇总统计信息
        """
        try:
            stats = self.collector.get_summary_stats()
            return jsonify({
                "status": "success",
                "data": stats,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting summary stats: {e}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    def get_model_stats(self) -> Response:
        """
        获取所有模型的统计信息
        
        Returns:
            JSON响应包含模型统计信息
        """
        try:
            model_stats = self.collector.get_model_statistics()
            
            # 转换为可序列化的格式
            serializable_stats = {}
            for model, stats in model_stats.items():
                serializable_stats[model] = {
                    "model_name": stats.model_name,
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "total_input_tokens": stats.total_input_tokens,
                    "total_output_tokens": stats.total_output_tokens,
                    "total_tokens": stats.total_tokens,
                    "total_cost": round(stats.total_cost, 6),
                    "average_latency": round(stats.average_latency, 3),
                    "error_rate": round(stats.error_rate, 4),
                    "last_used": stats.last_used.isoformat() if stats.last_used else None
                }
            
            return jsonify({
                "status": "success",
                "data": serializable_stats,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    def get_model_detail(self, model_name: str) -> Response:
        """
        获取指定模型的详细统计信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            JSON响应包含模型详细统计信息
        """
        try:
            model_stats = self.collector.get_model_statistics(model_name)
            
            if model_name not in model_stats:
                return jsonify({
                    "status": "error",
                    "message": f"Model '{model_name}' not found",
                    "timestamp": datetime.now().isoformat()
                }), 404
            
            stats = model_stats[model_name]
            serializable_stats = {
                "model_name": stats.model_name,
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "total_input_tokens": stats.total_input_tokens,
                "total_output_tokens": stats.total_output_tokens,
                "total_tokens": stats.total_tokens,
                "total_cost": round(stats.total_cost, 6),
                "average_latency": round(stats.average_latency, 3),
                "error_rate": round(stats.error_rate, 4),
                "last_used": stats.last_used.isoformat() if stats.last_used else None
            }
            
            return jsonify({
                "status": "success",
                "data": serializable_stats,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting model detail for {model_name}: {e}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    def get_time_window_stats(self) -> Response:
        """
        获取时间窗口统计信息
        
        Query Parameters:
            window_type: 窗口类型（"hour" 或 "day"），默认为 "hour"
            count: 返回的窗口数量，默认为 24
            
        Returns:
            JSON响应包含时间窗口统计信息
        """
        try:
            window_type = request.args.get('window_type', 'hour')
            count = int(request.args.get('count', 24))
            
            if window_type not in ['hour', 'day']:
                return jsonify({
                    "status": "error",
                    "message": "window_type must be 'hour' or 'day'",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            if count <= 0 or count > 168:  # 最多7天的小时数据
                return jsonify({
                    "status": "error",
                    "message": "count must be between 1 and 168",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            time_stats = self.collector.get_time_window_stats(window_type, count)
            
            # 转换为可序列化的格式
            serializable_stats = []
            for stats in time_stats:
                serializable_stats.append({
                    "window_start": stats.window_start.isoformat(),
                    "window_end": stats.window_end.isoformat(),
                    "total_requests": stats.total_requests,
                    "total_tokens": stats.total_tokens,
                    "total_cost": round(stats.total_cost, 6),
                    "models_used": stats.models_used
                })
            
            return jsonify({
                "status": "success",
                "data": {
                    "window_type": window_type,
                    "count": len(serializable_stats),
                    "windows": serializable_stats
                },
                "timestamp": datetime.now().isoformat()
            })
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": f"Invalid parameter: {e}",
                "timestamp": datetime.now().isoformat()
            }), 400
        except Exception as e:
            logger.error(f"Error getting time window stats: {e}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    def get_recent_records(self) -> Response:
        """
        获取最近的使用记录
        
        Query Parameters:
            count: 返回的记录数量，默认为 100，最大为 1000
            
        Returns:
            JSON响应包含最近的使用记录
        """
        try:
            count = int(request.args.get('count', 100))
            
            if count <= 0 or count > 1000:
                return jsonify({
                    "status": "error",
                    "message": "count must be between 1 and 1000",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            records = self.collector.get_recent_records(count)
            
            # 转换为可序列化的格式
            serializable_records = []
            for record in records:
                serializable_records.append({
                    "timestamp": record.timestamp.isoformat(),
                    "trace_id": record.trace_id,
                    "model": record.model,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "total_tokens": record.total_tokens,
                    "cost": round(record.cost, 6) if record.cost else None,
                    "duration": round(record.duration, 3),
                    "success": record.success,
                    "error": record.error
                })
            
            return jsonify({
                "status": "success",
                "data": {
                    "count": len(serializable_records),
                    "records": serializable_records
                },
                "timestamp": datetime.now().isoformat()
            })
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": f"Invalid parameter: {e}",
                "timestamp": datetime.now().isoformat()
            }), 400
        except Exception as e:
            logger.error(f"Error getting recent records: {e}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    def export_statistics(self) -> Response:
        """
        导出统计信息
        
        Query Parameters:
            format: 导出格式（"json" 或 "csv"），默认为 "json"
            
        Returns:
            导出的统计信息文件
        """
        try:
            format_type = request.args.get('format', 'json')
            
            if format_type not in ['json', 'csv']:
                return jsonify({
                    "status": "error",
                    "message": "format must be 'json' or 'csv'",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            exported_data = self.collector.export_statistics(format_type)
            
            if format_type == 'json':
                return Response(
                    exported_data,
                    mimetype='application/json',
                    headers={
                        'Content-Disposition': f'attachment; filename=token_statistics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    }
                )
            else:  # csv
                return Response(
                    exported_data,
                    mimetype='text/csv',
                    headers={
                        'Content-Disposition': f'attachment; filename=token_statistics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    }
                )
        except Exception as e:
            logger.error(f"Error exporting statistics: {e}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    def health_check(self) -> Response:
        """
        健康检查接口
        
        Returns:
            JSON响应包含健康状态信息
        """
        try:
            summary = self.collector.get_summary_stats()
            
            return jsonify({
                "status": "healthy",
                "service": "token_statistics",
                "uptime_hours": summary.get('uptime_hours', 0),
                "total_requests": summary.get('total_requests', 0),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                "status": "unhealthy",
                "service": "token_statistics",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    def cleanup_old_records(self) -> Response:
        """
        清理旧记录
        
        Request Body (JSON):
            days: 保留天数，默认为 7
            
        Returns:
            JSON响应包含清理结果
        """
        try:
            data = request.get_json() or {}
            days = data.get('days', 7)
            
            if not isinstance(days, int) or days <= 0 or days > 365:
                return jsonify({
                    "status": "error",
                    "message": "days must be an integer between 1 and 365",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            cleaned_count = self.collector.clear_old_records(days)
            
            return jsonify({
                "status": "success",
                "data": {
                    "cleaned_records": cleaned_count,
                    "retention_days": days
                },
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error cleaning old records: {e}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500


# 创建全局API实例
token_statistics_api = TokenStatisticsAPI()


def create_statistics_app() -> Flask:
    """
    创建Token统计Flask应用
    
    Returns:
        配置好的Flask应用实例
    """
    app = Flask(__name__)
    
    # 配置CORS
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    # 初始化API
    token_statistics_api.init_app(app)
    
    # 根路径重定向到健康检查
    @app.route('/')
    def index():
        return jsonify({
            "service": "HarborAI Token Statistics API",
            "version": "1.0.0",
            "endpoints": [
                "/api/statistics/summary",
                "/api/statistics/models",
                "/api/statistics/models/<model_name>",
                "/api/statistics/time-windows",
                "/api/statistics/records",
                "/api/statistics/export",
                "/api/statistics/health",
                "/api/statistics/cleanup"
            ],
            "timestamp": datetime.now().isoformat()
        })
    
    return app


if __name__ == '__main__':
    # 开发模式运行
    app = create_statistics_app()
    app.run(host='0.0.0.0', port=8080, debug=True)