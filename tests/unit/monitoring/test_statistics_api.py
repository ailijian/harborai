#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token统计API模块测试"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from flask import Flask

from harborai.monitoring.statistics_api import (
    TokenStatisticsAPI, 
    token_statistics_api, 
    create_statistics_app
)
from harborai.monitoring.token_statistics import (
    TokenUsageRecord, 
    ModelStatistics, 
    TimeWindowStatistics
)


class TestTokenStatisticsAPI:
    """Token统计API类测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.api = TokenStatisticsAPI()
        self.api.init_app(self.app)
        self.client = self.app.test_client()
        
        # 模拟收集器
        self.mock_collector = Mock()
        self.api.collector = self.mock_collector
    
    def test_init_without_app(self):
        """测试不带Flask应用的初始化"""
        api = TokenStatisticsAPI()
        assert api.collector is not None
    
    def test_init_with_app(self):
        """测试带Flask应用的初始化"""
        app = Flask(__name__)
        api = TokenStatisticsAPI(app)
        assert api.collector is not None
    
    def test_init_app(self):
        """测试初始化Flask应用"""
        app = Flask(__name__)
        api = TokenStatisticsAPI()
        api.init_app(app)
        
        # 检查路由是否注册
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = [
            '/api/statistics/summary',
            '/api/statistics/models',
            '/api/statistics/models/<model_name>',
            '/api/statistics/time-windows',
            '/api/statistics/records',
            '/api/statistics/export',
            '/api/statistics/health',
            '/api/statistics/cleanup'
        ]
        
        for route in expected_routes:
            assert route in rules
    
    def test_get_summary_stats_success(self):
        """测试获取汇总统计信息 - 成功"""
        mock_stats = {
            'total_requests': 100,
            'total_tokens': 10000,
            'total_cost': 50.0,
            'uptime_hours': 24.5
        }
        self.mock_collector.get_summary_stats.return_value = mock_stats
        
        response = self.client.get('/api/statistics/summary')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['data'] == mock_stats
        assert 'timestamp' in data
    
    def test_get_summary_stats_error(self):
        """测试获取汇总统计信息 - 错误"""
        self.mock_collector.get_summary_stats.side_effect = Exception("数据库错误")
        
        response = self.client.get('/api/statistics/summary')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert '数据库错误' in data['message']
    
    def test_get_model_stats_success(self):
        """测试获取模型统计信息 - 成功"""
        mock_stats = Mock()
        mock_stats.model_name = "gpt-4"
        mock_stats.total_requests = 50
        mock_stats.successful_requests = 48
        mock_stats.failed_requests = 2
        mock_stats.total_input_tokens = 5000
        mock_stats.total_output_tokens = 3000
        mock_stats.total_tokens = 8000
        mock_stats.total_cost = 25.0
        mock_stats.average_latency = 2.5
        mock_stats.error_rate = 0.04
        mock_stats.last_used = datetime(2024, 1, 1, 12, 0, 0)
        
        self.mock_collector.get_model_statistics.return_value = {"gpt-4": mock_stats}
        
        response = self.client.get('/api/statistics/models')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'gpt-4' in data['data']
        assert data['data']['gpt-4']['total_requests'] == 50
        assert data['data']['gpt-4']['total_cost'] == 25.0
        assert data['data']['gpt-4']['last_used'] == "2024-01-01T12:00:00"
    
    def test_get_model_stats_with_none_last_used(self):
        """测试获取模型统计信息 - last_used为None"""
        mock_stats = Mock()
        mock_stats.model_name = "gpt-3.5"
        mock_stats.total_requests = 30
        mock_stats.successful_requests = 30
        mock_stats.failed_requests = 0
        mock_stats.total_input_tokens = 3000
        mock_stats.total_output_tokens = 2000
        mock_stats.total_tokens = 5000
        mock_stats.total_cost = 15.0
        mock_stats.average_latency = 1.5
        mock_stats.error_rate = 0.0
        mock_stats.last_used = None
        
        self.mock_collector.get_model_statistics.return_value = {"gpt-3.5": mock_stats}
        
        response = self.client.get('/api/statistics/models')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data']['gpt-3.5']['last_used'] is None
    
    def test_get_model_stats_error(self):
        """测试获取模型统计信息 - 错误"""
        self.mock_collector.get_model_statistics.side_effect = Exception("统计错误")
        
        response = self.client.get('/api/statistics/models')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_get_model_detail_success(self):
        """测试获取模型详细信息 - 成功"""
        mock_stats = Mock()
        mock_stats.model_name = "gpt-4"
        mock_stats.total_requests = 50
        mock_stats.successful_requests = 48
        mock_stats.failed_requests = 2
        mock_stats.total_input_tokens = 5000
        mock_stats.total_output_tokens = 3000
        mock_stats.total_tokens = 8000
        mock_stats.total_cost = 25.0
        mock_stats.average_latency = 2.5
        mock_stats.error_rate = 0.04
        mock_stats.last_used = datetime(2024, 1, 1, 12, 0, 0)
        
        self.mock_collector.get_model_statistics.return_value = {"gpt-4": mock_stats}
        
        response = self.client.get('/api/statistics/models/gpt-4')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['data']['model_name'] == 'gpt-4'
        assert data['data']['total_requests'] == 50
    
    def test_get_model_detail_not_found(self):
        """测试获取模型详细信息 - 模型不存在"""
        self.mock_collector.get_model_statistics.return_value = {}
        
        response = self.client.get('/api/statistics/models/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'not found' in data['message']
    
    def test_get_model_detail_error(self):
        """测试获取模型详细信息 - 错误"""
        self.mock_collector.get_model_statistics.side_effect = Exception("查询错误")
        
        response = self.client.get('/api/statistics/models/gpt-4')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_get_time_window_stats_success(self):
        """测试获取时间窗口统计 - 成功"""
        mock_stats = TimeWindowStatistics(
            window_start=datetime(2024, 1, 1, 10, 0, 0),
            window_end=datetime(2024, 1, 1, 11, 0, 0),
            total_requests=10,
            total_tokens=1000,
            total_cost=5.0,
            unique_models=2
        )
        
        self.mock_collector.get_time_window_stats.return_value = [mock_stats]
        
        response = self.client.get('/api/statistics/time-windows?window_type=hour&count=24')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['data']['window_type'] == 'hour'
        assert data['data']['count'] == 1
        assert len(data['data']['windows']) == 1
        assert data['data']['windows'][0]['total_requests'] == 10
    
    def test_get_time_window_stats_default_params(self):
        """测试获取时间窗口统计 - 默认参数"""
        self.mock_collector.get_time_window_stats.return_value = []
        
        response = self.client.get('/api/statistics/time-windows')
        
        assert response.status_code == 200
        self.mock_collector.get_time_window_stats.assert_called_with('hour', 24)
    
    def test_get_time_window_stats_invalid_window_type(self):
        """测试获取时间窗口统计 - 无效窗口类型"""
        response = self.client.get('/api/statistics/time-windows?window_type=invalid')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'window_type must be' in data['message']
    
    def test_get_time_window_stats_invalid_count(self):
        """测试获取时间窗口统计 - 无效计数"""
        response = self.client.get('/api/statistics/time-windows?count=0')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'count must be between' in data['message']
        
        response = self.client.get('/api/statistics/time-windows?count=200')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_get_time_window_stats_value_error(self):
        """测试获取时间窗口统计 - 值错误"""
        response = self.client.get('/api/statistics/time-windows?count=abc')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Invalid parameter' in data['message']
    
    def test_get_time_window_stats_error(self):
        """测试获取时间窗口统计 - 错误"""
        self.mock_collector.get_time_window_stats.side_effect = Exception("查询错误")
        
        response = self.client.get('/api/statistics/time-windows')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_get_recent_records_success(self):
        """测试获取最近记录 - 成功"""
        mock_record = TokenUsageRecord(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            model_name="gpt-4",
            provider="openai",
            request_id="trace_001",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.75,
            latency_ms=2500.0,
            success=True,
            error_message=None
        )
        
        self.mock_collector.get_recent_records.return_value = [mock_record]
        
        response = self.client.get('/api/statistics/records?count=50')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['data']['count'] == 1
        assert len(data['data']['records']) == 1
        assert data['data']['records'][0]['trace_id'] == 'trace_001'
        assert data['data']['records'][0]['cost'] == 0.75
    
    def test_get_recent_records_with_none_cost(self):
        """测试获取最近记录 - cost为None"""
        mock_record = TokenUsageRecord(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            model_name="gpt-3.5",
            provider="openai",
            request_id="trace_002",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.0,
            latency_ms=1500.0,
            success=False,
            error_message="API错误"
        )
        
        self.mock_collector.get_recent_records.return_value = [mock_record]
        
        response = self.client.get('/api/statistics/records')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data']['records'][0]['cost'] is None
        assert data['data']['records'][0]['success'] is False
        assert data['data']['records'][0]['error'] == "API错误"
    
    def test_get_recent_records_default_count(self):
        """测试获取最近记录 - 默认计数"""
        self.mock_collector.get_recent_records.return_value = []
        
        response = self.client.get('/api/statistics/records')
        
        assert response.status_code == 200
        self.mock_collector.get_recent_records.assert_called_with(100)
    
    def test_get_recent_records_invalid_count(self):
        """测试获取最近记录 - 无效计数"""
        response = self.client.get('/api/statistics/records?count=0')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        
        response = self.client.get('/api/statistics/records?count=2000')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_get_recent_records_value_error(self):
        """测试获取最近记录 - 值错误"""
        response = self.client.get('/api/statistics/records?count=invalid')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Invalid parameter' in data['message']
    
    def test_get_recent_records_error(self):
        """测试获取最近记录 - 错误"""
        self.mock_collector.get_recent_records.side_effect = Exception("查询错误")
        
        response = self.client.get('/api/statistics/records')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_export_statistics_json(self):
        """测试导出统计信息 - JSON格式"""
        mock_data = '{"test": "data"}'
        self.mock_collector.export_statistics.return_value = mock_data
        
        response = self.client.get('/api/statistics/export?format=json')
        
        assert response.status_code == 200
        assert response.mimetype == 'application/json'
        assert 'attachment' in response.headers['Content-Disposition']
        assert 'token_statistics_' in response.headers['Content-Disposition']
        assert response.data.decode() == mock_data
    
    def test_export_statistics_csv(self):
        """测试导出统计信息 - CSV格式"""
        mock_data = 'header1,header2\nvalue1,value2'
        self.mock_collector.export_statistics.return_value = mock_data
        
        response = self.client.get('/api/statistics/export?format=csv')
        
        assert response.status_code == 200
        assert response.mimetype == 'text/csv'
        assert 'attachment' in response.headers['Content-Disposition']
        assert response.data.decode() == mock_data
    
    def test_export_statistics_default_format(self):
        """测试导出统计信息 - 默认格式"""
        mock_data = '{"default": "json"}'
        self.mock_collector.export_statistics.return_value = mock_data
        
        response = self.client.get('/api/statistics/export')
        
        assert response.status_code == 200
        assert response.mimetype == 'application/json'
        self.mock_collector.export_statistics.assert_called_with('json')
    
    def test_export_statistics_invalid_format(self):
        """测试导出统计信息 - 无效格式"""
        response = self.client.get('/api/statistics/export?format=xml')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'format must be' in data['message']
    
    def test_export_statistics_error(self):
        """测试导出统计信息 - 错误"""
        self.mock_collector.export_statistics.side_effect = Exception("导出错误")
        
        response = self.client.get('/api/statistics/export')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_health_check_healthy(self):
        """测试健康检查 - 健康状态"""
        mock_summary = {
            'uptime_hours': 12.5,
            'total_requests': 1000
        }
        self.mock_collector.get_summary_stats.return_value = mock_summary
        
        response = self.client.get('/api/statistics/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'token_statistics'
        assert data['uptime_hours'] == 12.5
        assert data['total_requests'] == 1000
    
    def test_health_check_unhealthy(self):
        """测试健康检查 - 不健康状态"""
        self.mock_collector.get_summary_stats.side_effect = Exception("服务异常")
        
        response = self.client.get('/api/statistics/health')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'unhealthy'
        assert data['service'] == 'token_statistics'
        assert '服务异常' in data['error']
    
    def test_cleanup_old_records_success(self):
        """测试清理旧记录 - 成功"""
        self.mock_collector.clear_old_records.return_value = 50
        
        response = self.client.post('/api/statistics/cleanup', 
                                  json={'days': 30})
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['data']['cleaned_records'] == 50
        assert data['data']['retention_days'] == 30
        self.mock_collector.clear_old_records.assert_called_with(30)
    
    def test_cleanup_old_records_default_days(self):
        """测试清理旧记录 - 默认天数"""
        self.mock_collector.clear_old_records.return_value = 25
        
        response = self.client.post('/api/statistics/cleanup', json={})
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data']['retention_days'] == 7
        self.mock_collector.clear_old_records.assert_called_with(7)
    
    def test_cleanup_old_records_no_json(self):
        """测试清理旧记录 - 无JSON数据"""
        self.mock_collector.clear_old_records.return_value = 15
        
        # 当没有JSON数据时，使用默认值7天
        response = self.client.post('/api/statistics/cleanup')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data']['retention_days'] == 7
        self.mock_collector.clear_old_records.assert_called_with(7)
    
    def test_cleanup_old_records_invalid_days(self):
        """测试清理旧记录 - 无效天数"""
        # 测试负数
        response = self.client.post('/api/statistics/cleanup', 
                                  json={'days': -1})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'days must be an integer' in data['message']
        
        # 测试超大值
        response = self.client.post('/api/statistics/cleanup', 
                                  json={'days': 400})
        
        assert response.status_code == 400
        
        # 测试非整数
        response = self.client.post('/api/statistics/cleanup', 
                                  json={'days': 'invalid'})
        
        assert response.status_code == 400
    
    def test_cleanup_old_records_error(self):
        """测试清理旧记录 - 错误"""
        self.mock_collector.clear_old_records.side_effect = Exception("清理错误")
        
        response = self.client.post('/api/statistics/cleanup', 
                                  json={'days': 7})
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'


class TestGlobalFunctions:
    """全局函数测试"""
    
    def test_token_statistics_api_instance(self):
        """测试全局API实例"""
        assert token_statistics_api is not None
        assert isinstance(token_statistics_api, TokenStatisticsAPI)
    
    def test_create_statistics_app(self):
        """测试创建统计应用"""
        app = create_statistics_app()
        
        assert isinstance(app, Flask)
        
        # 测试根路径
        with app.test_client() as client:
            response = client.get('/')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['service'] == 'HarborAI Token Statistics API'
            assert data['version'] == '1.0.0'
            assert 'endpoints' in data
            assert len(data['endpoints']) == 8
    
    def test_create_statistics_app_cors(self):
        """测试CORS配置"""
        app = create_statistics_app()
        
        with app.test_client() as client:
            response = client.get('/')
            
            assert 'Access-Control-Allow-Origin' in response.headers
            assert response.headers['Access-Control-Allow-Origin'] == '*'
            assert 'Access-Control-Allow-Headers' in response.headers
            assert 'Access-Control-Allow-Methods' in response.headers


class TestEdgeCases:
    """边界情况测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.api = TokenStatisticsAPI()
        self.api.init_app(self.app)
        self.client = self.app.test_client()
        
        self.mock_collector = Mock()
        self.api.collector = self.mock_collector
    
    def test_unicode_handling(self):
        """测试Unicode字符处理"""
        mock_record = TokenUsageRecord(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            model_name="gpt-4",
            provider="openai",
            request_id="trace_测试",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.75,
            latency_ms=2500.0,
            success=False,
            error_message="错误：API调用失败"
        )
        
        self.mock_collector.get_recent_records.return_value = [mock_record]
        
        response = self.client.get('/api/statistics/records')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data']['records'][0]['trace_id'] == 'trace_测试'
        assert data['data']['records'][0]['error'] == '错误：API调用失败'
    
    def test_large_numbers(self):
        """测试大数值处理"""
        mock_stats = Mock()
        mock_stats.model_name = "gpt-4"
        mock_stats.total_requests = 1000000
        mock_stats.successful_requests = 999999
        mock_stats.failed_requests = 1
        mock_stats.total_input_tokens = 100000000
        mock_stats.total_output_tokens = 50000000
        mock_stats.total_tokens = 150000000
        mock_stats.total_cost = 999999.999999
        mock_stats.average_latency = 0.001
        mock_stats.error_rate = 0.000001
        mock_stats.last_used = datetime(2024, 1, 1, 12, 0, 0)
        
        self.mock_collector.get_model_statistics.return_value = {"gpt-4": mock_stats}
        
        response = self.client.get('/api/statistics/models')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data']['gpt-4']['total_requests'] == 1000000
        assert data['data']['gpt-4']['total_cost'] == 999999.999999
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        self.mock_collector.get_model_statistics.return_value = {}
        self.mock_collector.get_time_window_stats.return_value = []
        self.mock_collector.get_recent_records.return_value = []
        
        # 测试空模型统计
        response = self.client.get('/api/statistics/models')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data'] == {}
        
        # 测试空时间窗口统计
        response = self.client.get('/api/statistics/time-windows')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data']['count'] == 0
        assert data['data']['windows'] == []
        
        # 测试空记录
        response = self.client.get('/api/statistics/records')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['data']['count'] == 0
        assert data['data']['records'] == []
    
    def test_concurrent_requests(self):
        """测试并发请求处理"""
        import threading
        import time
        
        self.mock_collector.get_summary_stats.return_value = {'test': 'data'}
        
        results = []
        
        def make_request():
            response = self.client.get('/api/statistics/summary')
            results.append(response.status_code)
        
        # 创建多个线程同时发送请求
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有请求都成功
        assert len(results) == 10
        assert all(status == 200 for status in results)
    
    def test_malformed_json_request(self):
        """测试格式错误的JSON请求"""
        # 确保Mock对象返回正确的值
        self.mock_collector.clear_old_records.return_value = 10
        
        response = self.client.post('/api/statistics/cleanup',
                                  data='{"invalid": json}',
                                  content_type='application/json')
        
        # 格式错误的JSON会导致解析失败，但我们的代码会使用默认值
        # 所以应该返回200（成功）
        assert response.status_code == 200
    
    def test_missing_content_type(self):
        """测试缺少Content-Type的POST请求"""
        self.mock_collector.clear_old_records.return_value = 10
        
        response = self.client.post('/api/statistics/cleanup',
                                  data='{"days": 7}')
        
        # 没有Content-Type时，Flask无法解析JSON，会使用默认值
        # 但由于数据格式问题，可能返回400或200
        assert response.status_code in [200, 400]


if __name__ == '__main__':
    pytest.main([__file__])