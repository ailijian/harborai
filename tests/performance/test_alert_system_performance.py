#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警系统性能测试

测试告警系统在高负载下的性能表现
"""

import pytest
import asyncio
import time
import psutil
import tempfile
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

from harborai.core.alerts.alert_manager import (
    AlertManager, AlertRule, AlertSeverity, AlertCondition
)
from harborai.core.alerts.notification_service import NotificationService
from harborai.core.alerts.suppression_manager import SuppressionManager
from harborai.core.alerts.alert_history import AlertHistory


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.start_cpu = None
        self.end_cpu = None
        
    def start_measurement(self):
        """开始测量"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = psutil.Process().cpu_percent()
        
    def end_measurement(self):
        """结束测量"""
        self.end_time = time.time()
        self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.end_cpu = psutil.Process().cpu_percent()
        
    @property
    def duration(self) -> float:
        """执行时间（秒）"""
        return self.end_time - self.start_time if self.end_time and self.start_time else 0
        
    @property
    def memory_usage(self) -> float:
        """内存使用量变化（MB）"""
        return self.end_memory - self.start_memory if self.end_memory and self.start_memory else 0
        
    @property
    def cpu_usage(self) -> float:
        """CPU使用率"""
        return self.end_cpu if self.end_cpu else 0


class TestAlertSystemPerformance:
    """告警系统性能测试"""
    
    @pytest.fixture
    async def temp_db(self):
        """临时数据库文件"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
            
    @pytest.fixture
    async def alert_system(self, temp_db):
        """完整的告警系统"""
        # 创建组件
        alert_manager = AlertManager()
        notification_service = NotificationService()
        suppression_manager = SuppressionManager()
        alert_history = AlertHistory(db_path=temp_db)
        
        # 初始化
        await alert_manager.initialize()
        await notification_service.initialize()
        await suppression_manager.initialize()
        await alert_history.initialize()
        
        # 设置依赖
        alert_manager.set_notification_service(notification_service)
        alert_manager.set_suppression_service(suppression_manager)
        alert_manager.history_service = alert_history
        
        # 注册模拟指标提供者
        mock_provider = Mock()
        mock_provider.get_metric = AsyncMock()
        alert_manager.register_metric_provider("test", mock_provider)
        
        return {
            "alert_manager": alert_manager,
            "notification_service": notification_service,
            "suppression_manager": suppression_manager,
            "alert_history": alert_history,
            "metric_provider": mock_provider
        }
        
    def create_test_rules(self, count: int) -> List[AlertRule]:
        """创建测试规则"""
        rules = []
        for i in range(count):
            rule = AlertRule(
                id=f"perf_rule_{i}",
                name=f"性能测试规则 {i}",
                description=f"性能测试规则 {i}",
                severity=AlertSeverity.HIGH if i % 2 == 0 else AlertSeverity.MEDIUM,
                condition=AlertCondition.THRESHOLD,
                metric=f"perf_metric_{i}",
                threshold=10.0,
                duration=timedelta(seconds=1),
                labels={"component": "perf", "index": str(i)},
                annotations={"summary": f"性能测试告警 {i}"}
            )
            rules.append(rule)
        return rules
        
    async def test_rule_loading_performance(self, alert_system):
        """测试规则加载性能"""
        alert_manager = alert_system["alert_manager"]
        metrics = PerformanceMetrics()
        
        # 测试不同数量的规则
        rule_counts = [100, 500, 1000, 2000]
        results = {}
        
        for count in rule_counts:
            rules = self.create_test_rules(count)
            
            metrics.start_measurement()
            
            # 批量添加规则
            for rule in rules:
                await alert_manager.add_rule(rule)
                
            metrics.end_measurement()
            
            results[count] = {
                "duration": metrics.duration,
                "memory_usage": metrics.memory_usage,
                "rules_per_second": count / metrics.duration if metrics.duration > 0 else 0
            }
            
            # 清理规则
            for rule in rules:
                await alert_manager.remove_rule(rule.id)
                
        # 验证性能要求
        for count, result in results.items():
            print(f"规则数量: {count}, 加载时间: {result['duration']:.2f}s, "
                  f"内存使用: {result['memory_usage']:.2f}MB, "
                  f"规则/秒: {result['rules_per_second']:.0f}")
            
            # 性能要求：每秒至少处理100个规则
            assert result["rules_per_second"] >= 100, f"规则加载性能不足: {result['rules_per_second']}"
            
            # 内存要求：每1000个规则不超过50MB
            memory_per_1000 = result["memory_usage"] * 1000 / count
            assert memory_per_1000 <= 50, f"内存使用过多: {memory_per_1000:.2f}MB/1000规则"
            
    async def test_alert_evaluation_performance(self, alert_system):
        """测试告警评估性能"""
        alert_manager = alert_system["alert_manager"]
        metric_provider = alert_system["metric_provider"]
        metrics = PerformanceMetrics()
        
        # 创建1000个规则
        rules = self.create_test_rules(1000)
        for rule in rules:
            await alert_manager.add_rule(rule)
            
        # 设置指标值（一半触发告警）
        def get_metric_side_effect(metric_name):
            if metric_name.startswith("perf_metric_"):
                index = int(metric_name.split("_")[-1])
                return 15.0 if index % 2 == 0 else 5.0
            return 0.0
            
        metric_provider.get_metric.side_effect = get_metric_side_effect
        
        metrics.start_measurement()
        
        # 启动告警管理器并等待评估
        await alert_manager.start()
        await asyncio.sleep(5)  # 等待多轮评估
        
        metrics.end_measurement()
        
        # 检查结果
        active_alerts = await alert_manager.get_active_alerts()
        expected_alerts = 500  # 一半的规则应该触发告警
        
        await alert_manager.stop()
        
        # 性能验证
        evaluation_rate = len(rules) * 5 / metrics.duration  # 5秒内的评估次数
        print(f"评估性能: {evaluation_rate:.0f} 规则评估/秒")
        print(f"内存使用: {metrics.memory_usage:.2f}MB")
        print(f"CPU使用率: {metrics.cpu_usage:.1f}%")
        print(f"活跃告警数: {len(active_alerts)}")
        
        # 性能要求
        assert evaluation_rate >= 1000, f"评估性能不足: {evaluation_rate}"
        assert len(active_alerts) == expected_alerts, f"告警数量不正确: {len(active_alerts)}"
        assert metrics.memory_usage <= 100, f"内存使用过多: {metrics.memory_usage}MB"
        
    async def test_concurrent_alert_handling(self, alert_system):
        """测试并发告警处理性能"""
        alert_manager = alert_system["alert_manager"]
        metric_provider = alert_system["metric_provider"]
        metrics = PerformanceMetrics()
        
        # 创建规则
        rules = self.create_test_rules(100)
        for rule in rules:
            await alert_manager.add_rule(rule)
            
        # 模拟并发指标更新
        async def update_metrics():
            for i in range(100):
                metric_provider.get_metric.return_value = 15.0 + i
                await asyncio.sleep(0.01)  # 10ms间隔
                
        metrics.start_measurement()
        
        # 启动告警管理器和并发指标更新
        await alert_manager.start()
        
        # 并发执行多个指标更新任务
        tasks = [update_metrics() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        await asyncio.sleep(2)  # 等待处理完成
        metrics.end_measurement()
        
        await alert_manager.stop()
        
        # 检查结果
        active_alerts = await alert_manager.get_active_alerts()
        
        print(f"并发处理时间: {metrics.duration:.2f}s")
        print(f"处理的告警数: {len(active_alerts)}")
        print(f"内存使用: {metrics.memory_usage:.2f}MB")
        
        # 性能要求：并发处理不应显著影响性能
        assert metrics.duration <= 5, f"并发处理时间过长: {metrics.duration}s"
        assert len(active_alerts) > 0, "没有处理任何告警"
        
    async def test_notification_performance(self, alert_system):
        """测试通知性能"""
        alert_manager = alert_system["alert_manager"]
        notification_service = alert_system["notification_service"]
        metric_provider = alert_system["metric_provider"]
        
        # 模拟快速通知发送
        notification_times = []
        
        async def mock_send_notification(*args, **kwargs):
            start = time.time()
            await asyncio.sleep(0.001)  # 模拟1ms的发送时间
            end = time.time()
            notification_times.append(end - start)
            return True
            
        notification_service.send_alert_notification = mock_send_notification
        
        # 创建规则并触发大量告警
        rules = self.create_test_rules(200)
        for rule in rules:
            await alert_manager.add_rule(rule)
            
        metric_provider.get_metric.return_value = 15.0
        
        metrics = PerformanceMetrics()
        metrics.start_measurement()
        
        await alert_manager.start()
        await asyncio.sleep(3)  # 等待告警触发和通知发送
        
        metrics.end_measurement()
        await alert_manager.stop()
        
        # 分析通知性能
        if notification_times:
            avg_notification_time = sum(notification_times) / len(notification_times)
            max_notification_time = max(notification_times)
            notification_rate = len(notification_times) / metrics.duration
            
            print(f"发送的通知数: {len(notification_times)}")
            print(f"平均通知时间: {avg_notification_time*1000:.2f}ms")
            print(f"最大通知时间: {max_notification_time*1000:.2f}ms")
            print(f"通知发送率: {notification_rate:.0f} 通知/秒")
            
            # 性能要求
            assert avg_notification_time <= 0.01, f"平均通知时间过长: {avg_notification_time*1000}ms"
            assert notification_rate >= 50, f"通知发送率过低: {notification_rate}"
            
    async def test_history_storage_performance(self, alert_system):
        """测试历史存储性能"""
        alert_history = alert_system["alert_history"]
        metrics = PerformanceMetrics()
        
        # 创建大量告警记录
        alert_count = 1000
        
        metrics.start_measurement()
        
        # 批量创建告警历史记录
        tasks = []
        for i in range(alert_count):
            # 模拟告警对象
            mock_alert = Mock()
            mock_alert.id = f"alert_{i}"
            mock_alert.rule_id = f"rule_{i % 100}"
            mock_alert.rule_name = f"规则 {i % 100}"
            mock_alert.severity = AlertSeverity.HIGH
            mock_alert.status = "firing"
            mock_alert.message = f"测试告警 {i}"
            mock_alert.metric_value = 15.0 + i
            mock_alert.threshold = 10.0
            mock_alert.labels = {"index": str(i)}
            mock_alert.annotations = {"summary": f"告警 {i}"}
            mock_alert.started_at = datetime.now()
            mock_alert.resolved_at = None
            mock_alert.acknowledged_at = None
            mock_alert.acknowledged_by = None
            mock_alert.escalation_level = 0
            
            tasks.append(alert_history.record_alert(mock_alert))
            
        # 并发执行存储
        await asyncio.gather(*tasks)
        
        metrics.end_measurement()
        
        # 测试查询性能
        query_start = time.time()
        records = await alert_history.get_alert_history(limit=100)
        query_time = time.time() - query_start
        
        # 测试统计性能
        stats_start = time.time()
        stats = await alert_history.get_statistics()
        stats_time = time.time() - stats_start
        
        print(f"存储时间: {metrics.duration:.2f}s")
        print(f"存储速率: {alert_count/metrics.duration:.0f} 记录/秒")
        print(f"查询时间: {query_time*1000:.2f}ms")
        print(f"统计时间: {stats_time*1000:.2f}ms")
        print(f"内存使用: {metrics.memory_usage:.2f}MB")
        
        # 性能要求
        storage_rate = alert_count / metrics.duration
        assert storage_rate >= 500, f"存储速率过低: {storage_rate}"
        assert query_time <= 0.1, f"查询时间过长: {query_time*1000}ms"
        assert stats_time <= 0.5, f"统计时间过长: {stats_time*1000}ms"
        assert len(records) == 100, "查询结果数量不正确"
        
    async def test_memory_usage_under_load(self, alert_system):
        """测试负载下的内存使用"""
        alert_manager = alert_system["alert_manager"]
        metric_provider = alert_system["metric_provider"]
        
        # 监控内存使用
        memory_samples = []
        
        async def monitor_memory():
            while True:
                memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(memory)
                await asyncio.sleep(0.5)
                
        # 启动内存监控
        monitor_task = asyncio.create_task(monitor_memory())
        
        try:
            # 逐步增加负载
            for rule_count in [100, 500, 1000, 2000]:
                print(f"测试 {rule_count} 个规则...")
                
                # 添加规则
                rules = self.create_test_rules(rule_count)
                for rule in rules:
                    await alert_manager.add_rule(rule)
                    
                # 设置指标值触发告警
                metric_provider.get_metric.return_value = 15.0
                
                # 运行一段时间
                await alert_manager.start()
                await asyncio.sleep(5)
                await alert_manager.stop()
                
                # 记录当前内存使用
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"规则数: {rule_count}, 内存使用: {current_memory:.2f}MB")
                
                # 清理规则
                for rule in rules:
                    await alert_manager.remove_rule(rule.id)
                    
                # 等待垃圾回收
                await asyncio.sleep(1)
                
        finally:
            monitor_task.cancel()
            
        # 分析内存使用趋势
        if len(memory_samples) > 10:
            initial_memory = memory_samples[0]
            peak_memory = max(memory_samples)
            final_memory = memory_samples[-1]
            
            print(f"初始内存: {initial_memory:.2f}MB")
            print(f"峰值内存: {peak_memory:.2f}MB")
            print(f"最终内存: {final_memory:.2f}MB")
            print(f"内存增长: {peak_memory - initial_memory:.2f}MB")
            
            # 内存要求：峰值内存不应超过500MB
            assert peak_memory <= 500, f"内存使用过多: {peak_memory}MB"
            
            # 内存泄漏检查：最终内存不应比初始内存高太多
            memory_leak = final_memory - initial_memory
            assert memory_leak <= 50, f"可能存在内存泄漏: {memory_leak}MB"
            
    async def test_response_time_percentiles(self, alert_system):
        """测试响应时间百分位数"""
        alert_manager = alert_system["alert_manager"]
        metric_provider = alert_system["metric_provider"]
        
        # 创建规则
        rules = self.create_test_rules(100)
        for rule in rules:
            await alert_manager.add_rule(rule)
            
        metric_provider.get_metric.return_value = 15.0
        
        # 测量多次操作的响应时间
        response_times = []
        
        await alert_manager.start()
        
        # 测试告警查询响应时间
        for _ in range(100):
            start = time.time()
            await alert_manager.get_active_alerts()
            end = time.time()
            response_times.append((end - start) * 1000)  # 转换为毫秒
            
        await alert_manager.stop()
        
        # 计算百分位数
        response_times.sort()
        p50 = response_times[len(response_times) // 2]
        p95 = response_times[int(len(response_times) * 0.95)]
        p99 = response_times[int(len(response_times) * 0.99)]
        avg = sum(response_times) / len(response_times)
        
        print(f"响应时间统计:")
        print(f"平均值: {avg:.2f}ms")
        print(f"P50: {p50:.2f}ms")
        print(f"P95: {p95:.2f}ms")
        print(f"P99: {p99:.2f}ms")
        
        # 性能要求
        assert avg <= 10, f"平均响应时间过长: {avg}ms"
        assert p95 <= 20, f"P95响应时间过长: {p95}ms"
        assert p99 <= 50, f"P99响应时间过长: {p99}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])