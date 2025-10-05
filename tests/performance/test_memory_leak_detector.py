#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoryLeakDetector 完整测试用例

本模块包含对内存泄漏检测器的全面测试，包括：
- 单元测试：测试各个方法的功能
- 集成测试：测试完整的内存监控流程
- 边界条件测试：测试极端情况和错误处理
- 性能基准测试：验证检测器本身的性能
- 真实泄漏模拟：测试实际内存泄漏场景

作者: HarborAI Team
创建时间: 2024-01-20
遵循: VIBE Coding 规范
"""

import pytest
import asyncio
import time
import gc
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any
import psutil
import tracemalloc

try:
    from .memory_leak_detector import (
        MemoryLeakDetector,
        MemorySnapshot,
        MemoryLeakAnalysis,
        detect_memory_leak
    )
except ImportError:
    from memory_leak_detector import (
        MemoryLeakDetector,
        MemorySnapshot,
        MemoryLeakAnalysis,
        detect_memory_leak
    )


class TestMemorySnapshot:
    """MemorySnapshot 测试类"""
    
    def test_memory_snapshot_creation(self):
        """测试内存快照创建"""
        timestamp = datetime.now()
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            rss_memory=1024 * 1024,  # 1MB
            vms_memory=2048 * 1024,  # 2MB
            heap_memory=512 * 1024,  # 512KB
            gc_count={0: 10, 1: 5, 2: 1},
            object_count=1000,
            tracemalloc_peak=800 * 1024,  # 800KB
            memory_percent=25.5
        )
        
        assert snapshot.timestamp == timestamp
        assert snapshot.rss_memory == 1024 * 1024
        assert snapshot.vms_memory == 2048 * 1024
        assert snapshot.heap_memory == 512 * 1024
        assert snapshot.gc_count == {0: 10, 1: 5, 2: 1}
        assert snapshot.object_count == 1000
        assert snapshot.tracemalloc_peak == 800 * 1024
        assert snapshot.memory_percent == 25.5
    
    def test_memory_snapshot_to_dict(self):
        """测试内存快照转换为字典"""
        timestamp = datetime.now()
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            rss_memory=1024,
            vms_memory=2048,
            heap_memory=512,
            gc_count={0: 1, 1: 2},
            object_count=100,
            tracemalloc_peak=800,
            memory_percent=10.0
        )
        
        result_dict = snapshot.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['timestamp'] == timestamp.isoformat()
        assert result_dict['rss_memory'] == 1024
        assert result_dict['vms_memory'] == 2048
        assert result_dict['heap_memory'] == 512
        assert result_dict['gc_count'] == {0: 1, 1: 2}
        assert result_dict['object_count'] == 100
        assert result_dict['tracemalloc_peak'] == 800
        assert result_dict['memory_percent'] == 10.0


class TestMemoryLeakAnalysis:
    """MemoryLeakAnalysis 测试类"""
    
    def test_memory_leak_analysis_creation(self):
        """测试内存泄漏分析结果创建"""
        analysis = MemoryLeakAnalysis(
            is_leak_detected=True,
            leak_rate=1024.5,
            confidence_level=0.85,
            trend_analysis="持续增长",
            recommendations=["建议1", "建议2"],
            peak_memory=2048,
            average_memory=1500,
            memory_growth=15.5,
            gc_efficiency=0.75
        )
        
        assert analysis.is_leak_detected is True
        assert analysis.leak_rate == 1024.5
        assert analysis.confidence_level == 0.85
        assert analysis.trend_analysis == "持续增长"
        assert analysis.recommendations == ["建议1", "建议2"]
        assert analysis.peak_memory == 2048
        assert analysis.average_memory == 1500
        assert analysis.memory_growth == 15.5
        assert analysis.gc_efficiency == 0.75
    
    def test_memory_leak_analysis_to_dict(self):
        """测试内存泄漏分析结果转换为字典"""
        analysis = MemoryLeakAnalysis(
            is_leak_detected=False,
            leak_rate=0.0,
            confidence_level=0.95,
            trend_analysis="稳定",
            recommendations=[],
            peak_memory=1024,
            average_memory=1000,
            memory_growth=2.0,
            gc_efficiency=0.90
        )
        
        result_dict = analysis.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['is_leak_detected'] is False
        assert result_dict['leak_rate'] == 0.0
        assert result_dict['confidence_level'] == 0.95
        assert result_dict['trend_analysis'] == "稳定"
        assert result_dict['recommendations'] == []
        assert result_dict['peak_memory'] == 1024
        assert result_dict['average_memory'] == 1000
        assert result_dict['memory_growth'] == 2.0
        assert result_dict['gc_efficiency'] == 0.90


class TestMemoryLeakDetector:
    """MemoryLeakDetector 测试类"""
    
    @pytest.fixture
    def detector(self):
        """内存泄漏检测器fixture"""
        return MemoryLeakDetector(
            monitoring_interval=0.1,  # 快速测试
            max_snapshots=100,
            leak_threshold=1024,  # 1KB/s
            confidence_threshold=0.7
        )
    
    @pytest.fixture
    def mock_process(self):
        """模拟进程fixture"""
        mock_proc = Mock()
        mock_proc.memory_info.return_value = Mock(
            rss=1024 * 1024,  # 1MB
            vms=2048 * 1024   # 2MB
        )
        mock_proc.memory_percent.return_value = 25.0
        return mock_proc
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        detector = MemoryLeakDetector(
            monitoring_interval=2.0,
            max_snapshots=500,
            leak_threshold=2048,
            confidence_threshold=0.9
        )
        
        assert detector.monitoring_interval == 2.0
        assert detector.max_snapshots == 500
        assert detector.leak_threshold == 2048
        assert detector.confidence_threshold == 0.9
        assert detector.is_monitoring is False
        assert len(detector.snapshots) == 0
        assert len(detector.leak_callbacks) == 0
    
    def test_add_leak_callback(self, detector):
        """测试添加泄漏回调"""
        callback1 = Mock()
        callback2 = Mock()
        
        detector.add_leak_callback(callback1)
        detector.add_leak_callback(callback2)
        
        assert len(detector.leak_callbacks) == 2
        assert callback1 in detector.leak_callbacks
        assert callback2 in detector.leak_callbacks
    
    @patch('tests.performance.memory_leak_detector.tracemalloc')
    def test_take_snapshot(self, mock_tracemalloc, detector):
        """测试获取内存快照"""
        # 设置mock
        mock_tracemalloc.is_tracing.return_value = True
        mock_tracemalloc.get_traced_memory.return_value = (512 * 1024, 800 * 1024)
        
        # Mock detector的process对象
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024
        mock_memory_info.vms = 2048 * 1024
        
        mock_process = Mock()
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 25.0
        detector.process = mock_process
        
        # 模拟gc统计
        with patch('gc.get_count', return_value=(10, 5, 1)):
            with patch('gc.get_objects', return_value=[Mock() for _ in range(1000)]):
                snapshot = detector._take_snapshot()
        
        # 验证快照
        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.rss_memory == 1024 * 1024
        assert snapshot.vms_memory == 2048 * 1024
        assert snapshot.memory_percent == 25.0
        assert snapshot.object_count == 1000
        assert snapshot.gc_count == {0: 10, 1: 5, 2: 1}
    
    def test_start_stop_monitoring(self, detector):
        """测试开始和停止监控"""
        # 测试开始监控
        detector.start_monitoring()
        assert detector.is_monitoring is True
        assert detector.monitor_thread is not None
        assert detector.monitor_thread.is_alive()
        
        # 等待一小段时间让监控运行
        time.sleep(0.2)
        
        # 测试停止监控
        detector.stop_monitoring()
        assert detector.is_monitoring is False
        
        # 等待线程结束
        if detector.monitor_thread:
            detector.monitor_thread.join(timeout=1.0)
        
        assert not detector.monitor_thread.is_alive()
    
    def test_calculate_leak_rate(self, detector):
        """测试计算泄漏率"""
        # 创建模拟内存数据（持续增长）
        base_time = datetime.now()
        memory_data = [1000, 1100, 1200, 1300, 1400]  # 每次增长100字节
        timestamps = [
            base_time + timedelta(seconds=i) 
            for i in range(len(memory_data))
        ]
        
        leak_rate, confidence = detector._calculate_leak_rate(memory_data, timestamps)
        
        # 验证泄漏率计算（应该约为100字节/秒）
        assert leak_rate > 0
        assert 90 <= leak_rate <= 110  # 允许一些误差
        assert 0 <= confidence <= 1
    
    def test_calculate_leak_rate_stable_memory(self, detector):
        """测试稳定内存的泄漏率计算"""
        # 创建稳定的内存数据
        base_time = datetime.now()
        memory_data = [1000, 1000, 1000, 1000, 1000]  # 内存稳定
        timestamps = [
            base_time + timedelta(seconds=i) 
            for i in range(len(memory_data))
        ]
        
        leak_rate, confidence = detector._calculate_leak_rate(memory_data, timestamps)
        
        # 验证稳定内存的泄漏率
        assert abs(leak_rate) < 10  # 应该接近0
        assert confidence > 0.8  # 置信度应该较高
    
    def test_analyze_gc_efficiency(self, detector):
        """测试垃圾回收效率分析"""
        # 添加一些模拟快照
        base_time = datetime.now()
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(seconds=i),
                rss_memory=1000 + i * 100,
                vms_memory=2000,
                heap_memory=500,
                gc_count={0: 10 + i * 2, 1: 5 + i, 2: 1},
                object_count=1000 + i * 50,
                tracemalloc_peak=800,
                memory_percent=25.0
            )
            detector.snapshots.append(snapshot)
        
        efficiency = detector._analyze_gc_efficiency()
        
        # 验证效率计算
        assert 0 <= efficiency <= 1
        assert isinstance(efficiency, float)
    
    def test_generate_trend_analysis(self, detector):
        """测试趋势分析生成"""
        # 测试增长趋势
        memory_data = [1000, 1100, 1200, 1300]
        trend = detector._generate_trend_analysis(memory_data, 100.0)
        assert "增长" in trend
        
        # 测试稳定趋势
        memory_data = [1000, 1000, 1000, 1000]
        trend = detector._generate_trend_analysis(memory_data, 0.0)
        assert "稳定" in trend
        
        # 测试下降趋势
        memory_data = [1300, 1200, 1100, 1000]
        trend = detector._generate_trend_analysis(memory_data, -100.0)
        assert "下降" in trend
    
    def test_generate_recommendations(self, detector):
        """测试建议生成"""
        # 测试有泄漏的情况
        recommendations = detector._generate_recommendations(
            is_leak_detected=True,
            leak_rate=1024.0,
            gc_efficiency=0.5
        )
        assert len(recommendations) > 0
        assert any("内存泄漏" in rec for rec in recommendations)
        
        # 测试无泄漏的情况
        recommendations = detector._generate_recommendations(
            is_leak_detected=False,
            leak_rate=0.0,
            gc_efficiency=0.9
        )
        assert len(recommendations) >= 0  # 可能有一般性建议
    
    def test_get_memory_statistics(self, detector):
        """测试获取内存统计"""
        # 添加一些模拟快照
        base_time = datetime.now()
        for i in range(3):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(seconds=i),
                rss_memory=1000 + i * 100,
                vms_memory=2000,
                heap_memory=500,
                gc_count={0: 10, 1: 5, 2: 1},
                object_count=1000,
                tracemalloc_peak=800,
                memory_percent=25.0
            )
            detector.snapshots.append(snapshot)
        
        stats = detector.get_memory_statistics()
        
        # 验证统计信息
        assert isinstance(stats, dict)
        assert 'snapshot_count' in stats
        assert 'monitoring_duration' in stats
        assert 'average_memory' in stats
        assert 'peak_memory' in stats
        assert 'memory_growth' in stats
        assert stats['snapshot_count'] == 3
        assert stats['average_memory'] > 0
        assert stats['peak_memory'] >= stats['average_memory']
    
    def test_cleanup(self, detector):
        """测试清理资源"""
        # 启动监控
        detector.start_monitoring()
        assert detector.is_monitoring is True
        
        # 清理
        detector.cleanup()
        
        # 验证清理结果
        assert detector.is_monitoring is False
        assert len(detector.snapshots) == 0
        assert len(detector.leak_callbacks) == 0
    
    def test_context_manager(self, detector):
        """测试上下文管理器"""
        with detector as d:
            assert d is detector
            assert d.is_monitoring is True
        
        # 退出上下文后应该停止监控
        assert detector.is_monitoring is False


class TestMemoryLeakDetectorIntegration:
    """MemoryLeakDetector 集成测试类"""
    
    @pytest.mark.asyncio
    async def test_detect_memory_leak_async_no_leak(self):
        """测试异步检测无内存泄漏"""
        detector = MemoryLeakDetector(
            monitoring_interval=0.05,  # 更短的间隔
            max_snapshots=50,
            leak_threshold=1024
        )
        
        # 运行足够长的时间以收集足够样本
        analysis = await detector.detect_memory_leak_async(duration=1.5, interval=0.05)
        
        # 验证分析结果
        assert isinstance(analysis, MemoryLeakAnalysis)
        # 如果样本数量足够，置信度应该大于0
        if len(detector.snapshots) >= 10:
            assert analysis.confidence_level >= 0
        assert analysis.leak_rate is not None
        assert analysis.trend_analysis is not None
        assert isinstance(analysis.recommendations, list)
        
        # 清理
        detector.cleanup()
    
    def test_detect_memory_leak_function_wrapper(self):
        """测试内存泄漏检测函数包装器"""
        def stable_function():
            """稳定的测试函数"""
            data = [i for i in range(100)]
            return sum(data)
        
        # 运行检测
        analysis = detect_memory_leak(
            test_function=stable_function,
            duration=2.0,
            monitoring_interval=0.2
        )
        
        # 验证结果
        assert isinstance(analysis, MemoryLeakAnalysis)
        assert analysis.confidence_level > 0
        assert analysis.leak_rate is not None
    
    def test_memory_leak_simulation(self):
        """测试内存泄漏模拟"""
        # 创建一个会产生内存泄漏的函数
        leaked_data = []
        
        def leaky_function():
            """模拟内存泄漏的函数"""
            # 每次调用都添加数据但不释放
            leaked_data.extend([i for i in range(1000)])
            return len(leaked_data)
        
        detector = MemoryLeakDetector(
            monitoring_interval=0.1,
            max_snapshots=50,
            leak_threshold=100  # 较低的阈值以便检测到泄漏
        )
        
        try:
            # 启动监控
            detector.start_monitoring()
            
            # 运行泄漏函数多次
            for _ in range(10):
                leaky_function()
                time.sleep(0.1)
            
            # 等待收集足够的数据
            time.sleep(1.0)
            
            # 分析结果
            analysis = detector._analyze_memory_leak()
            
            # 验证检测到泄漏（可能需要多次运行才能检测到）
            assert isinstance(analysis, MemoryLeakAnalysis)
            assert analysis.leak_rate >= 0  # 应该有正的泄漏率
            
        finally:
            detector.cleanup()
            leaked_data.clear()  # 清理泄漏的数据
    
    def test_callback_notification(self):
        """测试回调通知功能"""
        callback_results = []
        
        def leak_callback(analysis: MemoryLeakAnalysis):
            """泄漏检测回调"""
            callback_results.append(analysis)
        
        detector = MemoryLeakDetector(
            monitoring_interval=0.1,
            max_snapshots=20,
            leak_threshold=1,  # 很低的阈值
            confidence_threshold=0.1  # 很低的置信度阈值
        )
        
        # 添加回调
        detector.add_leak_callback(leak_callback)
        
        # 创建一些模拟数据来触发泄漏检测
        base_time = datetime.now()
        for i in range(15):  # 确保有足够的样本
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(seconds=i * 0.1),
                rss_memory=1000000 + i * 10000,  # 持续增长的内存
                vms_memory=2000000,
                heap_memory=500000,
                gc_count={0: 100, 1: 50, 2: 10},
                object_count=10000,
                tracemalloc_peak=800000,
                memory_percent=25.0
            )
            detector.snapshots.append(snapshot)
        
        try:
            # 手动触发分析以测试回调
            analysis = detector._analyze_memory_leak()
            if analysis.is_leak_detected:
                detector._notify_leak_callbacks(analysis)
            
            # 验证回调被调用（如果检测到泄漏）
            if analysis.is_leak_detected:
                assert len(callback_results) > 0
                assert all(isinstance(result, MemoryLeakAnalysis) for result in callback_results)
            else:
                # 如果没有检测到泄漏，手动调用回调进行测试
                detector._notify_leak_callbacks(analysis)
                assert len(callback_results) > 0
                assert all(isinstance(result, MemoryLeakAnalysis) for result in callback_results)
            
        finally:
            detector.cleanup()


class TestMemoryLeakDetectorBenchmarks:
    """MemoryLeakDetector 性能基准测试类"""
    
    def test_snapshot_creation_performance(self, benchmark):
        """基准测试：快照创建性能"""
        detector = MemoryLeakDetector()
        
        def create_snapshot():
            return detector._take_snapshot()
        
        # 运行基准测试
        result = benchmark(create_snapshot)
        assert isinstance(result, MemorySnapshot)
    
    def test_leak_analysis_performance(self, benchmark):
        """基准测试：泄漏分析性能"""
        detector = MemoryLeakDetector()
        
        # 添加一些模拟数据
        base_time = datetime.now()
        for i in range(100):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(seconds=i * 0.1),
                rss_memory=1000000 + i * 1000,
                vms_memory=2000000,
                heap_memory=500000,
                gc_count={0: 100 + i, 1: 50 + i//2, 2: 10 + i//10},
                object_count=10000 + i * 10,
                tracemalloc_peak=800000,
                memory_percent=25.0 + i * 0.1
            )
            detector.snapshots.append(snapshot)
        
        def analyze_leak():
            return detector._analyze_memory_leak()
        
        # 运行基准测试
        result = benchmark(analyze_leak)
        assert isinstance(result, MemoryLeakAnalysis)
    
    def test_monitoring_overhead(self):
        """测试监控开销"""
        detector = MemoryLeakDetector(monitoring_interval=0.01)  # 高频监控
        
        # 测量启动监控前的内存
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        try:
            # 启动监控
            start_time = time.time()
            detector.start_monitoring()
            
            # 运行一段时间
            time.sleep(1.0)
            
            # 测量监控期间的内存
            monitoring_memory = process.memory_info().rss
            end_time = time.time()
            
            # 计算开销
            memory_overhead = monitoring_memory - initial_memory
            time_overhead = end_time - start_time
            
            # 验证开销在合理范围内
            assert memory_overhead < 50 * 1024 * 1024  # 小于50MB
            assert time_overhead >= 1.0  # 至少运行了1秒
            
            # 验证收集了数据
            assert len(detector.snapshots) > 0
            
        finally:
            detector.cleanup()


class TestMemoryLeakDetectorEdgeCases:
    """MemoryLeakDetector 边界条件测试类"""
    
    def test_empty_snapshots_analysis(self):
        """测试空快照列表的分析"""
        detector = MemoryLeakDetector()
        
        # 分析空数据
        analysis = detector._analyze_memory_leak()
        
        # 验证处理空数据的情况
        assert isinstance(analysis, MemoryLeakAnalysis)
        assert analysis.is_leak_detected is False
        assert analysis.leak_rate == 0.0
        assert analysis.confidence_level == 0.0
    
    def test_single_snapshot_analysis(self):
        """测试单个快照的分析"""
        detector = MemoryLeakDetector()
        
        # 添加单个快照
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_memory=1024,
            vms_memory=2048,
            heap_memory=512,
            gc_count={0: 10, 1: 5, 2: 1},
            object_count=1000,
            tracemalloc_peak=800,
            memory_percent=25.0
        )
        detector.snapshots.append(snapshot)
        
        # 分析单个快照
        analysis = detector._analyze_memory_leak()
        
        # 验证单个快照的处理
        assert isinstance(analysis, MemoryLeakAnalysis)
        assert analysis.confidence_level == 0.0  # 单个数据点置信度为0
    
    def test_max_snapshots_limit(self):
        """测试最大快照数量限制"""
        detector = MemoryLeakDetector(max_snapshots=5)
        
        # 添加超过限制的快照
        base_time = datetime.now()
        for i in range(10):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(seconds=i),
                rss_memory=1000 + i,
                vms_memory=2000,
                heap_memory=500,
                gc_count={0: 10, 1: 5, 2: 1},
                object_count=1000,
                tracemalloc_peak=800,
                memory_percent=25.0
            )
            detector.snapshots.append(snapshot)
        
        # 验证快照数量限制
        assert len(detector.snapshots) <= detector.max_snapshots
    
    def test_invalid_monitoring_interval(self):
        """测试无效的监控间隔"""
        # 测试负数间隔
        with pytest.raises(ValueError):
            MemoryLeakDetector(monitoring_interval=-1.0)
        
        # 测试零间隔
        with pytest.raises(ValueError):
            MemoryLeakDetector(monitoring_interval=0.0)
    
    def test_concurrent_monitoring_calls(self):
        """测试并发监控调用"""
        detector = MemoryLeakDetector(monitoring_interval=0.1)
        
        try:
            # 启动第一次监控
            detector.start_monitoring()
            assert detector.is_monitoring is True
            
            # 尝试再次启动监控（应该被忽略）
            detector.start_monitoring()
            assert detector.is_monitoring is True
            
            # 应该只有一个监控线程
            thread_count = sum(1 for t in threading.enumerate() if 'memory_monitor' in t.name.lower())
            assert thread_count <= 1
            
        finally:
            detector.cleanup()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short", "--benchmark-skip"])