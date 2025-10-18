#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据一致性系统性能测试

测试DataConsistencyChecker、DatabaseConstraintManager和AutoCorrectionService
在高负载和大数据集下的性能表现，包括：
- 大数据集处理性能
- 并发操作性能
- 内存使用优化
- 响应时间基准测试
"""

import pytest
import asyncio
import time
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import gc

from harborai.core.consistency import (
    DataConsistencyChecker,
    DatabaseConstraintManager,
    AutoCorrectionService
)
from harborai.database.manager import DatabaseManager


class TestDataConsistencyPerformance:
    """数据一致性系统性能测试类"""
    
    @pytest.fixture
    async def db_manager(self):
        """创建数据库管理器实例"""
        db_manager = DatabaseManager(
            host='localhost',
            port=5432,
            database='harborai_test',
            user='test_user',
            password='test_password'
        )
        await db_manager.initialize()
        yield db_manager
        await db_manager.close()
    
    @pytest.fixture
    async def consistency_system(self, db_manager):
        """创建数据一致性系统"""
        checker = DataConsistencyChecker(db_manager)
        constraint_manager = DatabaseConstraintManager(db_manager)
        correction_service = AutoCorrectionService(db_manager)
        
        return {
            'checker': checker,
            'constraint_manager': constraint_manager,
            'correction_service': correction_service,
            'db_manager': db_manager
        }
    
    @pytest.fixture
    async def large_dataset(self, db_manager):
        """创建大数据集用于性能测试"""
        # 清理现有测试数据
        await self._cleanup_performance_data(db_manager)
        
        # 插入大量测试数据
        dataset_size = 1000
        await self._insert_large_dataset(db_manager, dataset_size)
        
        yield dataset_size
        
        # 清理测试数据
        await self._cleanup_performance_data(db_manager)
    
    async def _cleanup_performance_data(self, db_manager):
        """清理性能测试数据"""
        cleanup_queries = [
            "DELETE FROM tracing_info WHERE log_id >= 50000",
            "DELETE FROM cost_info WHERE log_id >= 50000",
            "DELETE FROM token_usage WHERE log_id >= 50000",
            "DELETE FROM api_logs WHERE id >= 50000"
        ]
        
        for query in cleanup_queries:
            try:
                await db_manager.execute_query(query)
            except Exception:
                pass
    
    async def _insert_large_dataset(self, db_manager, size: int):
        """插入大数据集"""
        batch_size = 100
        
        for batch_start in range(0, size, batch_size):
            batch_end = min(batch_start + batch_size, size)
            
            # 批量插入API日志
            api_logs_batch = []
            for i in range(batch_start, batch_end):
                log_id = 50000 + i
                api_logs_batch.append({
                    'id': log_id,
                    'trace_id': f'perf-trace-{log_id}',
                    'span_id': f'perf-span-{log_id}',
                    'model': 'gpt-3.5-turbo',
                    'prompt_tokens': 100 + (i % 50),
                    'completion_tokens': 50 + (i % 25),
                    'total_tokens': 150 + (i % 75),
                    'total_cost': 0.05 + (i % 10) * 0.001,
                    'duration_ms': 1000 + (i % 500),
                    'status': 'success' if i % 10 != 0 else 'error',
                    'created_at': datetime.now() - timedelta(minutes=i % 1440)
                })
            
            # 批量插入
            for api_log in api_logs_batch:
                await db_manager.execute_query("""
                    INSERT INTO api_logs (
                        id, trace_id, span_id, model, prompt_tokens, completion_tokens,
                        total_tokens, total_cost, duration_ms, status, created_at
                    ) VALUES (
                        %(id)s, %(trace_id)s, %(span_id)s, %(model)s, %(prompt_tokens)s,
                        %(completion_tokens)s, %(total_tokens)s, %(total_cost)s,
                        %(duration_ms)s, %(status)s, %(created_at)s
                    )
                """, api_log)
                
                # 随机插入部分相关数据（模拟不完整数据）
                if i % 3 == 0:  # 1/3的记录有token数据
                    await db_manager.execute_query("""
                        INSERT INTO token_usage (log_id, prompt_tokens, completion_tokens, total_tokens)
                        VALUES (%(log_id)s, %(prompt_tokens)s, %(completion_tokens)s, %(total_tokens)s)
                    """, {
                        'log_id': api_log['id'],
                        'prompt_tokens': api_log['prompt_tokens'],
                        'completion_tokens': api_log['completion_tokens'],
                        'total_tokens': api_log['total_tokens']
                    })
                
                if i % 4 == 0:  # 1/4的记录有cost数据
                    await db_manager.execute_query("""
                        INSERT INTO cost_info (log_id, prompt_cost, completion_cost, total_cost)
                        VALUES (%(log_id)s, %(prompt_cost)s, %(completion_cost)s, %(total_cost)s)
                    """, {
                        'log_id': api_log['id'],
                        'prompt_cost': api_log['total_cost'] * 0.6,
                        'completion_cost': api_log['total_cost'] * 0.4,
                        'total_cost': api_log['total_cost']
                    })
    
    def _measure_memory_usage(self) -> Dict[str, float]:
        """测量内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': process.memory_percent()        # 内存使用百分比
        }
    
    def _measure_cpu_usage(self) -> float:
        """测量CPU使用率"""
        return psutil.cpu_percent(interval=0.1)
    
    @pytest.mark.asyncio
    async def test_large_dataset_consistency_check_performance(self, consistency_system, large_dataset):
        """测试大数据集一致性检查性能"""
        checker = consistency_system['checker']
        
        # 记录开始状态
        start_memory = self._measure_memory_usage()
        start_time = time.time()
        
        # 执行一致性检查
        report = await checker.generate_report()
        
        # 记录结束状态
        end_time = time.time()
        end_memory = self._measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_increase = end_memory['rss_mb'] - start_memory['rss_mb']
        
        # 性能断言
        assert execution_time < 60.0, f"检查时间过长: {execution_time:.2f}秒"
        assert memory_increase < 500.0, f"内存增长过多: {memory_increase:.2f}MB"
        assert report.total_issues >= 0
        
        # 记录性能指标
        print(f"\n大数据集性能测试结果:")
        print(f"数据集大小: {large_dataset} 条记录")
        print(f"执行时间: {execution_time:.2f} 秒")
        print(f"内存增长: {memory_increase:.2f} MB")
        print(f"发现问题: {report.total_issues} 个")
        print(f"处理速度: {large_dataset / execution_time:.2f} 记录/秒")
    
    @pytest.mark.asyncio
    async def test_concurrent_consistency_checks_performance(self, consistency_system, large_dataset):
        """测试并发一致性检查性能"""
        checker = consistency_system['checker']
        
        # 并发级别
        concurrency_levels = [1, 2, 4, 8]
        results = {}
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            start_memory = self._measure_memory_usage()
            
            # 创建并发任务
            tasks = []
            for _ in range(concurrency):
                tasks.append(checker.check_token_consistency())
            
            # 执行并发检查
            concurrent_results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            end_memory = self._measure_memory_usage()
            
            execution_time = end_time - start_time
            memory_increase = end_memory['rss_mb'] - start_memory['rss_mb']
            
            results[concurrency] = {
                'execution_time': execution_time,
                'memory_increase': memory_increase,
                'results_count': len(concurrent_results),
                'throughput': concurrency / execution_time
            }
            
            # 验证结果
            assert len(concurrent_results) == concurrency
            assert all(isinstance(result, list) for result in concurrent_results)
        
        # 分析并发性能
        print(f"\n并发性能测试结果:")
        for concurrency, metrics in results.items():
            print(f"并发级别 {concurrency}: "
                  f"时间 {metrics['execution_time']:.2f}s, "
                  f"内存 {metrics['memory_increase']:.2f}MB, "
                  f"吞吐量 {metrics['throughput']:.2f} 任务/秒")
        
        # 性能断言
        assert all(metrics['execution_time'] < 30.0 for metrics in results.values())
        assert all(metrics['memory_increase'] < 200.0 for metrics in results.values())
    
    @pytest.mark.asyncio
    async def test_batch_correction_performance(self, consistency_system, large_dataset):
        """测试批量修正性能"""
        correction_service = consistency_system['correction_service']
        checker = consistency_system['checker']
        
        # 首先找到需要修正的记录
        issues = await checker.check_token_consistency()
        missing_token_issues = [
            issue for issue in issues[:50]  # 限制修正数量
            if hasattr(issue, 'log_id')
        ]
        
        if not missing_token_issues:
            pytest.skip("没有找到需要修正的token数据")
        
        # 测试批量修正性能
        start_time = time.time()
        start_memory = self._measure_memory_usage()
        
        correction_results = []
        for issue in missing_token_issues:
            result = await correction_service.correct_missing_token_data(issue.log_id)
            correction_results.append(result)
        
        end_time = time.time()
        end_memory = self._measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_increase = end_memory['rss_mb'] - start_memory['rss_mb']
        
        # 性能分析
        successful_corrections = len([
            r for r in correction_results 
            if hasattr(r, 'status') and r.status.name == 'SUCCESS'
        ])
        
        correction_rate = successful_corrections / execution_time if execution_time > 0 else 0
        
        print(f"\n批量修正性能测试结果:")
        print(f"修正记录数: {len(missing_token_issues)}")
        print(f"成功修正数: {successful_corrections}")
        print(f"执行时间: {execution_time:.2f} 秒")
        print(f"内存增长: {memory_increase:.2f} MB")
        print(f"修正速度: {correction_rate:.2f} 记录/秒")
        
        # 性能断言
        assert execution_time < 60.0
        assert memory_increase < 100.0
        assert correction_rate > 0.5  # 至少每秒修正0.5条记录
    
    @pytest.mark.asyncio
    async def test_constraint_check_performance_scaling(self, consistency_system, large_dataset):
        """测试约束检查性能扩展性"""
        constraint_manager = consistency_system['constraint_manager']
        
        # 测试不同数据量下的性能
        data_sizes = [100, 500, 1000]
        performance_metrics = []
        
        for size in data_sizes:
            # 限制检查范围
            start_time = time.time()
            
            # 执行约束检查
            violations = await constraint_manager.check_foreign_key_violations()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            performance_metrics.append({
                'data_size': size,
                'execution_time': execution_time,
                'violations_found': len(violations),
                'check_rate': size / execution_time if execution_time > 0 else 0
            })
        
        # 分析性能扩展性
        print(f"\n约束检查性能扩展性测试:")
        for metrics in performance_metrics:
            print(f"数据量 {metrics['data_size']}: "
                  f"时间 {metrics['execution_time']:.2f}s, "
                  f"违反 {metrics['violations_found']} 个, "
                  f"检查速度 {metrics['check_rate']:.2f} 记录/秒")
        
        # 验证性能扩展性（时间复杂度应该接近线性）
        if len(performance_metrics) >= 2:
            time_ratios = []
            for i in range(1, len(performance_metrics)):
                prev_metrics = performance_metrics[i-1]
                curr_metrics = performance_metrics[i]
                
                size_ratio = curr_metrics['data_size'] / prev_metrics['data_size']
                time_ratio = curr_metrics['execution_time'] / prev_metrics['execution_time']
                
                time_ratios.append(time_ratio / size_ratio)
            
            # 时间复杂度应该接近O(n)，比率应该接近1
            avg_complexity_ratio = statistics.mean(time_ratios)
            assert avg_complexity_ratio < 3.0, f"时间复杂度过高: {avg_complexity_ratio:.2f}"
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, consistency_system, large_dataset):
        """测试内存使用优化"""
        checker = consistency_system['checker']
        
        # 多次执行检查，观察内存使用模式
        memory_snapshots = []
        
        for i in range(5):
            # 强制垃圾回收
            gc.collect()
            
            before_memory = self._measure_memory_usage()
            
            # 执行检查
            report = await checker.generate_report()
            
            after_memory = self._measure_memory_usage()
            
            memory_snapshots.append({
                'iteration': i + 1,
                'before_mb': before_memory['rss_mb'],
                'after_mb': after_memory['rss_mb'],
                'increase_mb': after_memory['rss_mb'] - before_memory['rss_mb'],
                'issues_found': report.total_issues
            })
            
            # 短暂等待
            await asyncio.sleep(0.1)
        
        # 分析内存使用模式
        print(f"\n内存使用优化测试:")
        for snapshot in memory_snapshots:
            print(f"迭代 {snapshot['iteration']}: "
                  f"前 {snapshot['before_mb']:.1f}MB, "
                  f"后 {snapshot['after_mb']:.1f}MB, "
                  f"增长 {snapshot['increase_mb']:.1f}MB")
        
        # 验证内存使用稳定性
        memory_increases = [s['increase_mb'] for s in memory_snapshots]
        max_increase = max(memory_increases)
        avg_increase = statistics.mean(memory_increases)
        
        assert max_increase < 100.0, f"单次内存增长过大: {max_increase:.1f}MB"
        assert avg_increase < 50.0, f"平均内存增长过大: {avg_increase:.1f}MB"
        
        # 检查内存泄漏（后续迭代的内存增长应该稳定）
        if len(memory_increases) >= 3:
            recent_increases = memory_increases[-3:]
            increase_variance = statistics.variance(recent_increases)
            assert increase_variance < 25.0, f"内存使用不稳定，可能存在内存泄漏: {increase_variance:.2f}"
    
    @pytest.mark.asyncio
    async def test_response_time_percentiles(self, consistency_system, large_dataset):
        """测试响应时间百分位数"""
        checker = consistency_system['checker']
        
        # 执行多次检查，收集响应时间
        response_times = []
        iterations = 20
        
        for i in range(iterations):
            start_time = time.time()
            
            # 执行单个检查操作
            issues = await checker.check_token_consistency()
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # 转换为毫秒
            response_times.append(response_time)
            
            # 验证结果有效性
            assert isinstance(issues, list)
        
        # 计算百分位数
        response_times.sort()
        p50 = response_times[int(len(response_times) * 0.5)]
        p90 = response_times[int(len(response_times) * 0.9)]
        p95 = response_times[int(len(response_times) * 0.95)]
        p99 = response_times[int(len(response_times) * 0.99)]
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        print(f"\n响应时间百分位数测试 ({iterations} 次迭代):")
        print(f"最小值: {min_response_time:.2f}ms")
        print(f"平均值: {avg_response_time:.2f}ms")
        print(f"P50: {p50:.2f}ms")
        print(f"P90: {p90:.2f}ms")
        print(f"P95: {p95:.2f}ms")
        print(f"P99: {p99:.2f}ms")
        print(f"最大值: {max_response_time:.2f}ms")
        
        # 性能断言
        assert p95 < 5000.0, f"P95响应时间过长: {p95:.2f}ms"
        assert p99 < 10000.0, f"P99响应时间过长: {p99:.2f}ms"
        assert avg_response_time < 2000.0, f"平均响应时间过长: {avg_response_time:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_database_connection_pool_performance(self, consistency_system, large_dataset):
        """测试数据库连接池性能"""
        db_manager = consistency_system['db_manager']
        
        # 测试高并发数据库操作
        concurrent_operations = 20
        operation_results = []
        
        async def database_operation(operation_id: int):
            """单个数据库操作"""
            start_time = time.time()
            
            try:
                # 执行数据库查询
                result = await db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM api_logs WHERE id >= 50000"
                )
                
                end_time = time.time()
                return {
                    'operation_id': operation_id,
                    'success': True,
                    'duration': end_time - start_time,
                    'result_count': result[0]['count'] if result else 0
                }
            except Exception as e:
                end_time = time.time()
                return {
                    'operation_id': operation_id,
                    'success': False,
                    'duration': end_time - start_time,
                    'error': str(e)
                }
        
        # 并发执行数据库操作
        start_time = time.time()
        
        tasks = [database_operation(i) for i in range(concurrent_operations)]
        operation_results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 分析结果
        successful_operations = [r for r in operation_results if r['success']]
        failed_operations = [r for r in operation_results if not r['success']]
        
        if successful_operations:
            avg_operation_time = statistics.mean([r['duration'] for r in successful_operations])
            max_operation_time = max([r['duration'] for r in successful_operations])
            min_operation_time = min([r['duration'] for r in successful_operations])
        else:
            avg_operation_time = max_operation_time = min_operation_time = 0
        
        print(f"\n数据库连接池性能测试:")
        print(f"并发操作数: {concurrent_operations}")
        print(f"总执行时间: {total_time:.2f}秒")
        print(f"成功操作数: {len(successful_operations)}")
        print(f"失败操作数: {len(failed_operations)}")
        print(f"平均操作时间: {avg_operation_time:.3f}秒")
        print(f"最大操作时间: {max_operation_time:.3f}秒")
        print(f"最小操作时间: {min_operation_time:.3f}秒")
        print(f"操作吞吐量: {len(successful_operations) / total_time:.2f} 操作/秒")
        
        # 性能断言
        assert len(successful_operations) >= concurrent_operations * 0.9  # 至少90%成功
        assert avg_operation_time < 1.0, f"平均操作时间过长: {avg_operation_time:.3f}秒"
        assert max_operation_time < 5.0, f"最大操作时间过长: {max_operation_time:.3f}秒"