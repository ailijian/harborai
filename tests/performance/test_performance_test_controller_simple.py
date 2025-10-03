#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试控制器简化测试模块

测试performance_test_controller.py的基本功能
避免复杂依赖导致的AST递归问题
遵循VIBE Coding规范
"""

import pytest
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_performance_controller_import():
    """测试性能控制器模块是否可以正常导入"""
    try:
        import performance_test_controller
        assert hasattr(performance_test_controller, 'TestType')
        assert hasattr(performance_test_controller, 'TestStatus')
        assert hasattr(performance_test_controller, 'TestResult')
        assert hasattr(performance_test_controller, 'TestConfiguration')
        assert hasattr(performance_test_controller, 'PerformanceTestController')
    except ImportError as e:
        pytest.fail(f"无法导入performance_test_controller模块: {e}")


def test_test_type_enum():
    """测试TestType枚举"""
    try:
        from performance_test_controller import TestType
        
        # 检查枚举值
        assert TestType.EXECUTION_EFFICIENCY.value == "execution_efficiency"
        assert TestType.MEMORY_LEAK.value == "memory_leak"
        assert TestType.RESOURCE_UTILIZATION.value == "resource_utilization"
        assert TestType.RESPONSE_TIME.value == "response_time"
        assert TestType.CONCURRENCY.value == "concurrency"
        assert TestType.INTEGRATION.value == "integration"
        assert TestType.ALL.value == "all"
        
    except ImportError as e:
        pytest.fail(f"无法导入TestType: {e}")


def test_test_status_enum():
    """测试TestStatus枚举"""
    try:
        from performance_test_controller import TestStatus
        
        # 检查枚举值
        assert TestStatus.PENDING.value == "pending"
        assert TestStatus.RUNNING.value == "running"
        assert TestStatus.COMPLETED.value == "completed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.SKIPPED.value == "skipped"
        
    except ImportError as e:
        pytest.fail(f"无法导入TestStatus: {e}")


def test_test_result_dataclass():
    """测试TestResult数据类"""
    try:
        from performance_test_controller import TestResult, TestType, TestStatus
        from datetime import datetime
        
        # 创建测试结果实例
        result = TestResult(
            test_type=TestType.EXECUTION_EFFICIENCY,
            test_name="test_basic_execution",
            status=TestStatus.COMPLETED,
            start_time=datetime.now()
        )
        
        assert result.test_type == TestType.EXECUTION_EFFICIENCY
        assert result.test_name == "test_basic_execution"
        assert result.status == TestStatus.COMPLETED
        assert result.start_time is not None
        assert result.end_time is None
        assert result.duration is None
        assert result.metrics is None
        assert result.error_message is None
        
    except ImportError as e:
        pytest.fail(f"无法导入TestResult: {e}")


def test_test_configuration_dataclass():
    """测试TestConfiguration数据类"""
    try:
        from performance_test_controller import TestConfiguration, TestType
        
        # 创建测试配置实例
        config = TestConfiguration(
            test_type=TestType.EXECUTION_EFFICIENCY,
            duration=60,
            iterations=100
        )
        
        assert config.test_type == TestType.EXECUTION_EFFICIENCY
        assert config.duration == 60
        assert config.iterations == 100
        
    except ImportError as e:
        pytest.fail(f"无法导入TestConfiguration: {e}")


def test_performance_config_dataclass():
    """测试PerformanceConfig数据类"""
    try:
        from performance_test_controller import PerformanceConfig
        
        # 创建性能配置实例
        config = PerformanceConfig()
        
        # 检查默认值
        assert config.output_dir is not None
        assert config.max_workers > 0
        assert config.timeout > 0
        assert config.enable_monitoring is True
        assert config.enable_profiling is False
        
    except ImportError as e:
        pytest.fail(f"无法导入PerformanceConfig: {e}")


@pytest.mark.integration
def test_performance_controller_basic_creation():
    """测试性能控制器基本创建功能"""
    try:
        from performance_test_controller import PerformanceTestController, PerformanceConfig
        import tempfile
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PerformanceConfig(output_dir=Path(temp_dir))
            controller = PerformanceTestController(config)
            
            assert controller is not None
            assert controller.config == config
            assert controller.output_dir == Path(temp_dir)
            
    except ImportError as e:
        pytest.fail(f"无法导入PerformanceTestController: {e}")
    except Exception as e:
        pytest.fail(f"创建PerformanceTestController时发生错误: {e}")


def test_file_size_check():
    """检查performance_test_controller.py文件大小，防止AST递归问题"""
    controller_file = current_dir / "performance_test_controller.py"
    
    if controller_file.exists():
        file_size = controller_file.stat().st_size
        line_count = len(controller_file.read_text(encoding='utf-8').splitlines())
        
        # 检查文件大小和行数
        print(f"performance_test_controller.py 文件大小: {file_size} bytes")
        print(f"performance_test_controller.py 行数: {line_count}")
        
        # 如果文件过大，可能导致AST递归问题
        if file_size > 100000:  # 100KB
            pytest.skip(f"文件过大 ({file_size} bytes)，可能导致AST递归问题")
        
        if line_count > 1000:  # 1000行
            pytest.skip(f"文件行数过多 ({line_count} 行)，可能导致AST递归问题")
    else:
        pytest.fail("performance_test_controller.py 文件不存在")


def test_module_complexity_check():
    """检查模块复杂度，识别可能的AST递归问题"""
    try:
        import ast
        
        controller_file = current_dir / "performance_test_controller.py"
        if not controller_file.exists():
            pytest.fail("performance_test_controller.py 文件不存在")
        
        # 解析AST
        with open(controller_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        try:
            tree = ast.parse(source_code)
            
            # 统计AST节点
            class NodeCounter(ast.NodeVisitor):
                def __init__(self):
                    self.count = 0
                    self.max_depth = 0
                    self.current_depth = 0
                
                def visit(self, node):
                    self.count += 1
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
            
            counter = NodeCounter()
            counter.visit(tree)
            
            print(f"AST节点总数: {counter.count}")
            print(f"AST最大深度: {counter.max_depth}")
            
            # 检查是否可能导致递归问题
            if counter.count > 5000:
                pytest.skip(f"AST节点过多 ({counter.count})，可能导致递归问题")
            
            if counter.max_depth > 100:
                pytest.skip(f"AST深度过深 ({counter.max_depth})，可能导致递归问题")
                
        except RecursionError:
            pytest.fail("解析AST时发生递归错误，确认存在AST递归问题")
        except Exception as e:
            pytest.fail(f"解析AST时发生错误: {e}")
            
    except ImportError:
        pytest.skip("无法导入ast模块")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])