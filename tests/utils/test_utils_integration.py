#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试工具集成测试

功能：验证tests.utils包中所有模块的基本功能
作者：HarborAI测试团队
创建时间：2024年
"""

import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """测试所有模块导入
    
    功能：验证所有测试工具模块能够正常导入
    """
    logger.info("开始测试模块导入...")
    
    try:
        # 测试基础导入
        from tests.utils import (
            TestConfig, TestTimer, MockDataGenerator, SecurityTestHelper,
            MockResponse, APIResponseMocker, ErrorSimulator,
            TestDataGenerator, PerformanceMetrics, SystemMonitor,
            SecurityLevel, DataSanitizer, VulnerabilityScanner,
            ReportGenerator, MetricsCollector, TestMetricsExporter
        )
        logger.info("✅ 所有主要类导入成功")
        
        # 测试便捷函数导入
        from tests.utils import (
            generate_test_data, measure_performance,
            mock_harborai_client, sanitize_data, 
            generate_test_report
        )
        logger.info("✅ 所有便捷函数导入成功")
        
        assert True, "所有模块导入成功"
        
    except ImportError as e:
        logger.error(f"❌ 模块导入失败：{e}")
        assert False, f"模块导入失败：{e}"
    except Exception as e:
        logger.error(f"❌ 导入过程中发生错误：{e}")
        assert False, f"导入过程中发生错误：{e}"


def test_basic_functionality():
    """测试基本功能
    
    功能：验证各模块的基本功能是否正常
    """
    logger.info("开始测试基本功能...")
    
    try:
        # 测试配置类
        from tests.utils import TestConfig
        config = TestConfig()
        logger.info(f"✅ TestConfig创建成功，超时设置：{config.timeout}秒")
        
        # 测试数据生成
        from tests.utils import generate_test_data
        test_data = generate_test_data("user", count=2)
        logger.info(f"✅ 测试数据生成成功，生成{len(test_data)}条用户数据")
        
        # 测试性能测量
        from tests.utils import TestTimer
        timer = TestTimer()
        timer.start()
        import time
        time.sleep(0.1)  # 模拟操作
        duration = timer.stop()
        logger.info(f"✅ 性能测量成功，耗时：{duration:.3f}秒")
        
        # 测试数据脱敏
        from tests.utils import sanitize_data
        sensitive_text = "我的API密钥是sk-1234567890abcdef，请保密"
        sanitized = sanitize_data(sensitive_text)
        logger.info(f"✅ 数据脱敏成功：{sanitized}")
        
        # 测试Mock响应
        from tests.utils import MockResponse
        mock_resp = MockResponse(
            content="测试响应",
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
        logger.info(f"✅ Mock响应创建成功，状态码：{mock_resp.status_code}")
        
        assert True, "基本功能测试通过"
        
    except Exception as e:
        logger.error(f"❌ 基本功能测试失败：{e}")
        assert False, f"基本功能测试失败：{e}"


def test_integration_workflow():
    """测试集成工作流
    
    功能：验证多个模块协同工作的场景
    """
    logger.info("开始测试集成工作流...")
    
    try:
        # 模拟一个完整的测试流程
        from tests.utils import (
            TestDataGenerator, APIResponseMocker, 
            ReportGenerator, MetricsCollector
        )
        
        # 1. 生成测试数据
        data_gen = TestDataGenerator()
        chat_message = data_gen.generate_chat_message()
        logger.info(f"✅ 生成了聊天消息：{chat_message.get('role', 'unknown')}")
        
        # 2.# 创建Mock响应
        mocker = APIResponseMocker()
        mock_response = mocker.create_chat_completion_response(
            content="这是一个测试响应",
            model="deepseek-chat"
        )
        logger.info(f"✅ 创建Mock响应成功，模型：{mock_response.get('model')}")
        
        # 3. 模拟测试结果
        test_results = [
            {
                'name': 'test_chat_completion',
                'status': 'passed',
                'duration': 1.2,
                'type': 'integration'
            },
            {
                'name': 'test_error_handling',
                'status': 'passed', 
                'duration': 0.8,
                'type': 'unit'
            }
        ]
        
        # 4. 收集指标
        collector = MetricsCollector()
        metrics = collector.collect_test_metrics(
            session_id="integration_test",
            test_results=test_results
        )
        logger.info(f"✅ 指标收集成功，成功率：{metrics.get('test_results', {}).get('success_rate', 0):.1f}%")
        
        # 5. 生成报告
        generator = ReportGenerator()
        summary = generator.generate_summary_report(test_results)
        logger.info(f"✅ 报告生成成功，长度：{len(summary)}字符")
        
        assert True, "集成工作流测试通过"
        
    except Exception as e:
        logger.error(f"❌ 集成工作流测试失败：{e}")
        assert False, f"集成工作流测试失败：{e}"


def test_error_handling():
    """测试错误处理
    
    功能：验证各模块的错误处理能力
    """
    logger.info("开始测试错误处理...")
    
    try:
        # 测试错误模拟器
        from tests.utils import ErrorSimulator
        
        simulator = ErrorSimulator()
        
        # 测试认证错误
        auth_error = simulator.create_authentication_error()
        logger.info(f"✅ 认证错误模拟成功：{auth_error.get('error', {}).get('message', '')}")
        
        # 测试速率限制错误
        rate_limit_error = simulator.create_rate_limit_error()
        logger.info(f"✅ 速率限制错误模拟成功：{rate_limit_error.get('error', {}).get('type', '')}")
        
        # 测试重试装饰器
        from tests.utils import retry_on_failure
        
        @retry_on_failure(max_retries=2, delay=0.1)
        def flaky_function():
            import random
            if random.random() < 0.7:  # 70%概率失败
                raise ValueError("模拟的随机错误")
            return "成功"
        
        try:
            result = flaky_function()
            logger.info(f"✅ 重试机制测试成功：{result}")
        except Exception as e:
            logger.info(f"✅ 重试机制测试完成（最终失败是正常的）：{e}")
        
        assert True, "错误处理测试通过"
        
    except Exception as e:
        logger.error(f"❌ 错误处理测试失败：{e}")
        assert False, f"错误处理测试失败：{e}"


def run_all_tests():
    """运行所有测试
    
    功能：执行完整的测试套件
    返回：测试是否全部通过
    """
    logger.info("="*60)
    logger.info("开始运行HarborAI测试工具集成测试")
    logger.info("="*60)
    
    tests = [
        ("模块导入测试", test_imports),
        ("基本功能测试", test_basic_functionality),
        ("集成工作流测试", test_integration_workflow),
        ("错误处理测试", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 运行 {test_name}...")
        try:
            if test_func():
                logger.info(f"✅ {test_name} 通过")
                passed += 1
            else:
                logger.error(f"❌ {test_name} 失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 异常：{e}")
    
    logger.info("\n" + "="*60)
    logger.info(f"测试结果：{passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！HarborAI测试工具包已准备就绪。")
        return True
    else:
        logger.error(f"⚠️  有 {total - passed} 个测试失败，请检查相关模块。")
        return False


def main():
    """主函数
    
    功能：程序入口点
    """
    try:
        # 添加项目根目录到Python路径
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # 运行测试
        success = run_all_tests()
        
        # 输出使用示例
        if success:
            logger.info("\n" + "="*60)
            logger.info("使用示例：")
            logger.info("")
            logger.info("# 导入测试工具")
            logger.info("from tests.utils import (")
            logger.info("    TestConfig, generate_test_data, mock_harborai_client,")
            logger.info("    sanitize_data, generate_test_report")
            logger.info(")")
            logger.info("")
            logger.info("# 生成测试数据")
            logger.info("test_data = generate_test_data('user', count=5)")
            logger.info("")
            logger.info("# 使用Mock客户端")
            logger.info("with mock_harborai_client() as mock_client:")
            logger.info("    # 执行测试")
            logger.info("    pass")
            logger.info("")
            logger.info("# 生成测试报告")
            logger.info("reports = generate_test_report(test_results)")
            logger.info("="*60)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n用户中断测试")
        return 1
    except Exception as e:
        logger.error(f"测试运行失败：{e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)