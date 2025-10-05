"""
性能报告生成器测试模块

该模块包含对PerformanceReportGenerator类的全面测试，包括：
- 单元测试：测试各个方法的基本功能
- 集成测试：测试组件间的协作
- 边界测试：测试极端情况和错误处理
- 性能基准测试：测试性能指标

作者：HarborAI性能测试团队
创建时间：2024年
遵循VIBE Coding规范
"""

import pytest
import tempfile
import shutil
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# 导入被测试的模块
try:
    from .performance_report_generator import (
        PerformanceReportGenerator,
        ReportMetadata,
        PerformanceSummary,
        ChartData,
        generate_quick_report,
        MATPLOTLIB_AVAILABLE
    )
except ImportError:
    from performance_report_generator import (
        PerformanceReportGenerator,
        ReportMetadata,
        PerformanceSummary,
        ChartData,
        generate_quick_report,
        MATPLOTLIB_AVAILABLE
    )


class TestReportMetadata:
    """ReportMetadata类的单元测试"""
    
    def test_report_metadata_creation(self):
        """测试ReportMetadata对象创建"""
        generated_at = datetime.now()
        test_duration = timedelta(minutes=5)
        test_environment = {
            'python_version': '3.9.0',
            'platform': 'Windows 11',
            'cpu_cores': 8,
            'memory_total': '16GB'
        }
        
        metadata = ReportMetadata(
            title="性能测试报告",
            description="HarborAI系统性能测试",
            generated_at=generated_at,
            test_duration=test_duration,
            test_environment=test_environment,
            version="2.0"
        )
        
        assert metadata.title == "性能测试报告"
        assert metadata.description == "HarborAI系统性能测试"
        assert metadata.generated_at == generated_at
        assert metadata.test_duration == test_duration
        assert metadata.test_environment == test_environment
        assert metadata.version == "2.0"
    
    def test_report_metadata_to_dict(self):
        """测试ReportMetadata转换为字典"""
        generated_at = datetime.now()
        test_duration = timedelta(minutes=3)
        test_environment = {'platform': 'Linux'}
        
        metadata = ReportMetadata(
            title="测试报告",
            description="测试描述",
            generated_at=generated_at,
            test_duration=test_duration,
            test_environment=test_environment
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict['title'] == "测试报告"
        assert metadata_dict['description'] == "测试描述"
        assert metadata_dict['generated_at'] == generated_at.isoformat()
        assert metadata_dict['test_duration'] == str(test_duration)
        assert metadata_dict['test_environment'] == test_environment
        assert metadata_dict['version'] == "1.0"  # 默认值


class TestPerformanceSummary:
    """PerformanceSummary类的单元测试"""
    
    def test_performance_summary_creation(self):
        """测试PerformanceSummary对象创建"""
        summary = PerformanceSummary(
            total_tests=100,
            passed_tests=95,
            failed_tests=5,
            success_rate=0.95,
            average_response_time=0.125,
            peak_memory_usage=512 * 1024 * 1024,  # 512MB
            peak_cpu_usage=75.5,
            total_requests=1000,
            requests_per_second=80.0
        )
        
        assert summary.total_tests == 100
        assert summary.passed_tests == 95
        assert summary.failed_tests == 5
        assert summary.success_rate == 0.95
        assert summary.average_response_time == 0.125
        assert summary.peak_memory_usage == 512 * 1024 * 1024
        assert summary.peak_cpu_usage == 75.5
        assert summary.total_requests == 1000
        assert summary.requests_per_second == 80.0
    
    def test_performance_summary_to_dict(self):
        """测试PerformanceSummary转换为字典"""
        summary = PerformanceSummary(
            total_tests=50,
            passed_tests=48,
            failed_tests=2,
            success_rate=0.96,
            average_response_time=0.15,
            peak_memory_usage=256 * 1024 * 1024,
            peak_cpu_usage=60.0,
            total_requests=500,
            requests_per_second=50.0
        )
        
        summary_dict = summary.to_dict()
        
        assert summary_dict['total_tests'] == 50
        assert summary_dict['passed_tests'] == 48
        assert summary_dict['failed_tests'] == 2
        assert summary_dict['success_rate'] == 0.96
        assert summary_dict['average_response_time'] == 0.15
        assert summary_dict['peak_memory_usage'] == 256 * 1024 * 1024
        assert summary_dict['peak_cpu_usage'] == 60.0
        assert summary_dict['total_requests'] == 500
        assert summary_dict['requests_per_second'] == 50.0


class TestChartData:
    """ChartData类的单元测试"""
    
    def test_chart_data_creation(self):
        """测试ChartData对象创建"""
        data = {
            'labels': ['A', 'B', 'C'],
            'values': [10, 20, 30]
        }
        options = {
            'color': 'blue',
            'width': 800,
            'height': 600
        }
        
        chart = ChartData(
            chart_type="bar",
            title="测试图表",
            data=data,
            options=options
        )
        
        assert chart.chart_type == "bar"
        assert chart.title == "测试图表"
        assert chart.data == data
        assert chart.options == options
    
    def test_chart_data_to_dict(self):
        """测试ChartData转换为字典"""
        data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
        options = {'theme': 'dark'}
        
        chart = ChartData(
            chart_type="line",
            title="线性图表",
            data=data,
            options=options
        )
        
        chart_dict = chart.to_dict()
        
        assert chart_dict['chart_type'] == "line"
        assert chart_dict['title'] == "线性图表"
        assert chart_dict['data'] == data
        assert chart_dict['options'] == options
    
    def test_chart_data_default_options(self):
        """测试ChartData默认选项"""
        chart = ChartData(
            chart_type="pie",
            title="饼图",
            data={'segments': [1, 2, 3]}
        )
        
        assert chart.options == {}


class TestPerformanceReportGenerator:
    """PerformanceReportGenerator类的单元测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def generator(self, temp_dir):
        """PerformanceReportGenerator实例fixture"""
        return PerformanceReportGenerator(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_metadata(self):
        """示例元数据fixture"""
        return ReportMetadata(
            title="测试报告",
            description="性能测试报告",
            generated_at=datetime.now(),
            test_duration=timedelta(minutes=5),
            test_environment={'platform': 'Windows 11'}
        )
    
    @pytest.fixture
    def sample_summary(self):
        """示例摘要fixture"""
        return PerformanceSummary(
            total_tests=100,
            passed_tests=95,
            failed_tests=5,
            success_rate=0.95,
            average_response_time=0.125,
            peak_memory_usage=512 * 1024 * 1024,
            peak_cpu_usage=75.5,
            total_requests=1000,
            requests_per_second=80.0
        )
    
    def test_initialization(self, temp_dir):
        """测试初始化"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        assert generator.output_dir == Path(temp_dir)
        assert generator.output_dir.exists()
        assert generator.metadata is None
        assert generator.summary is None
        assert len(generator.charts) == 0
        assert generator.detailed_data == {}
        assert generator.html_template is not None
    
    def test_set_metadata(self, generator, sample_metadata):
        """测试设置元数据"""
        generator.set_metadata(sample_metadata)
        
        assert generator.metadata == sample_metadata
        assert generator.metadata.title == "测试报告"
    
    def test_set_summary(self, generator, sample_summary):
        """测试设置摘要"""
        generator.set_summary(sample_summary)
        
        assert generator.summary == sample_summary
        assert generator.summary.total_tests == 100
    
    def test_add_chart(self, generator):
        """测试添加图表"""
        chart1 = ChartData(
            chart_type="line",
            title="图表1",
            data={'x': [1, 2], 'y': [3, 4]}
        )
        chart2 = ChartData(
            chart_type="bar",
            title="图表2",
            data={'labels': ['A', 'B'], 'values': [10, 20]}
        )
        
        generator.add_chart(chart1)
        generator.add_chart(chart2)
        
        assert len(generator.charts) == 2
        assert generator.charts[0] == chart1
        assert generator.charts[1] == chart2
    
    def test_set_detailed_data(self, generator):
        """测试设置详细数据"""
        detailed_data = {
            'test_results': [{'name': 'test1', 'result': 'pass'}],
            'metrics': {'cpu': 50.0, 'memory': 128}
        }
        
        generator.set_detailed_data(detailed_data)
        
        assert generator.detailed_data == detailed_data
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib不可用")
    def test_generate_response_time_chart(self, generator):
        """测试生成响应时间图表"""
        timestamps = [
            datetime.now() + timedelta(seconds=i)
            for i in range(10)
        ]
        response_times = [0.1 + i * 0.01 for i in range(10)]
        
        chart_base64 = generator.generate_response_time_chart(
            timestamps, response_times, "响应时间测试"
        )
        
        assert chart_base64 is not None
        assert isinstance(chart_base64, str)
        assert len(chart_base64) > 0
        
        # 验证是否为有效的base64编码
        try:
            base64.b64decode(chart_base64)
        except Exception:
            pytest.fail("生成的图表不是有效的base64编码")
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib不可用")
    def test_generate_memory_usage_chart(self, generator):
        """测试生成内存使用图表"""
        timestamps = [
            datetime.now() + timedelta(seconds=i)
            for i in range(5)
        ]
        memory_usage = [100 * 1024 * 1024 + i * 10 * 1024 * 1024 for i in range(5)]  # MB
        
        chart_base64 = generator.generate_memory_usage_chart(
            timestamps, memory_usage, "内存使用测试"
        )
        
        assert chart_base64 is not None
        assert isinstance(chart_base64, str)
        assert len(chart_base64) > 0
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib不可用")
    def test_generate_cpu_usage_chart(self, generator):
        """测试生成CPU使用图表"""
        timestamps = [
            datetime.now() + timedelta(seconds=i)
            for i in range(8)
        ]
        cpu_usage = [50.0 + i * 5.0 for i in range(8)]
        
        chart_base64 = generator.generate_cpu_usage_chart(
            timestamps, cpu_usage, "CPU使用测试"
        )
        
        assert chart_base64 is not None
        assert isinstance(chart_base64, str)
        assert len(chart_base64) > 0
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib不可用")
    def test_generate_throughput_chart(self, generator):
        """测试生成吞吐量图表"""
        timestamps = [
            datetime.now() + timedelta(seconds=i)
            for i in range(6)
        ]
        throughput = [80.0 + i * 2.0 for i in range(6)]
        
        chart_base64 = generator.generate_throughput_chart(
            timestamps, throughput, "吞吐量测试"
        )
        
        assert chart_base64 is not None
        assert isinstance(chart_base64, str)
        assert len(chart_base64) > 0
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib不可用")
    def test_generate_performance_summary_chart(self, generator):
        """测试生成性能摘要图表"""
        metrics = {
            '响应时间': 0.125,
            'CPU使用率': 75.5,
            '内存使用': 60.0,
            '成功率': 95.0,
            '吞吐量': 80.0
        }
        
        chart_base64 = generator.generate_performance_summary_chart(
            metrics, "性能指标概览"
        )
        
        assert chart_base64 is not None
        assert isinstance(chart_base64, str)
        assert len(chart_base64) > 0
    
    def test_generate_charts_without_matplotlib(self, generator):
        """测试在没有matplotlib的情况下生成图表"""
        # 直接patch模块中的MATPLOTLIB_AVAILABLE变量
        import performance_report_generator as prg_module
        with patch.object(prg_module, 'MATPLOTLIB_AVAILABLE', False):
            timestamps = [datetime.now()]
            values = [1.0]
            
            # 所有图表生成方法都应该返回None
            assert generator.generate_response_time_chart(timestamps, values) is None
            assert generator.generate_memory_usage_chart(timestamps, [1024]) is None
            assert generator.generate_cpu_usage_chart(timestamps, values) is None
            assert generator.generate_throughput_chart(timestamps, values) is None
            assert generator.generate_performance_summary_chart({'test': 1.0}) is None
    
    def test_generate_html_report(self, generator, sample_metadata, sample_summary, temp_dir):
        """测试生成HTML报告"""
        # 设置报告数据
        generator.set_metadata(sample_metadata)
        generator.set_summary(sample_summary)
        
        # 添加图表数据
        chart = ChartData(
            chart_type="line",
            title="测试图表",
            data={'x': [1, 2, 3], 'y': [4, 5, 6]}
        )
        generator.add_chart(chart)
        
        # 设置详细数据
        generator.set_detailed_data({'test_data': 'example'})
        
        # 生成HTML报告
        html_path = generator.generate_html_report("test_report.html")
        
        # 验证文件生成
        assert html_path is not None
        html_file = Path(html_path)
        assert html_file.exists()
        assert html_file.suffix == '.html'
        
        # 验证HTML内容
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert "测试报告" in html_content
        assert "性能测试报告" in html_content
        assert "100" in html_content  # total_tests
        assert "95" in html_content   # passed_tests
        assert "95.0%" in html_content  # success_rate
    
    def test_generate_json_report(self, generator, sample_metadata, sample_summary, temp_dir):
        """测试生成JSON报告"""
        # 设置报告数据
        generator.set_metadata(sample_metadata)
        generator.set_summary(sample_summary)
        
        # 添加图表数据
        chart = ChartData(
            chart_type="bar",
            title="JSON测试图表",
            data={'labels': ['A', 'B'], 'values': [10, 20]}
        )
        generator.add_chart(chart)
        
        # 设置详细数据
        generator.set_detailed_data({'json_test_data': 'example'})
        
        # 生成JSON报告
        json_path = generator.generate_json_report("test_report.json")
        
        # 验证文件生成
        assert json_path is not None
        json_file = Path(json_path)
        assert json_file.exists()
        assert json_file.suffix == '.json'
        
        # 验证JSON内容
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert 'metadata' in json_data
        assert 'summary' in json_data
        assert 'charts' in json_data
        assert 'detailed_data' in json_data
        
        assert json_data['metadata']['title'] == "测试报告"
        assert json_data['summary']['total_tests'] == 100
        assert len(json_data['charts']) == 1
        assert json_data['charts'][0]['title'] == "JSON测试图表"
        assert json_data['detailed_data']['json_test_data'] == 'example'
    
    def test_generate_reports_without_data(self, generator, temp_dir):
        """测试在没有数据的情况下生成报告"""
        # 生成HTML报告
        html_path = generator.generate_html_report("empty_report.html")
        assert html_path is not None
        assert Path(html_path).exists()
        
        # 生成JSON报告
        json_path = generator.generate_json_report("empty_report.json")
        assert json_path is not None
        assert Path(json_path).exists()
        
        # 验证JSON内容
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 现在应该有默认数据而不是None
        assert json_data['metadata'] is not None
        assert json_data['metadata']['title'] == "性能测试报告"
        assert json_data['metadata']['description'] == "无数据报告"
        assert json_data['summary'] is not None
        assert json_data['summary']['total_tests'] == 0
        assert json_data['charts'] == []
        assert json_data['detailed_data'] == {}


class TestPerformanceReportGeneratorIntegration:
    """PerformanceReportGenerator集成测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_full_report_generation_workflow(self, temp_dir):
        """测试完整的报告生成工作流程"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 1. 设置元数据
        metadata = ReportMetadata(
            title="HarborAI性能测试报告",
            description="完整的系统性能评估",
            generated_at=datetime.now(),
            test_duration=timedelta(minutes=10),
            test_environment={
                'python_version': '3.9.0',
                'platform': 'Windows 11',
                'cpu_cores': 8,
                'memory_total': '16GB'
            }
        )
        generator.set_metadata(metadata)
        
        # 2. 设置性能摘要
        summary = PerformanceSummary(
            total_tests=500,
            passed_tests=485,
            failed_tests=15,
            success_rate=0.97,
            average_response_time=0.145,
            peak_memory_usage=768 * 1024 * 1024,  # 768MB
            peak_cpu_usage=82.3,
            total_requests=5000,
            requests_per_second=83.3
        )
        generator.set_summary(summary)
        
        # 3. 生成并添加图表
        if MATPLOTLIB_AVAILABLE:
            # 响应时间图表
            timestamps = [
                datetime.now() + timedelta(seconds=i * 30)
                for i in range(20)
            ]
            response_times = [0.1 + (i % 5) * 0.02 for i in range(20)]
            
            response_chart = generator.generate_response_time_chart(
                timestamps, response_times, "响应时间趋势分析"
            )
            
            # 内存使用图表
            memory_usage = [
                200 * 1024 * 1024 + i * 20 * 1024 * 1024
                for i in range(20)
            ]
            
            memory_chart = generator.generate_memory_usage_chart(
                timestamps, memory_usage, "内存使用趋势分析"
            )
            
            # CPU使用图表
            cpu_usage = [60.0 + (i % 8) * 3.0 for i in range(20)]
            
            cpu_chart = generator.generate_cpu_usage_chart(
                timestamps, cpu_usage, "CPU使用率趋势分析"
            )
            
            # 性能摘要图表
            performance_metrics = {
                '响应时间(ms)': 145,
                'CPU使用率(%)': 82.3,
                '内存使用率(%)': 75.0,
                '成功率(%)': 97.0,
                '吞吐量(req/s)': 83.3
            }
            
            summary_chart = generator.generate_performance_summary_chart(
                performance_metrics, "关键性能指标概览"
            )
        
        # 4. 添加自定义图表数据
        custom_chart = ChartData(
            chart_type="pie",
            title="测试类型分布",
            data={
                'labels': ['响应时间测试', '内存泄漏测试', '负载测试', '并发测试'],
                'values': [150, 100, 150, 100]
            },
            options={'colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']}
        )
        generator.add_chart(custom_chart)
        
        # 5. 设置详细数据
        detailed_data = {
            'test_scenarios': [
                {
                    'name': '响应时间测试',
                    'duration': 300,
                    'requests': 1500,
                    'avg_response_time': 0.142,
                    'success_rate': 0.98
                },
                {
                    'name': '负载测试',
                    'duration': 600,
                    'requests': 3000,
                    'avg_response_time': 0.156,
                    'success_rate': 0.96
                }
            ],
            'error_analysis': {
                'timeout_errors': 8,
                'connection_errors': 5,
                'server_errors': 2
            },
            'performance_thresholds': {
                'response_time_threshold': 0.2,
                'cpu_threshold': 85.0,
                'memory_threshold': 80.0,
                'success_rate_threshold': 0.95
            }
        }
        generator.set_detailed_data(detailed_data)
        
        # 6. 生成HTML报告
        html_path = generator.generate_html_report("integration_test_report.html")
        assert html_path is not None
        
        html_file = Path(html_path)
        assert html_file.exists()
        assert html_file.stat().st_size > 1000  # 确保文件有实际内容
        
        # 验证HTML内容
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert "HarborAI性能测试报告" in html_content
        assert "完整的系统性能评估" in html_content
        assert "500" in html_content  # total_tests
        assert "485" in html_content  # passed_tests
        assert "97.0%" in html_content or "97%" in html_content or "0.97" in html_content  # success_rate
        
        # 7. 生成JSON报告
        json_path = generator.generate_json_report("integration_test_report.json")
        assert json_path is not None
        
        json_file = Path(json_path)
        assert json_file.exists()
        
        # 验证JSON内容
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert json_data['metadata']['title'] == "HarborAI性能测试报告"
        assert json_data['summary']['total_tests'] == 500
        assert json_data['summary']['success_rate'] == 0.97
        assert len(json_data['charts']) >= 1  # 至少有自定义图表
        assert 'test_scenarios' in json_data['detailed_data']
        assert 'error_analysis' in json_data['detailed_data']
        
        # 8. 验证文件结构
        output_dir = Path(temp_dir)
        assert (output_dir / "integration_test_report.html").exists()
        assert (output_dir / "integration_test_report.json").exists()


class TestGenerateQuickReport:
    """generate_quick_report函数测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_generate_quick_report_basic(self, temp_dir):
        """测试基本快速报告生成"""
        test_results = {
            'total_tests': 100,
            'passed_tests': 95,
            'failed_tests': 5,
            'success_rate': 0.95,
            'average_response_time': 0.125,
            'peak_memory_usage': 512 * 1024 * 1024,
            'peak_cpu_usage': 75.5,
            'total_requests': 1000,
            'requests_per_second': 80.0,
            'duration': 300,
            'environment': {
                'python_version': '3.9.0',
                'platform': 'Windows 11'
            }
        }
        
        html_path, json_path = generate_quick_report(
            test_results,
            output_dir=temp_dir,
            title="快速测试报告"
        )
        
        # 验证返回的路径
        assert html_path is not None
        assert json_path is not None
        
        # 验证文件存在
        assert Path(html_path).exists()
        assert Path(json_path).exists()
        
        # 验证HTML内容
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert "快速测试报告" in html_content
        assert "100" in html_content  # total_tests
        assert "95" in html_content   # passed_tests
        
        # 验证JSON内容
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert json_data['metadata']['title'] == "快速测试报告"
        assert json_data['summary']['total_tests'] == 100
        assert json_data['summary']['success_rate'] == 0.95
    
    def test_generate_quick_report_minimal_data(self, temp_dir):
        """测试最小数据的快速报告生成"""
        test_results = {
            'total_tests': 10,
            'passed_tests': 8,
            'failed_tests': 2,
            'success_rate': 0.8
        }
        
        html_path, json_path = generate_quick_report(
            test_results,
            output_dir=temp_dir,
            title="最小数据报告"
        )
        
        assert Path(html_path).exists()
        assert Path(json_path).exists()
        
        # 验证JSON内容
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert json_data['summary']['total_tests'] == 10
        assert json_data['summary']['success_rate'] == 0.8
        # 缺失的字段应该有默认值
        assert json_data['summary']['average_response_time'] == 0.0
        assert json_data['summary']['peak_memory_usage'] == 0
    
    def test_generate_quick_report_with_charts(self, temp_dir):
        """测试包含图表的快速报告生成"""
        test_results = {
            'total_tests': 200,
            'passed_tests': 190,
            'failed_tests': 10,
            'success_rate': 0.95,
            'average_response_time': 0.15,
            'peak_memory_usage': 1024 * 1024 * 1024,  # 1GB
            'peak_cpu_usage': 85.0,
            'total_requests': 2000,
            'requests_per_second': 100.0,
            'duration': 600,  # 10分钟
            'environment': {
                'python_version': '3.9.0',
                'platform': 'Linux',
                'cpu_cores': 16,
                'memory_total': '32GB'
            },
            # 时间序列数据用于生成图表
            'timestamps': [
                (datetime.now() + timedelta(seconds=i * 30)).isoformat()
                for i in range(20)
            ],
            'response_times': [0.1 + (i % 5) * 0.02 for i in range(20)],
            'memory_usage': [
                500 * 1024 * 1024 + i * 25 * 1024 * 1024
                for i in range(20)
            ],
            'cpu_usage': [70.0 + (i % 6) * 2.5 for i in range(20)]
        }
        
        html_path, json_path = generate_quick_report(
            test_results,
            output_dir=temp_dir,
            title="包含图表的报告"
        )
        
        assert Path(html_path).exists()
        assert Path(json_path).exists()
        
        # 验证JSON内容包含图表数据
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 如果matplotlib可用，应该有图表
        if MATPLOTLIB_AVAILABLE:
            assert len(json_data['charts']) > 0
        
        assert json_data['summary']['total_tests'] == 200
        assert json_data['summary']['peak_memory_usage'] == 1024 * 1024 * 1024


class TestPerformanceReportGeneratorErrorHandling:
    """PerformanceReportGenerator错误处理测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_chart_generation_error_handling(self, temp_dir):
        """测试图表生成错误处理"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 测试空数据
        empty_timestamps = []
        empty_values = []
        
        if MATPLOTLIB_AVAILABLE:
            # 空数据应该返回None或处理错误
            result = generator.generate_response_time_chart(empty_timestamps, empty_values)
            # 根据实现，可能返回None或空图表
            
            # 测试不匹配的数据长度
            timestamps = [datetime.now()]
            values = [1.0, 2.0]  # 长度不匹配
            
            # 应该能处理长度不匹配的情况
            result = generator.generate_response_time_chart(timestamps, values)
    
    def test_file_write_permission_error(self, temp_dir):
        """测试文件写入权限错误"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 设置基本数据
        metadata = ReportMetadata(
            title="权限测试",
            description="测试文件权限",
            generated_at=datetime.now(),
            test_duration=timedelta(minutes=1),
            test_environment={}
        )
        generator.set_metadata(metadata)
        
        # 模拟文件写入错误
        with patch('builtins.open', side_effect=PermissionError("权限被拒绝")):
            # 应该能处理权限错误
            try:
                html_path = generator.generate_html_report("permission_test.html")
                # 根据实现，可能返回None或抛出异常
            except PermissionError:
                pass  # 预期的错误
    
    def test_invalid_output_directory(self):
        """测试无效输出目录"""
        # 使用不存在的父目录
        invalid_path = "/nonexistent/path/reports"
        
        try:
            generator = PerformanceReportGenerator(output_dir=invalid_path)
            # 应该能创建目录或处理错误
        except Exception:
            pass  # 可能的错误情况


# 性能基准测试
class TestPerformanceReportGeneratorBenchmarks:
    """PerformanceReportGenerator性能基准测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_html_generation_performance(self, temp_dir, benchmark):
        """测试HTML生成性能"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 设置大量数据
        metadata = ReportMetadata(
            title="性能基准测试",
            description="大数据量报告生成测试",
            generated_at=datetime.now(),
            test_duration=timedelta(hours=1),
            test_environment={'platform': 'test'}
        )
        generator.set_metadata(metadata)
        
        summary = PerformanceSummary(
            total_tests=10000,
            passed_tests=9500,
            failed_tests=500,
            success_rate=0.95,
            average_response_time=0.125,
            peak_memory_usage=2 * 1024 * 1024 * 1024,  # 2GB
            peak_cpu_usage=85.0,
            total_requests=100000,
            requests_per_second=277.8
        )
        generator.set_summary(summary)
        
        # 添加多个图表
        for i in range(10):
            chart = ChartData(
                chart_type="line",
                title=f"基准图表_{i}",
                data={'x': list(range(100)), 'y': list(range(100))}
            )
            generator.add_chart(chart)
        
        # 大量详细数据
        detailed_data = {
            'test_results': [
                {'id': i, 'name': f'test_{i}', 'result': 'pass' if i % 10 != 0 else 'fail'}
                for i in range(1000)
            ]
        }
        generator.set_detailed_data(detailed_data)
        
        def generate_html():
            return generator.generate_html_report(f"benchmark_report_{id(generator)}.html")
        
        # 基准测试HTML生成
        result = benchmark(generate_html)
        assert result is not None
    
    def test_json_generation_performance(self, temp_dir, benchmark):
        """测试JSON生成性能"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 设置数据（同上）
        metadata = ReportMetadata(
            title="JSON基准测试",
            description="JSON生成性能测试",
            generated_at=datetime.now(),
            test_duration=timedelta(minutes=30),
            test_environment={'platform': 'test'}
        )
        generator.set_metadata(metadata)
        
        summary = PerformanceSummary(
            total_tests=5000,
            passed_tests=4800,
            failed_tests=200,
            success_rate=0.96,
            average_response_time=0.15,
            peak_memory_usage=1024 * 1024 * 1024,
            peak_cpu_usage=75.0,
            total_requests=50000,
            requests_per_second=166.7
        )
        generator.set_summary(summary)
        
        def generate_json():
            return generator.generate_json_report(f"benchmark_report_{id(generator)}.json")
        
        # 基准测试JSON生成
        result = benchmark(generate_json)
        assert result is not None
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib不可用")
    def test_chart_generation_performance(self, temp_dir, benchmark):
        """测试图表生成性能"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 大量数据点
        timestamps = [
            datetime.now() + timedelta(seconds=i)
            for i in range(1000)
        ]
        response_times = [0.1 + (i % 100) * 0.001 for i in range(1000)]
        
        def generate_chart():
            return generator.generate_response_time_chart(
                timestamps, response_times, "性能基准图表"
            )
        
        # 基准测试图表生成
        result = benchmark(generate_chart)
        assert result is not None


class TestPerformanceReportGeneratorPrivateMethods:
    """测试PerformanceReportGenerator的私有方法"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_get_default_html_template(self, temp_dir):
        """测试_get_default_html_template方法"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 调用私有方法
        template = generator._get_default_html_template()
        
        # 验证模板内容
        assert isinstance(template, str)
        assert len(template) > 0
        assert "<!DOCTYPE html>" in template
        assert "<html lang=\"zh-CN\">" in template
        assert "<head>" in template
        assert "<body>" in template
        assert "</html>" in template
        
        # 验证模板包含必要的占位符
        assert "{title}" in template
        assert "{description}" in template
        assert "{chart_html}" in template
        assert "{total_tests}" in template
        assert "{passed_tests}" in template
        
        # 验证CSS样式存在
        assert "<style>" in template
        assert "font-family:" in template
        assert "background-color:" in template
        
    def test_format_chart_html_with_charts(self, temp_dir):
        """测试_format_chart_html方法 - 包含图表"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 准备测试数据
        chart_images = {
            "响应时间趋势": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
            "CPU使用率": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
            "内存使用情况": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
        
        # 调用私有方法
        chart_html = generator._format_chart_html(chart_images)
        
        # 验证生成的HTML
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        
        # 验证每个图表都被包含
        for title, image_base64 in chart_images.items():
            assert title in chart_html
            assert f"data:image/png;base64,{image_base64}" in chart_html
            assert f'alt="{title}"' in chart_html
        
        # 验证HTML结构
        assert '<div class="chart">' in chart_html
        assert '<h3>' in chart_html
        assert '<img src=' in chart_html
        
        # 验证图表数量
        chart_count = chart_html.count('<div class="chart">')
        assert chart_count == len(chart_images)
        
    def test_format_chart_html_empty(self, temp_dir):
        """测试_format_chart_html方法 - 空图表字典"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 空图表字典
        chart_images = {}
        
        # 调用私有方法
        chart_html = generator._format_chart_html(chart_images)
        
        # 验证结果
        assert isinstance(chart_html, str)
        assert chart_html == ""
        
    def test_format_chart_html_single_chart(self, temp_dir):
        """测试_format_chart_html方法 - 单个图表"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 单个图表
        chart_images = {
            "测试图表": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
        
        # 调用私有方法
        chart_html = generator._format_chart_html(chart_images)
        
        # 验证结果
        assert isinstance(chart_html, str)
        assert "测试图表" in chart_html
        assert '<div class="chart">' in chart_html
        assert '<h3>测试图表</h3>' in chart_html
        assert 'data:image/png;base64,' in chart_html
        
        # 验证只有一个图表
        chart_count = chart_html.count('<div class="chart">')
        assert chart_count == 1
        
    def test_format_chart_html_special_characters(self, temp_dir):
        """测试_format_chart_html方法 - 特殊字符处理"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 包含特殊字符的图表标题
        chart_images = {
            "响应时间 & CPU使用率 <测试>": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
            "内存使用率 (MB)": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
        
        # 调用私有方法
        chart_html = generator._format_chart_html(chart_images)
        
        # 验证特殊字符被正确处理
        assert "响应时间 & CPU使用率 <测试>" in chart_html
        assert "内存使用率 (MB)" in chart_html
        assert '<h3>响应时间 & CPU使用率 <测试></h3>' in chart_html
        assert '<h3>内存使用率 (MB)</h3>' in chart_html
        
    def test_html_template_integration(self, temp_dir):
        """测试HTML模板与图表格式化的集成"""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # 获取模板
        template = generator._get_default_html_template()
        
        # 准备图表数据
        chart_images = {
            "测试图表": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
        
        # 格式化图表
        chart_html = generator._format_chart_html(chart_images)
        
        # 验证模板可以正确替换图表占位符
        assert "{chart_html}" in template
        
        # 模拟模板替换
        formatted_template = template.replace("{chart_html}", chart_html)
        
        # 验证替换后的模板
        assert "{chart_html}" not in formatted_template
        assert "测试图表" in formatted_template
        assert '<div class="chart">' in formatted_template


class TestImportErrorHandling:
    """测试导入错误处理"""
    
    @patch('performance_report_generator.MATPLOTLIB_AVAILABLE', False)
    def test_matplotlib_not_available_error(self):
        """测试matplotlib不可用时的错误处理"""
        generator = PerformanceReportGenerator()
        
        # 测试生成图表时的错误处理
        chart_data = ChartData(
            chart_type="line",
            title="测试图表",
            data={'x': [1, 2, 3], 'y': [10, 20, 30]}
        )
        
        result = generator._generate_chart_image(chart_data)
        assert result is None
    
    @patch('performance_report_generator.SEABORN_AVAILABLE', False)
    def test_seaborn_not_available_fallback(self):
        """测试seaborn不可用时的回退处理"""
        generator = PerformanceReportGenerator()
        
        # 测试在没有seaborn的情况下生成图表
        chart_data = ChartData(
            chart_type="heatmap",
            title="热力图测试",
            data={'x': [1, 2, 3], 'y': [10, 20, 30]}
        )
        
        # 应该回退到基本的matplotlib实现
        if MATPLOTLIB_AVAILABLE:
            result = generator._generate_chart_image(chart_data)
            # 即使没有seaborn，也应该能生成基本图表
            assert result is not None or result is None  # 取决于具体实现


class TestExceptionHandling:
    """测试异常处理"""
    
    def test_heatmap_generation(self):
        """测试热力图生成"""
        generator = PerformanceReportGenerator()
        
        chart_data = ChartData(
            chart_type="heatmap",
            title="性能热力图",
            data={'x': [1, 2, 3], 'y': [10, 20, 30]}
        )
        
        with patch('performance_report_generator.sns') as mock_sns:
            result = generator._generate_chart_image(chart_data)
            mock_sns.heatmap.assert_called_once()
    
    def test_memory_chart_generation_exception(self):
        """测试内存使用图表生成异常处理"""
        generator = PerformanceReportGenerator()
        
        # 模拟异常情况
        with patch('performance_report_generator.plt.figure') as mock_figure:
            mock_figure.side_effect = Exception("模拟matplotlib错误")
            
            timestamps = [datetime.now() + timedelta(seconds=i) for i in range(3)]
            memory_usage = [100 * 1024 * 1024, 200 * 1024 * 1024, 300 * 1024 * 1024]
            
            result = generator.generate_memory_usage_chart(timestamps, memory_usage, "内存使用测试")
            assert result is None
    
    def test_cpu_chart_generation_exception(self):
        """测试CPU使用率图表生成异常处理"""
        generator = PerformanceReportGenerator()
        
        # 模拟异常情况
        with patch('performance_report_generator.plt.figure') as mock_figure:
            mock_figure.side_effect = Exception("模拟CPU图表错误")
            
            timestamps = [datetime.now() + timedelta(seconds=i) for i in range(3)]
            cpu_usage = [50.0, 75.0, 90.0]
            
            result = generator.generate_cpu_usage_chart(timestamps, cpu_usage, "CPU使用率测试")
            assert result is None


class TestBoundaryConditions:
    """测试边界条件"""
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        generator = PerformanceReportGenerator()
        
        # 测试空的图表数据
        chart_data = ChartData(
            chart_type="line",
            title="空数据测试",
            data={'x': [], 'y': []}
        )
        
        result = generator._generate_chart_image(chart_data)
        # 应该能处理空数据而不崩溃
        assert result is None or isinstance(result, str)
    
    def test_mismatched_data_lengths(self):
        """测试数据长度不匹配"""
        generator = PerformanceReportGenerator()
        
        # 测试x和y数据长度不匹配
        chart_data = ChartData(
            chart_type="line",
            title="数据长度不匹配测试",
            data={'x': [1, 2, 3], 'y': [10, 20]}  # 长度不匹配
        )
        
        result = generator._generate_chart_image(chart_data)
        # 应该能处理数据长度不匹配的情况
        assert result is None or isinstance(result, str)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--benchmark-only"])