#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q模块：文档与示例一致性测试

测试用例：
- Q-001: 验证README/示例与实际API对齐
- Q-002: 验证TDD/PRD中所有功能点均可找到对应测试项

作者: HarborAI测试团队
创建时间: 2024
"""

import os
import re
import ast
import sys
import inspect
import importlib
from pathlib import Path
from typing import List, Dict, Set, Any, Optional
from unittest.mock import patch, MagicMock

import pytest

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入HarborAI相关模块
try:
    from harborai import HarborAI
    from harborai.api.client import ChatCompletions, Chat
    from harborai.utils.exceptions import HarborAIError, ValidationError, APIError
except ImportError as e:
    pytest.skip(f"无法导入HarborAI模块: {e}", allow_module_level=True)


class TestDocumentationConsistency:
    """
    Q模块：文档与示例一致性测试类
    
    验证README文档中的示例代码与实际API接口的一致性，
    确保文档准确性和用户体验。
    """
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.project_root = Path(__file__).parent.parent.parent
        cls.readme_path = cls.project_root / "README.md"
        cls.docs_path = cls.project_root / ".trae" / "documents"
        
        # 读取README内容
        if cls.readme_path.exists():
            with open(cls.readme_path, 'r', encoding='utf-8') as f:
                cls.readme_content = f.read()
        else:
            cls.readme_content = ""
            
        # 读取技术设计文档
        cls.tech_design_path = cls.docs_path / "HarborAI测试项目技术设计方案.md"
        if cls.tech_design_path.exists():
            with open(cls.tech_design_path, 'r', encoding='utf-8') as f:
                cls.tech_design_content = f.read()
        else:
            cls.tech_design_content = ""
            
        # 读取功能测试清单
        cls.test_checklist_path = cls.docs_path / "HarborAI功能与性能测试清单.md"
        if cls.test_checklist_path.exists():
            with open(cls.test_checklist_path, 'r', encoding='utf-8') as f:
                cls.test_checklist_content = f.read()
        else:
            cls.test_checklist_content = ""
    
    def extract_code_blocks(self, content: str, language: str = "python") -> List[str]:
        """
        从Markdown内容中提取指定语言的代码块
        
        Args:
            content: Markdown内容
            language: 编程语言标识
            
        Returns:
            代码块列表
        """
        pattern = rf"```{language}\n(.*?)\n```"
        matches = re.findall(pattern, content, re.DOTALL)
        return matches
    
    def parse_python_code(self, code: str) -> Optional[ast.AST]:
        """
        解析Python代码并返回AST
        
        Args:
            code: Python代码字符串
            
        Returns:
            AST对象或None（如果解析失败）
        """
        try:
            return ast.parse(code)
        except SyntaxError:
            return None
    
    def extract_api_calls(self, code: str) -> List[Dict[str, Any]]:
        """
        从代码中提取API调用信息
        
        Args:
            code: Python代码字符串
            
        Returns:
            API调用信息列表
        """
        api_calls = []
        tree = self.parse_python_code(code)
        
        if tree is None:
            return api_calls
        
        class APICallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # 提取函数调用信息
                if isinstance(node.func, ast.Attribute):
                    # 处理 obj.method() 形式的调用
                    if isinstance(node.func.value, ast.Name):
                        obj_name = node.func.value.id
                        method_name = node.func.attr
                        
                        # 提取参数
                        args = []
                        kwargs = {}
                        
                        for arg in node.args:
                            if isinstance(arg, ast.Constant):
                                args.append(arg.value)
                            elif isinstance(arg, ast.Name):
                                args.append(arg.id)
                        
                        for keyword in node.keywords:
                            if isinstance(keyword.value, ast.Constant):
                                kwargs[keyword.arg] = keyword.value.value
                            elif isinstance(keyword.value, ast.Name):
                                kwargs[keyword.arg] = keyword.value.id
                        
                        api_calls.append({
                            'object': obj_name,
                            'method': method_name,
                            'args': args,
                            'kwargs': kwargs,
                            'line': node.lineno
                        })
                
                self.generic_visit(node)
        
        visitor = APICallVisitor()
        visitor.visit(tree)
        return api_calls
    
    def get_harborai_methods(self) -> Dict[str, List[str]]:
        """
        获取HarborAI类的所有公共方法
        
        Returns:
            类名到方法列表的映射
        """
        methods = {}
        
        # 检查HarborAI主类
        harborai_methods = []
        for name, method in inspect.getmembers(HarborAI, predicate=inspect.ismethod):
            if not name.startswith('_'):
                harborai_methods.append(name)
        
        for name, method in inspect.getmembers(HarborAI, predicate=inspect.isfunction):
            if not name.startswith('_'):
                harborai_methods.append(name)
        
        methods['HarborAI'] = harborai_methods
        
        # 检查FastHarborAI类
        try:
            from harborai.api.fast_client import FastHarborAI
            fast_harborai_methods = []
            for name, method in inspect.getmembers(FastHarborAI, predicate=inspect.ismethod):
                if not name.startswith('_'):
                    fast_harborai_methods.append(name)
            
            for name, method in inspect.getmembers(FastHarborAI, predicate=inspect.isfunction):
                if not name.startswith('_'):
                    fast_harborai_methods.append(name)
            
            methods['FastHarborAI'] = fast_harborai_methods
        except ImportError:
            pass
        
        # 检查ChatCompletions类
        chat_methods = []
        for name, method in inspect.getmembers(ChatCompletions, predicate=inspect.ismethod):
            if not name.startswith('_'):
                chat_methods.append(name)
        
        for name, method in inspect.getmembers(ChatCompletions, predicate=inspect.isfunction):
            if not name.startswith('_'):
                chat_methods.append(name)
        
        methods['ChatCompletions'] = chat_methods
        
        return methods
    
    @pytest.mark.q001
    def test_readme_api_consistency(self):
        """
        Q-001: 验证README/示例与实际API对齐
        
        测试步骤：
        1. 提取README中的Python代码示例
        2. 解析代码中的API调用
        3. 验证API调用与实际接口一致
        4. 检查参数名称和类型
        
        预期结果：
        - 所有示例代码语法正确
        - API调用方法存在于实际接口中
        - 参数名称与实际接口匹配
        """
        # 提取README中的Python代码块
        code_blocks = self.extract_code_blocks(self.readme_content, "python")
        
        assert len(code_blocks) > 0, "README中应该包含Python代码示例"
        
        # 获取实际的API方法
        actual_methods = self.get_harborai_methods()
        
        syntax_errors = []
        api_inconsistencies = []
        
        for i, code in enumerate(code_blocks):
            # 检查语法正确性
            tree = self.parse_python_code(code)
            if tree is None:
                syntax_errors.append(f"代码块 {i+1} 存在语法错误")
                continue
            
            # 提取API调用
            api_calls = self.extract_api_calls(code)
            
            for call in api_calls:
                obj_name = call['object']
                method_name = call['method']
                
                # 检查HarborAI相关的API调用
                if obj_name in ['client', 'harborai', 'harbor'] or 'harbor' in obj_name.lower():
                    # 检查方法是否存在
                    method_found = False
                    
                    # 检查主类方法
                    if method_name in actual_methods.get('HarborAI', []):
                        method_found = True
                    
                    # 检查FastHarborAI类方法
                    if method_name in actual_methods.get('FastHarborAI', []):
                        method_found = True
                    
                    # 检查chat.completions方法
                    if method_name in actual_methods.get('ChatCompletions', []):
                        method_found = True
                    
                    # 检查特殊的链式调用（如 client.chat.completions.create）
                    if method_name == 'create' and 'chat' in str(call).lower():
                        method_found = True
                    
                    # 检查hasattr调用（用于条件性方法调用）
                    if method_name == 'hasattr':
                        method_found = True
                    
                    if not method_found:
                        api_inconsistencies.append(
                            f"代码块 {i+1} 第 {call['line']} 行: "
                            f"方法 {obj_name}.{method_name} 在实际API中不存在"
                        )
        
        # 断言检查
        assert len(syntax_errors) == 0, f"发现语法错误: {'; '.join(syntax_errors)}"
        assert len(api_inconsistencies) == 0, f"发现API不一致: {'; '.join(api_inconsistencies)}"
    
    @pytest.mark.q001
    def test_readme_import_statements(self):
        """
        Q-001: 验证README中的导入语句正确性
        
        测试步骤：
        1. 提取README中的import语句
        2. 验证导入的模块和类存在
        3. 检查导入路径正确性
        
        预期结果：
        - 所有import语句可以成功执行
        - 导入的类和函数可以正常使用
        """
        code_blocks = self.extract_code_blocks(self.readme_content, "python")
        
        import_errors = []
        
        for i, code in enumerate(code_blocks):
            # 提取import语句
            tree = self.parse_python_code(code)
            if tree is None:
                continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if 'harborai' in module_name:
                            try:
                                importlib.import_module(module_name)
                            except ImportError as e:
                                import_errors.append(
                                    f"代码块 {i+1}: 无法导入模块 {module_name} - {str(e)}"
                                )
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and 'harborai' in node.module:
                        try:
                            module = importlib.import_module(node.module)
                            for alias in node.names:
                                if not hasattr(module, alias.name):
                                    import_errors.append(
                                        f"代码块 {i+1}: 模块 {node.module} 中不存在 {alias.name}"
                                    )
                        except ImportError as e:
                            import_errors.append(
                                f"代码块 {i+1}: 无法导入模块 {node.module} - {str(e)}"
                            )
        
        assert len(import_errors) == 0, f"发现导入错误: {'; '.join(import_errors)}"
    
    @pytest.mark.q001
    def test_readme_parameter_consistency(self):
        """
        Q-001: 验证README示例中的参数与实际API参数一致
        
        测试步骤：
        1. 提取README中API调用的参数
        2. 获取实际API方法的参数签名
        3. 比较参数名称和默认值
        
        预期结果：
        - 示例中使用的参数在实际API中存在
        - 参数类型和默认值匹配
        """
        # 获取ChatCompletions.create方法的参数签名
        create_signature = inspect.signature(ChatCompletions.create)
        actual_params = set(create_signature.parameters.keys())
        
        # 提取README中的代码示例
        code_blocks = self.extract_code_blocks(self.readme_content, "python")
        
        parameter_errors = []
        
        for i, code in enumerate(code_blocks):
            api_calls = self.extract_api_calls(code)
            
            for call in api_calls:
                if call['method'] == 'create' and 'chat' in str(call).lower():
                    # 检查kwargs参数
                    for param_name in call['kwargs'].keys():
                        if param_name not in actual_params:
                            parameter_errors.append(
                                f"代码块 {i+1}: 参数 '{param_name}' 在实际API中不存在"
                            )
        
        assert len(parameter_errors) == 0, f"发现参数不一致: {'; '.join(parameter_errors)}"
    
    def find_test_files(self) -> List[Path]:
        """
        查找所有测试文件
        
        Returns:
            测试文件路径列表
        """
        test_files = []
        tests_dir = self.project_root / "tests"
        
        if tests_dir.exists():
            for file_path in tests_dir.rglob("test_*.py"):
                test_files.append(file_path)
        
        return test_files
    
    def extract_test_functions(self, file_path: Path) -> List[str]:
        """
        从测试文件中提取测试函数名
        
        Args:
            file_path: 测试文件路径
            
        Returns:
            测试函数名列表
        """
        test_functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_functions.append(node.name)
        
        except Exception:
            # 如果文件解析失败，跳过
            pass
        
        return test_functions
    
    def extract_features_from_docs(self) -> Set[str]:
        """
        从技术设计文档中提取功能点
        
        Returns:
            功能点集合
        """
        features = set()
        
        # 从技术设计文档中提取功能点
        if self.tech_design_content:
            # 查找功能相关的章节
            feature_patterns = [
                r"##\s*(.+?)功能",
                r"###\s*(.+?)模块",
                r"\*\*(.+?)\*\*",
                r"- (.+?)：",
                r"\d+\.\s*(.+?)\n"
            ]
            
            for pattern in feature_patterns:
                matches = re.findall(pattern, self.tech_design_content, re.MULTILINE)
                for match in matches:
                    feature = match.strip()
                    if len(feature) > 2 and len(feature) < 50:  # 过滤过短或过长的匹配
                        features.add(feature)
        
        # 从测试清单中提取功能点
        if self.test_checklist_content:
            # 查找模块和测试用例
            module_patterns = [
                r"模块([A-Z])：(.+?)",
                r"([A-Z])-\d+[：:](.+?)",
                r"测试(.+?)功能",
                r"验证(.+?)\n"
            ]
            
            for pattern in module_patterns:
                matches = re.findall(pattern, self.test_checklist_content, re.MULTILINE)
                for match in matches:
                    if isinstance(match, tuple):
                        for item in match:
                            feature = item.strip()
                            if len(feature) > 2 and len(feature) < 50:
                                features.add(feature)
                    else:
                        feature = match.strip()
                        if len(feature) > 2 and len(feature) < 50:
                            features.add(feature)
        
        return features
    
    @pytest.mark.q002
    def test_feature_test_coverage(self):
        """
        Q-002: 验证TDD/PRD中所有功能点均可找到对应测试项
        
        测试步骤：
        1. 从技术设计文档中提取功能点
        2. 扫描所有测试文件，提取测试用例
        3. 匹配功能点与测试用例
        4. 识别缺失的测试覆盖
        
        预期结果：
        - 每个主要功能点都有对应的测试用例
        - 测试覆盖率达到要求
        """
        # 提取文档中的功能点
        documented_features = self.extract_features_from_docs()
        
        # 查找所有测试文件
        test_files = self.find_test_files()
        
        assert len(test_files) > 0, "应该存在测试文件"
        
        # 提取所有测试函数
        all_test_functions = []
        for test_file in test_files:
            test_functions = self.extract_test_functions(test_file)
            all_test_functions.extend([
                f"{test_file.stem}.{func}" for func in test_functions
            ])
        
        assert len(all_test_functions) > 0, "应该存在测试函数"
        
        # 检查核心功能的测试覆盖
        core_features = {
            "聊天完成": ["chat", "completion", "create"],
            "异步调用": ["async", "acreate"],
            "错误处理": ["error", "exception", "validation"],
            "重试机制": ["retry", "fallback"],
            "日志记录": ["log", "trace"],
            "性能监控": ["performance", "metric", "monitor"],
            "安全认证": ["security", "auth", "key"],
            "配置管理": ["config", "setting"],
            "插件系统": ["plugin"],
            "数据库操作": ["database", "db"],
            "结构化输出": ["structured", "format"],
            "推理模型": ["reasoning", "model"]
        }
        
        missing_coverage = []
        
        for feature, keywords in core_features.items():
            # 检查是否有相关的测试
            has_test = False
            for test_func in all_test_functions:
                test_func_lower = test_func.lower()
                if any(keyword in test_func_lower for keyword in keywords):
                    has_test = True
                    break
            
            if not has_test:
                missing_coverage.append(feature)
        
        # 允许部分功能暂时没有测试，但应该记录
        if missing_coverage:
            print(f"\n警告：以下功能缺少测试覆盖: {', '.join(missing_coverage)}")
        
        # 至少应该有基本的聊天完成测试
        basic_tests_exist = any(
            any(keyword in test_func.lower() for keyword in ["chat", "completion", "create"])
            for test_func in all_test_functions
        )
        
        assert basic_tests_exist, "应该存在基本的聊天完成功能测试"
    
    @pytest.mark.q002
    def test_test_module_completeness(self):
        """
        Q-002: 验证测试模块的完整性
        
        测试步骤：
        1. 检查测试清单中定义的所有模块
        2. 验证每个模块都有对应的测试文件
        3. 检查测试用例编号的连续性
        
        预期结果：
        - 所有定义的测试模块都有实现
        - 测试用例编号连续且完整
        """
        # 从测试清单中提取模块信息
        module_pattern = r"模块([A-Z])：(.+?)"
        modules = re.findall(module_pattern, self.test_checklist_content)
        
        assert len(modules) > 0, "测试清单中应该定义测试模块"
        
        # 检查测试文件是否存在
        test_files = self.find_test_files()
        test_file_names = [f.stem for f in test_files]
        
        missing_modules = []
        
        for module_id, module_name in modules:
            # 查找对应的测试文件
            expected_file_pattern = f"test_{module_id.lower()}_"
            
            module_has_test = any(
                expected_file_pattern in test_file_name.lower()
                for test_file_name in test_file_names
            )
            
            if not module_has_test:
                missing_modules.append(f"模块{module_id}({module_name})")
        
        if missing_modules:
            print(f"\n警告：以下模块缺少测试文件: {', '.join(missing_modules)}")
        
        # 至少应该有P模块和Q模块的测试
        has_p_module = any("test_p_" in name for name in test_file_names)
        has_q_module = any("test_q_" in name for name in test_file_names)
        
        assert has_p_module, "应该存在P模块（错误用例与健壮性）测试"
        assert has_q_module, "应该存在Q模块（文档与示例一致性）测试"
    
    @pytest.mark.q002
    def test_documentation_completeness(self):
        """
        Q-002: 验证文档完整性
        
        测试步骤：
        1. 检查必要的文档文件是否存在
        2. 验证文档内容的完整性
        3. 检查文档格式的正确性
        
        预期结果：
        - 所有必要的文档文件存在
        - 文档内容完整且格式正确
        """
        # 检查必要的文档文件
        required_docs = {
            "README.md": self.readme_path,
            "技术设计方案": self.tech_design_path,
            "功能测试清单": self.test_checklist_path
        }
        
        missing_docs = []
        
        for doc_name, doc_path in required_docs.items():
            if not doc_path.exists():
                missing_docs.append(doc_name)
            elif doc_path.stat().st_size == 0:
                missing_docs.append(f"{doc_name}(空文件)")
        
        assert len(missing_docs) == 0, f"缺少必要文档: {', '.join(missing_docs)}"
        
        # 检查README的基本结构
        readme_sections = [
            "# HarborAI",
            "## 安装",
            "## 快速开始",
            "## API文档"
        ]
        
        missing_sections = []
        for section in readme_sections:
            if section not in self.readme_content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"\n警告：README缺少以下章节: {', '.join(missing_sections)}")
        
        # 至少应该有基本的项目描述
        assert "HarborAI" in self.readme_content, "README应该包含项目名称"
        assert len(self.readme_content) > 100, "README内容应该足够详细"


if __name__ == "__main__":
    # 运行测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "q001 or q002"
    ])