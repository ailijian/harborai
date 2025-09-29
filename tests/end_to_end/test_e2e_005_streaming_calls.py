#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI E2E-005 流式调用测试用例

测试目标：
1. 验证流式调用的chunk结构与OpenAI一致
2. 测试所有7个模型的流式输出功能
3. 验证delta.content逐步输出和完整性
4. 使用.env文件中配置的API密钥和端点

测试范围：
- DeepSeek: deepseek-chat, deepseek-reasoner
- 豆包: doubao-pro-4k, doubao-pro-32k, doubao-pro-128k
- 文心一言: ernie-4.0-8k, ernie-3.5-8k
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入HarborAI客户端
try:
    from harborai import HarborAI
    from harborai.core.models import is_reasoning_model, get_model_capabilities
except ImportError as e:
    print(f"❌ 导入HarborAI失败: {e}")
    print("请确保HarborAI包已正确安装")
    sys.exit(1)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

class StreamingTestCase:
    """流式调用测试用例类"""
    
    def __init__(self):
        """初始化测试用例"""
        self.client = None
        self.test_results = []
        self.models_to_test = [
            # DeepSeek模型
            "deepseek-chat",
            "deepseek-reasoner",
            # 豆包模型
            "doubao-1-5-pro-32k-character-250715",
            "doubao-seed-1-6-250615",
            # 文心一言模型
            "ernie-3.5-8k",
            "ernie-4.0-turbo-8k",
            "ernie-x1-turbo-32k"
        ]
        
    def setup_client(self) -> bool:
        """设置HarborAI客户端"""
        try:
            # 检查必要的环境变量
            required_vars = [
                "DEEPSEEK_API_KEY", "DOUBAO_API_KEY", "WENXIN_API_KEY"
            ]
            
            missing_vars = []
            for var in required_vars:
                value = os.getenv(var)
                if not value:
                    missing_vars.append(var)
                else:
                    print(f"✅ {var}: {value[:10]}...{value[-4:] if len(value) > 14 else value}")
            
            if missing_vars:
                print(f"⚠️  缺少环境变量: {', '.join(missing_vars)}")
                print("继续测试，但可能会有API密钥验证失败...")
            
            # 初始化HarborAI客户端
            self.client = HarborAI()
            
            # 打印调试信息
            print(f"✅ HarborAI客户端初始化成功")
            available_models = self.client.get_available_models()
            print(f"📋 可用模型: {available_models}")
            print(f"🔌 已加载插件: {list(self.client.client_manager.plugins.keys())}")
            
            # 检查插件配置
            from harborai.config.settings import get_settings
            settings = get_settings()
            
            for plugin_name in ["deepseek", "doubao", "wenxin"]:
                plugin_config = settings.get_plugin_config(plugin_name)
                print(f"🔧 {plugin_name} 插件配置: {plugin_config}")
                # 检查API密钥是否存在
                api_key = plugin_config.get('api_key')
                print(f"插件 {plugin_name} API密钥存在: {bool(api_key)}, 长度: {len(api_key) if api_key else 0}")
            
            # 尝试手动实例化豆包插件以查看详细错误
            try:
                from harborai.core.plugins.doubao_plugin import DoubaoPlugin
                doubao_config = {
                    'api_key': os.getenv('DOUBAO_API_KEY'),
                    'base_url': os.getenv('DOUBAO_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
                }
                print(f"🔍 尝试手动实例化豆包插件，配置: {doubao_config}")
                doubao_plugin = DoubaoPlugin('doubao', **doubao_config)
                print(f"✅ 豆包插件手动实例化成功")
            except Exception as e:
                print(f"❌ 豆包插件手动实例化失败: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
            
            return True
            
        except Exception as e:
            print(f"❌ 客户端初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_chunk_structure(self, chunk) -> bool:
        """验证chunk结构是否符合OpenAI标准"""
        try:
            # 如果chunk是对象，尝试转换为字典
            if hasattr(chunk, '__dict__'):
                chunk_dict = chunk.__dict__ if hasattr(chunk, '__dict__') else {}
                # 尝试使用属性访问
                chunk_data = {
                    'id': getattr(chunk, 'id', None),
                    'object': getattr(chunk, 'object', None),
                    'created': getattr(chunk, 'created', None),
                    'model': getattr(chunk, 'model', None),
                    'choices': getattr(chunk, 'choices', None)
                }
            elif isinstance(chunk, dict):
                chunk_data = chunk
            else:
                print(f"❌ Chunk类型不支持: {type(chunk)}")
                return False
            
            required_fields = ["id", "object", "created", "model", "choices"]
            
            # 检查必需字段
            for field in required_fields:
                if field not in chunk_data or chunk_data[field] is None:
                    print(f"❌ Chunk缺少必需字段: {field}")
                    return False
            
            # 验证object字段
            if chunk_data["object"] != "chat.completion.chunk":
                print(f"❌ object字段值错误: {chunk_data['object']}，期望: chat.completion.chunk")
                return False
            
            # 验证choices结构
            choices = chunk_data["choices"]
            if not isinstance(choices, list) or len(choices) == 0:
                print(f"❌ choices字段格式错误")
                return False
            
            choice = choices[0]
            # 检查choice是否有delta字段
            if hasattr(choice, 'delta'):
                return True
            elif isinstance(choice, dict) and "delta" in choice:
                return True
            else:
                print(f"❌ choice缺少delta字段")
                return False
                
        except Exception as e:
            print(f"❌ Chunk结构验证出错: {e}")
            return False
            
        return True
    
    def test_streaming_call(self, model: str) -> Dict[str, Any]:
        """测试单个模型的流式调用"""
        print(f"\n🔄 测试模型: {model}")
        
        test_result = {
            "model": model,
            "success": False,
            "chunks_received": 0,
            "total_content": "",
            "first_chunk_time": None,
            "total_time": None,
            "error": None,
            "chunk_structure_valid": True,
            "is_reasoning_model": is_reasoning_model(model),
            "test_type": "streaming"  # 所有模型都使用流式输出测试，包括推理模型
        }
        
        try:
            start_time = time.time()
            
            # 构造测试消息
            messages = [
                {
                    "role": "user",
                    "content": "请简要介绍一下人工智能的发展历程，大约100字左右。"
                }
            ]
            
            # 所有模型都使用流式调用，包括推理模型
            if is_reasoning_model(model):
                print(f"🧠 检测到推理模型 {model}，使用流式调用测试推理过程和结果输出")
                # 推理模型使用流式调用，不包含temperature参数（避免警告）
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=200
                )
            else:
                print(f"💬 常规模型 {model}，使用流式调用")
                # 常规模型使用流式调用，包含temperature参数
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=200,
                    temperature=0.7
                )
            
            # 处理流式响应（统一处理逻辑）
            reasoning_content_chunks = []
            content_chunks = []
            
            for chunk in stream:
                if test_result["first_chunk_time"] is None:
                    test_result["first_chunk_time"] = time.time() - start_time
                
                test_result["chunks_received"] += 1
                
                # 验证chunk结构
                if not self.validate_chunk_structure(chunk):
                    test_result["chunk_structure_valid"] = False
                
                # 提取内容
                try:
                    choices = None
                    if hasattr(chunk, 'choices'):
                        choices = chunk.choices
                    elif isinstance(chunk, dict) and "choices" in chunk:
                        choices = chunk["choices"]
                    
                    if choices and len(choices) > 0:
                        choice = choices[0]
                        delta = None
                        
                        if hasattr(choice, 'delta'):
                            delta = choice.delta
                        elif isinstance(choice, dict) and "delta" in choice:
                            delta = choice["delta"]
                        
                        if delta:
                            # 提取常规内容
                            content = None
                            if hasattr(delta, 'content'):
                                content = delta.content
                            elif isinstance(delta, dict) and "content" in delta:
                                content = delta["content"]
                            
                            if content:
                                test_result["total_content"] += content
                                content_chunks.append(content)
                                print(f"📝 接收内容片段: {content[:50]}{'...' if len(content) > 50 else ''}")
                            
                            # 对于推理模型，提取推理内容
                            if is_reasoning_model(model):
                                reasoning_content = None
                                if hasattr(delta, 'reasoning_content'):
                                    reasoning_content = delta.reasoning_content
                                elif isinstance(delta, dict) and "reasoning_content" in delta:
                                    reasoning_content = delta["reasoning_content"]
                                
                                if reasoning_content:
                                    reasoning_content_chunks.append(reasoning_content)
                                    print(f"🧠 接收推理片段: {reasoning_content[:50]}{'...' if len(reasoning_content) > 50 else ''}")
                                    
                except Exception as e:
                    print(f"⚠️  内容提取出错: {e}")
            
            # 记录推理模型的推理内容统计
            if is_reasoning_model(model) and reasoning_content_chunks:
                total_reasoning_content = "".join(reasoning_content_chunks)
                test_result["total_reasoning_content"] = total_reasoning_content
                test_result["reasoning_chunks_received"] = len(reasoning_content_chunks)
                print(f"🧠 推理内容总长度: {len(total_reasoning_content)}字符，分{len(reasoning_content_chunks)}个chunk")
            
            # 记录内容统计
            if content_chunks:
                test_result["content_chunks_received"] = len(content_chunks)
                print(f"📝 内容总长度: {len(test_result['total_content'])}字符，分{len(content_chunks)}个chunk")
            
            test_result["total_time"] = time.time() - start_time
            test_result["success"] = True
            
            print(f"✅ {model} 测试成功")
            print(f"   - 接收chunks: {test_result['chunks_received']}")
            print(f"   - 首个chunk时间: {test_result['first_chunk_time']:.3f}s")
            print(f"   - 总耗时: {test_result['total_time']:.3f}s")
            print(f"   - 内容长度: {len(test_result['total_content'])}字符")
            print(f"   - Chunk结构有效: {test_result['chunk_structure_valid']}")
            
            # 对于推理模型，显示推理内容统计
            if is_reasoning_model(model) and "total_reasoning_content" in test_result:
                print(f"   - 推理内容长度: {len(test_result['total_reasoning_content'])}字符")
                print(f"   - 推理chunk数量: {test_result.get('reasoning_chunks_received', 0)}个")
                print(f"   - 内容chunk数量: {test_result.get('content_chunks_received', 0)}个")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"❌ {model} 测试失败: {e}")
        
        return test_result
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """运行所有模型的流式调用测试"""
        print("🚀 开始执行E2E-005流式调用测试")
        print(f"📋 待测试模型数量: {len(self.models_to_test)}")
        
        for model in self.models_to_test:
            result = self.test_streaming_call(model)
            self.test_results.append(result)
            
            # 模型间间隔，避免请求过于频繁
            time.sleep(1)
        
        return self.test_results
    
    def generate_test_report(self) -> None:
        """生成测试报告"""
        print("\n" + "="*80)
        print("📊 E2E-005 流式调用测试报告")
        print("="*80)
        
        successful_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]
        
        print(f"\n📈 测试概览:")
        print(f"   - 总测试数: {len(self.test_results)}")
        print(f"   - 成功: {len(successful_tests)}")
        print(f"   - 失败: {len(failed_tests)}")
        print(f"   - 成功率: {len(successful_tests)/len(self.test_results)*100:.1f}%")
        
        if successful_tests:
            print(f"\n✅ 成功的测试:")
            for result in successful_tests:
                print(f"   - {result['model']}: {result['chunks_received']}个chunks, "
                      f"{result['total_time']:.3f}s, {len(result['total_content'])}字符")
        
        if failed_tests:
            print(f"\n❌ 失败的测试:")
            for result in failed_tests:
                print(f"   - {result['model']}: {result['error']}")
        
        # 性能统计
        if successful_tests:
            avg_first_chunk = sum(r["first_chunk_time"] for r in successful_tests) / len(successful_tests)
            avg_total_time = sum(r["total_time"] for r in successful_tests) / len(successful_tests)
            avg_chunks = sum(r["chunks_received"] for r in successful_tests) / len(successful_tests)
            
            print(f"\n📊 性能统计 (成功测试):")
            print(f"   - 平均首个chunk时间: {avg_first_chunk:.3f}s")
            print(f"   - 平均总耗时: {avg_total_time:.3f}s")
            print(f"   - 平均chunk数量: {avg_chunks:.1f}")
        
        # Chunk结构验证
        structure_valid_count = sum(1 for r in successful_tests if r["chunk_structure_valid"])
        print(f"\n🔍 Chunk结构验证:")
        print(f"   - 结构有效的测试: {structure_valid_count}/{len(successful_tests)}")
        
        print("\n" + "="*80)

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(level=logging.DEBUG)
    
    print("🎯 HarborAI E2E-005 流式调用测试")
    print("测试目标: 验证流式调用功能与OpenAI兼容性")
    
    # 创建测试实例
    test_case = StreamingTestCase()
    
    # 设置客户端
    if not test_case.setup_client():
        print("❌ 测试终止: 客户端设置失败")
        return
    
    try:
        # 运行所有测试
        test_case.run_all_tests()
        
        # 生成测试报告
        test_case.generate_test_report()
        
        # 保存测试结果到JSON文件
        results_file = project_root / "tests" / "end_to_end" / "e2e_005_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "test_name": "E2E-005 流式调用测试",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": test_case.test_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 测试结果已保存到: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试执行出错: {e}")
        import traceback
        traceback.print_exc()

def test_streaming_calls():
    """pytest测试函数 - 流式调用测试"""
    # 配置日志
    logging.basicConfig(level=logging.DEBUG)
    
    print("🎯 HarborAI E2E-005 流式调用测试")
    print("测试目标: 验证流式调用功能与OpenAI兼容性")
    
    # 创建测试实例
    test_case = StreamingTestCase()
    
    # 设置客户端
    assert test_case.setup_client(), "客户端设置失败"
    
    # 运行所有测试
    test_case.run_all_tests()
    
    # 生成测试报告
    test_case.generate_test_report()
    
    # 保存测试结果到JSON文件
    results_file = project_root / "tests" / "end_to_end" / "e2e_005_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_name": "E2E-005 流式调用测试",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": test_case.test_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 测试结果已保存到: {results_file}")
    
    # 验证测试结果
    successful_tests = [r for r in test_case.test_results if r["success"]]
    failed_tests = [r for r in test_case.test_results if not r["success"]]
    
    # 确保至少有一些测试成功
    assert len(successful_tests) > 0, f"所有测试都失败了: {[r['error'] for r in failed_tests]}"
    
    print(f"\n✅ 测试完成: {len(successful_tests)}/{len(test_case.test_results)} 成功")

if __name__ == "__main__":
    # 运行同步测试
    main()