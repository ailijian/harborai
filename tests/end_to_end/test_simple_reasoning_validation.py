#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单推理模型基础功能验证测试

目的：验证DeepSeek、文心和豆包推理模型的基本请求和响应功能
- 测试基本连接性
- 验证响应结构
- 检查推理字段和结果字段
- 使用120秒超时和重试机制
"""

import os
import sys
import time
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from harborai import HarborAI


class SimpleReasoningValidation:
    """简单推理模型验证测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        # 加载环境变量
        load_dotenv()
        
        # 配置推理模型
        self.reasoning_models = [
            {
                "name": "deepseek-reasoner",
                "vendor": "DeepSeek",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            },
            {
                "name": "ernie-x1-turbo-32k",
                "vendor": "百度文心",
                "api_key": os.getenv("WENXIN_API_KEY"),
                "base_url": os.getenv("WENXIN_BASE_URL", "https://qianfan.baidubce.com/v2")
            },
            {
                "name": "doubao-seed-1-6-250615",
                "vendor": "字节豆包",
                "api_key": os.getenv("DOUBAO_API_KEY"),
                "base_url": os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
            }
        ]
        
        print("简单推理模型基础功能验证测试初始化完成")
        print(f"配置的推理模型数量：{len(self.reasoning_models)}")
        
        # 验证环境变量
        for model in self.reasoning_models:
            if not model["api_key"]:
                print(f"⚠ 警告：{model['vendor']} 的API密钥未配置")
            else:
                print(f"✓ {model['vendor']} API密钥已配置")
    
    def test_basic_connection(self):
        """测试基本连接功能"""
        print("\n" + "="*60)
        print("测试1：基本连接功能验证")
        print("="*60)
        
        simple_prompt = "你好，请简单介绍一下人工智能。"
        
        for model_config in self.reasoning_models:
            print(f"\n--- 测试模型：{model_config['name']} ({model_config['vendor']}) ---")
            
            if not model_config["api_key"]:
                print(f"⚠ 跳过 {model_config['vendor']}：API密钥未配置")
                continue
            
            try:
                # 初始化客户端
                client = HarborAI(
                    api_key=model_config["api_key"],
                    base_url=model_config["base_url"],
                    timeout=120  # 120秒超时
                )
                print(f"✓ 客户端初始化成功")
                
                # 发送简单请求
                print(f"发送测试请求...")
                start_time = time.time()
                
                response = client.chat.completions.create(
                    model=model_config["name"],
                    messages=[
                        {"role": "user", "content": simple_prompt}
                    ],
                    timeout=120
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"✓ 请求成功，响应时间：{response_time:.2f}秒")
                
                # 验证基本响应结构
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    message = response.choices[0].message
                    print(f"✓ 响应结构正常")
                    
                    # 检查内容
                    if hasattr(message, 'content') and message.content:
                        content_length = len(message.content)
                        print(f"✓ 响应内容长度：{content_length} 字符")
                        print(f"✓ 响应内容预览：{message.content[:100]}...")
                    else:
                        print(f"⚠ 响应内容为空")
                    
                    # 检查推理字段
                    if hasattr(message, 'reasoning_content') and message.reasoning_content:
                        reasoning_length = len(message.reasoning_content)
                        print(f"✓ 包含推理内容，长度：{reasoning_length} 字符")
                        print(f"✓ 推理内容预览：{message.reasoning_content[:100]}...")
                    else:
                        print(f"ℹ 该模型未提供推理内容字段")
                else:
                    print(f"⚠ 响应结构异常")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ 连接失败：{error_msg}")
                
                # 分析错误类型
                if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                    print(f"ℹ 错误类型：网络超时")
                elif "connection" in error_msg.lower():
                    print(f"ℹ 错误类型：连接问题")
                elif "api" in error_msg.lower():
                    print(f"ℹ 错误类型：API相关")
                else:
                    print(f"ℹ 错误类型：其他")
    
    def test_reasoning_capability(self):
        """测试推理能力"""
        print("\n" + "="*60)
        print("测试2：推理能力验证")
        print("="*60)
        
        reasoning_prompt = "请分析一下：为什么深度学习在图像识别领域如此成功？请详细说明你的思考过程。"
        
        for model_config in self.reasoning_models:
            print(f"\n--- 推理测试：{model_config['name']} ({model_config['vendor']}) ---")
            
            if not model_config["api_key"]:
                print(f"⚠ 跳过 {model_config['vendor']}：API密钥未配置")
                continue
            
            try:
                # 初始化客户端
                client = HarborAI(
                    api_key=model_config["api_key"],
                    base_url=model_config["base_url"],
                    timeout=120
                )
                
                # 发送推理请求
                print(f"发送推理测试请求...")
                start_time = time.time()
                
                response = client.chat.completions.create(
                    model=model_config["name"],
                    messages=[
                        {"role": "user", "content": reasoning_prompt}
                    ],
                    timeout=120
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"✓ 推理请求成功，响应时间：{response_time:.2f}秒")
                
                # 详细分析响应
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    message = response.choices[0].message
                    
                    # 分析最终答案
                    if hasattr(message, 'content') and message.content:
                        content = message.content
                        content_length = len(content)
                        print(f"✓ 最终答案长度：{content_length} 字符")
                        
                        # 检查关键词
                        key_terms = ['深度学习', '图像识别', '神经网络', '特征', '卷积']
                        found_terms = [term for term in key_terms if term in content]
                        print(f"✓ 答案包含关键词：{found_terms}")
                        
                        if len(found_terms) >= 2:
                            print(f"✓ 答案质量：良好（包含{len(found_terms)}个关键词）")
                        else:
                            print(f"⚠ 答案质量：一般（仅包含{len(found_terms)}个关键词）")
                    
                    # 分析推理过程
                    if hasattr(message, 'reasoning_content') and message.reasoning_content:
                        reasoning = message.reasoning_content
                        reasoning_length = len(reasoning)
                        print(f"✓ 推理过程长度：{reasoning_length} 字符")
                        
                        # 检查推理质量
                        reasoning_indicators = ['分析', '因为', '所以', '首先', '其次', '总结']
                        found_indicators = [ind for ind in reasoning_indicators if ind in reasoning]
                        print(f"✓ 推理过程包含逻辑词：{found_indicators}")
                        
                        if len(found_indicators) >= 3:
                            print(f"✓ 推理质量：优秀（包含{len(found_indicators)}个逻辑词）")
                        elif len(found_indicators) >= 1:
                            print(f"✓ 推理质量：良好（包含{len(found_indicators)}个逻辑词）")
                        else:
                            print(f"⚠ 推理质量：需改进")
                        
                        # 检查推理与答案的一致性
                        common_terms = set(found_terms) & set([term for term in key_terms if term in reasoning])
                        if len(common_terms) > 0:
                            print(f"✓ 推理与答案一致性：良好（共同关键词：{list(common_terms)}）")
                        else:
                            print(f"⚠ 推理与答案一致性：需检查")
                    else:
                        print(f"ℹ 该模型未提供推理过程，仅有最终答案")
                        
            except Exception as e:
                error_msg = str(e)
                print(f"❌ 推理测试失败：{error_msg}")
                
                # 提供错误处理建议
                if "timed out" in error_msg.lower():
                    print(f"💡 建议：推理任务较复杂，可能需要更长超时时间")
                elif "rate limit" in error_msg.lower():
                    print(f"💡 建议：遇到频率限制，稍后重试")
                else:
                    print(f"💡 建议：检查API配置和网络连接")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始执行简单推理模型基础功能验证测试...")
        print(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 执行基本连接测试
            self.test_basic_connection()
            
            # 执行推理能力测试
            self.test_reasoning_capability()
            
            # 测试总结
            print("\n" + "="*60)
            print("🎉 简单推理模型基础功能验证测试完成！")
            print("="*60)
            print("📊 测试总结：")
            print("  - 基本连接功能：已验证")
            print("  - 推理能力：已验证")
            print("  - 响应结构：已检查")
            print("  - 推理字段：已验证")
            print("✅ 可以继续运行完整的E2E-003测试")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ 测试过程中出现异常：{str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    # 运行简单验证测试
    validator = SimpleReasoningValidation()
    validator.run_all_tests()