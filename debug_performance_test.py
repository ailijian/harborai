#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试性能测试中的问题
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

def debug_performance_test():
    """调试性能测试流程"""
    try:
        from harborai import HarborAI
        from harborai.config.performance import PerformanceMode
        
        print("=== 调试性能测试流程 ===")
        
        # 1. 设置FAST模式
        print("1. 设置FAST模式...")
        from harborai.config.performance import get_performance_config, reset_performance_config, PerformanceMode
        perf_config = reset_performance_config(PerformanceMode.FAST)
        print(f"   当前模式: {perf_config.mode.value}")
        print(f"   性能配置: {perf_config.__dict__}")
        
        client = HarborAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model="deepseek-chat",
            mode="fast"
        )
        
        # 2. 创建测试Schema
        schema = {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "情感分析结果"
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "情感倾向"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "置信度"
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词列表"
                }
            },
            "required": ["analysis", "sentiment", "confidence"]
        }
        
        user_query = "这是一个测试查询，用于评估情感分析功能"
        
        print("2. 第一次调用（应该填充缓存）...")
        
        # 检查快速结构化输出路径的条件
        response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
        structured_provider = "agently"
        stream = None
        
        print(f"   检查快速结构化输出条件:")
        print(f"   - perf_config.mode.value == 'fast': {perf_config.mode.value == 'fast'}")
        print(f"   - response_format: {bool(response_format)}")
        print(f"   - response_format.get('type') == 'json_schema': {response_format.get('type') == 'json_schema'}")
        print(f"   - structured_provider == 'agently': {structured_provider == 'agently'}")
        print(f"   - not stream: {not stream}")
        
        should_use_fast_structured = (
            perf_config.mode.value == "fast" and 
            response_format and 
            response_format.get("type") == "json_schema" and
            structured_provider == "agently" and
            not stream
        )
        print(f"   - 应该使用快速结构化输出: {should_use_fast_structured}")
        
        # 添加日志级别以捕获警告信息
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        try:
            print("   开始第一次调用...")
            print(f"   client.chat.completions类型: {type(client.chat.completions)}")
            print(f"   client.chat.completions.create方法: {client.chat.completions.create}")
            
            result1 = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": user_query}],
                response_format={"type": "json_schema", "json_schema": {"schema": schema}},
                structured_provider="agently"
            )
            print(f"   第一次调用成功: {result1.choices[0].message.content}")
        except Exception as e:
            print(f"   第一次调用失败: {e}")
            traceback.print_exc()
            return
        
        print("3. 第二次调用（应该使用快速路径）...")
        try:
            print("   开始第二次调用...")
            result2 = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": user_query}],
                response_format={"type": "json_schema", "json_schema": {"schema": schema}},
                structured_provider="agently"
            )
            print(f"   第二次调用成功: {result2.choices[0].message.content}")
        except Exception as e:
            print(f"   第二次调用失败: {e}")
            print(f"   错误详情: {traceback.format_exc()}")
        
        print("4. 检查快速处理器状态...")
        try:
            # 获取快速处理器实例
            from harborai.core.fast_structured_output import get_fast_structured_output_processor
            processor = get_fast_structured_output_processor()
            
            if processor:
                stats = processor.get_performance_stats()
                print(f"   性能统计: {stats}")
                
                # 检查缓存状态
                can_use_fast = processor._can_use_fast_path(
                    schema, 
                    os.getenv("DEEPSEEK_API_KEY"),
                    os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    "deepseek-chat"
                )
                print(f"   可以使用快速路径: {can_use_fast}")
                
                # 检查缓存内容
                cached_schema = processor._cache_manager.schema_cache.get_converted_schema(schema)
                print(f"   缓存的Schema: {cached_schema is not None}")
                if cached_schema:
                    print(f"     Schema内容: {cached_schema}")
            else:
                print("   快速处理器未初始化")
                
        except Exception as e:
            print(f"   检查快速处理器失败: {e}")
            print(f"   错误详情: {traceback.format_exc()}")
        
        print("=== 调试完成 ===")
        
    except Exception as e:
        print(f"❌ 调试过程出错: {e}")
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_performance_test()