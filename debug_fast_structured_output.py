#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试快速结构化输出的详细脚本
"""

import traceback
import json
from harborai.core.fast_structured_output import FastStructuredOutputProcessor
from harborai.core.client_manager import ClientManager

def debug_fast_structured_output():
    """调试快速结构化输出处理"""
    try:
        print("=== 开始调试快速结构化输出 ===")
        
        # 创建客户端管理器
        print("1. 创建客户端管理器...")
        client_manager = ClientManager()
        print(f"   客户端管理器创建成功: {client_manager}")
        
        # 创建快速处理器
        print("2. 创建快速处理器...")
        processor = FastStructuredOutputProcessor(client_manager=client_manager)
        print(f"   快速处理器创建成功: {processor}")
        print(f"   _cache_manager: {processor._cache_manager}")
        print(f"   _client_pool: {processor._client_pool}")
        print(f"   client_manager: {processor.client_manager}")
        
        # 测试Schema（使用性能测试中的复杂Schema）
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
        
        # 检查是否可以使用快速路径
        print("3. 检查快速路径可用性...")
        can_use_fast = processor._can_use_fast_path(
            schema=schema,
            api_key="test_key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
        print(f"   可以使用快速路径: {can_use_fast}")
        
        # 如果不能使用快速路径，先填充缓存
        if not can_use_fast:
            print("4. 填充缓存...")
            # 转换Schema并缓存
            agently_schema = processor._convert_json_schema_to_agently(schema)
            print(f"   转换后的Schema: {agently_schema}")
            
            if processor._cache_manager:
                processor._cache_manager.schema_cache.set_converted_schema(schema, agently_schema)
                print("   Schema已缓存")
                
                # 缓存配置
                config_data = {
                    'api_key_hash': hash("test_key"),
                    'base_url': "https://api.deepseek.com",
                    'model': "deepseek-chat"
                }
                processor._cache_manager.config_cache.set_config(config_data, config_data)
                print("   配置已缓存")
                
                # 再次检查快速路径
                can_use_fast = processor._can_use_fast_path(
                    schema=schema,
                    api_key="test_key",
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat"
                )
                print(f"   缓存后可以使用快速路径: {can_use_fast}")
        
        # 5. 测试快速路径逻辑（不实际调用API）
        print("5. 测试快速路径逻辑...")
        try:
            # 检查缓存是否正确设置
            cached_schema = processor._cache_manager.schema_cache.get_converted_schema(schema)
            print(f"   缓存的Schema: {cached_schema}")
            print(f"   Schema类型: {type(cached_schema)}")
            
            # 检查Schema内容
            if cached_schema:
                for key, value in cached_schema.items():
                    print(f"     {key}: {value} (类型: {type(value)})")
                    if isinstance(value, tuple) and len(value) >= 2:
                        print(f"       元组内容: {value[0]} (类型: {type(value[0])}), {value[1]} (类型: {type(value[1])})")
            
            config_data = {
                'api_key_hash': hash("test_key"),
                'base_url': "https://api.deepseek.com",
                'model': "deepseek-chat"
            }
            cached_config = processor._cache_manager.config_cache.get_config(config_data)
            print(f"   缓存的配置: {cached_config}")
            
            # 再次检查快速路径
            can_use_fast_after_cache = processor._can_use_fast_path(
                schema=schema,
                api_key="test_key",
                base_url="https://api.deepseek.com",
                model="deepseek-chat"
            )
            print(f"   缓存后可以使用快速路径: {can_use_fast_after_cache}")
            
            if can_use_fast_after_cache:
                print("   ✅ 快速路径逻辑正常工作")
            else:
                print("   ❌ 快速路径逻辑存在问题")
                
        except Exception as e:
            print(f"   测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("=== 调试完成 ===")
        
    except Exception as e:
        print(f"调试过程中发生错误: {e}")
        print("详细堆栈跟踪:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_fast_structured_output()