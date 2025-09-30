#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试Agently执行中的'NoneType' object is not subscriptable错误
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

def debug_agently_execution():
    """调试Agently执行过程"""
    try:
        from harborai.core.fast_structured_output import FastStructuredOutputProcessor
        from harborai.core.client_manager import ClientManager
        
        print("=== 调试Agently执行错误 ===")
        
        # 1. 初始化组件
        print("1. 初始化组件...")
        client_manager = ClientManager()
        processor = FastStructuredOutputProcessor(client_manager=client_manager)
        
        # 2. 测试Schema（使用性能测试中的复杂Schema）
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
        
        # 3. 配置参数
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = "deepseek-chat"
        user_query = "这是一个测试查询"
        
        print(f"   API Key: {api_key[:10] if api_key else 'None'}...")
        print(f"   Base URL: {base_url}")
        print(f"   Model: {model}")
        
        # 4. 填充缓存
        print("2. 填充缓存...")
        agently_schema = processor._convert_schema_with_cache(schema)
        config_data = processor._process_config_with_cache(api_key, base_url, model)
        
        print(f"   Schema已缓存: {agently_schema is not None}")
        print(f"   配置已缓存: {config_data is not None}")
        
        # 5. 测试快速路径
        print("3. 测试快速路径...")
        can_use_fast = processor._can_use_fast_path(schema, api_key, base_url, model)
        print(f"   可以使用快速路径: {can_use_fast}")
        
        if can_use_fast:
            print("4. 执行快速路径...")
            
            # 手动执行快速路径的关键步骤
            try:
                # 获取缓存的Schema和配置
                cached_schema = processor._cache_manager.schema_cache.get_converted_schema(schema)
                print(f"   缓存的Schema: {type(cached_schema)} - {cached_schema}")
                
                config_data = {
                    'api_key_hash': hash(api_key) if api_key else None,
                    'base_url': base_url,
                    'model': model
                }
                cached_config = processor._cache_manager.config_cache.get_config(config_data)
                print(f"   缓存的配置: {type(cached_config)} - {cached_config}")
                
                # 获取客户端池
                if processor._client_pool:
                    print("   使用客户端池...")
                    
                    # 获取provider
                    try:
                        plugin = client_manager.get_plugin_for_model(model)
                        provider = plugin.name if plugin else "unknown"
                        print(f"   Provider: {provider}")
                    except Exception as e:
                        print(f"   获取provider失败: {e}")
                        provider = "unknown"
                    
                    # 创建客户端配置
                    from harborai.core.agently_client_pool import create_agently_client_config
                    client_config = create_agently_client_config(
                        provider=provider,
                        api_key=api_key,
                        base_url=base_url,
                        model=model
                    )
                    print(f"   客户端配置: {client_config}")
                    
                    # 获取客户端
                    with processor._client_pool.get_client_context(client_config) as agently_client:
                        print(f"   Agently客户端: {type(agently_client)} - {agently_client}")
                        
                        # 检查客户端是否为None
                        if agently_client is None:
                            print("   ❌ Agently客户端为None!")
                            return
                        
                        # 检查Schema格式
                        print(f"   Schema格式检查:")
                        for key, value in cached_schema.items():
                            print(f"     {key}: {type(value)} - {value}")
                            if isinstance(value, (list, tuple)) and len(value) > 0:
                                print(f"       第一个元素: {type(value[0])} - {value[0]}")
                        
                        # 尝试执行Agently请求
                        print("   执行Agently请求...")
                        try:
                            result = (
                                agently_client
                                .input(user_query)
                                .output(cached_schema)
                                .start()
                            )
                            print(f"   ✅ 执行成功: {result}")
                        except Exception as e:
                            print(f"   ❌ 执行失败: {e}")
                            print(f"   错误类型: {type(e)}")
                            print(f"   错误详情: {traceback.format_exc()}")
                            
                            # 检查是否是subscriptable错误
                            if "'NoneType' object is not subscriptable" in str(e):
                                print("   🔍 发现subscriptable错误，检查相关变量:")
                                print(f"     agently_client: {agently_client}")
                                print(f"     cached_schema: {cached_schema}")
                                print(f"     user_query: {user_query}")
                
            except Exception as e:
                print(f"   ❌ 快速路径执行失败: {e}")
                print(f"   错误详情: {traceback.format_exc()}")
        
        print("=== 调试完成 ===")
        
    except Exception as e:
        print(f"❌ 调试过程出错: {e}")
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_agently_execution()