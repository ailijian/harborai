#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试time错误的脚本
"""

import traceback

try:
    from harborai.core.fast_structured_output import FastStructuredOutputProcessor
    from harborai.core.client_manager import ClientManager
    
    # 创建客户端管理器
    client_manager = ClientManager()
    
    # 创建快速处理器
    processor = FastStructuredOutputProcessor(client_manager=client_manager)
    
    # 测试处理
    result = processor.process_structured_output(
        user_query="测试查询",
        schema={"type": "object", "properties": {"result": {"type": "string"}}},
        api_key="test",
        base_url="test",
        model="test"
    )
    
    print("测试成功")
    
except Exception as e:
    print(f"错误: {e}")
    print("详细堆栈跟踪:")
    traceback.print_exc()