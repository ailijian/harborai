#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'e:/project/harborai')
from tests.end_to_end.test_e2e_008_native_structured_output import *
import traceback

def test_doubao_only():
    client = HarborAI()
    models = get_test_models()
    
    # 只测试豆包模型
    doubao_models = [m for m in models if "doubao" in m["model"]]
    
    for model_config in doubao_models:
        print(f'\n[DEBUG] 测试豆包模型: {model_config["name"]} ({model_config["model"]})')
        try:
            test_native_structured_output_with_model(client, model_config)
            print(f'SUCCESS: {model_config["model"]} 测试成功')
        except Exception as e:
            print(f'FAILED: 豆包模型 {model_config["model"]} 测试失败:')
            print(f'   错误类型: {type(e).__name__}')
            print(f'   错误信息: {str(e)}')
            print(f'   详细堆栈:')
            traceback.print_exc()
            print("-" * 50)

if __name__ == "__main__":
    test_doubao_only()