#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'e:/project/harborai')
from tests.end_to_end.test_e2e_008_native_structured_output import *
import traceback

def debug_test():
    client = HarborAI()
    models = get_test_models()
    
    success_count = 0
    total_count = len(models)
    
    for model_config in models:
        try:
            print(f'\n测试模型: {model_config["name"]} ({model_config["model"]})')
            
            # 专门检查豆包模型
            if "doubao" in model_config["model"]:
                print(f"🔍 豆包模型详细测试: {model_config['model']}")
                try:
                    test_native_structured_output_with_model(client, model_config)
                    print(f'✅ {model_config["model"]} 测试成功')
                    success_count += 1
                except Exception as e:
                    print(f'❌ 豆包模型 {model_config["model"]} 测试失败:')
                    print(f'   错误类型: {type(e).__name__}')
                    print(f'   错误信息: {str(e)}')
                    print(f'   详细堆栈:')
                    traceback.print_exc()
                    print("-" * 50)
            else:
                test_native_structured_output_with_model(client, model_config)
                print(f'✅ {model_config["model"]} 测试成功')
                success_count += 1
                
        except Exception as e:
            print(f'❌ {model_config["model"]} 测试失败: {str(e)}')
            if "doubao" not in model_config["model"]:
                traceback.print_exc()
    
    print(f'\n测试总结: {success_count}/{total_count} 成功 ({success_count/total_count*100:.1f}%)')

if __name__ == "__main__":
    debug_test()