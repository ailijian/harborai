#!/usr/bin/env python3
"""
检查 HarborAI 中可用的模型
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ 环境变量已加载")
except ImportError:
    print("⚠ python-dotenv未安装，直接使用环境变量")

from harborai import HarborAI

def main():
    """检查可用模型"""
    print("=== 检查 HarborAI 可用模型 ===")
    
    try:
        # 初始化 HarborAI 客户端
        client = HarborAI()
        
        # 获取可用模型
        models = client.client_manager.get_available_models()
        
        print(f"\n总共找到 {len(models)} 个模型:")
        
        # 按提供商分组显示
        providers = {}
        for model in models:
            provider = getattr(model, 'provider', 'unknown')
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model)
        
        for provider, provider_models in providers.items():
            print(f"\n--- {provider.upper()} 提供商 ({len(provider_models)} 个模型) ---")
            for model in provider_models:
                supports_structured = getattr(model, 'supports_structured_output', False)
                print(f"  - {model.id} (结构化输出: {'✓' if supports_structured else '✗'})")
        
        # 特别检查豆包模型
        doubao_models = [m for m in models if 'doubao' in m.id.lower()]
        print(f"\n--- 豆包模型详情 ({len(doubao_models)} 个) ---")
        for model in doubao_models:
            print(f"  模型ID: {model.id}")
            print(f"  提供商: {getattr(model, 'provider', 'unknown')}")
            print(f"  支持结构化输出: {getattr(model, 'supports_structured_output', False)}")
            print(f"  支持流式输出: {getattr(model, 'supports_streaming', False)}")
            print(f"  最大令牌: {getattr(model, 'max_tokens', 'unknown')}")
            print()
            
    except Exception as e:
        print(f"✗ 检查模型失败: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")

if __name__ == "__main__":
    main()