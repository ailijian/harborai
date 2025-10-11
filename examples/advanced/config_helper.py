#!/usr/bin/env python3
"""
配置助手模块

从环境变量读取API配置，并提供可用模型的配置信息。
"""

import os
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass

# 加载环境变量
try:
    from dotenv import load_dotenv
    # 加载项目根目录的 .env 文件
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
    print(f"✅ 已加载环境配置: {env_path}")
except ImportError:
    print("⚠️  python-dotenv 未安装，将直接使用系统环境变量")

@dataclass
class ModelConfig:
    """模型配置"""
    vendor: str
    model: str
    api_key: str
    base_url: str
    is_reasoning: bool = False
    cost_per_token: float = 0.0001  # 默认成本

# 可用模型配置
AVAILABLE_MODELS = [
    {'vendor': 'deepseek', 'model': 'deepseek-chat', 'is_reasoning': False},
    {'vendor': 'deepseek', 'model': 'deepseek-reasoner', 'is_reasoning': True},
    {'vendor': 'ernie', 'model': 'ernie-3.5-8k', 'is_reasoning': False},
    {'vendor': 'ernie', 'model': 'ernie-4.0-turbo-8k', 'is_reasoning': False},
    {'vendor': 'ernie', 'model': 'ernie-x1-turbo-32k', 'is_reasoning': True},
    {'vendor': 'doubao', 'model': 'doubao-1-5-pro-32k-character-250715', 'is_reasoning': False},
    {'vendor': 'doubao', 'model': 'doubao-seed-1-6-250615', 'is_reasoning': True}
]

def get_api_config(vendor: str) -> Optional[Dict[str, str]]:
    """获取指定厂商的API配置"""
    configs = {
        'deepseek': {
            'api_key': os.getenv('DEEPSEEK_API_KEY'),
            'base_url': os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        },
        'ernie': {
            'api_key': os.getenv('WENXIN_API_KEY'),
            'base_url': os.getenv('WENXIN_BASE_URL', 'https://qianfan.baidubce.com/v2')
        },
        'doubao': {
            'api_key': os.getenv('DOUBAO_API_KEY'),
            'base_url': os.getenv('DOUBAO_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
        }
    }
    
    config = configs.get(vendor)
    if config and config['api_key']:
        return config
    return None

def get_model_configs() -> List[ModelConfig]:
    """获取所有可用的模型配置"""
    model_configs = []
    
    for model_info in AVAILABLE_MODELS:
        vendor = model_info['vendor']
        api_config = get_api_config(vendor)
        
        if api_config:
            model_config = ModelConfig(
                vendor=vendor,
                model=model_info['model'],
                api_key=api_config['api_key'],
                base_url=api_config['base_url'],
                is_reasoning=model_info['is_reasoning']
            )
            model_configs.append(model_config)
    
    return model_configs

def get_primary_model_config() -> Optional[ModelConfig]:
    """获取主要模型配置（优先使用 deepseek-chat）"""
    model_configs = get_model_configs()
    
    if not model_configs:
        return None
    
    # 优先使用 deepseek-chat
    for config in model_configs:
        if config.vendor == 'deepseek' and config.model == 'deepseek-chat':
            return config
    
    # 如果没有 deepseek-chat，返回第一个可用的
    return model_configs[0]

def get_fallback_models() -> List[str]:
    """获取降级模型列表"""
    model_configs = get_model_configs()
    return [config.model for config in model_configs]

def print_available_models():
    """打印可用模型信息"""
    model_configs = get_model_configs()
    
    if not model_configs:
        print("❌ 没有找到可用的模型配置，请检查环境变量设置")
        print("\n需要设置的环境变量:")
        print("- DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL")
        print("- WENXIN_API_KEY 和 WENXIN_BASE_URL")
        print("- DOUBAO_API_KEY 和 DOUBAO_BASE_URL")
        return
    
    print(f"✅ 找到 {len(model_configs)} 个可用模型:")
    for i, config in enumerate(model_configs, 1):
        reasoning_tag = " (推理模型)" if config.is_reasoning else ""
        masked_key = f"{config.api_key[:6]}...{config.api_key[-4:]}" if len(config.api_key) > 10 else "无效密钥"
        print(f"  {i}. {config.vendor}/{config.model}{reasoning_tag}")
        print(f"     API Key: {masked_key}")
        print(f"     Base URL: {config.base_url}")

if __name__ == "__main__":
    print("🔧 HarborAI 配置助手")
    print("=" * 40)
    print_available_models()
    
    primary = get_primary_model_config()
    if primary:
        print(f"\n🎯 主要模型: {primary.vendor}/{primary.model}")
    
    fallback_models = get_fallback_models()
    if len(fallback_models) > 1:
        print(f"🔄 降级模型: {', '.join(fallback_models[1:])}")