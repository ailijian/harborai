#!/usr/bin/env python3
"""
配置管理助手

这个示例展示了 HarborAI 的配置管理功能，包括：
1. 多模型配置管理
2. 环境变量配置
3. 动态配置切换
4. 配置验证和测试
5. 最佳实践示例

场景：
- 多环境部署（开发、测试、生产）
- 多模型供应商管理
- 动态配置切换
- 配置安全管理

价值：
- 简化配置管理流程
- 提高配置安全性
- 支持多环境部署
- 降低配置错误风险
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# 正确的 HarborAI 导入方式
from harborai import HarborAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """模型供应商枚举"""
    DEEPSEEK = "deepseek"
    ERNIE = "ernie"
    DOUBAO = "doubao"
    OPENAI = "openai"

@dataclass
class ModelConfig:
    """模型配置"""
    provider: ModelProvider
    model_name: str
    api_key_env: str
    base_url_env: str
    default_base_url: str
    description: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 30.0

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.configs = self._load_default_configs()
        self.current_config: Optional[ModelConfig] = None
        self.client: Optional[HarborAI] = None
    
    def _load_default_configs(self) -> Dict[ModelProvider, ModelConfig]:
        """加载默认配置"""
        return {
            ModelProvider.DEEPSEEK: ModelConfig(
                provider=ModelProvider.DEEPSEEK,
                model_name="deepseek-chat",
                api_key_env="DEEPSEEK_API_KEY",
                base_url_env="DEEPSEEK_BASE_URL",
                default_base_url="https://api.deepseek.com",
                description="DeepSeek 聊天模型 - 高性价比，支持中文",
                max_tokens=8192,
                temperature=0.7,
                timeout=30.0
            ),
            ModelProvider.ERNIE: ModelConfig(
                provider=ModelProvider.ERNIE,
                model_name="ernie-3.5-8k",
                api_key_env="ERNIE_API_KEY",
                base_url_env="ERNIE_BASE_URL",
                default_base_url="https://aip.baidubce.com",
                description="百度文心一言 - 中文优化，企业级",
                max_tokens=8192,
                temperature=0.7,
                timeout=30.0
            ),
            ModelProvider.DOUBAO: ModelConfig(
                provider=ModelProvider.DOUBAO,
                model_name="doubao-1-5-pro-32k-character-250715",
                api_key_env="DOUBAO_API_KEY",
                base_url_env="DOUBAO_BASE_URL",
                default_base_url="https://ark.cn-beijing.volces.com",
                description="字节跳动豆包 - 长上下文，多模态",
                max_tokens=32768,
                temperature=0.7,
                timeout=45.0
            ),
            ModelProvider.OPENAI: ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                api_key_env="OPENAI_API_KEY",
                base_url_env="OPENAI_BASE_URL",
                default_base_url="https://api.openai.com/v1",
                description="OpenAI GPT-3.5 - 通用模型，广泛支持",
                max_tokens=4096,
                temperature=0.7,
                timeout=30.0
            )
        }
    
    def get_available_providers(self) -> List[ModelProvider]:
        """获取可用的模型供应商"""
        available = []
        for provider, config in self.configs.items():
            if os.getenv(config.api_key_env):
                available.append(provider)
        return available
    
    def get_config(self, provider: ModelProvider) -> Optional[ModelConfig]:
        """获取指定供应商的配置"""
        return self.configs.get(provider)
    
    def is_provider_configured(self, provider: ModelProvider) -> bool:
        """检查供应商是否已配置"""
        config = self.get_config(provider)
        if not config:
            return False
        return bool(os.getenv(config.api_key_env))
    
    def get_primary_provider(self) -> Optional[ModelProvider]:
        """获取主要供应商（优先级：DeepSeek > Ernie > Doubao > OpenAI）"""
        priority_order = [
            ModelProvider.DEEPSEEK,
            ModelProvider.ERNIE,
            ModelProvider.DOUBAO,
            ModelProvider.OPENAI
        ]
        
        for provider in priority_order:
            if self.is_provider_configured(provider):
                return provider
        
        return None
    
    def create_client(self, provider: Optional[ModelProvider] = None) -> Tuple[Optional[HarborAI], Optional[ModelConfig]]:
        """创建 HarborAI 客户端"""
        if provider is None:
            provider = self.get_primary_provider()
        
        if provider is None:
            return None, None
        
        config = self.get_config(provider)
        if not config or not self.is_provider_configured(provider):
            return None, None
        
        api_key = os.getenv(config.api_key_env)
        base_url = os.getenv(config.base_url_env, config.default_base_url)
        
        try:
            client = HarborAI(
                api_key=api_key,
                base_url=base_url
            )
            
            self.current_config = config
            self.client = client
            
            return client, config
            
        except Exception as e:
            logger.error(f"创建客户端失败 ({provider.value}): {e}")
            return None, None
    
    def get_fallback_models(self, exclude_provider: Optional[ModelProvider] = None) -> List[str]:
        """获取降级模型列表"""
        fallback_models = []
        
        for provider in [ModelProvider.DEEPSEEK, ModelProvider.ERNIE, ModelProvider.DOUBAO]:
            if provider == exclude_provider:
                continue
            
            if self.is_provider_configured(provider):
                config = self.get_config(provider)
                if config:
                    fallback_models.append(config.model_name)
        
        return fallback_models
    
    def print_configuration_status(self):
        """打印配置状态"""
        print("\n📋 HarborAI 配置状态")
        print("=" * 50)
        
        available_providers = self.get_available_providers()
        primary_provider = self.get_primary_provider()
        
        print(f"🔍 已配置的供应商: {len(available_providers)}/{len(self.configs)}")
        
        for provider, config in self.configs.items():
            status = "✅" if provider in available_providers else "❌"
            primary_mark = "🌟" if provider == primary_provider else "  "
            
            print(f"{primary_mark} {status} {config.provider.value.upper()}")
            print(f"     模型: {config.model_name}")
            print(f"     描述: {config.description}")
            print(f"     环境变量: {config.api_key_env}")
            
            if provider in available_providers:
                base_url = os.getenv(config.base_url_env, config.default_base_url)
                print(f"     API地址: {base_url}")
            else:
                print(f"     状态: 未配置 API Key")
            print()
        
        if primary_provider:
            print(f"🎯 主要供应商: {primary_provider.value.upper()}")
            fallback_models = self.get_fallback_models(primary_provider)
            if fallback_models:
                print(f"🔄 降级模型: {', '.join(fallback_models)}")
        else:
            print("⚠️ 警告: 未配置任何供应商")
            print("请设置至少一个 API Key:")
            for config in self.configs.values():
                print(f"   export {config.api_key_env}=your_api_key")

async def demo_basic_configuration():
    """演示基础配置"""
    print("\n🔧 演示基础配置管理")
    print("=" * 50)
    
    config_manager = ConfigManager()
    
    # 显示配置状态
    config_manager.print_configuration_status()
    
    # 创建客户端
    client, config = config_manager.create_client()
    
    if not client or not config:
        print("❌ 无法创建客户端，请检查配置")
        return
    
    print(f"✅ 成功创建客户端")
    print(f"   供应商: {config.provider.value.upper()}")
    print(f"   模型: {config.model_name}")
    
    # 测试基础调用
    try:
        response = await client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": "你好，请简单介绍一下你自己"}],
            max_tokens=100,
            temperature=config.temperature,
            timeout=config.timeout
        )
        
        content = response.choices[0].message.content if response.choices else "无响应"
        print(f"✅ 测试调用成功")
        print(f"   响应: {content[:100]}...")
        
        if response.usage:
            print(f"   Token使用: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"❌ 测试调用失败: {e}")

async def demo_multi_provider_switching():
    """演示多供应商切换"""
    print("\n🔄 演示多供应商切换")
    print("=" * 50)
    
    config_manager = ConfigManager()
    available_providers = config_manager.get_available_providers()
    
    if len(available_providers) < 2:
        print("⚠️ 需要至少配置2个供应商才能演示切换功能")
        print(f"当前已配置: {[p.value for p in available_providers]}")
        return
    
    test_message = "请用一句话解释什么是人工智能"
    
    for provider in available_providers[:3]:  # 最多测试3个供应商
        print(f"\n🔄 切换到 {provider.value.upper()}")
        
        client, config = config_manager.create_client(provider)
        
        if not client or not config:
            print(f"❌ 无法创建 {provider.value} 客户端")
            continue
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            response = await client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": test_message}],
                max_tokens=100,
                temperature=0.7,
                timeout=config.timeout
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            content = response.choices[0].message.content if response.choices else "无响应"
            
            print(f"   ✅ 响应时间: {response_time:.2f}秒")
            print(f"   📝 内容: {content[:80]}...")
            
            if response.usage:
                print(f"   🔢 Token: {response.usage.total_tokens}")
            
        except Exception as e:
            print(f"   ❌ 调用失败: {e}")

async def demo_fallback_configuration():
    """演示降级配置"""
    print("\n🛡️ 演示降级配置")
    print("=" * 50)
    
    config_manager = ConfigManager()
    client, config = config_manager.create_client()
    
    if not client or not config:
        print("❌ 无法创建客户端")
        return
    
    # 获取降级模型列表
    fallback_models = config_manager.get_fallback_models(config.provider)
    
    print(f"🎯 主模型: {config.model_name}")
    print(f"🔄 降级模型: {fallback_models}")
    
    if not fallback_models:
        print("⚠️ 未配置降级模型，建议配置多个供应商")
        return
    
    # 测试带降级的调用
    try:
        response = await client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": "测试降级机制：请简单回答什么是机器学习"}],
            fallback=fallback_models,
            retry_policy={
                "max_attempts": 3,
                "base_delay": 1.0,
                "max_delay": 5.0
            },
            max_tokens=100,
            timeout=30.0
        )
        
        content = response.choices[0].message.content if response.choices else "无响应"
        print(f"✅ 降级配置测试成功")
        print(f"   响应: {content[:100]}...")
        
    except Exception as e:
        print(f"❌ 降级配置测试失败: {e}")

async def demo_environment_configuration():
    """演示环境配置"""
    print("\n🌍 演示环境配置管理")
    print("=" * 50)
    
    # 模拟不同环境的配置
    environments = {
        "development": {
            "description": "开发环境 - 快速响应，成本优化",
            "preferred_providers": [ModelProvider.DEEPSEEK, ModelProvider.ERNIE],
            "max_tokens": 1024,
            "temperature": 0.8,
            "timeout": 15.0
        },
        "testing": {
            "description": "测试环境 - 稳定性优先",
            "preferred_providers": [ModelProvider.ERNIE, ModelProvider.DEEPSEEK],
            "max_tokens": 2048,
            "temperature": 0.5,
            "timeout": 30.0
        },
        "production": {
            "description": "生产环境 - 高可用，多降级",
            "preferred_providers": [ModelProvider.DEEPSEEK, ModelProvider.ERNIE, ModelProvider.DOUBAO],
            "max_tokens": 4096,
            "temperature": 0.7,
            "timeout": 45.0
        }
    }
    
    current_env = os.getenv("HARBORAI_ENV", "development")
    env_config = environments.get(current_env, environments["development"])
    
    print(f"🔍 当前环境: {current_env}")
    print(f"📝 环境描述: {env_config['description']}")
    print(f"🎯 首选供应商: {[p.value for p in env_config['preferred_providers']]}")
    print(f"⚙️ 配置参数:")
    print(f"   - max_tokens: {env_config['max_tokens']}")
    print(f"   - temperature: {env_config['temperature']}")
    print(f"   - timeout: {env_config['timeout']}s")
    
    # 根据环境配置创建客户端
    config_manager = ConfigManager()
    
    # 尝试使用首选供应商
    client = None
    config = None
    
    for provider in env_config['preferred_providers']:
        if config_manager.is_provider_configured(provider):
            client, config = config_manager.create_client(provider)
            if client and config:
                print(f"✅ 使用供应商: {provider.value.upper()}")
                break
    
    if not client or not config:
        print("❌ 无法根据环境配置创建客户端")
        return
    
    # 测试环境配置
    try:
        response = await client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": f"这是{current_env}环境的测试"}],
            max_tokens=env_config['max_tokens'],
            temperature=env_config['temperature'],
            timeout=env_config['timeout']
        )
        
        content = response.choices[0].message.content if response.choices else "无响应"
        print(f"✅ 环境配置测试成功")
        print(f"   响应: {content[:80]}...")
        
    except Exception as e:
        print(f"❌ 环境配置测试失败: {e}")

async def demo_configuration_validation():
    """演示配置验证"""
    print("\n✅ 演示配置验证")
    print("=" * 50)
    
    config_manager = ConfigManager()
    
    print("🔍 验证所有供应商配置...")
    
    validation_results = {}
    
    for provider in ModelProvider:
        print(f"\n📋 验证 {provider.value.upper()}:")
        
        config = config_manager.get_config(provider)
        if not config:
            print("   ❌ 配置不存在")
            validation_results[provider] = False
            continue
        
        # 检查环境变量
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            print(f"   ❌ 缺少环境变量: {config.api_key_env}")
            validation_results[provider] = False
            continue
        
        print(f"   ✅ API Key: 已设置")
        
        # 检查 Base URL
        base_url = os.getenv(config.base_url_env, config.default_base_url)
        print(f"   ✅ Base URL: {base_url}")
        
        # 尝试创建客户端
        try:
            client, _ = config_manager.create_client(provider)
            if client:
                print(f"   ✅ 客户端创建: 成功")
                
                # 简单连接测试
                try:
                    response = await client.chat.completions.create(
                        model=config.model_name,
                        messages=[{"role": "user", "content": "测试"}],
                        max_tokens=10,
                        timeout=10.0
                    )
                    print(f"   ✅ 连接测试: 成功")
                    validation_results[provider] = True
                    
                except Exception as e:
                    print(f"   ❌ 连接测试: 失败 - {e}")
                    validation_results[provider] = False
            else:
                print(f"   ❌ 客户端创建: 失败")
                validation_results[provider] = False
                
        except Exception as e:
            print(f"   ❌ 客户端创建: 失败 - {e}")
            validation_results[provider] = False
    
    # 汇总验证结果
    valid_providers = [p for p, valid in validation_results.items() if valid]
    invalid_providers = [p for p, valid in validation_results.items() if not valid]
    
    print(f"\n📊 验证汇总:")
    print(f"   ✅ 有效供应商: {len(valid_providers)}")
    print(f"   ❌ 无效供应商: {len(invalid_providers)}")
    
    if valid_providers:
        print(f"   🎯 可用: {[p.value for p in valid_providers]}")
    
    if invalid_providers:
        print(f"   ⚠️ 需修复: {[p.value for p in invalid_providers]}")

async def demo_best_practices():
    """演示最佳实践"""
    print("\n💡 演示配置最佳实践")
    print("=" * 50)
    
    print("📋 HarborAI 配置最佳实践:")
    print()
    
    print("1. 🔐 安全配置:")
    print("   - 使用环境变量存储 API Key")
    print("   - 不要在代码中硬编码密钥")
    print("   - 使用 .env 文件管理本地配置")
    print("   - 生产环境使用密钥管理服务")
    print()
    
    print("2. 🔄 多供应商配置:")
    print("   - 配置多个模型供应商作为备选")
    print("   - 设置合理的降级顺序")
    print("   - 根据成本和性能选择主供应商")
    print("   - 定期测试所有供应商的可用性")
    print()
    
    print("3. 🌍 环境管理:")
    print("   - 为不同环境设置不同的配置")
    print("   - 开发环境优化响应速度")
    print("   - 生产环境优化稳定性和成本")
    print("   - 使用环境变量区分配置")
    print()
    
    print("4. ⚙️ 参数调优:")
    print("   - 根据业务需求调整 max_tokens")
    print("   - 创意任务使用较高 temperature")
    print("   - 分析任务使用较低 temperature")
    print("   - 设置合理的超时时间")
    print()
    
    print("5. 📊 监控和维护:")
    print("   - 定期验证配置有效性")
    print("   - 监控 API 调用成功率")
    print("   - 跟踪成本和使用量")
    print("   - 及时更新过期的配置")
    print()
    
    # 示例配置文件
    print("📄 示例 .env 配置文件:")
    print("```")
    print("# HarborAI 配置")
    print("HARBORAI_ENV=production")
    print()
    print("# DeepSeek 配置")
    print("DEEPSEEK_API_KEY=your_deepseek_api_key")
    print("DEEPSEEK_BASE_URL=https://api.deepseek.com")
    print()
    print("# 百度文心一言配置")
    print("ERNIE_API_KEY=your_ernie_api_key")
    print("ERNIE_BASE_URL=https://aip.baidubce.com")
    print()
    print("# 字节跳动豆包配置")
    print("DOUBAO_API_KEY=your_doubao_api_key")
    print("DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com")
    print("```")

async def main():
    """主函数"""
    print("🔧 HarborAI 配置管理演示")
    print("=" * 60)
    
    demos = [
        ("基础配置管理", demo_basic_configuration),
        ("多供应商切换", demo_multi_provider_switching),
        ("降级配置", demo_fallback_configuration),
        ("环境配置管理", demo_environment_configuration),
        ("配置验证", demo_configuration_validation),
        ("最佳实践", demo_best_practices)
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
            await asyncio.sleep(1)  # 避免请求过于频繁
        except Exception as e:
            print(f"❌ {name} 演示失败: {e}")
    
    print("\n🎉 配置管理演示完成！")
    print("\n💡 关键要点:")
    print("1. 使用环境变量管理敏感配置")
    print("2. 配置多个供应商确保高可用")
    print("3. 根据环境调整配置参数")
    print("4. 定期验证配置有效性")
    print("5. 遵循安全配置最佳实践")

if __name__ == "__main__":
    asyncio.run(main())