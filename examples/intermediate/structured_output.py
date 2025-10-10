#!/usr/bin/env python3
"""
HarborAI 结构化输出示例

这个示例展示了如何使用HarborAI获取结构化的JSON输出，
确保AI响应符合预定义的数据格式和验证规则。

场景描述:
- JSON Schema验证
- Pydantic模型定义
- 结构化数据提取
- 格式一致性保证

应用价值:
- 确保输出格式一致
- 便于数据处理和存储
- 减少解析错误
- 支持类型安全
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加本地源码路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from harborai import HarborAI
    from pydantic import BaseModel, Field, validator
    import jsonschema
except ImportError as e:
    print(f"❌ 无法导入 HarborAI，请检查路径配置")
    print(f"缺失模块: {e}")
    exit(1)


def create_client() -> HarborAI:
    """
    创建HarborAI客户端
    
    Returns:
        HarborAI: 配置好的客户端实例
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    if not api_key:
        raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")
    
    return HarborAI(
        api_key=api_key,
        base_url=base_url
    )


# ==================== Pydantic 模型定义 ====================

class SkillLevel(str, Enum):
    """技能水平枚举"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Skill(BaseModel):
    """技能模型"""
    name: str = Field(..., description="技能名称")
    level: SkillLevel = Field(..., description="技能水平")
    years_experience: int = Field(..., ge=0, le=50, description="经验年数")


class Contact(BaseModel):
    """联系方式模型"""
    email: Optional[str] = Field(None, description="邮箱地址")
    phone: Optional[str] = Field(None, description="电话号码")
    linkedin: Optional[str] = Field(None, description="LinkedIn链接")


class Person(BaseModel):
    """人员信息模型"""
    name: str = Field(..., min_length=1, max_length=100, description="姓名")
    age: int = Field(..., ge=18, le=100, description="年龄")
    occupation: str = Field(..., description="职业")
    skills: List[Skill] = Field(..., description="技能列表")
    contact: Optional[Contact] = Field(None, description="联系方式")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")
    
    @validator('skills')
    def validate_skills(cls, v):
        if len(v) == 0:
            raise ValueError("至少需要一个技能")
        return v


class Product(BaseModel):
    """产品信息模型"""
    name: str = Field(..., description="产品名称")
    category: str = Field(..., description="产品类别")
    price: float = Field(..., ge=0, description="价格")
    description: str = Field(..., description="产品描述")
    features: List[str] = Field(..., description="产品特性")
    rating: Optional[float] = Field(None, ge=0, le=5, description="评分")
    in_stock: bool = Field(..., description="是否有库存")


class AnalysisResult(BaseModel):
    """分析结果模型"""
    summary: str = Field(..., description="分析摘要")
    key_points: List[str] = Field(..., description="关键点")
    sentiment: str = Field(..., description="情感倾向")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    recommendations: List[str] = Field(..., description="建议")


# ==================== JSON Schema 定义 ====================

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 100},
        "age": {"type": "integer", "minimum": 18, "maximum": 100},
        "occupation": {"type": "string"},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced", "expert"]},
                    "years_experience": {"type": "integer", "minimum": 0, "maximum": 50}
                },
                "required": ["name", "level", "years_experience"]
            },
            "minItems": 1
        },
        "contact": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
                "phone": {"type": "string"},
                "linkedin": {"type": "string"}
            }
        },
        "bio": {"type": "string", "maxLength": 500}
    },
    "required": ["name", "age", "occupation", "skills"]
}


def extract_structured_data_with_schema(client: HarborAI, prompt: str, schema: dict, model: str = "deepseek-chat") -> Dict[str, Any]:
    """
    使用JSON Schema提取结构化数据
    
    Args:
        client: HarborAI客户端
        prompt: 提示词
        schema: JSON Schema定义
        model: 使用的模型
        
    Returns:
        Dict: 提取的结构化数据
    """
    print(f"\n📋 使用JSON Schema提取结构化数据")
    print(f"🎯 提示: {prompt[:100]}...")
    
    # 构建包含schema的提示
    schema_prompt = f"""
请根据以下描述提取信息，并严格按照JSON Schema格式返回：

描述: {prompt}

JSON Schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}

请只返回符合schema的JSON数据，不要包含其他文字说明。
"""
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": schema_prompt}],
            temperature=0.1,  # 低温度确保格式一致性
            # max_tokens 默认无限制，由模型厂商控制
        )
        
        elapsed_time = time.time() - start_time
        content = response.choices[0].message.content.strip()
        
        # 尝试解析JSON
        try:
            # 清理可能的markdown格式
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            
            # 验证schema
            jsonschema.validate(data, schema)
            
            print(f"✅ 结构化数据提取成功 (耗时: {elapsed_time:.2f}秒)")
            print(f"📊 提取的数据:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            return {
                "success": True,
                "data": data,
                "elapsed_time": elapsed_time,
                "raw_content": content
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"原始内容: {content}")
            return {
                "success": False,
                "error": f"JSON解析失败: {e}",
                "raw_content": content,
                "elapsed_time": elapsed_time
            }
            
        except jsonschema.ValidationError as e:
            print(f"❌ Schema验证失败: {e}")
            return {
                "success": False,
                "error": f"Schema验证失败: {e}",
                "data": data,
                "elapsed_time": elapsed_time
            }
            
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }


def extract_structured_data_with_pydantic(client: HarborAI, prompt: str, model_class: BaseModel, model: str = "deepseek-chat") -> Dict[str, Any]:
    """
    使用Pydantic模型提取结构化数据
    
    Args:
        client: HarborAI客户端
        prompt: 提示词
        model_class: Pydantic模型类
        model: 使用的模型
        
    Returns:
        Dict: 提取的结构化数据
    """
    print(f"\n🏗️  使用Pydantic模型提取结构化数据")
    print(f"🎯 提示: {prompt[:100]}...")
    print(f"📝 模型类: {model_class.__name__}")
    
    # 生成schema
    schema = model_class.schema()
    
    # 构建提示
    pydantic_prompt = f"""
请根据以下描述提取信息，并严格按照指定格式返回JSON数据：

描述: {prompt}

数据格式要求:
{json.dumps(schema, indent=2, ensure_ascii=False)}

请只返回符合格式的JSON数据，不要包含其他文字说明。
"""
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": pydantic_prompt}],
            temperature=0.1,
            # max_tokens 默认无限制，由模型厂商控制
        )
        
        elapsed_time = time.time() - start_time
        content = response.choices[0].message.content.strip()
        
        # 清理和解析JSON
        try:
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            
            # 使用Pydantic验证
            validated_data = model_class(**data)
            
            print(f"✅ Pydantic数据提取成功 (耗时: {elapsed_time:.2f}秒)")
            print(f"📊 验证后的数据:")
            print(validated_data.json(indent=2, ensure_ascii=False))
            
            return {
                "success": True,
                "data": validated_data.dict(),
                "validated_object": validated_data,
                "elapsed_time": elapsed_time,
                "raw_content": content
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            return {
                "success": False,
                "error": f"JSON解析失败: {e}",
                "raw_content": content,
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            print(f"❌ Pydantic验证失败: {e}")
            return {
                "success": False,
                "error": f"Pydantic验证失败: {e}",
                "raw_content": content,
                "elapsed_time": elapsed_time
            }
            
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }


async def batch_structured_extraction(client: HarborAI, prompts: List[Dict[str, Any]], model: str = "deepseek-chat"):
    """
    批量结构化数据提取
    
    Args:
        client: HarborAI客户端
        prompts: 提示列表，每个包含prompt和model_class
        model: 使用的模型
    """
    print("\n" + "="*60)
    print("📦 批量结构化数据提取")
    print("="*60)
    
    async def extract_single(prompt_info: Dict[str, Any], index: int):
        """提取单个数据"""
        prompt = prompt_info["prompt"]
        model_class = prompt_info["model_class"]
        
        print(f"\n🔄 处理第 {index+1} 个提取任务...")
        
        # 由于是演示，这里使用同步调用
        # 实际应用中可以实现真正的异步版本
        result = extract_structured_data_with_pydantic(client, prompt, model_class, model)
        result["index"] = index
        result["prompt"] = prompt[:50] + "..."
        result["model_class"] = model_class.__name__
        
        return result
    
    # 模拟异步处理
    start_time = time.time()
    results = []
    
    for i, prompt_info in enumerate(prompts):
        result = await asyncio.create_task(
            asyncio.to_thread(extract_single, prompt_info, i)
        )
        results.append(result)
    
    total_time = time.time() - start_time
    
    # 统计结果
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n📊 批量提取完成:")
    print(f"   总耗时: {total_time:.2f}秒")
    print(f"   成功: {len(successful)}/{len(prompts)}")
    print(f"   失败: {len(failed)}")
    
    if successful:
        avg_time = sum(r["elapsed_time"] for r in successful) / len(successful)
        print(f"   平均耗时: {avg_time:.2f}秒")
    
    return results


def compare_extraction_methods(client: HarborAI, prompt: str, model: str = "deepseek-chat"):
    """
    对比不同的结构化提取方法
    
    Args:
        client: HarborAI客户端
        prompt: 测试提示
        model: 使用的模型
    """
    print("\n" + "="*60)
    print("⚖️  结构化提取方法对比")
    print("="*60)
    
    # 方法1: JSON Schema
    print("\n🔹 方法1: JSON Schema验证")
    schema_result = extract_structured_data_with_schema(client, prompt, PERSON_SCHEMA, model)
    
    # 方法2: Pydantic模型
    print("\n🔹 方法2: Pydantic模型验证")
    pydantic_result = extract_structured_data_with_pydantic(client, prompt, Person, model)
    
    # 对比分析
    print(f"\n📊 方法对比:")
    print("-" * 40)
    
    if schema_result["success"] and pydantic_result["success"]:
        print(f"JSON Schema耗时: {schema_result['elapsed_time']:.2f}秒")
        print(f"Pydantic耗时: {pydantic_result['elapsed_time']:.2f}秒")
        
        # 数据一致性检查
        schema_data = schema_result["data"]
        pydantic_data = pydantic_result["data"]
        
        consistent = schema_data == pydantic_data
        print(f"数据一致性: {'✅ 一致' if consistent else '❌ 不一致'}")
        
        if not consistent:
            print("差异分析:")
            for key in set(schema_data.keys()) | set(pydantic_data.keys()):
                if schema_data.get(key) != pydantic_data.get(key):
                    print(f"  {key}: Schema={schema_data.get(key)} vs Pydantic={pydantic_data.get(key)}")
    
    return schema_result, pydantic_result


def interactive_structured_extraction(client: HarborAI, model: str = "deepseek-chat"):
    """
    交互式结构化数据提取
    
    Args:
        client: HarborAI客户端
        model: 使用的模型
    """
    print("\n" + "="*60)
    print("🎯 交互式结构化数据提取")
    print("="*60)
    print("💡 输入描述文本，AI将提取结构化信息")
    print("💡 输入 'quit' 或 'exit' 退出")
    print("💡 输入 'models' 查看可用的数据模型")
    
    available_models = {
        "1": ("Person", Person, "人员信息"),
        "2": ("Product", Product, "产品信息"),
        "3": ("AnalysisResult", AnalysisResult, "分析结果")
    }
    
    while True:
        try:
            print(f"\n可用数据模型:")
            for key, (name, cls, desc) in available_models.items():
                print(f"  {key}. {name} - {desc}")
            
            model_choice = input("\n选择数据模型 (1-3): ").strip()
            
            if model_choice.lower() in ['quit', 'exit', '退出']:
                print("👋 交互式提取结束！")
                break
            
            if model_choice not in available_models:
                print("❌ 无效的模型选择")
                continue
            
            model_name, model_class, model_desc = available_models[model_choice]
            print(f"✅ 已选择: {model_name} - {model_desc}")
            
            user_input = input(f"\n📝 请输入{model_desc}的描述: ").strip()
            
            if not user_input:
                continue
            
            # 提取结构化数据
            result = extract_structured_data_with_pydantic(client, user_input, model_class, model)
            
            if result["success"]:
                print(f"\n🎉 提取成功！可以继续输入其他描述...")
            else:
                print(f"\n❌ 提取失败: {result['error']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 交互式提取被中断！")
            break
        except Exception as e:
            print(f"\n❌ 处理出错: {e}")


async def main():
    """主函数"""
    print("="*60)
    print("📋 HarborAI 结构化输出示例")
    print("="*60)
    
    try:
        # 创建客户端
        client = create_client()
        print("✅ HarborAI 客户端初始化成功")
        
        # 测试数据
        test_prompts = [
            "张三是一名30岁的软件工程师，精通Python和JavaScript，有5年开发经验。他的邮箱是zhangsan@example.com，LinkedIn是linkedin.com/in/zhangsan。他擅长后端开发和数据分析。",
            "李四，25岁，UI/UX设计师，专业技能包括Figma(高级)、Photoshop(专家级)、用户研究(中级)。工作3年，电话13800138000。",
            "王五是一位28岁的数据科学家，在机器学习方面有4年经验(高级水平)，Python编程6年(专家级)，SQL数据库3年(中级)。邮箱wangwu@data.com。"
        ]
        
        # 1. JSON Schema提取示例
        print("\n🔹 1. JSON Schema结构化提取")
        schema_result = extract_structured_data_with_schema(client, test_prompts[0], PERSON_SCHEMA)
        
        # 2. Pydantic模型提取示例
        print("\n🔹 2. Pydantic模型结构化提取")
        pydantic_result = extract_structured_data_with_pydantic(client, test_prompts[1], Person)
        
        # 3. 方法对比
        print("\n🔹 3. 提取方法对比")
        compare_extraction_methods(client, test_prompts[2])
        
        # 4. 批量提取
        print("\n🔹 4. 批量结构化提取")
        batch_prompts = [
            {"prompt": test_prompts[0], "model_class": Person},
            {"prompt": "iPhone 15 Pro是苹果公司的旗舰手机，售价999美元，具有A17芯片、钛金属机身、48MP相机等特性，评分4.5分，目前有库存。", "model_class": Product},
            {"prompt": "这篇文章分析了AI技术的发展趋势，主要观点包括：1)AI将改变各行各业，2)需要关注伦理问题，3)技术发展迅速。整体情感积极，建议企业尽早布局AI。", "model_class": AnalysisResult}
        ]
        await batch_structured_extraction(client, batch_prompts)
        
        # 5. 交互式提取
        print("\n🔹 5. 交互式结构化提取")
        choice = input("是否开始交互式结构化提取？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_structured_extraction(client)
        
        print(f"\n🎉 所有结构化输出示例执行完成！")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        print("\n💡 请检查:")
        print("1. 是否正确配置了环境变量")
        print("2. 是否安装了所有依赖包")
        print("3. 网络连接是否正常")
        print("4. API密钥是否有效")


if __name__ == "__main__":
    asyncio.run(main())