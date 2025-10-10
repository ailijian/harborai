#!/usr/bin/env python3
"""
HarborAI 智能内容生成系统

场景描述:
构建多媒体内容创作平台，支持文章、营销文案、产品描述等多种内容的智能生成。
适用于媒体出版、电商营销、企业宣传等多种业务场景。

应用价值:
- 大幅提升内容创作效率
- 保证内容质量和一致性
- 降低内容制作成本
- 支持个性化和规模化内容生产

核心功能:
1. 多类型内容模板和生成策略
2. 内容质量评估和优化建议
3. 批量内容生成和管理
4. SEO优化和关键词集成
5. 内容版本控制和协作
"""

import asyncio
import json
import time
import uuid
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
from collections import defaultdict
import hashlib
import threading
# 添加本地源码路径
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from harborai import HarborAI
    from harborai.core.base_plugin import ChatCompletion
except ImportError:
    print("❌ 无法导入 HarborAI，请检查路径配置")
    exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """内容类型"""
    ARTICLE = "article"                 # 文章
    BLOG_POST = "blog_post"            # 博客文章
    PRODUCT_DESC = "product_desc"       # 产品描述
    MARKETING_COPY = "marketing_copy"   # 营销文案
    SOCIAL_MEDIA = "social_media"       # 社交媒体
    EMAIL = "email"                     # 邮件
    PRESS_RELEASE = "press_release"     # 新闻稿
    TECHNICAL_DOC = "technical_doc"     # 技术文档

class ContentStyle(Enum):
    """内容风格"""
    PROFESSIONAL = "professional"       # 专业
    CASUAL = "casual"                   # 随意
    FORMAL = "formal"                   # 正式
    CREATIVE = "creative"               # 创意
    PERSUASIVE = "persuasive"          # 说服性
    INFORMATIVE = "informative"        # 信息性
    ENTERTAINING = "entertaining"       # 娱乐性

class ContentStatus(Enum):
    """内容状态"""
    DRAFT = "draft"                     # 草稿
    REVIEW = "review"                   # 审核中
    APPROVED = "approved"               # 已批准
    PUBLISHED = "published"             # 已发布
    ARCHIVED = "archived"               # 已归档

class QualityLevel(Enum):
    """质量等级"""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    VERY_POOR = 1

@dataclass
class ContentRequirement:
    """内容需求"""
    content_type: ContentType
    title: str
    target_audience: str
    key_points: List[str]
    style: ContentStyle
    word_count: int
    keywords: List[str] = field(default_factory=list)
    tone: str = "neutral"
    language: str = "zh-CN"
    seo_requirements: Dict[str, Any] = field(default_factory=dict)
    brand_guidelines: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    priority: int = 3  # 1-5, 5最高

@dataclass
class ContentPiece:
    """内容作品"""
    id: str
    requirement: ContentRequirement
    content: str
    status: ContentStatus
    created_at: datetime
    updated_at: datetime
    author: str = "AI"
    version: int = 1
    quality_score: Optional[float] = None
    seo_score: Optional[float] = None
    readability_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)
    revisions: List[str] = field(default_factory=list)

@dataclass
class QualityAssessment:
    """质量评估"""
    overall_score: float
    readability: float
    coherence: float
    relevance: float
    originality: float
    engagement: float
    seo_optimization: float
    suggestions: List[str]
    strengths: List[str]
    weaknesses: List[str]

class ContentTemplateManager:
    """内容模板管理器"""
    
    def __init__(self):
        self.templates = {
            ContentType.ARTICLE: {
                "structure": ["引言", "主体段落", "结论"],
                "prompt": """
请根据以下要求创作一篇{word_count}字的{style}风格文章：

标题: {title}
目标受众: {target_audience}
关键要点: {key_points}
关键词: {keywords}
语调: {tone}

文章结构要求:
1. 引人入胜的开头
2. 逻辑清晰的主体内容
3. 有力的结论
4. 自然融入关键词
5. 符合SEO最佳实践

请确保内容原创、准确、有价值。
"""
            },
            
            ContentType.PRODUCT_DESC: {
                "structure": ["产品概述", "核心特性", "使用场景", "购买理由"],
                "prompt": """
请为以下产品创作{word_count}字的{style}风格产品描述：

产品名称: {title}
目标客户: {target_audience}
核心卖点: {key_points}
关键词: {keywords}
语调: {tone}

描述要求:
1. 突出产品独特价值
2. 解决客户痛点
3. 激发购买欲望
4. 包含技术规格（如适用）
5. 优化搜索引擎可见性

请确保描述准确、吸引人且具有说服力。
"""
            },
            
            ContentType.MARKETING_COPY: {
                "structure": ["吸引注意", "激发兴趣", "建立渴望", "促成行动"],
                "prompt": """
请创作{word_count}字的{style}风格营销文案：

活动/产品: {title}
目标受众: {target_audience}
核心信息: {key_points}
关键词: {keywords}
语调: {tone}

文案要求:
1. 强有力的标题
2. 引人注目的开头
3. 清晰的价值主张
4. 紧迫感和稀缺性
5. 明确的行动号召

请确保文案具有强烈的说服力和转化能力。
"""
            },
            
            ContentType.SOCIAL_MEDIA: {
                "structure": ["钩子", "内容", "互动"],
                "prompt": """
请创作{word_count}字的{style}风格社交媒体内容：

主题: {title}
目标受众: {target_audience}
关键信息: {key_points}
话题标签: {keywords}
语调: {tone}

内容要求:
1. 吸引眼球的开头
2. 简洁有力的表达
3. 鼓励互动和分享
4. 适当使用表情符号
5. 包含相关话题标签

请确保内容有趣、易分享且符合平台特点。
"""
            }
        }
    
    def get_template(self, content_type: ContentType) -> Dict[str, Any]:
        """获取内容模板"""
        return self.templates.get(content_type, self.templates[ContentType.ARTICLE])
    
    def format_prompt(self, requirement: ContentRequirement) -> str:
        """格式化提示词"""
        template = self.get_template(requirement.content_type)
        
        return template["prompt"].format(
            word_count=requirement.word_count,
            style=requirement.style.value,
            title=requirement.title,
            target_audience=requirement.target_audience,
            key_points=", ".join(requirement.key_points),
            keywords=", ".join(requirement.keywords),
            tone=requirement.tone
        )

class ContentGenerator:
    """内容生成器"""
    
    def __init__(self, client: HarborAI):
        self.client = client
        self.template_manager = ContentTemplateManager()
    
    async def generate_content(self, requirement: ContentRequirement) -> str:
        """生成内容"""
        try:
            # 获取格式化的提示词
            prompt = self.template_manager.format_prompt(requirement)
            
            # 调用AI生成内容
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的内容创作专家，擅长创作各种类型的高质量内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=min(requirement.word_count * 2, 4000)  # 预留足够的token
            )
            
            content = response.choices[0].message.content.strip()
            
            # 后处理：确保内容符合要求
            content = self._post_process_content(content, requirement)
            
            return content
            
        except Exception as e:
            logger.error(f"内容生成失败: {str(e)}")
            raise e
    
    def _post_process_content(self, content: str, requirement: ContentRequirement) -> str:
        """后处理内容"""
        # 确保关键词自然融入
        for keyword in requirement.keywords:
            if keyword.lower() not in content.lower():
                # 如果关键词未出现，尝试在合适位置插入
                sentences = content.split('。')
                if len(sentences) > 1:
                    # 在第一段插入关键词
                    first_paragraph = sentences[0]
                    if len(first_paragraph) > 50:
                        insertion_point = len(first_paragraph) // 2
                        first_paragraph = (
                            first_paragraph[:insertion_point] + 
                            f"，{keyword}" + 
                            first_paragraph[insertion_point:]
                        )
                        sentences[0] = first_paragraph
                        content = '。'.join(sentences)
        
        return content
    
    async def generate_variations(self, requirement: ContentRequirement, count: int = 3) -> List[str]:
        """生成多个版本"""
        variations = []
        
        for i in range(count):
            # 调整温度参数以获得不同的变体
            temperature = 0.5 + (i * 0.2)
            
            try:
                prompt = self.template_manager.format_prompt(requirement)
                
                response = await self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": f"你是一个专业的内容创作专家。请创作第{i+1}个版本，确保与之前的版本有所不同。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=min(requirement.word_count * 2, 4000)
                )
                
                content = response.choices[0].message.content.strip()
                content = self._post_process_content(content, requirement)
                variations.append(content)
                
            except Exception as e:
                logger.error(f"生成变体 {i+1} 失败: {str(e)}")
                continue
        
        return variations

class QualityAssessor:
    """质量评估器"""
    
    def __init__(self, client: HarborAI):
        self.client = client
    
    async def assess_quality(self, content: str, requirement: ContentRequirement) -> QualityAssessment:
        """评估内容质量"""
        try:
            assessment_prompt = f"""
请对以下内容进行全面的质量评估：

内容类型: {requirement.content_type.value}
目标受众: {requirement.target_audience}
要求字数: {requirement.word_count}
关键词: {', '.join(requirement.keywords)}

内容:
{content}

请从以下维度评分（1-10分）并提供具体建议：

1. 可读性 (readability): 语言流畅度、句式多样性
2. 连贯性 (coherence): 逻辑结构、段落衔接
3. 相关性 (relevance): 与主题的匹配度
4. 原创性 (originality): 内容的独特性和新颖性
5. 吸引力 (engagement): 读者参与度和兴趣度
6. SEO优化 (seo_optimization): 关键词使用、搜索友好性

请以JSON格式返回评估结果：
{{
    "scores": {{
        "readability": 8.5,
        "coherence": 9.0,
        "relevance": 8.0,
        "originality": 7.5,
        "engagement": 8.5,
        "seo_optimization": 7.0
    }},
    "overall_score": 8.1,
    "suggestions": ["具体改进建议1", "具体改进建议2"],
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["不足1", "不足2"]
}}
"""
            
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的内容质量评估专家。"},
                    {"role": "user", "content": assessment_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content
            
            # 解析JSON结果
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                scores = result_data.get("scores", {})
                
                return QualityAssessment(
                    overall_score=result_data.get("overall_score", 0.0),
                    readability=scores.get("readability", 0.0),
                    coherence=scores.get("coherence", 0.0),
                    relevance=scores.get("relevance", 0.0),
                    originality=scores.get("originality", 0.0),
                    engagement=scores.get("engagement", 0.0),
                    seo_optimization=scores.get("seo_optimization", 0.0),
                    suggestions=result_data.get("suggestions", []),
                    strengths=result_data.get("strengths", []),
                    weaknesses=result_data.get("weaknesses", [])
                )
            
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
        
        # 返回默认评估
        return QualityAssessment(
            overall_score=5.0,
            readability=5.0,
            coherence=5.0,
            relevance=5.0,
            originality=5.0,
            engagement=5.0,
            seo_optimization=5.0,
            suggestions=["无法完成自动评估，请人工审核"],
            strengths=[],
            weaknesses=[]
        )

class SEOOptimizer:
    """SEO优化器"""
    
    def __init__(self, client: HarborAI):
        self.client = client
    
    async def optimize_for_seo(self, content: str, keywords: List[str], target_audience: str) -> Dict[str, Any]:
        """SEO优化分析"""
        try:
            seo_prompt = f"""
请对以下内容进行SEO优化分析：

目标关键词: {', '.join(keywords)}
目标受众: {target_audience}

内容:
{content}

请分析并提供以下信息：

1. 关键词密度分析
2. 标题优化建议
3. 元描述建议
4. 内部链接机会
5. 结构化数据建议
6. 用户体验优化建议

请以JSON格式返回：
{{
    "keyword_density": {{"关键词": "密度%"}},
    "title_suggestions": ["标题建议1", "标题建议2"],
    "meta_description": "推荐的元描述",
    "internal_links": ["链接机会1", "链接机会2"],
    "structured_data": ["结构化数据建议"],
    "ux_improvements": ["用户体验改进建议"],
    "seo_score": 8.5,
    "optimization_tips": ["优化技巧1", "优化技巧2"]
}}
"""
            
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个SEO优化专家。"},
                    {"role": "user", "content": seo_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content
            
            # 解析JSON结果
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.error(f"SEO优化分析失败: {str(e)}")
        
        # 返回默认结果
        return {
            "keyword_density": {},
            "title_suggestions": [],
            "meta_description": "",
            "internal_links": [],
            "structured_data": [],
            "ux_improvements": [],
            "seo_score": 5.0,
            "optimization_tips": []
        }

class ContentManager:
    """内容管理器"""
    
    def __init__(self, db_path: str = "content_system.db"):
        self.db_path = db_path
        self.content_cache: Dict[str, ContentPiece] = {}
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建内容表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_pieces (
                id TEXT PRIMARY KEY,
                content_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                author TEXT NOT NULL,
                version INTEGER NOT NULL,
                quality_score REAL,
                seo_score REAL,
                readability_score REAL,
                requirement_data TEXT NOT NULL,
                metadata TEXT,
                feedback TEXT,
                revisions TEXT
            )
        """)
        
        # 创建内容统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_stats (
                date DATE PRIMARY KEY,
                total_generated INTEGER DEFAULT 0,
                total_approved INTEGER DEFAULT 0,
                total_published INTEGER DEFAULT 0,
                avg_quality_score REAL DEFAULT 0.0,
                avg_generation_time REAL DEFAULT 0.0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_content(self, content_piece: ContentPiece):
        """保存内容"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO content_pieces 
            (id, content_type, title, content, status, created_at, updated_at, 
             author, version, quality_score, seo_score, readability_score,
             requirement_data, metadata, feedback, revisions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            content_piece.id,
            content_piece.requirement.content_type.value,
            content_piece.requirement.title,
            content_piece.content,
            content_piece.status.value,
            content_piece.created_at,
            content_piece.updated_at,
            content_piece.author,
            content_piece.version,
            content_piece.quality_score,
            content_piece.seo_score,
            content_piece.readability_score,
            json.dumps(asdict(content_piece.requirement)),
            json.dumps(content_piece.metadata),
            json.dumps(content_piece.feedback),
            json.dumps(content_piece.revisions)
        ))
        
        conn.commit()
        conn.close()
        
        # 更新缓存
        self.content_cache[content_piece.id] = content_piece
    
    def get_content(self, content_id: str) -> Optional[ContentPiece]:
        """获取内容"""
        if content_id in self.content_cache:
            return self.content_cache[content_id]
        
        # 从数据库加载
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM content_pieces WHERE id = ?", (content_id,))
        row = cursor.fetchone()
        
        if row:
            # 重构ContentRequirement
            requirement_data = json.loads(row[12])
            requirement = ContentRequirement(
                content_type=ContentType(requirement_data["content_type"]),
                title=requirement_data["title"],
                target_audience=requirement_data["target_audience"],
                key_points=requirement_data["key_points"],
                style=ContentStyle(requirement_data["style"]),
                word_count=requirement_data["word_count"],
                keywords=requirement_data.get("keywords", []),
                tone=requirement_data.get("tone", "neutral"),
                language=requirement_data.get("language", "zh-CN"),
                seo_requirements=requirement_data.get("seo_requirements", {}),
                brand_guidelines=requirement_data.get("brand_guidelines", {}),
                deadline=datetime.fromisoformat(requirement_data["deadline"]) if requirement_data.get("deadline") else None,
                priority=requirement_data.get("priority", 3)
            )
            
            content_piece = ContentPiece(
                id=row[0],
                requirement=requirement,
                content=row[3],
                status=ContentStatus(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                author=row[7],
                version=row[8],
                quality_score=row[9],
                seo_score=row[10],
                readability_score=row[11],
                metadata=json.loads(row[13] or "{}"),
                feedback=json.loads(row[14] or "[]"),
                revisions=json.loads(row[15] or "[]")
            )
            
            self.content_cache[content_id] = content_piece
            conn.close()
            return content_piece
        
        conn.close()
        return None
    
    def get_content_by_status(self, status: ContentStatus) -> List[ContentPiece]:
        """按状态获取内容"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM content_pieces WHERE status = ?", (status.value,))
        content_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return [self.get_content(content_id) for content_id in content_ids if self.get_content(content_id)]
    
    def update_content_status(self, content_id: str, status: ContentStatus):
        """更新内容状态"""
        content = self.get_content(content_id)
        if content:
            content.status = status
            content.updated_at = datetime.now()
            self.save_content(content)
    
    def add_feedback(self, content_id: str, feedback: str):
        """添加反馈"""
        content = self.get_content(content_id)
        if content:
            content.feedback.append(feedback)
            content.updated_at = datetime.now()
            self.save_content(content)
    
    def get_content_stats(self) -> Dict[str, Any]:
        """获取内容统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总内容数
        cursor.execute("SELECT COUNT(*) FROM content_pieces")
        total_content = cursor.fetchone()[0]
        
        # 各状态内容数
        cursor.execute("""
            SELECT status, COUNT(*) FROM content_pieces GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())
        
        # 平均质量分
        cursor.execute("""
            SELECT AVG(quality_score) FROM content_pieces 
            WHERE quality_score IS NOT NULL
        """)
        avg_quality = cursor.fetchone()[0] or 0.0
        
        # 今日生成数
        today = datetime.now().date()
        cursor.execute("""
            SELECT COUNT(*) FROM content_pieces 
            WHERE DATE(created_at) = ?
        """, (today,))
        today_generated = cursor.fetchone()[0]
        
        # 内容类型分布
        cursor.execute("""
            SELECT content_type, COUNT(*) FROM content_pieces GROUP BY content_type
        """)
        type_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_content": total_content,
            "status_distribution": status_counts,
            "average_quality": round(avg_quality, 2),
            "today_generated": today_generated,
            "type_distribution": type_distribution
        }

class ContentGenerationSystem:
    """内容生成系统"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.deepseek.com/v1",
                 db_path: str = "content_system.db"):
        
        # 初始化组件
        self.client = HarborAI(api_key=api_key, base_url=base_url)
        self.generator = ContentGenerator(self.client)
        self.quality_assessor = QualityAssessor(self.client)
        self.seo_optimizer = SEOOptimizer(self.client)
        self.content_manager = ContentManager(db_path)
        
        # 性能统计
        self.stats = {
            "total_generated": 0,
            "total_generation_time": 0.0,
            "quality_scores": [],
            "seo_scores": []
        }
    
    async def create_content(self, requirement: ContentRequirement) -> ContentPiece:
        """创建内容"""
        start_time = time.time()
        
        try:
            # 生成内容
            content = await self.generator.generate_content(requirement)
            
            # 创建内容对象
            content_piece = ContentPiece(
                id=str(uuid.uuid4()),
                requirement=requirement,
                content=content,
                status=ContentStatus.DRAFT,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # 质量评估
            quality_assessment = await self.quality_assessor.assess_quality(content, requirement)
            content_piece.quality_score = quality_assessment.overall_score
            content_piece.readability_score = quality_assessment.readability
            
            # SEO优化分析
            if requirement.keywords:
                seo_analysis = await self.seo_optimizer.optimize_for_seo(
                    content, requirement.keywords, requirement.target_audience
                )
                content_piece.seo_score = seo_analysis.get("seo_score", 0.0)
                content_piece.metadata["seo_analysis"] = seo_analysis
            
            # 保存质量评估结果
            content_piece.metadata["quality_assessment"] = asdict(quality_assessment)
            
            # 保存内容
            self.content_manager.save_content(content_piece)
            
            # 更新统计
            generation_time = time.time() - start_time
            self.stats["total_generated"] += 1
            self.stats["total_generation_time"] += generation_time
            self.stats["quality_scores"].append(quality_assessment.overall_score)
            if content_piece.seo_score:
                self.stats["seo_scores"].append(content_piece.seo_score)
            
            logger.info(f"内容生成完成: {content_piece.id} (质量分: {quality_assessment.overall_score:.1f})")
            
            return content_piece
            
        except Exception as e:
            logger.error(f"内容创建失败: {str(e)}")
            raise e
    
    async def create_content_variations(self, requirement: ContentRequirement, count: int = 3) -> List[ContentPiece]:
        """创建内容变体"""
        variations = []
        
        # 生成多个版本的内容
        content_list = await self.generator.generate_variations(requirement, count)
        
        for i, content in enumerate(content_list):
            # 创建内容对象
            content_piece = ContentPiece(
                id=str(uuid.uuid4()),
                requirement=requirement,
                content=content,
                status=ContentStatus.DRAFT,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version=i + 1
            )
            
            # 质量评估
            quality_assessment = await self.quality_assessor.assess_quality(content, requirement)
            content_piece.quality_score = quality_assessment.overall_score
            content_piece.metadata["quality_assessment"] = asdict(quality_assessment)
            
            # 保存内容
            self.content_manager.save_content(content_piece)
            variations.append(content_piece)
        
        return variations
    
    async def batch_generate_content(self, requirements: List[ContentRequirement]) -> List[ContentPiece]:
        """批量生成内容"""
        logger.info(f"开始批量生成 {len(requirements)} 个内容")
        
        # 并发生成内容
        tasks = [self.create_content(req) for req in requirements]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤成功的结果
        successful_results = [
            result for result in results 
            if isinstance(result, ContentPiece)
        ]
        
        # 记录失败的任务
        failed_count = len(results) - len(successful_results)
        if failed_count > 0:
            logger.warning(f"批量生成完成，成功: {len(successful_results)}, 失败: {failed_count}")
        else:
            logger.info(f"批量生成全部成功: {len(successful_results)} 个内容")
        
        return successful_results
    
    async def optimize_content(self, content_id: str) -> ContentPiece:
        """优化内容"""
        content = self.content_manager.get_content(content_id)
        if not content:
            raise ValueError(f"内容不存在: {content_id}")
        
        # 获取优化建议
        quality_assessment = await self.quality_assessor.assess_quality(
            content.content, content.requirement
        )
        
        if quality_assessment.suggestions:
            # 根据建议重新生成内容
            optimization_prompt = f"""
请根据以下建议优化内容：

原内容:
{content.content}

优化建议:
{chr(10).join(quality_assessment.suggestions)}

请保持原有的核心信息和结构，重点改进建议中提到的问题。
"""
            
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的内容编辑，擅长根据反馈优化内容。"},
                    {"role": "user", "content": optimization_prompt}
                ],
                temperature=0.5,
                max_tokens=4000
            )
            
            optimized_content = response.choices[0].message.content.strip()
            
            # 创建新版本
            content.content = optimized_content
            content.version += 1
            content.updated_at = datetime.now()
            content.revisions.append(f"优化版本 {content.version}")
            
            # 重新评估质量
            new_assessment = await self.quality_assessor.assess_quality(
                optimized_content, content.requirement
            )
            content.quality_score = new_assessment.overall_score
            content.metadata["quality_assessment"] = asdict(new_assessment)
            
            # 保存优化后的内容
            self.content_manager.save_content(content)
            
            logger.info(f"内容优化完成: {content_id} (质量分: {content.quality_score:.1f})")
        
        return content
    
    def get_content_by_id(self, content_id: str) -> Optional[ContentPiece]:
        """根据ID获取内容"""
        return self.content_manager.get_content(content_id)
    
    def get_content_by_status(self, status: ContentStatus) -> List[ContentPiece]:
        """根据状态获取内容"""
        return self.content_manager.get_content_by_status(status)
    
    def approve_content(self, content_id: str) -> bool:
        """批准内容"""
        try:
            self.content_manager.update_content_status(content_id, ContentStatus.APPROVED)
            return True
        except Exception as e:
            logger.error(f"批准内容失败: {str(e)}")
            return False
    
    def publish_content(self, content_id: str) -> bool:
        """发布内容"""
        try:
            self.content_manager.update_content_status(content_id, ContentStatus.PUBLISHED)
            return True
        except Exception as e:
            logger.error(f"发布内容失败: {str(e)}")
            return False
    
    def add_feedback(self, content_id: str, feedback: str) -> bool:
        """添加反馈"""
        try:
            self.content_manager.add_feedback(content_id, feedback)
            return True
        except Exception as e:
            logger.error(f"添加反馈失败: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        content_stats = self.content_manager.get_content_stats()
        
        avg_generation_time = 0.0
        if self.stats["total_generated"] > 0:
            avg_generation_time = self.stats["total_generation_time"] / self.stats["total_generated"]
        
        avg_quality = 0.0
        if self.stats["quality_scores"]:
            avg_quality = sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"])
        
        avg_seo = 0.0
        if self.stats["seo_scores"]:
            avg_seo = sum(self.stats["seo_scores"]) / len(self.stats["seo_scores"])
        
        return {
            "generation_stats": {
                "total_generated": self.stats["total_generated"],
                "average_generation_time": round(avg_generation_time, 3),
                "average_quality_score": round(avg_quality, 2),
                "average_seo_score": round(avg_seo, 2)
            },
            "content_stats": content_stats
        }

# 演示函数
async def demo_basic_content_generation():
    """演示基础内容生成"""
    print("\n📝 基础内容生成演示")
    print("=" * 50)
    
    # 创建内容生成系统
    system = ContentGenerationSystem(api_key="your-deepseek-key")
    
    # 创建内容需求
    requirement = ContentRequirement(
        content_type=ContentType.ARTICLE,
        title="人工智能在医疗领域的应用",
        target_audience="医疗从业者和技术爱好者",
        key_points=[
            "AI诊断技术的发展",
            "机器学习在药物研发中的作用",
            "智能医疗设备的普及",
            "数据隐私和伦理考虑"
        ],
        style=ContentStyle.PROFESSIONAL,
        word_count=800,
        keywords=["人工智能", "医疗", "机器学习", "智能诊断"],
        tone="专业且易懂"
    )
    
    print(f"📋 内容需求:")
    print(f"   类型: {requirement.content_type.value}")
    print(f"   标题: {requirement.title}")
    print(f"   目标受众: {requirement.target_audience}")
    print(f"   字数要求: {requirement.word_count}")
    print(f"   关键词: {', '.join(requirement.keywords)}")
    
    # 生成内容
    print(f"\n🔄 正在生成内容...")
    start_time = time.time()
    
    content_piece = await system.create_content(requirement)
    
    generation_time = time.time() - start_time
    
    print(f"✅ 内容生成完成!")
    print(f"   内容ID: {content_piece.id}")
    print(f"   生成时间: {generation_time:.3f}s")
    print(f"   质量评分: {content_piece.quality_score:.1f}/10")
    print(f"   SEO评分: {content_piece.seo_score:.1f}/10" if content_piece.seo_score else "   SEO评分: 未评估")
    print(f"   字数: {len(content_piece.content)}")
    
    # 显示内容预览
    preview = content_piece.content[:200] + "..." if len(content_piece.content) > 200 else content_piece.content
    print(f"\n📄 内容预览:")
    print(f"   {preview}")
    
    # 显示质量评估详情
    if "quality_assessment" in content_piece.metadata:
        assessment = content_piece.metadata["quality_assessment"]
        print(f"\n📊 质量评估详情:")
        print(f"   可读性: {assessment['readability']:.1f}/10")
        print(f"   连贯性: {assessment['coherence']:.1f}/10")
        print(f"   相关性: {assessment['relevance']:.1f}/10")
        print(f"   原创性: {assessment['originality']:.1f}/10")
        print(f"   吸引力: {assessment['engagement']:.1f}/10")
        
        if assessment['suggestions']:
            print(f"   改进建议: {', '.join(assessment['suggestions'][:2])}")
    
    return system, content_piece

async def demo_content_variations():
    """演示内容变体生成"""
    print("\n🎭 内容变体生成演示")
    print("=" * 50)
    
    system = ContentGenerationSystem(api_key="your-deepseek-key")
    
    # 创建营销文案需求
    requirement = ContentRequirement(
        content_type=ContentType.MARKETING_COPY,
        title="智能手表新品发布",
        target_audience="科技爱好者和健身人群",
        key_points=[
            "7天超长续航",
            "专业健康监测",
            "时尚外观设计",
            "智能语音助手"
        ],
        style=ContentStyle.PERSUASIVE,
        word_count=300,
        keywords=["智能手表", "健康监测", "长续航", "时尚"],
        tone="激动人心且具有说服力"
    )
    
    print(f"📋 营销文案需求:")
    print(f"   产品: {requirement.title}")
    print(f"   目标受众: {requirement.target_audience}")
    print(f"   核心卖点: {', '.join(requirement.key_points)}")
    
    # 生成多个变体
    print(f"\n🔄 正在生成 3 个文案变体...")
    
    variations = await system.create_content_variations(requirement, count=3)
    
    print(f"✅ 变体生成完成!")
    
    for i, variation in enumerate(variations):
        print(f"\n📝 变体 {i+1} (ID: {variation.id[:8]}...):")
        print(f"   质量评分: {variation.quality_score:.1f}/10")
        
        # 显示内容预览
        preview = variation.content[:150] + "..." if len(variation.content) > 150 else variation.content
        print(f"   内容预览: {preview}")
    
    # 选择最佳变体
    best_variation = max(variations, key=lambda x: x.quality_score)
    print(f"\n🏆 推荐最佳变体: {best_variation.id[:8]}... (质量分: {best_variation.quality_score:.1f})")
    
    return variations

async def demo_batch_generation():
    """演示批量内容生成"""
    print("\n⚡ 批量内容生成演示")
    print("=" * 50)
    
    system = ContentGenerationSystem(api_key="your-deepseek-key")
    
    # 创建多个内容需求
    requirements = [
        ContentRequirement(
            content_type=ContentType.PRODUCT_DESC,
            title="无线蓝牙耳机",
            target_audience="音乐爱好者",
            key_points=["高音质", "降噪功能", "长续航"],
            style=ContentStyle.PERSUASIVE,
            word_count=200,
            keywords=["蓝牙耳机", "降噪", "音质"]
        ),
        ContentRequirement(
            content_type=ContentType.SOCIAL_MEDIA,
            title="春季新品上市",
            target_audience="时尚年轻人",
            key_points=["春季新款", "限时优惠", "潮流设计"],
            style=ContentStyle.CASUAL,
            word_count=100,
            keywords=["春季", "新品", "优惠"]
        ),
        ContentRequirement(
            content_type=ContentType.BLOG_POST,
            title="远程办公效率提升指南",
            target_audience="职场人士",
            key_points=["时间管理", "工具推荐", "沟通技巧"],
            style=ContentStyle.INFORMATIVE,
            word_count=600,
            keywords=["远程办公", "效率", "时间管理"]
        )
    ]
    
    print(f"📋 批量生成任务:")
    for i, req in enumerate(requirements):
        print(f"   {i+1}. {req.content_type.value}: {req.title}")
    
    # 批量生成
    print(f"\n🔄 正在批量生成 {len(requirements)} 个内容...")
    start_time = time.time()
    
    results = await system.batch_generate_content(requirements)
    
    total_time = time.time() - start_time
    
    print(f"✅ 批量生成完成!")
    print(f"   成功生成: {len(results)} 个内容")
    print(f"   总耗时: {total_time:.3f}s")
    print(f"   平均耗时: {total_time/len(results):.3f}s/个")
    
    # 显示结果摘要
    total_quality = sum(content.quality_score for content in results)
    avg_quality = total_quality / len(results) if results else 0
    
    print(f"\n📊 生成结果摘要:")
    print(f"   平均质量分: {avg_quality:.1f}/10")
    
    for content in results:
        print(f"   - {content.requirement.content_type.value}: {content.quality_score:.1f}/10")
    
    return results

async def demo_content_optimization():
    """演示内容优化"""
    print("\n🔧 内容优化演示")
    print("=" * 50)
    
    system = ContentGenerationSystem(api_key="your-deepseek-key")
    
    # 先生成一个内容
    requirement = ContentRequirement(
        content_type=ContentType.ARTICLE,
        title="区块链技术的未来发展",
        target_audience="技术从业者",
        key_points=["去中心化应用", "智能合约", "数字货币"],
        style=ContentStyle.PROFESSIONAL,
        word_count=500,
        keywords=["区块链", "去中心化", "智能合约"]
    )
    
    print(f"📝 生成初始内容...")
    original_content = await system.create_content(requirement)
    
    print(f"✅ 初始内容生成完成:")
    print(f"   内容ID: {original_content.id}")
    print(f"   质量评分: {original_content.quality_score:.1f}/10")
    
    # 显示优化建议
    if "quality_assessment" in original_content.metadata:
        assessment = original_content.metadata["quality_assessment"]
        if assessment["suggestions"]:
            print(f"\n💡 优化建议:")
            for i, suggestion in enumerate(assessment["suggestions"][:3]):
                print(f"   {i+1}. {suggestion}")
    
    # 执行优化
    print(f"\n🔄 正在优化内容...")
    optimized_content = await system.optimize_content(original_content.id)
    
    print(f"✅ 内容优化完成:")
    print(f"   优化前质量分: {original_content.quality_score:.1f}/10")
    print(f"   优化后质量分: {optimized_content.quality_score:.1f}/10")
    print(f"   质量提升: {optimized_content.quality_score - original_content.quality_score:+.1f}")
    print(f"   当前版本: {optimized_content.version}")
    
    return optimized_content

async def demo_content_workflow():
    """演示内容工作流"""
    print("\n🔄 内容工作流演示")
    print("=" * 50)
    
    system = ContentGenerationSystem(api_key="your-deepseek-key")
    
    # 1. 创建内容
    requirement = ContentRequirement(
        content_type=ContentType.MARKETING_COPY,
        title="夏季促销活动",
        target_audience="购物爱好者",
        key_points=["全场5折", "限时3天", "包邮服务"],
        style=ContentStyle.PERSUASIVE,
        word_count=250,
        keywords=["夏季促销", "5折", "限时"]
    )
    
    print(f"📝 1. 创建内容...")
    content = await system.create_content(requirement)
    print(f"   状态: {content.status.value}")
    print(f"   质量分: {content.quality_score:.1f}/10")
    
    # 2. 添加反馈
    print(f"\n💬 2. 添加编辑反馈...")
    feedback = "文案很有吸引力，但建议增加更多紧迫感的表达"
    system.add_feedback(content.id, feedback)
    print(f"   反馈已添加: {feedback}")
    
    # 3. 优化内容
    print(f"\n🔧 3. 根据反馈优化内容...")
    optimized_content = await system.optimize_content(content.id)
    print(f"   优化后质量分: {optimized_content.quality_score:.1f}/10")
    
    # 4. 审核批准
    print(f"\n✅ 4. 审核批准...")
    system.approve_content(content.id)
    approved_content = system.get_content_by_id(content.id)
    print(f"   状态: {approved_content.status.value}")
    
    # 5. 发布内容
    print(f"\n🚀 5. 发布内容...")
    system.publish_content(content.id)
    published_content = system.get_content_by_id(content.id)
    print(f"   状态: {published_content.status.value}")
    
    # 6. 查看工作流统计
    print(f"\n📊 6. 工作流统计:")
    stats = system.get_system_stats()
    print(f"   总生成数: {stats['generation_stats']['total_generated']}")
    print(f"   平均质量分: {stats['generation_stats']['average_quality_score']}")
    print(f"   状态分布: {stats['content_stats']['status_distribution']}")
    
    return published_content

async def demo_seo_optimization():
    """演示SEO优化"""
    print("\n🔍 SEO优化演示")
    print("=" * 50)
    
    system = ContentGenerationSystem(api_key="your-api-key-here")
    
    # 创建需要SEO优化的内容
    requirement = ContentRequirement(
        content_type=ContentType.BLOG_POST,
        title="如何选择最适合的云服务器",
        target_audience="中小企业IT管理员",
        key_points=[
            "性能需求评估",
            "成本效益分析",
            "安全性考虑",
            "扩展性规划"
        ],
        style=ContentStyle.INFORMATIVE,
        word_count=800,
        keywords=["云服务器", "选择指南", "性能评估", "成本分析"],
        seo_requirements={
            "target_keywords": ["云服务器", "服务器选择", "云计算"],
            "meta_description_length": 160,
            "title_length": 60
        }
    )
    
    print(f"📝 生成SEO优化内容...")
    content = await system.create_content(requirement)
    
    print(f"✅ 内容生成完成:")
    print(f"   质量评分: {content.quality_score:.1f}/10")
    print(f"   SEO评分: {content.seo_score:.1f}/10")
    
    # 显示SEO分析结果
    if "seo_analysis" in content.metadata:
        seo_analysis = content.metadata["seo_analysis"]
        
        print(f"\n🔍 SEO分析结果:")
        
        # 关键词密度
        if "keyword_density" in seo_analysis:
            print(f"   关键词密度:")
            for keyword, density in seo_analysis["keyword_density"].items():
                print(f"     - {keyword}: {density}")
        
        # 标题建议
        if seo_analysis.get("title_suggestions"):
            print(f"   标题优化建议:")
            for i, suggestion in enumerate(seo_analysis["title_suggestions"][:2]):
                print(f"     {i+1}. {suggestion}")
        
        # 元描述
        if seo_analysis.get("meta_description"):
            print(f"   推荐元描述: {seo_analysis['meta_description'][:100]}...")
        
        # 优化技巧
        if seo_analysis.get("optimization_tips"):
            print(f"   优化技巧:")
            for i, tip in enumerate(seo_analysis["optimization_tips"][:2]):
                print(f"     {i+1}. {tip}")
    
    return content

async def main():
    """主演示函数"""
    print("📝 HarborAI 智能内容生成系统演示")
    print("=" * 60)
    
    try:
        # 基础内容生成演示
        system, content = await demo_basic_content_generation()
        
        # 内容变体生成演示
        await demo_content_variations()
        
        # 批量内容生成演示
        await demo_batch_generation()
        
        # 内容优化演示
        await demo_content_optimization()
        
        # 内容工作流演示
        await demo_content_workflow()
        
        # SEO优化演示
        await demo_seo_optimization()
        
        # 显示最终统计
        final_stats = system.get_system_stats()
        print(f"\n📊 系统最终统计:")
        print(f"   总生成内容: {final_stats['generation_stats']['total_generated']}")
        print(f"   平均生成时间: {final_stats['generation_stats']['average_generation_time']:.3f}s")
        print(f"   平均质量分: {final_stats['generation_stats']['average_quality_score']:.1f}/10")
        print(f"   平均SEO分: {final_stats['generation_stats']['average_seo_score']:.1f}/10")
        print(f"   内容类型分布: {final_stats['content_stats']['type_distribution']}")
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境部署建议:")
        print("   1. 建立内容审核和版本控制流程")
        print("   2. 配置多级质量检查和人工审核")
        print("   3. 实现内容模板和品牌指南管理")
        print("   4. 添加内容性能追踪和A/B测试")
        print("   5. 集成CMS和发布平台API")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())