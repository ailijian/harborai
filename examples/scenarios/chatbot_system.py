#!/usr/bin/env python3
"""
HarborAI 智能聊天机器人系统

场景描述:
构建企业级智能客服系统，支持多轮对话、上下文记忆、情感分析和智能路由。
适用于客户服务、技术支持、销售咨询等多种业务场景。

应用价值:
- 提升客户服务效率和满意度
- 降低人工客服成本
- 提供7x24小时不间断服务
- 积累客户交互数据用于业务优化

核心功能:
1. 多轮对话管理和上下文保持
2. 用户意图识别和情感分析
3. 智能问题路由和升级机制
4. 对话历史存储和分析
5. 实时性能监控和优化
"""

import asyncio
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import re
from collections import defaultdict, deque
import threading
from harborai import HarborAI
from harborai.core.base_plugin import ChatCompletion

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntentType(Enum):
    """意图类型"""
    GREETING = "greeting"           # 问候
    QUESTION = "question"           # 询问
    COMPLAINT = "complaint"         # 投诉
    PRAISE = "praise"              # 表扬
    REQUEST = "request"            # 请求
    GOODBYE = "goodbye"            # 告别
    UNKNOWN = "unknown"            # 未知

class EmotionType(Enum):
    """情感类型"""
    POSITIVE = "positive"          # 积极
    NEGATIVE = "negative"          # 消极
    NEUTRAL = "neutral"            # 中性
    ANGRY = "angry"               # 愤怒
    HAPPY = "happy"               # 开心
    CONFUSED = "confused"         # 困惑

class ConversationStatus(Enum):
    """对话状态"""
    ACTIVE = "active"             # 活跃
    WAITING = "waiting"           # 等待
    ESCALATED = "escalated"       # 已升级
    RESOLVED = "resolved"         # 已解决
    CLOSED = "closed"             # 已关闭

class Priority(Enum):
    """优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Message:
    """消息"""
    id: str
    conversation_id: str
    sender: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    intent: Optional[IntentType] = None
    emotion: Optional[EmotionType] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """对话上下文"""
    conversation_id: str
    user_id: str
    status: ConversationStatus
    priority: Priority
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    escalation_reason: Optional[str] = None
    satisfaction_score: Optional[float] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class IntentAnalysisResult:
    """意图分析结果"""
    intent: IntentType
    confidence: float
    entities: Dict[str, Any]
    emotion: EmotionType
    emotion_confidence: float

class IntentClassifier:
    """意图分类器"""
    
    def __init__(self, client: HarborAI):
        self.client = client
        
        # 意图识别提示模板
        self.intent_prompt = """
你是一个专业的客服意图分析助手。请分析用户消息的意图和情感。

用户消息: "{message}"

请以JSON格式返回分析结果:
{{
    "intent": "greeting|question|complaint|praise|request|goodbye|unknown",
    "confidence": 0.0-1.0,
    "entities": {{"关键实体": "值"}},
    "emotion": "positive|negative|neutral|angry|happy|confused",
    "emotion_confidence": 0.0-1.0,
    "reasoning": "分析理由"
}}

分析要点:
1. 准确识别用户的主要意图
2. 提取关键实体信息（产品、服务、时间等）
3. 分析用户的情感倾向
4. 给出置信度评分
"""
    
    async def analyze_intent(self, message: str, context: List[Message] = None) -> IntentAnalysisResult:
        """分析用户意图"""
        try:
            # 构建上下文信息
            context_info = ""
            if context:
                recent_messages = context[-3:]  # 最近3条消息
                context_info = "\n".join([
                    f"{msg.sender}: {msg.content}" 
                    for msg in recent_messages
                ])
                context_info = f"\n\n对话上下文:\n{context_info}"
            
            # 调用AI进行意图分析
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.intent_prompt.format(message=message) + context_info},
                    {"role": "user", "content": f"请分析这条消息: {message}"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # 解析结果
            result_text = response.choices[0].message.content
            
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                
                return IntentAnalysisResult(
                    intent=IntentType(result_data.get("intent", "unknown")),
                    confidence=float(result_data.get("confidence", 0.0)),
                    entities=result_data.get("entities", {}),
                    emotion=EmotionType(result_data.get("emotion", "neutral")),
                    emotion_confidence=float(result_data.get("emotion_confidence", 0.0))
                )
            
        except Exception as e:
            logger.error(f"意图分析失败: {str(e)}")
        
        # 返回默认结果
        return IntentAnalysisResult(
            intent=IntentType.UNKNOWN,
            confidence=0.0,
            entities={},
            emotion=EmotionType.NEUTRAL,
            emotion_confidence=0.0
        )

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, db_path: str = "chatbot.db"):
        self.db_path = db_path
        self.active_conversations: Dict[str, ConversationContext] = {}
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建对话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                user_profile TEXT,
                session_data TEXT,
                escalation_reason TEXT,
                satisfaction_score REAL,
                tags TEXT
            )
        """)
        
        # 创建消息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                intent TEXT,
                emotion TEXT,
                confidence REAL,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_conversation(self, user_id: str, priority: Priority = Priority.MEDIUM) -> ConversationContext:
        """创建新对话"""
        conversation_id = str(uuid.uuid4())
        now = datetime.now()
        
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            status=ConversationStatus.ACTIVE,
            priority=priority,
            created_at=now,
            updated_at=now
        )
        
        self.active_conversations[conversation_id] = context
        self._save_conversation(context)
        
        logger.info(f"创建新对话: {conversation_id} (用户: {user_id})")
        return context
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """获取对话"""
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # 从数据库加载
        return self._load_conversation(conversation_id)
    
    def add_message(self, conversation_id: str, message: Message):
        """添加消息"""
        context = self.get_conversation(conversation_id)
        if context:
            context.messages.append(message)
            context.updated_at = datetime.now()
            
            # 更新活跃对话
            self.active_conversations[conversation_id] = context
            
            # 保存到数据库
            self._save_message(message)
            self._save_conversation(context)
    
    def update_conversation_status(self, conversation_id: str, status: ConversationStatus, reason: str = None):
        """更新对话状态"""
        context = self.get_conversation(conversation_id)
        if context:
            context.status = status
            context.updated_at = datetime.now()
            
            if status == ConversationStatus.ESCALATED:
                context.escalation_reason = reason
            
            self._save_conversation(context)
            logger.info(f"对话状态更新: {conversation_id} -> {status.value}")
    
    def set_satisfaction_score(self, conversation_id: str, score: float):
        """设置满意度评分"""
        context = self.get_conversation(conversation_id)
        if context:
            context.satisfaction_score = score
            context.updated_at = datetime.now()
            self._save_conversation(context)
    
    def _save_conversation(self, context: ConversationContext):
        """保存对话到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO conversations 
            (id, user_id, status, priority, created_at, updated_at, 
             user_profile, session_data, escalation_reason, satisfaction_score, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            context.conversation_id,
            context.user_id,
            context.status.value,
            context.priority.value,
            context.created_at,
            context.updated_at,
            json.dumps(context.user_profile),
            json.dumps(context.session_data),
            context.escalation_reason,
            context.satisfaction_score,
            json.dumps(context.tags)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_message(self, message: Message):
        """保存消息到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages 
            (id, conversation_id, sender, content, timestamp, intent, emotion, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message.id,
            message.conversation_id,
            message.sender,
            message.content,
            message.timestamp,
            message.intent.value if message.intent else None,
            message.emotion.value if message.emotion else None,
            message.confidence,
            json.dumps(message.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _load_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """从数据库加载对话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 加载对话基本信息
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # 构建对话上下文
        context = ConversationContext(
            conversation_id=row[0],
            user_id=row[1],
            status=ConversationStatus(row[2]),
            priority=Priority(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            updated_at=datetime.fromisoformat(row[5]),
            user_profile=json.loads(row[6] or "{}"),
            session_data=json.loads(row[7] or "{}"),
            escalation_reason=row[8],
            satisfaction_score=row[9],
            tags=json.loads(row[10] or "[]")
        )
        
        # 加载消息
        cursor.execute("""
            SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp
        """, (conversation_id,))
        
        for msg_row in cursor.fetchall():
            message = Message(
                id=msg_row[0],
                conversation_id=msg_row[1],
                sender=msg_row[2],
                content=msg_row[3],
                timestamp=datetime.fromisoformat(msg_row[4]),
                intent=IntentType(msg_row[5]) if msg_row[5] else None,
                emotion=EmotionType(msg_row[6]) if msg_row[6] else None,
                confidence=msg_row[7] or 0.0,
                metadata=json.loads(msg_row[8] or "{}")
            )
            context.messages.append(message)
        
        conn.close()
        return context
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总对话数
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        # 各状态对话数
        cursor.execute("""
            SELECT status, COUNT(*) FROM conversations GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())
        
        # 平均满意度
        cursor.execute("""
            SELECT AVG(satisfaction_score) FROM conversations 
            WHERE satisfaction_score IS NOT NULL
        """)
        avg_satisfaction = cursor.fetchone()[0] or 0.0
        
        # 今日对话数
        today = datetime.now().date()
        cursor.execute("""
            SELECT COUNT(*) FROM conversations 
            WHERE DATE(created_at) = ?
        """, (today,))
        today_conversations = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_conversations": total_conversations,
            "status_distribution": status_counts,
            "average_satisfaction": round(avg_satisfaction, 2),
            "today_conversations": today_conversations,
            "active_conversations": len(self.active_conversations)
        }

class ResponseGenerator:
    """回复生成器"""
    
    def __init__(self, client: HarborAI):
        self.client = client
        
        # 回复生成提示模板
        self.response_prompt = """
你是一个专业、友好的客服助手。请根据用户的消息和对话上下文生成合适的回复。

用户意图: {intent}
用户情感: {emotion}
对话上下文: {context}

用户消息: "{message}"

回复要求:
1. 语气友好、专业
2. 针对用户意图给出准确回复
3. 考虑用户情感状态
4. 保持对话连贯性
5. 如果无法解决问题，建议转人工客服

请生成一个合适的回复:
"""
    
    async def generate_response(self, 
                              message: str, 
                              intent: IntentType, 
                              emotion: EmotionType,
                              context: List[Message] = None) -> str:
        """生成回复"""
        try:
            # 构建上下文信息
            context_info = "无"
            if context:
                recent_messages = context[-5:]  # 最近5条消息
                context_info = "\n".join([
                    f"{msg.sender}: {msg.content}" 
                    for msg in recent_messages
                ])
            
            # 调用AI生成回复
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.response_prompt.format(
                        intent=intent.value,
                        emotion=emotion.value,
                        context=context_info,
                        message=message
                    )},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"回复生成失败: {str(e)}")
            return "抱歉，我遇到了一些技术问题。请稍后再试，或联系人工客服。"

class EscalationManager:
    """升级管理器"""
    
    def __init__(self):
        self.escalation_rules = {
            # 情感触发规则
            EmotionType.ANGRY: {"threshold": 0.7, "priority": Priority.HIGH},
            EmotionType.NEGATIVE: {"threshold": 0.8, "priority": Priority.MEDIUM},
            
            # 意图触发规则
            IntentType.COMPLAINT: {"threshold": 0.6, "priority": Priority.HIGH},
            
            # 关键词触发规则
            "keywords": {
                "投诉": Priority.HIGH,
                "退款": Priority.MEDIUM,
                "故障": Priority.HIGH,
                "紧急": Priority.URGENT,
                "法律": Priority.URGENT
            }
        }
    
    def should_escalate(self, 
                       message: str, 
                       intent: IntentType, 
                       emotion: EmotionType,
                       confidence: float,
                       conversation_context: ConversationContext) -> Tuple[bool, str, Priority]:
        """判断是否需要升级"""
        
        # 检查情感触发
        if emotion in self.escalation_rules:
            rule = self.escalation_rules[emotion]
            if confidence >= rule["threshold"]:
                return True, f"检测到{emotion.value}情感，置信度{confidence:.2f}", rule["priority"]
        
        # 检查意图触发
        if intent in self.escalation_rules:
            rule = self.escalation_rules[intent]
            if confidence >= rule["threshold"]:
                return True, f"检测到{intent.value}意图，置信度{confidence:.2f}", rule["priority"]
        
        # 检查关键词触发
        for keyword, priority in self.escalation_rules["keywords"].items():
            if keyword in message:
                return True, f"检测到关键词: {keyword}", priority
        
        # 检查对话轮次（超过10轮未解决）
        if len(conversation_context.messages) > 20:  # 10轮对话=20条消息
            return True, "对话轮次过多，可能需要人工介入", Priority.MEDIUM
        
        # 检查重复问题
        user_messages = [msg.content for msg in conversation_context.messages if msg.sender == "user"]
        if len(user_messages) >= 3:
            recent_messages = user_messages[-3:]
            if len(set(recent_messages)) == 1:  # 连续3次相同问题
                return True, "用户重复提问，可能不满意回复", Priority.MEDIUM
        
        return False, "", Priority.LOW

class ChatbotSystem:
    """聊天机器人系统"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.deepseek.com/v1",
                 db_path: str = "chatbot.db"):
        
        # 初始化组件
        self.client = HarborAI(api_key=api_key, base_url=base_url)
        self.intent_classifier = IntentClassifier(self.client)
        self.conversation_manager = ConversationManager(db_path)
        self.response_generator = ResponseGenerator(self.client)
        self.escalation_manager = EscalationManager()
        
        # 性能统计
        self.stats = {
            "total_messages": 0,
            "total_response_time": 0.0,
            "intent_accuracy": 0.0,
            "escalation_rate": 0.0,
            "satisfaction_scores": []
        }
    
    async def process_message(self, 
                            user_id: str, 
                            message: str,
                            conversation_id: str = None) -> Dict[str, Any]:
        """处理用户消息"""
        start_time = time.time()
        
        try:
            # 获取或创建对话
            if conversation_id:
                context = self.conversation_manager.get_conversation(conversation_id)
                if not context:
                    context = self.conversation_manager.create_conversation(user_id)
            else:
                context = self.conversation_manager.create_conversation(user_id)
            
            # 创建用户消息
            user_message = Message(
                id=str(uuid.uuid4()),
                conversation_id=context.conversation_id,
                sender="user",
                content=message,
                timestamp=datetime.now()
            )
            
            # 意图分析
            intent_result = await self.intent_classifier.analyze_intent(
                message, context.messages
            )
            
            # 更新消息的意图和情感信息
            user_message.intent = intent_result.intent
            user_message.emotion = intent_result.emotion
            user_message.confidence = intent_result.confidence
            user_message.metadata = {
                "entities": intent_result.entities,
                "emotion_confidence": intent_result.emotion_confidence
            }
            
            # 添加用户消息到对话
            self.conversation_manager.add_message(context.conversation_id, user_message)
            
            # 检查是否需要升级
            should_escalate, escalation_reason, priority = self.escalation_manager.should_escalate(
                message, intent_result.intent, intent_result.emotion, 
                intent_result.confidence, context
            )
            
            if should_escalate:
                # 升级到人工客服
                self.conversation_manager.update_conversation_status(
                    context.conversation_id, 
                    ConversationStatus.ESCALATED, 
                    escalation_reason
                )
                
                response_text = f"我理解您的问题很重要。为了更好地帮助您，我已经将您的问题转接给人工客服。客服人员会尽快与您联系。\n\n升级原因: {escalation_reason}"
            else:
                # 生成AI回复
                response_text = await self.response_generator.generate_response(
                    message, intent_result.intent, intent_result.emotion, context.messages
                )
            
            # 创建助手回复消息
            assistant_message = Message(
                id=str(uuid.uuid4()),
                conversation_id=context.conversation_id,
                sender="assistant",
                content=response_text,
                timestamp=datetime.now(),
                metadata={
                    "escalated": should_escalate,
                    "escalation_reason": escalation_reason if should_escalate else None
                }
            )
            
            # 添加助手消息到对话
            self.conversation_manager.add_message(context.conversation_id, assistant_message)
            
            # 更新统计
            response_time = time.time() - start_time
            self.stats["total_messages"] += 1
            self.stats["total_response_time"] += response_time
            
            return {
                "conversation_id": context.conversation_id,
                "response": response_text,
                "intent": intent_result.intent.value,
                "emotion": intent_result.emotion.value,
                "confidence": intent_result.confidence,
                "escalated": should_escalate,
                "escalation_reason": escalation_reason if should_escalate else None,
                "response_time": response_time,
                "entities": intent_result.entities
            }
            
        except Exception as e:
            logger.error(f"消息处理失败: {str(e)}")
            return {
                "error": str(e),
                "response": "抱歉，系统遇到了问题。请稍后再试。"
            }
    
    async def get_conversation_history(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """获取对话历史"""
        context = self.conversation_manager.get_conversation(conversation_id)
        if not context:
            return None
        
        return {
            "conversation_id": context.conversation_id,
            "user_id": context.user_id,
            "status": context.status.value,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat(),
            "satisfaction_score": context.satisfaction_score,
            "messages": [
                {
                    "id": msg.id,
                    "sender": msg.sender,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "intent": msg.intent.value if msg.intent else None,
                    "emotion": msg.emotion.value if msg.emotion else None,
                    "confidence": msg.confidence
                }
                for msg in context.messages
            ]
        }
    
    async def set_satisfaction_rating(self, conversation_id: str, rating: float) -> bool:
        """设置满意度评分"""
        try:
            self.conversation_manager.set_satisfaction_score(conversation_id, rating)
            self.stats["satisfaction_scores"].append(rating)
            return True
        except Exception as e:
            logger.error(f"设置满意度评分失败: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        conversation_stats = self.conversation_manager.get_conversation_stats()
        
        avg_response_time = 0.0
        if self.stats["total_messages"] > 0:
            avg_response_time = self.stats["total_response_time"] / self.stats["total_messages"]
        
        avg_satisfaction = 0.0
        if self.stats["satisfaction_scores"]:
            avg_satisfaction = sum(self.stats["satisfaction_scores"]) / len(self.stats["satisfaction_scores"])
        
        return {
            "message_stats": {
                "total_messages": self.stats["total_messages"],
                "average_response_time": round(avg_response_time, 3)
            },
            "conversation_stats": conversation_stats,
            "satisfaction": {
                "average_score": round(avg_satisfaction, 2),
                "total_ratings": len(self.stats["satisfaction_scores"])
            }
        }

# 演示函数
async def demo_basic_conversation():
    """演示基础对话功能"""
    print("\n🤖 基础对话功能演示")
    print("=" * 50)
    
    # 创建聊天机器人系统
    chatbot = ChatbotSystem(api_key="your-deepseek-key")
    
    # 模拟用户对话
    user_id = "user_001"
    test_messages = [
        "你好，我想咨询一下产品信息",
        "我的订单什么时候能到？",
        "我对服务很不满意，要投诉！",
        "谢谢你的帮助"
    ]
    
    conversation_id = None
    
    for i, message in enumerate(test_messages):
        print(f"\n👤 用户: {message}")
        
        # 处理消息
        result = await chatbot.process_message(user_id, message, conversation_id)
        
        if "error" in result:
            print(f"❌ 错误: {result['error']}")
            continue
        
        conversation_id = result["conversation_id"]
        
        print(f"🤖 助手: {result['response']}")
        print(f"📊 意图: {result['intent']} (置信度: {result['confidence']:.2f})")
        print(f"😊 情感: {result['emotion']}")
        print(f"⏱️ 响应时间: {result['response_time']:.3f}s")
        
        if result.get("escalated"):
            print(f"🚨 已升级: {result['escalation_reason']}")
    
    # 设置满意度评分
    await chatbot.set_satisfaction_rating(conversation_id, 4.5)
    print(f"\n⭐ 用户满意度评分: 4.5/5.0")
    
    return chatbot, conversation_id

async def demo_intent_analysis():
    """演示意图分析功能"""
    print("\n🧠 意图分析功能演示")
    print("=" * 50)
    
    chatbot = ChatbotSystem(api_key="your-deepseek-key")
    
    # 测试不同类型的消息
    test_cases = [
        ("你好", "问候"),
        ("我的订单在哪里？", "询问"),
        ("这个产品太差了，我要退款！", "投诉"),
        ("服务很棒，谢谢！", "表扬"),
        ("请帮我取消订单", "请求"),
        ("再见", "告别"),
        ("asdfghjkl", "未知")
    ]
    
    for message, expected_type in test_cases:
        print(f"\n📝 测试消息: {message}")
        print(f"🎯 预期类型: {expected_type}")
        
        # 分析意图
        result = await chatbot.intent_classifier.analyze_intent(message)
        
        print(f"✅ 识别意图: {result.intent.value}")
        print(f"📊 置信度: {result.confidence:.2f}")
        print(f"😊 情感: {result.emotion.value} (置信度: {result.emotion_confidence:.2f})")
        
        if result.entities:
            print(f"🏷️ 实体: {result.entities}")

async def demo_escalation_system():
    """演示升级系统功能"""
    print("\n🚨 升级系统功能演示")
    print("=" * 50)
    
    chatbot = ChatbotSystem(api_key="your-deepseek-key")
    
    # 测试升级触发场景
    escalation_cases = [
        ("我非常愤怒！你们的服务太差了！", "情感触发"),
        ("我要投诉你们公司！", "意图触发"),
        ("这是紧急情况，需要立即处理！", "关键词触发"),
        ("我已经问了很多次了，为什么还是没有解决？", "重复问题")
    ]
    
    user_id = "escalation_test_user"
    
    for message, trigger_type in escalation_cases:
        print(f"\n📝 测试消息: {message}")
        print(f"🎯 触发类型: {trigger_type}")
        
        # 处理消息
        result = await chatbot.process_message(user_id, message)
        
        print(f"🤖 回复: {result['response']}")
        
        if result.get("escalated"):
            print(f"🚨 升级触发: {result['escalation_reason']}")
        else:
            print("✅ 未触发升级")

async def demo_conversation_management():
    """演示对话管理功能"""
    print("\n💬 对话管理功能演示")
    print("=" * 50)
    
    chatbot = ChatbotSystem(api_key="your-deepseek-key")
    
    # 创建多个对话
    conversations = []
    for i in range(3):
        user_id = f"user_{i+1:03d}"
        
        # 发送初始消息
        result = await chatbot.process_message(user_id, f"你好，我是用户{i+1}")
        conversations.append(result["conversation_id"])
        
        # 继续对话
        await chatbot.process_message(user_id, "我有一个问题需要咨询", result["conversation_id"])
        
        # 设置满意度评分
        await chatbot.set_satisfaction_rating(result["conversation_id"], 4.0 + i * 0.5)
    
    # 显示对话历史
    for i, conv_id in enumerate(conversations):
        print(f"\n📋 对话 {i+1} 历史:")
        history = await chatbot.get_conversation_history(conv_id)
        
        if history:
            print(f"   用户ID: {history['user_id']}")
            print(f"   状态: {history['status']}")
            print(f"   满意度: {history['satisfaction_score']}")
            print(f"   消息数: {len(history['messages'])}")
    
    # 显示系统统计
    stats = chatbot.get_system_stats()
    print(f"\n📊 系统统计:")
    print(f"   总消息数: {stats['message_stats']['total_messages']}")
    print(f"   平均响应时间: {stats['message_stats']['average_response_time']}s")
    print(f"   总对话数: {stats['conversation_stats']['total_conversations']}")
    print(f"   平均满意度: {stats['satisfaction']['average_score']}/5.0")

async def demo_performance_monitoring():
    """演示性能监控功能"""
    print("\n📈 性能监控功能演示")
    print("=" * 50)
    
    chatbot = ChatbotSystem(api_key="your-deepseek-key")
    
    # 模拟高并发场景
    print("🔄 模拟高并发消息处理...")
    
    async def process_user_session(user_id: str, message_count: int):
        """模拟用户会话"""
        conversation_id = None
        for i in range(message_count):
            message = f"这是用户{user_id}的第{i+1}条消息"
            result = await chatbot.process_message(user_id, message, conversation_id)
            conversation_id = result.get("conversation_id")
            
            # 随机设置满意度评分
            if i == message_count - 1:  # 最后一条消息后评分
                import random
                rating = random.uniform(3.0, 5.0)
                await chatbot.set_satisfaction_rating(conversation_id, rating)
    
    # 并发处理多个用户会话
    import asyncio
    tasks = []
    for i in range(10):  # 10个并发用户
        user_id = f"perf_user_{i+1:03d}"
        task = asyncio.create_task(process_user_session(user_id, 3))
        tasks.append(task)
    
    start_time = time.time()
    await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # 显示性能统计
    stats = chatbot.get_system_stats()
    
    print(f"\n📊 性能统计结果:")
    print(f"   总处理时间: {total_time:.3f}s")
    print(f"   总消息数: {stats['message_stats']['total_messages']}")
    print(f"   平均响应时间: {stats['message_stats']['average_response_time']:.3f}s")
    print(f"   消息处理速率: {stats['message_stats']['total_messages'] / total_time:.1f} msg/s")
    print(f"   总对话数: {stats['conversation_stats']['total_conversations']}")
    print(f"   今日对话数: {stats['conversation_stats']['today_conversations']}")
    print(f"   活跃对话数: {stats['conversation_stats']['active_conversations']}")
    print(f"   平均满意度: {stats['satisfaction']['average_score']}/5.0")
    print(f"   满意度样本数: {stats['satisfaction']['total_ratings']}")

async def main():
    """主演示函数"""
    print("🤖 HarborAI 智能聊天机器人系统演示")
    print("=" * 60)
    
    try:
        # 基础对话功能演示
        chatbot, conversation_id = await demo_basic_conversation()
        
        # 意图分析功能演示
        await demo_intent_analysis()
        
        # 升级系统功能演示
        await demo_escalation_system()
        
        # 对话管理功能演示
        await demo_conversation_management()
        
        # 性能监控功能演示
        await demo_performance_monitoring()
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境部署建议:")
        print("   1. 配置数据库连接池和缓存")
        print("   2. 实现负载均衡和故障转移")
        print("   3. 添加详细的日志和监控")
        print("   4. 设置API限流和安全防护")
        print("   5. 定期备份对话数据和模型优化")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())