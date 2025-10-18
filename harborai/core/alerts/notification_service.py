#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通知服务

负责告警通知的发送，支持多种通知渠道和模板
"""

import asyncio
import logging
import smtplib
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DINGTALK = "dingtalk"
    WECHAT = "wechat"
    SMS = "sms"
    CONSOLE = "console"


class NotificationPriority(Enum):
    """通知优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(Enum):
    """通知状态"""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"
    SUPPRESSED = "suppressed"


@dataclass
class NotificationTemplate:
    """通知模板"""
    id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    format_type: str = "text"  # text, html, markdown
    variables: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """渲染模板"""
        try:
            subject = self.subject_template.format(**context)
            body = self.body_template.format(**context)
            return {"subject": subject, "body": body}
        except KeyError as e:
            logger.error(f"模板渲染失败，缺少变量: {e}")
            return {"subject": "告警通知", "body": str(context)}


@dataclass
class NotificationConfig:
    """通知配置"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[int] = None  # 每分钟最大发送数
    retry_count: int = 3
    retry_delay: int = 60  # 重试延迟（秒）
    timeout: int = 30  # 超时时间（秒）


@dataclass
class NotificationRecord:
    """通知记录"""
    id: str
    alert_id: str
    channel: NotificationChannel
    recipient: str
    subject: str
    body: str
    status: str  # pending, sent, failed, retrying
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class NotificationResult:
    """通知结果"""
    message_id: str
    status: NotificationStatus
    channel: NotificationChannel
    recipients: List[str]
    sent_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationService:
    """通知服务"""
    
    def __init__(self):
        self.configs: Dict[NotificationChannel, NotificationConfig] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.records: List[NotificationRecord] = []
        self.rate_limiters: Dict[NotificationChannel, Dict[str, List[datetime]]] = {}
        self.retry_queue: List[NotificationRecord] = []
        self.running = False
        
    async def initialize(self):
        """初始化通知服务"""
        logger.info("初始化通知服务")
        await self._load_default_configs()
        await self._load_default_templates()
        
    async def add_config(self, config: NotificationConfig):
        """添加通知配置"""
        self.configs[config.channel] = config
        logger.info(f"添加通知配置: {config.channel.value}")
        
    async def add_template(self, template: NotificationTemplate):
        """添加通知模板"""
        self.templates[template.id] = template
        logger.info(f"添加通知模板: {template.name} ({template.id})")
        
    async def send_alert_notification(self, alert, rule):
        """发送告警通知"""
        try:
            # 获取通知渠道
            channels = rule.notification_channels or ["console"]
            
            # 准备通知上下文
            context = self._prepare_alert_context(alert, rule)
            
            # 发送到各个渠道
            for channel_name in channels:
                try:
                    channel = NotificationChannel(channel_name)
                    await self._send_notification(channel, alert, context, rule.notification_template)
                except ValueError:
                    logger.warning(f"未知的通知渠道: {channel_name}")
                    
        except Exception as e:
            logger.error(f"发送告警通知失败: {e}")
            
    async def send_resolution_notification(self, alert, reason: str):
        """发送解决通知"""
        try:
            # 查找原始告警的通知记录
            alert_records = [r for r in self.records if r.alert_id == alert.id]
            if not alert_records:
                return
                
            # 准备解决通知上下文
            context = self._prepare_resolution_context(alert, reason)
            
            # 发送解决通知到相同渠道
            channels = set(r.channel for r in alert_records)
            for channel in channels:
                await self._send_notification(channel, alert, context, "resolution")
                
        except Exception as e:
            logger.error(f"发送解决通知失败: {e}")
            
    async def send_custom_notification(
        self,
        channel: NotificationChannel,
        recipient: str,
        subject: str,
        body: str,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> bool:
        """发送自定义通知"""
        try:
            record = NotificationRecord(
                id=f"custom_{datetime.now().timestamp()}",
                alert_id="custom",
                channel=channel,
                recipient=recipient,
                subject=subject,
                body=body,
                status="pending"
            )
            
            success = await self._send_notification_record(record)
            return success
            
        except Exception as e:
            logger.error(f"发送自定义通知失败: {e}")
            return False
            
    async def _send_notification(
        self,
        channel: NotificationChannel,
        alert,
        context: Dict[str, Any],
        template_id: Optional[str] = None
    ):
        """发送通知到指定渠道"""
        try:
            # 检查渠道配置
            if channel not in self.configs or not self.configs[channel].enabled:
                logger.warning(f"通知渠道未配置或已禁用: {channel.value}")
                return
                
            config = self.configs[channel]
            
            # 获取模板
            template = self._get_template(channel, template_id)
            if not template:
                logger.warning(f"未找到通知模板: {channel.value}, {template_id}")
                return
                
            # 渲染模板
            rendered = template.render(context)
            
            # 获取收件人
            recipients = self._get_recipients(channel, config, alert)
            
            # 发送通知
            for recipient in recipients:
                # 检查速率限制
                if not await self._check_rate_limit(channel, recipient):
                    logger.warning(f"通知速率限制: {channel.value} -> {recipient}")
                    continue
                    
                record = NotificationRecord(
                    id=f"{alert.id}_{channel.value}_{recipient}_{datetime.now().timestamp()}",
                    alert_id=alert.id,
                    channel=channel,
                    recipient=recipient,
                    subject=rendered["subject"],
                    body=rendered["body"],
                    status="pending"
                )
                
                await self._send_notification_record(record)
                
        except Exception as e:
            logger.error(f"发送通知失败 {channel.value}: {e}")
            
    async def _send_notification_record(self, record: NotificationRecord) -> bool:
        """发送通知记录"""
        try:
            config = self.configs[record.channel]
            
            # 根据渠道类型发送通知
            if record.channel == NotificationChannel.EMAIL:
                success = await self._send_email(record, config)
            elif record.channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook(record, config)
            elif record.channel == NotificationChannel.SLACK:
                success = await self._send_slack(record, config)
            elif record.channel == NotificationChannel.DINGTALK:
                success = await self._send_dingtalk(record, config)
            elif record.channel == NotificationChannel.WECHAT:
                success = await self._send_wechat(record, config)
            elif record.channel == NotificationChannel.SMS:
                success = await self._send_sms(record, config)
            elif record.channel == NotificationChannel.CONSOLE:
                success = await self._send_console(record, config)
            else:
                logger.warning(f"不支持的通知渠道: {record.channel.value}")
                success = False
            
            # 更新记录状态
            if success:
                record.status = "sent"
                record.sent_at = datetime.now()
                logger.info(f"通知发送成功: {record.channel.value} -> {record.recipient}")
            else:
                record.status = "failed"
                record.error_message = "发送失败"
                
                # 添加到重试队列
                if record.retry_count < config.retry_count:
                    record.retry_count += 1
                    record.status = "retrying"
                    self.retry_queue.append(record)
                    logger.info(f"通知加入重试队列: {record.id}")
            
            self.records.append(record)
            return success
            
        except Exception as e:
            logger.error(f"发送通知记录失败: {e}")
            record.status = "failed"
            record.error_message = str(e)
            self.records.append(record)
            return False

    async def _send_email(self, record: NotificationRecord, config: NotificationConfig) -> bool:
        """发送邮件通知"""
        try:
            smtp_config = config.config
            
            # 创建邮件
            msg = MIMEMultipart('alternative')
            msg['From'] = smtp_config.get('from_email', smtp_config.get('from'))
            msg['To'] = record.recipient
            msg['Subject'] = Header(record.subject, 'utf-8')
            
            # 添加邮件头信息
            msg['X-Priority'] = '1'  # 高优先级
            msg['X-MSMail-Priority'] = 'High'
            msg['Importance'] = 'High'
            
            # 检查邮件内容格式
            template = self._get_template(NotificationChannel.EMAIL, None)
            if template and template.format_type == 'html':
                # HTML格式邮件
                html_part = MIMEText(record.body, 'html', 'utf-8')
                msg.attach(html_part)
                
                # 同时添加纯文本版本
                import re
                text_body = re.sub(r'<[^>]+>', '', record.body)  # 简单的HTML标签移除
                text_part = MIMEText(text_body, 'plain', 'utf-8')
                msg.attach(text_part)
            else:
                # 纯文本邮件
                text_part = MIMEText(record.body, 'plain', 'utf-8')
                msg.attach(text_part)
            
            # 发送邮件
            smtp_host = smtp_config.get('smtp_server', smtp_config.get('host'))
            smtp_port = smtp_config.get('smtp_port', smtp_config.get('port', 587))
            
            if smtp_config.get('use_ssl', False):
                # 使用SSL连接
                with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                    if smtp_config.get('username'):
                        server.login(smtp_config['username'], smtp_config['password'])
                    server.send_message(msg)
            else:
                # 使用TLS连接
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    if smtp_config.get('use_tls', True):
                        server.starttls()
                    if smtp_config.get('username'):
                        server.login(smtp_config['username'], smtp_config['password'])
                    server.send_message(msg)
                
            logger.info(f"邮件发送成功: {record.recipient}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP认证失败: {e}")
            return False
        except smtplib.SMTPRecipientsRefused as e:
            logger.error(f"收件人被拒绝: {e}")
            return False
        except smtplib.SMTPServerDisconnected as e:
            logger.error(f"SMTP服务器连接断开: {e}")
            return False
        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
            return False
            
    async def _send_webhook(self, record: NotificationRecord, config: NotificationConfig) -> bool:
        """发送Webhook通知"""
        try:
            webhook_config = config.config
            url = webhook_config['url']
            
            payload = {
                "alert_id": record.alert_id,
                "subject": record.subject,
                "body": record.body,
                "timestamp": datetime.now().isoformat(),
                "channel": record.channel.value
            }
            
            # 添加自定义字段
            if 'custom_fields' in webhook_config:
                payload.update(webhook_config['custom_fields'])
                
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout),
                    headers=webhook_config.get('headers', {})
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"发送Webhook失败: {e}")
            return False
            
    async def _send_slack(self, record: NotificationRecord, config: NotificationConfig) -> bool:
        """发送Slack通知"""
        try:
            slack_config = config.config
            webhook_url = slack_config['webhook_url']
            
            # 构建Slack消息
            payload = {
                "text": record.subject,
                "username": slack_config.get('username', 'HarborAI监控'),
                "icon_emoji": slack_config.get('icon_emoji', ':warning:'),
                "attachments": [
                    {
                        "color": self._get_slack_color(record),
                        "title": record.subject,
                        "text": record.body,
                        "fields": [
                            {
                                "title": "告警ID",
                                "value": record.alert_id,
                                "short": True
                            },
                            {
                                "title": "通知渠道",
                                "value": record.channel.value,
                                "short": True
                            },
                            {
                                "title": "发送时间",
                                "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "HarborAI监控系统",
                        "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            # 添加自定义字段
            if 'custom_fields' in slack_config:
                payload['attachments'][0]['fields'].extend(slack_config['custom_fields'])
            
            # 添加频道配置
            if 'channel' in slack_config:
                payload['channel'] = slack_config['channel']
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"发送Slack通知失败: {e}")
            return False
            
    async def _send_dingtalk(self, record: NotificationRecord, config: NotificationConfig) -> bool:
        """发送钉钉通知"""
        try:
            import time
            import hmac
            import hashlib
            import base64
            import urllib.parse
            
            dingtalk_config = config.config
            webhook_url = dingtalk_config['webhook_url']
            
            # 如果配置了secret，添加签名验证
            if 'secret' in dingtalk_config:
                timestamp = str(round(time.time() * 1000))
                secret = dingtalk_config['secret']
                secret_enc = secret.encode('utf-8')
                string_to_sign = f'{timestamp}\n{secret}'
                string_to_sign_enc = string_to_sign.encode('utf-8')
                hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
                sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
                webhook_url = f"{webhook_url}&timestamp={timestamp}&sign={sign}"
            
            # 根据告警级别选择不同的消息类型
            if dingtalk_config.get('use_markdown', False):
                payload = {
                    "msgtype": "markdown",
                    "markdown": {
                        "title": record.subject,
                        "text": f"## {record.subject}\n\n{record.body}"
                    }
                }
            else:
                payload = {
                    "msgtype": "text",
                    "text": {
                        "content": f"{record.subject}\n\n{record.body}"
                    }
                }
            
            # 添加@所有人或特定人员
            if dingtalk_config.get('at_all', False):
                payload["at"] = {"isAtAll": True}
            elif dingtalk_config.get('at_mobiles'):
                payload["at"] = {"atMobiles": dingtalk_config['at_mobiles']}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"发送钉钉通知失败: {e}")
            return False
            
    async def _send_wechat(self, record: NotificationRecord, config: NotificationConfig) -> bool:
        """发送微信通知"""
        try:
            wechat_config = config.config
            
            # 企业微信机器人Webhook
            if 'webhook_url' in wechat_config:
                webhook_url = wechat_config['webhook_url']
                
                # 构建微信消息
                if wechat_config.get('use_markdown', False):
                    payload = {
                        "msgtype": "markdown",
                        "markdown": {
                            "content": f"## {record.subject}\n\n{record.body}"
                        }
                    }
                else:
                    payload = {
                        "msgtype": "text",
                        "text": {
                            "content": f"{record.subject}\n\n{record.body}",
                            "mentioned_list": wechat_config.get('mentioned_list', []),
                            "mentioned_mobile_list": wechat_config.get('mentioned_mobile_list', [])
                        }
                    }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=config.timeout)
                    ) as response:
                        result = await response.json()
                        return result.get('errcode', 1) == 0
            
            # 企业微信应用消息
            elif 'corp_id' in wechat_config and 'corp_secret' in wechat_config:
                # 获取access_token
                token_url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
                token_params = {
                    "corpid": wechat_config['corp_id'],
                    "corpsecret": wechat_config['corp_secret']
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(token_url, params=token_params) as response:
                        token_result = await response.json()
                        if token_result.get('errcode', 1) != 0:
                            logger.error(f"获取微信access_token失败: {token_result}")
                            return False
                        
                        access_token = token_result['access_token']
                    
                    # 发送消息
                    send_url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}"
                    message_payload = {
                        "touser": record.recipient,
                        "msgtype": "text",
                        "agentid": wechat_config['agent_id'],
                        "text": {
                            "content": f"{record.subject}\n\n{record.body}"
                        }
                    }
                    
                    async with session.post(send_url, json=message_payload) as response:
                        result = await response.json()
                        return result.get('errcode', 1) == 0
            
            else:
                logger.error("微信通知配置不完整")
                return False
                
        except Exception as e:
            logger.error(f"发送微信通知失败: {e}")
            return False

    async def _send_sms(self, record: NotificationRecord, config: NotificationConfig) -> bool:
        """发送短信通知"""
        try:
            sms_config = config.config
            provider = sms_config.get('provider', 'aliyun')
            
            if provider == 'aliyun':
                return await self._send_aliyun_sms(record, sms_config)
            elif provider == 'tencent':
                return await self._send_tencent_sms(record, sms_config)
            elif provider == 'twilio':
                return await self._send_twilio_sms(record, sms_config)
            else:
                logger.error(f"不支持的短信服务商: {provider}")
                return False
                
        except Exception as e:
            logger.error(f"发送短信通知失败: {e}")
            return False

    async def _send_aliyun_sms(self, record: NotificationRecord, sms_config: Dict[str, Any]) -> bool:
        """发送阿里云短信"""
        try:
            import hashlib
            import hmac
            import base64
            import urllib.parse
            from datetime import datetime
            import uuid
            
            # 阿里云短信API参数
            access_key_id = sms_config['access_key_id']
            access_key_secret = sms_config['access_key_secret']
            sign_name = sms_config['sign_name']
            template_code = sms_config['template_code']
            
            # 构建请求参数
            params = {
                'Action': 'SendSms',
                'Version': '2017-05-25',
                'RegionId': 'cn-hangzhou',
                'PhoneNumbers': record.recipient,
                'SignName': sign_name,
                'TemplateCode': template_code,
                'TemplateParam': json.dumps({
                    'subject': record.subject,
                    'content': record.body[:50]  # 短信内容限制
                }),
                'AccessKeyId': access_key_id,
                'Format': 'JSON',
                'SignatureMethod': 'HMAC-SHA1',
                'SignatureVersion': '1.0',
                'SignatureNonce': str(uuid.uuid4()),
                'Timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            
            # 计算签名
            sorted_params = sorted(params.items())
            query_string = '&'.join([f'{k}={urllib.parse.quote_plus(str(v))}' for k, v in sorted_params])
            string_to_sign = f'POST&%2F&{urllib.parse.quote_plus(query_string)}'
            signature = base64.b64encode(
                hmac.new(
                    (access_key_secret + '&').encode('utf-8'),
                    string_to_sign.encode('utf-8'),
                    hashlib.sha1
                ).digest()
            ).decode('utf-8')
            
            params['Signature'] = signature
            
            # 发送请求
            url = 'https://dysmsapi.aliyuncs.com/'
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=params) as response:
                    result = await response.json()
                    return result.get('Code') == 'OK'
                    
        except Exception as e:
            logger.error(f"发送阿里云短信失败: {e}")
            return False

    async def _send_tencent_sms(self, record: NotificationRecord, sms_config: Dict[str, Any]) -> bool:
        """发送腾讯云短信"""
        try:
            import hashlib
            import hmac
            import time
            
            # 腾讯云短信API参数
            secret_id = sms_config['secret_id']
            secret_key = sms_config['secret_key']
            sdk_app_id = sms_config['sdk_app_id']
            template_id = sms_config['template_id']
            sign_name = sms_config['sign_name']
            
            # 构建请求
            endpoint = "sms.tencentcloudapi.com"
            service = "sms"
            version = "2021-01-11"
            action = "SendSms"
            region = "ap-guangzhou"
            
            timestamp = int(time.time())
            date = time.strftime('%Y-%m-%d', time.gmtime(timestamp))
            
            # 请求体
            payload = {
                "PhoneNumberSet": [record.recipient],
                "SmsSdkAppId": sdk_app_id,
                "TemplateId": template_id,
                "TemplateParamSet": [record.subject, record.body[:50]],
                "SignName": sign_name
            }
            
            payload_json = json.dumps(payload, separators=(',', ':'))
            
            # 计算签名
            algorithm = "TC3-HMAC-SHA256"
            canonical_request = f"POST\n/\n\ncontent-type:application/json; charset=utf-8\nhost:{endpoint}\n\ncontent-type;host\n{hashlib.sha256(payload_json.encode('utf-8')).hexdigest()}"
            credential_scope = f"{date}/{service}/tc3_request"
            string_to_sign = f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
            
            secret_date = hmac.new(("TC3" + secret_key).encode('utf-8'), date.encode('utf-8'), hashlib.sha256).digest()
            secret_service = hmac.new(secret_date, service.encode('utf-8'), hashlib.sha256).digest()
            secret_signing = hmac.new(secret_service, "tc3_request".encode('utf-8'), hashlib.sha256).digest()
            signature = hmac.new(secret_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            
            authorization = f"{algorithm} Credential={secret_id}/{credential_scope}, SignedHeaders=content-type;host, Signature={signature}"
            
            headers = {
                "Authorization": authorization,
                "Content-Type": "application/json; charset=utf-8",
                "Host": endpoint,
                "X-TC-Action": action,
                "X-TC-Timestamp": str(timestamp),
                "X-TC-Version": version,
                "X-TC-Region": region
            }
            
            # 发送请求
            url = f"https://{endpoint}/"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payload_json, headers=headers) as response:
                    result = await response.json()
                    return 'Error' not in result.get('Response', {})
                    
        except Exception as e:
            logger.error(f"发送腾讯云短信失败: {e}")
            return False

    async def _send_twilio_sms(self, record: NotificationRecord, sms_config: Dict[str, Any]) -> bool:
        """发送Twilio短信"""
        try:
            import base64
            
            account_sid = sms_config['account_sid']
            auth_token = sms_config['auth_token']
            from_number = sms_config['from_number']
            
            # 构建认证头
            credentials = f"{account_sid}:{auth_token}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "To": record.recipient,
                "From": from_number,
                "Body": f"{record.subject}\n{record.body}"
            }
            
            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"发送Twilio短信失败: {e}")
            return False

    async def _send_console(self, record: NotificationRecord, config: NotificationConfig) -> bool:
        """发送控制台通知"""
        try:
            print(f"\n{'='*50}")
            print(f"告警通知: {record.subject}")
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"收件人: {record.recipient}")
            print(f"内容: {record.body}")
            print(f"{'='*50}\n")
            return True
            
        except Exception as e:
            logger.error(f"发送控制台通知失败: {e}")
            return False
            
    def _prepare_alert_context(self, alert, rule) -> Dict[str, Any]:
        """准备告警通知上下文"""
        return {
            "alert_id": alert.id,
            "alert_name": alert.rule_name,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "message": alert.message,
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "started_at": alert.started_at.strftime('%Y-%m-%d %H:%M:%S'),
            "labels": json.dumps(alert.labels, ensure_ascii=False),
            "annotations": json.dumps(alert.annotations, ensure_ascii=False),
            "rule_description": rule.description
        }
        
    def _prepare_resolution_context(self, alert, reason: str) -> Dict[str, Any]:
        """准备解决通知上下文"""
        duration = ""
        if alert.resolved_at and alert.started_at:
            delta = alert.resolved_at - alert.started_at
            duration = str(delta)
            
        return {
            "alert_id": alert.id,
            "alert_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "started_at": alert.started_at.strftime('%Y-%m-%d %H:%M:%S'),
            "resolved_at": alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S') if alert.resolved_at else "",
            "duration": duration,
            "reason": reason
        }
        
    def _get_template(self, channel: NotificationChannel, template_id: Optional[str]) -> Optional[NotificationTemplate]:
        """获取通知模板"""
        if template_id and template_id in self.templates:
            return self.templates[template_id]
            
        # 查找默认模板
        for template in self.templates.values():
            if template.channel == channel and "default" in template.id:
                return template
                
        return None
        
    def _get_recipients(self, channel: NotificationChannel, config: NotificationConfig, alert) -> List[str]:
        """获取收件人列表"""
        recipients = config.config.get('recipients', [])
        
        # 根据告警严重程度过滤收件人
        if 'severity_recipients' in config.config:
            severity_recipients = config.config['severity_recipients'].get(alert.severity.value, [])
            recipients.extend(severity_recipients)
            
        return list(set(recipients))  # 去重
        
    async def _check_rate_limit(self, channel: NotificationChannel, recipient: str) -> bool:
        """检查速率限制"""
        config = self.configs[channel]
        if not config.rate_limit:
            return True
            
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # 初始化速率限制器
        if channel not in self.rate_limiters:
            self.rate_limiters[channel] = {}
        if recipient not in self.rate_limiters[channel]:
            self.rate_limiters[channel][recipient] = []
            
        # 清理过期记录
        timestamps = self.rate_limiters[channel][recipient]
        timestamps[:] = [ts for ts in timestamps if ts > minute_ago]
        
        # 检查是否超过限制
        if len(timestamps) >= config.rate_limit:
            return False
            
        # 记录当前时间
        timestamps.append(now)
        return True
        
    def _get_slack_color(self, record: NotificationRecord) -> str:
        """获取Slack消息颜色"""
        if "critical" in record.subject.lower():
            return "danger"
        elif "high" in record.subject.lower():
            return "warning"
        elif "resolved" in record.subject.lower():
            return "good"
        else:
            return "#439FE0"
            
    async def _load_default_configs(self):
        """加载默认通知配置"""
        # 控制台通知配置
        console_config = NotificationConfig(
            channel=NotificationChannel.CONSOLE,
            enabled=True,
            config={"recipients": ["console"]},
            rate_limit=None
        )
        await self.add_config(console_config)
        
        # 邮件通知配置（需要用户配置）
        email_config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            enabled=False,  # 默认禁用，需要配置
            config={
                "host": "smtp.example.com",
                "port": 587,
                "use_tls": True,
                "username": "",
                "password": "",
                "from": "alerts@example.com",
                "recipients": []
            },
            rate_limit=10  # 每分钟最多10封邮件
        )
        await self.add_config(email_config)
        
        # Webhook通知配置
        webhook_config = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            enabled=False,
            config={
                "url": "",
                "headers": {"Content-Type": "application/json"},
                "custom_fields": {}
            },
            rate_limit=60
        )
        await self.add_config(webhook_config)
        
        # 微信通知配置
        wechat_config = NotificationConfig(
            channel=NotificationChannel.WECHAT,
            enabled=False,
            config={
                # 企业微信机器人Webhook方式
                "webhook_url": "",
                "use_markdown": False,
                "mentioned_list": [],
                "mentioned_mobile_list": [],
                # 企业微信应用消息方式
                "corp_id": "",
                "corp_secret": "",
                "agent_id": "",
                "recipients": []
            },
            rate_limit=20  # 每分钟最多20条消息
        )
        await self.add_config(wechat_config)
        
        # 短信通知配置
        sms_config = NotificationConfig(
            channel=NotificationChannel.SMS,
            enabled=False,
            config={
                "provider": "aliyun",  # aliyun, tencent, twilio
                # 阿里云短信配置
                "access_key_id": "",
                "access_key_secret": "",
                "sign_name": "",
                "template_code": "",
                # 腾讯云短信配置
                "secret_id": "",
                "secret_key": "",
                "sdk_app_id": "",
                "template_id": "",
                # Twilio短信配置
                "account_sid": "",
                "auth_token": "",
                "from_number": "",
                "recipients": []
            },
            rate_limit=5  # 每分钟最多5条短信
        )
        await self.add_config(sms_config)
        
    async def _load_default_templates(self):
        """加载默认通知模板"""
        # 控制台告警模板
        console_alert_template = NotificationTemplate(
            id="console_alert_default",
            name="控制台告警模板",
            channel=NotificationChannel.CONSOLE,
            subject_template="🚨 {severity} 告警: {alert_name}",
            body_template="""告警详情:
- 告警名称: {alert_name}
- 严重程度: {severity}
- 状态: {status}
- 消息: {message}
- 指标值: {metric_value}
- 阈值: {threshold}
- 开始时间: {started_at}
- 标签: {labels}
- 注释: {annotations}
- 描述: {rule_description}""",
            format_type="text"
        )
        await self.add_template(console_alert_template)
        
        # 控制台解决模板
        console_resolution_template = NotificationTemplate(
            id="console_resolution_default",
            name="控制台解决模板",
            channel=NotificationChannel.CONSOLE,
            subject_template="✅ 告警已解决: {alert_name}",
            body_template="""告警解决详情:
- 告警名称: {alert_name}
- 严重程度: {severity}
- 消息: {message}
- 开始时间: {started_at}
- 解决时间: {resolved_at}
- 持续时间: {duration}
- 解决原因: {reason}""",
            format_type="text"
        )
        await self.add_template(console_resolution_template)
        
        # 邮件告警模板
        email_alert_template = NotificationTemplate(
            id="email_alert_default",
            name="邮件告警模板",
            channel=NotificationChannel.EMAIL,
            subject_template="[{severity}] HarborAI告警: {alert_name}",
            body_template="""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>告警通知</title>
</head>
<body>
    <h2 style="color: #d32f2f;">🚨 系统告警通知</h2>
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
        <tr><td><strong>告警名称</strong></td><td>{alert_name}</td></tr>
        <tr><td><strong>严重程度</strong></td><td>{severity}</td></tr>
        <tr><td><strong>状态</strong></td><td>{status}</td></tr>
        <tr><td><strong>消息</strong></td><td>{message}</td></tr>
        <tr><td><strong>指标值</strong></td><td>{metric_value}</td></tr>
        <tr><td><strong>阈值</strong></td><td>{threshold}</td></tr>
        <tr><td><strong>开始时间</strong></td><td>{started_at}</td></tr>
        <tr><td><strong>描述</strong></td><td>{rule_description}</td></tr>
    </table>
    <p><strong>标签:</strong> {labels}</p>
    <p><strong>注释:</strong> {annotations}</p>
    <hr>
    <p><small>此邮件由HarborAI监控系统自动发送</small></p>
</body>
</html>""",
            format_type="html"
        )
        await self.add_template(email_alert_template)
        
        # 邮件解决模板
        email_resolution_template = NotificationTemplate(
            id="email_resolution_default",
            name="邮件解决模板",
            channel=NotificationChannel.EMAIL,
            subject_template="[已解决] HarborAI告警: {alert_name}",
            body_template="""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>告警解决通知</title>
</head>
<body>
    <h2 style="color: #4caf50;">✅ 告警已解决</h2>
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
        <tr><td><strong>告警名称</strong></td><td>{alert_name}</td></tr>
        <tr><td><strong>严重程度</strong></td><td>{severity}</td></tr>
        <tr><td><strong>消息</strong></td><td>{message}</td></tr>
        <tr><td><strong>开始时间</strong></td><td>{started_at}</td></tr>
        <tr><td><strong>解决时间</strong></td><td>{resolved_at}</td></tr>
        <tr><td><strong>持续时间</strong></td><td>{duration}</td></tr>
        <tr><td><strong>解决原因</strong></td><td>{reason}</td></tr>
    </table>
    <hr>
    <p><small>此邮件由HarborAI监控系统自动发送</small></p>
</body>
</html>""",
            format_type="html"
        )
        await self.add_template(email_resolution_template)
        
        # Slack告警模板
        slack_alert_template = NotificationTemplate(
            id="slack_alert_default",
            name="Slack告警模板",
            channel=NotificationChannel.SLACK,
            subject_template="🚨 {severity} 告警: {alert_name}",
            body_template="""*告警详情:*
• *告警名称:* {alert_name}
• *严重程度:* {severity}
• *状态:* {status}
• *消息:* {message}
• *指标值:* {metric_value}
• *阈值:* {threshold}
• *开始时间:* {started_at}
• *描述:* {rule_description}

*标签:* {labels}
*注释:* {annotations}""",
            format_type="markdown"
        )
        await self.add_template(slack_alert_template)
        
        # Slack解决模板
        slack_resolution_template = NotificationTemplate(
            id="slack_resolution_default",
            name="Slack解决模板",
            channel=NotificationChannel.SLACK,
            subject_template="✅ 告警已解决: {alert_name}",
            body_template="""*告警解决详情:*
• *告警名称:* {alert_name}
• *严重程度:* {severity}
• *消息:* {message}
• *开始时间:* {started_at}
• *解决时间:* {resolved_at}
• *持续时间:* {duration}
• *解决原因:* {reason}""",
            format_type="markdown"
        )
        await self.add_template(slack_resolution_template)
        
        # 钉钉告警模板
        dingtalk_alert_template = NotificationTemplate(
            id="dingtalk_alert_default",
            name="钉钉告警模板",
            channel=NotificationChannel.DINGTALK,
            subject_template="🚨 {severity} 告警: {alert_name}",
            body_template="""告警详情:
告警名称: {alert_name}
严重程度: {severity}
状态: {status}
消息: {message}
指标值: {metric_value}
阈值: {threshold}
开始时间: {started_at}
描述: {rule_description}

标签: {labels}
注释: {annotations}""",
            format_type="text"
        )
        await self.add_template(dingtalk_alert_template)
        
        # 钉钉解决模板
        dingtalk_resolution_template = NotificationTemplate(
            id="dingtalk_resolution_default",
            name="钉钉解决模板",
            channel=NotificationChannel.DINGTALK,
            subject_template="✅ 告警已解决: {alert_name}",
            body_template="""告警解决详情:
告警名称: {alert_name}
严重程度: {severity}
消息: {message}
开始时间: {started_at}
解决时间: {resolved_at}
持续时间: {duration}
解决原因: {reason}""",
            format_type="text"
        )
        await self.add_template(dingtalk_resolution_template)
        
        # 微信告警模板
        wechat_alert_template = NotificationTemplate(
            id="wechat_alert_default",
            name="微信告警模板",
            channel=NotificationChannel.WECHAT,
            subject_template="🚨 {severity} 告警: {alert_name}",
            body_template="""告警详情:
告警名称: {alert_name}
严重程度: {severity}
状态: {status}
消息: {message}
指标值: {metric_value}
阈值: {threshold}
开始时间: {started_at}
描述: {rule_description}

标签: {labels}
注释: {annotations}""",
            format_type="text"
        )
        await self.add_template(wechat_alert_template)
        
        # 微信解决模板
        wechat_resolution_template = NotificationTemplate(
            id="wechat_resolution_default",
            name="微信解决模板",
            channel=NotificationChannel.WECHAT,
            subject_template="✅ 告警已解决: {alert_name}",
            body_template="""告警解决详情:
告警名称: {alert_name}
严重程度: {severity}
消息: {message}
开始时间: {started_at}
解决时间: {resolved_at}
持续时间: {duration}
解决原因: {reason}""",
            format_type="text"
        )
        await self.add_template(wechat_resolution_template)
        
        # 短信告警模板
        sms_alert_template = NotificationTemplate(
            id="sms_alert_default",
            name="短信告警模板",
            channel=NotificationChannel.SMS,
            subject_template="{severity}告警:{alert_name}",
            body_template="【HarborAI】{severity}告警:{alert_name},指标值:{metric_value},阈值:{threshold},时间:{started_at}",
            format_type="text"
        )
        await self.add_template(sms_alert_template)
        
        # 短信解决模板
        sms_resolution_template = NotificationTemplate(
            id="sms_resolution_default",
            name="短信解决模板",
            channel=NotificationChannel.SMS,
            subject_template="告警已解决:{alert_name}",
            body_template="【HarborAI】告警已解决:{alert_name},持续时间:{duration},解决时间:{resolved_at}",
            format_type="text"
        )
        await self.add_template(sms_resolution_template)

        logger.info("加载了默认通知模板")

    async def get_notification_statistics(self) -> Dict[str, Any]:
        """获取通知统计信息"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_records = [r for r in self.records if r.created_at >= last_24h]
        
        # 按渠道统计
        channel_stats = {}
        for channel in NotificationChannel:
            channel_records = [r for r in recent_records if r.channel == channel]
            channel_stats[channel.value] = {
                "total": len(channel_records),
                "sent": len([r for r in channel_records if r.status == "sent"]),
                "failed": len([r for r in channel_records if r.status == "failed"]),
                "pending": len([r for r in channel_records if r.status == "pending"])
            }
            
        # 按状态统计
        status_stats = {}
        for status in ["sent", "failed", "pending", "retrying"]:
            status_stats[status] = len([r for r in recent_records if r.status == status])
            
        return {
            "total_notifications_24h": len(recent_records),
            "channel_statistics": channel_stats,
            "status_statistics": status_stats,
            "retry_queue_size": len(self.retry_queue),
            "success_rate": (
                status_stats.get("sent", 0) / len(recent_records) * 100
                if recent_records else 0
            )
        }

    async def _send_wechat(self, record: NotificationRecord, config: NotificationConfig) -> bool:
        """发送微信通知"""
        try:
            wechat_config = config.config
            
            # 企业微信机器人Webhook
            if 'webhook_url' in wechat_config:
                webhook_url = wechat_config['webhook_url']
                
                # 构建微信消息
                if wechat_config.get('use_markdown', False):
                    payload = {
                        "msgtype": "markdown",
                        "markdown": {
                            "content": f"## {record.subject}\n\n{record.body}"
                        }
                    }
                else:
                    payload = {
                        "msgtype": "text",
                        "text": {
                            "content": f"{record.subject}\n\n{record.body}",
                            "mentioned_list": wechat_config.get('mentioned_list', []),
                            "mentioned_mobile_list": wechat_config.get('mentioned_mobile_list', [])
                        }
                    }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=config.timeout)
                    ) as response:
                        result = await response.json()
                        return result.get('errcode', 1) == 0
            
            # 企业微信应用消息
            elif 'corp_id' in wechat_config and 'corp_secret' in wechat_config:
                # 获取access_token
                token_url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
                token_params = {
                    "corpid": wechat_config['corp_id'],
                    "corpsecret": wechat_config['corp_secret']
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(token_url, params=token_params) as response:
                        token_result = await response.json()
                        if token_result.get('errcode', 1) != 0:
                            logger.error(f"获取微信access_token失败: {token_result}")
                            return False
                        
                        access_token = token_result['access_token']
                    
                    # 发送消息
                    send_url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}"
                    message_payload = {
                        "touser": record.recipient,
                        "msgtype": "text",
                        "agentid": wechat_config['agent_id'],
                        "text": {
                            "content": f"{record.subject}\n\n{record.body}"
                        }
                    }
                    
                    async with session.post(send_url, json=message_payload) as response:
                        result = await response.json()
                        return result.get('errcode', 1) == 0
            
            else:
                logger.error("微信通知配置不完整")
                return False
                
        except Exception as e:
            logger.error(f"发送微信通知失败: {e}")
            return False

    async def _send_sms(self, record: NotificationRecord, config: NotificationConfig) -> bool:
        """发送短信通知"""
        try:
            sms_config = config.config
            provider = sms_config.get('provider', 'aliyun')
            
            if provider == 'aliyun':
                return await self._send_aliyun_sms(record, sms_config)
            elif provider == 'tencent':
                return await self._send_tencent_sms(record, sms_config)
            elif provider == 'twilio':
                return await self._send_twilio_sms(record, sms_config)
            else:
                logger.error(f"不支持的短信服务商: {provider}")
                return False
                
        except Exception as e:
            logger.error(f"发送短信通知失败: {e}")
            return False

    async def _send_aliyun_sms(self, record: NotificationRecord, sms_config: Dict[str, Any]) -> bool:
        """发送阿里云短信"""
        try:
            import hashlib
            import hmac
            import base64
            import urllib.parse
            from datetime import datetime
            import uuid
            
            # 阿里云短信API参数
            access_key_id = sms_config['access_key_id']
            access_key_secret = sms_config['access_key_secret']
            sign_name = sms_config['sign_name']
            template_code = sms_config['template_code']
            
            # 构建请求参数
            params = {
                'Action': 'SendSms',
                'Version': '2017-05-25',
                'RegionId': 'cn-hangzhou',
                'PhoneNumbers': record.recipient,
                'SignName': sign_name,
                'TemplateCode': template_code,
                'TemplateParam': json.dumps({
                    'subject': record.subject,
                    'content': record.body[:50]  # 短信内容限制
                }),
                'AccessKeyId': access_key_id,
                'Format': 'JSON',
                'SignatureMethod': 'HMAC-SHA1',
                'SignatureVersion': '1.0',
                'SignatureNonce': str(uuid.uuid4()),
                'Timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            
            # 计算签名
            sorted_params = sorted(params.items())
            query_string = '&'.join([f'{k}={urllib.parse.quote_plus(str(v))}' for k, v in sorted_params])
            string_to_sign = f'POST&%2F&{urllib.parse.quote_plus(query_string)}'
            signature = base64.b64encode(
                hmac.new(
                    (access_key_secret + '&').encode('utf-8'),
                    string_to_sign.encode('utf-8'),
                    hashlib.sha1
                ).digest()
            ).decode('utf-8')
            
            params['Signature'] = signature
            
            # 发送请求
            url = 'https://dysmsapi.aliyuncs.com/'
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=params) as response:
                    result = await response.json()
                    return result.get('Code') == 'OK'
                    
        except Exception as e:
            logger.error(f"发送阿里云短信失败: {e}")
            return False

    async def _send_tencent_sms(self, record: NotificationRecord, sms_config: Dict[str, Any]) -> bool:
        """发送腾讯云短信"""
        try:
            import hashlib
            import hmac
            import time
            
            # 腾讯云短信API参数
            secret_id = sms_config['secret_id']
            secret_key = sms_config['secret_key']
            sdk_app_id = sms_config['sdk_app_id']
            template_id = sms_config['template_id']
            sign_name = sms_config['sign_name']
            
            # 构建请求
            endpoint = "sms.tencentcloudapi.com"
            service = "sms"
            version = "2021-01-11"
            action = "SendSms"
            region = "ap-guangzhou"
            
            timestamp = int(time.time())
            date = time.strftime('%Y-%m-%d', time.gmtime(timestamp))
            
            # 请求体
            payload = {
                "PhoneNumberSet": [record.recipient],
                "SmsSdkAppId": sdk_app_id,
                "TemplateId": template_id,
                "TemplateParamSet": [record.subject, record.body[:50]],
                "SignName": sign_name
            }
            
            payload_json = json.dumps(payload, separators=(',', ':'))
            
            # 计算签名
            algorithm = "TC3-HMAC-SHA256"
            canonical_request = f"POST\n/\n\ncontent-type:application/json; charset=utf-8\nhost:{endpoint}\n\ncontent-type;host\n{hashlib.sha256(payload_json.encode('utf-8')).hexdigest()}"
            credential_scope = f"{date}/{service}/tc3_request"
            string_to_sign = f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
            
            secret_date = hmac.new(("TC3" + secret_key).encode('utf-8'), date.encode('utf-8'), hashlib.sha256).digest()
            secret_service = hmac.new(secret_date, service.encode('utf-8'), hashlib.sha256).digest()
            secret_signing = hmac.new(secret_service, "tc3_request".encode('utf-8'), hashlib.sha256).digest()
            signature = hmac.new(secret_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            
            authorization = f"{algorithm} Credential={secret_id}/{credential_scope}, SignedHeaders=content-type;host, Signature={signature}"
            
            headers = {
                "Authorization": authorization,
                "Content-Type": "application/json; charset=utf-8",
                "Host": endpoint,
                "X-TC-Action": action,
                "X-TC-Timestamp": str(timestamp),
                "X-TC-Version": version,
                "X-TC-Region": region
            }
            
            # 发送请求
            url = f"https://{endpoint}/"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payload_json, headers=headers) as response:
                    result = await response.json()
                    return 'Error' not in result.get('Response', {})
                    
        except Exception as e:
            logger.error(f"发送腾讯云短信失败: {e}")
            return False

    async def _send_twilio_sms(self, record: NotificationRecord, sms_config: Dict[str, Any]) -> bool:
        """发送Twilio短信"""
        try:
            import base64
            
            account_sid = sms_config['account_sid']
            auth_token = sms_config['auth_token']
            from_number = sms_config['from_number']
            
            # 构建认证头
            credentials = f"{account_sid}:{auth_token}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "To": record.recipient,
                "From": from_number,
                "Body": f"{record.subject}\n{record.body}"
            }
            
            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"发送Twilio短信失败: {e}")
            return False

        # Slack通知配置
        slack_config = NotificationConfig(
            channel=NotificationChannel.SLACK,
            enabled=False,
            config={
                "webhook_url": "",
                "username": "HarborAI监控",
                "icon_emoji": ":warning:",
                "recipients": []
            },
            rate_limit=30
        )
        await self.add_config(slack_config)
        
        # 钉钉通知配置
        dingtalk_config = NotificationConfig(
            channel=NotificationChannel.DINGTALK,
            enabled=False,
            config={
                "webhook_url": "",
                "secret": "",  # 可选的安全设置
                "use_markdown": False,
                "at_all": False,
                "at_mobiles": [],
                "recipients": []
            },
            rate_limit=20
        )
        await self.add_config(dingtalk_config)
        
        # 微信通知配置
        wechat_config = NotificationConfig(
            channel=NotificationChannel.WECHAT,
            enabled=False,
            config={
                # 企业微信机器人方式
                "webhook_url": "",
                "use_markdown": False,
                "mentioned_list": [],
                "mentioned_mobile_list": [],
                # 企业微信应用方式
                "corp_id": "",
                "corp_secret": "",
                "agent_id": "",
                "recipients": []
            },
            rate_limit=20
        )
        await self.add_config(wechat_config)
        
        # 短信通知配置
        sms_config = NotificationConfig(
            channel=NotificationChannel.SMS,
            enabled=False,
            config={
                "provider": "aliyun",  # aliyun, tencent, twilio
                # 阿里云配置
                "access_key_id": "",
                "access_key_secret": "",
                "sign_name": "",
                "template_code": "",
                # 腾讯云配置
                "secret_id": "",
                "secret_key": "",
                "sdk_app_id": "",
                "template_id": "",
                # Twilio配置
                "account_sid": "",
                "auth_token": "",
                "from_number": "",
                "recipients": []
            },
            rate_limit=5  # 每分钟最多5条短信
        )
        await self.add_config(sms_config)
        
        logger.info("加载了默认通知配置")

        # Slack解决模板
        slack_resolution_template = NotificationTemplate(
            id="slack_resolution_default",
            name="Slack解决模板",
            channel=NotificationChannel.SLACK,
            subject_template="✅ 告警已解决: {alert_name}",
            body_template="""*告警解决详情:*
• *告警名称:* {alert_name}
• *严重程度:* {severity}
• *状态:* {status}
• *消息:* {message}
• *指标值:* {metric_value}
• *阈值:* {threshold}
• *开始时间:* {started_at}
• *描述:* {rule_description}

*标签:* {labels}
*注释:* {annotations}""",
            format_type="markdown"
        )
        await self.add_template(slack_resolution_template)
        
        # 钉钉解决模板
        dingtalk_resolution_template = NotificationTemplate(
            id="dingtalk_resolution_default",
            name="钉钉解决模板",
            channel=NotificationChannel.DINGTALK,
            subject_template="✅ 告警已解决: {alert_name}",
            body_template="""告警解决详情:
告警名称: {alert_name}
严重程度: {severity}
状态: {status}
消息: {message}
指标值: {metric_value}
阈值: {threshold}
开始时间: {started_at}
描述: {rule_description}

标签: {labels}
注释: {annotations}""",
            format_type="text"
        )
        await self.add_template(dingtalk_resolution_template)
        
        # 短信解决模板
        sms_resolution_template = NotificationTemplate(
            id="sms_resolution_default",
            name="短信解决模板",
            channel=NotificationChannel.SMS,
            subject_template="告警已解决:{alert_name}",
            body_template="【HarborAI】告警已解决:{alert_name},持续时间:{duration},解决时间:{resolved_at}",
            format_type="text"
        )
        await self.add_template(sms_resolution_template)

        logger.info("加载了默认通知模板")

    async def get_notification_statistics(self) -> Dict[str, Any]:
        """获取通知统计信息"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_records = [r for r in self.records if r.created_at >= last_24h]
        
        # 按渠道统计
        channel_stats = {}
        for channel in NotificationChannel:
            channel_records = [r for r in recent_records if r.channel == channel]
            channel_stats[channel.value] = {
                "total": len(channel_records),
                "sent": len([r for r in channel_records if r.status == "sent"]),
                "failed": len([r for r in channel_records if r.status == "failed"]),
                "pending": len([r for r in channel_records if r.status == "pending"])
            }
            
        # 按状态统计
        status_stats = {}
        for status in ["sent", "failed", "pending", "retrying"]:
            status_stats[status] = len([r for r in recent_records if r.status == status])
            
        return {
            "total_notifications_24h": len(recent_records),
            "channel_statistics": channel_stats,
            "status_statistics": status_stats,
            "retry_queue_size": len(self.retry_queue),
            "success_rate": (
                status_stats.get("sent", 0) / len(recent_records) * 100
                if recent_records else 0
            )
        }