#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通知服务单元测试

测试通知服务的各种通知渠道和功能
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from harborai.core.alerts.notification_service import (
    NotificationService, NotificationChannel, NotificationPriority,
    NotificationTemplate, NotificationResult, NotificationStatus
)
from harborai.core.alerts.alert_manager import Alert, AlertSeverity, AlertStatus


class TestNotificationService:
    """通知服务测试"""
    
    @pytest.fixture
    async def notification_service(self):
        """通知服务实例"""
        service = NotificationService()
        await service.initialize()
        return service
        
    @pytest.fixture
    def sample_alert(self):
        """示例告警"""
        return Alert(
            id="test_alert_001",
            rule_id="test_rule",
            name="测试告警",
            description="这是一个测试告警",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            metric="cpu_usage",
            value=85.0,
            threshold=80.0,
            labels={"component": "system", "env": "production"},
            annotations={
                "summary": "CPU使用率过高",
                "description": "系统CPU使用率达到85%，超过阈值80%",
                "runbook": "检查系统负载和进程"
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
    @pytest.fixture
    def console_channel(self):
        """控制台通知渠道"""
        return NotificationChannel(
            name="console",
            type="console",
            enabled=True,
            config={}
        )
        
    @pytest.fixture
    def email_channel(self):
        """邮件通知渠道"""
        return NotificationChannel(
            name="email",
            type="email",
            enabled=True,
            config={
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "alerts@example.com",
                "password": "password123",
                "from_email": "alerts@example.com",
                "to_emails": ["admin@example.com", "ops@example.com"]
            }
        )
        
    @pytest.fixture
    def webhook_channel(self):
        """Webhook通知渠道"""
        return NotificationChannel(
            name="webhook",
            type="webhook",
            enabled=True,
            config={
                "url": "https://hooks.example.com/webhook",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "timeout": 30
            }
        )
        
    @pytest.fixture
    def slack_channel(self):
        """Slack通知渠道"""
        return NotificationChannel(
            name="slack",
            type="slack",
            enabled=True,
            config={
                "webhook_url": "https://hooks.slack.com/services/TXXXXXXXX/BXXXXXXXX/XXXXXXXXXXXXXXXXXXXXXXXX",
                "channel": "#alerts",
                "username": "AlertBot"
            }
        )
        
    @pytest.fixture
    def dingtalk_channel(self):
        """钉钉通知渠道"""
        return NotificationChannel(
            name="dingtalk",
            type="dingtalk",
            enabled=True,
            config={
                "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=xxxxxxxx",
                "secret": "SECxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            }
        )

    async def test_initialization(self, notification_service):
        """测试初始化"""
        assert notification_service is not None
        assert notification_service.channels == {}
        assert notification_service.templates == {}
        assert notification_service.rate_limiter is not None
        
    async def test_add_channel(self, notification_service, console_channel):
        """测试添加通知渠道"""
        await notification_service.add_channel(console_channel)
        assert "console" in notification_service.channels
        assert notification_service.channels["console"] == console_channel
        
    async def test_add_duplicate_channel(self, notification_service, console_channel):
        """测试添加重复通知渠道"""
        await notification_service.add_channel(console_channel)
        
        # 添加同名渠道应该覆盖原有渠道
        new_channel = NotificationChannel(
            name="console",
            type="console",
            enabled=False,
            config={"debug": True}
        )
        await notification_service.add_channel(new_channel)
        assert notification_service.channels["console"] == new_channel
        
    async def test_remove_channel(self, notification_service, console_channel):
        """测试移除通知渠道"""
        await notification_service.add_channel(console_channel)
        await notification_service.remove_channel("console")
        assert "console" not in notification_service.channels
        
    async def test_remove_nonexistent_channel(self, notification_service):
        """测试移除不存在的通知渠道"""
        # 移除不存在的渠道不应该抛出异常
        await notification_service.remove_channel("nonexistent")
        
    async def test_get_channels(self, notification_service, console_channel, email_channel):
        """测试获取通知渠道"""
        await notification_service.add_channel(console_channel)
        await notification_service.add_channel(email_channel)
        
        channels = notification_service.get_channels()
        assert len(channels) == 2
        assert "console" in channels
        assert "email" in channels
        
    async def test_get_enabled_channels(self, notification_service):
        """测试获取启用的通知渠道"""
        enabled_channel = NotificationChannel(
            name="enabled",
            type="console",
            enabled=True,
            config={}
        )
        disabled_channel = NotificationChannel(
            name="disabled",
            type="console",
            enabled=False,
            config={}
        )
        
        await notification_service.add_channel(enabled_channel)
        await notification_service.add_channel(disabled_channel)
        
        enabled_channels = notification_service.get_enabled_channels()
        assert len(enabled_channels) == 1
        assert enabled_channels[0].name == "enabled"
        
    @patch('builtins.print')
    async def test_send_console_notification(self, mock_print, notification_service, console_channel, sample_alert):
        """测试控制台通知发送"""
        await notification_service.add_channel(console_channel)
        
        result = await notification_service.send_notification(
            channel_name="console",
            alert=sample_alert,
            priority=NotificationPriority.HIGH
        )
        
        assert result.status == NotificationStatus.SUCCESS
        assert result.channel_name == "console"
        mock_print.assert_called()
        
    @patch('smtplib.SMTP')
    async def test_send_email_notification(self, mock_smtp, notification_service, email_channel, sample_alert):
        """测试邮件通知发送"""
        # 设置mock
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        await notification_service.add_channel(email_channel)
        
        result = await notification_service.send_notification(
            channel_name="email",
            alert=sample_alert,
            priority=NotificationPriority.HIGH
        )
        
        assert result.status == NotificationStatus.SUCCESS
        assert result.channel_name == "email"
        
        # 验证SMTP调用
        mock_smtp.assert_called_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_with("alerts@example.com", "password123")
        mock_server.send_message.assert_called_once()
        
    @patch('aiohttp.ClientSession.post')
    async def test_send_webhook_notification(self, mock_post, notification_service, webhook_channel, sample_alert):
        """测试Webhook通知发送"""
        # 设置mock响应
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_post.return_value.__aenter__.return_value = mock_response
        
        await notification_service.add_channel(webhook_channel)
        
        result = await notification_service.send_notification(
            channel_name="webhook",
            alert=sample_alert,
            priority=NotificationPriority.HIGH
        )
        
        assert result.status == NotificationStatus.SUCCESS
        assert result.channel_name == "webhook"
        mock_post.assert_called_once()
        
    @patch('aiohttp.ClientSession.post')
    async def test_send_slack_notification(self, mock_post, notification_service, slack_channel, sample_alert):
        """测试Slack通知发送"""
        # 设置mock响应
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="ok")
        mock_post.return_value.__aenter__.return_value = mock_response
        
        await notification_service.add_channel(slack_channel)
        
        result = await notification_service.send_notification(
            channel_name="slack",
            alert=sample_alert,
            priority=NotificationPriority.HIGH
        )
        
        assert result.status == NotificationStatus.SUCCESS
        assert result.channel_name == "slack"
        mock_post.assert_called_once()
        
    @patch('aiohttp.ClientSession.post')
    async def test_send_dingtalk_notification(self, mock_post, notification_service, dingtalk_channel, sample_alert):
        """测试钉钉通知发送"""
        # 设置mock响应
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"errcode": 0, "errmsg": "ok"})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        await notification_service.add_channel(dingtalk_channel)
        
        result = await notification_service.send_notification(
            channel_name="dingtalk",
            alert=sample_alert,
            priority=NotificationPriority.HIGH
        )
        
        assert result.status == NotificationStatus.SUCCESS
        assert result.channel_name == "dingtalk"
        mock_post.assert_called_once()
        
    async def test_send_notification_to_nonexistent_channel(self, notification_service, sample_alert):
        """测试向不存在的渠道发送通知"""
        result = await notification_service.send_notification(
            channel_name="nonexistent",
            alert=sample_alert,
            priority=NotificationPriority.HIGH
        )
        
        assert result.status == NotificationStatus.FAILED
        assert "Channel not found" in result.error_message
        
    async def test_send_notification_to_disabled_channel(self, notification_service, sample_alert):
        """测试向禁用的渠道发送通知"""
        disabled_channel = NotificationChannel(
            name="disabled",
            type="console",
            enabled=False,
            config={}
        )
        await notification_service.add_channel(disabled_channel)
        
        result = await notification_service.send_notification(
            channel_name="disabled",
            alert=sample_alert,
            priority=NotificationPriority.HIGH
        )
        
        assert result.status == NotificationStatus.SKIPPED
        assert "Channel is disabled" in result.error_message
        
    async def test_broadcast_notification(self, notification_service, console_channel, sample_alert):
        """测试广播通知"""
        await notification_service.add_channel(console_channel)
        
        # 添加另一个渠道
        another_channel = NotificationChannel(
            name="another_console",
            type="console",
            enabled=True,
            config={}
        )
        await notification_service.add_channel(another_channel)
        
        with patch('builtins.print'):
            results = await notification_service.broadcast_notification(
                alert=sample_alert,
                priority=NotificationPriority.HIGH
            )
        
        assert len(results) == 2
        assert all(result.status == NotificationStatus.SUCCESS for result in results)
        
    async def test_broadcast_notification_with_channel_filter(self, notification_service, console_channel, sample_alert):
        """测试带渠道过滤的广播通知"""
        await notification_service.add_channel(console_channel)
        
        # 添加另一个渠道
        another_channel = NotificationChannel(
            name="another_console",
            type="console",
            enabled=True,
            config={}
        )
        await notification_service.add_channel(another_channel)
        
        with patch('builtins.print'):
            results = await notification_service.broadcast_notification(
                alert=sample_alert,
                priority=NotificationPriority.HIGH,
                channel_names=["console"]
            )
        
        assert len(results) == 1
        assert results[0].channel_name == "console"
        
    async def test_rate_limiting(self, notification_service, console_channel, sample_alert):
        """测试速率限制"""
        # 设置严格的速率限制
        notification_service.rate_limiter.max_notifications_per_minute = 1
        
        await notification_service.add_channel(console_channel)
        
        with patch('builtins.print'):
            # 第一次发送应该成功
            result1 = await notification_service.send_notification(
                channel_name="console",
                alert=sample_alert,
                priority=NotificationPriority.HIGH
            )
            assert result1.status == NotificationStatus.SUCCESS
            
            # 立即再次发送应该被限制
            result2 = await notification_service.send_notification(
                channel_name="console",
                alert=sample_alert,
                priority=NotificationPriority.HIGH
            )
            assert result2.status == NotificationStatus.RATE_LIMITED
            
    async def test_template_rendering(self, notification_service, sample_alert):
        """测试模板渲染"""
        template = NotificationTemplate(
            id="test_template",
            name="测试模板",
            channel="console",
            subject_template="告警: {{alert.name}}",
            body_template="告警详情: {{alert.description}}\n严重级别: {{alert.severity.value}}",
            format_type="text"
        )
        
        notification_service.templates["test_template"] = template
        
        subject, body = notification_service._render_template(template, sample_alert)
        
        assert subject == "告警: 测试告警"
        assert "告警详情: 这是一个测试告警" in body
        assert "严重级别: high" in body
        
    async def test_template_rendering_with_missing_variable(self, notification_service, sample_alert):
        """测试模板渲染缺失变量"""
        template = NotificationTemplate(
            id="test_template",
            name="测试模板",
            channel="console",
            subject_template="告警: {{alert.nonexistent_field}}",
            body_template="告警详情: {{alert.description}}",
            format_type="text"
        )
        
        notification_service.templates["test_template"] = template
        
        # 应该优雅处理缺失的变量
        subject, body = notification_service._render_template(template, sample_alert)
        
        assert "告警:" in subject  # 应该包含静态部分
        assert "告警详情: 这是一个测试告警" in body
        
    async def test_get_statistics(self, notification_service, console_channel, sample_alert):
        """测试获取统计信息"""
        await notification_service.add_channel(console_channel)
        
        with patch('builtins.print'):
            await notification_service.send_notification(
                channel_name="console",
                alert=sample_alert,
                priority=NotificationPriority.HIGH
            )
        
        stats = notification_service.get_statistics()
        
        assert "total_sent" in stats
        assert "success_count" in stats
        assert "failed_count" in stats
        assert "rate_limited_count" in stats
        assert "channels" in stats
        assert stats["total_sent"] >= 1
        assert stats["success_count"] >= 1
        
    async def test_error_handling_in_email_sending(self, notification_service, email_channel, sample_alert):
        """测试邮件发送错误处理"""
        await notification_service.add_channel(email_channel)
        
        with patch('smtplib.SMTP') as mock_smtp:
            # 模拟SMTP错误
            mock_smtp.side_effect = Exception("SMTP connection failed")
            
            result = await notification_service.send_notification(
                channel_name="email",
                alert=sample_alert,
                priority=NotificationPriority.HIGH
            )
            
            assert result.status == NotificationStatus.FAILED
            assert "SMTP connection failed" in result.error_message
            
    @patch('aiohttp.ClientSession.post')
    async def test_error_handling_in_webhook_sending(self, mock_post, notification_service, webhook_channel, sample_alert):
        """测试Webhook发送错误处理"""
        # 设置mock响应为错误状态
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_post.return_value.__aenter__.return_value = mock_response
        
        await notification_service.add_channel(webhook_channel)
        
        result = await notification_service.send_notification(
            channel_name="webhook",
            alert=sample_alert,
            priority=NotificationPriority.HIGH
        )
        
        assert result.status == NotificationStatus.FAILED
        assert "HTTP 500" in result.error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])