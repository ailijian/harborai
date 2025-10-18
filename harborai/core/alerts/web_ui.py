#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警系统Web界面

提供告警系统的Web管理界面
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Depends, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .alert_manager import AlertManager, AlertRule, AlertSeverity, AlertCondition, AlertStatus
from .notification_service import NotificationService
from .suppression_manager import SuppressionManager, SuppressionRule, SuppressionType
from .alert_history import AlertHistory
from .config import get_default_config


class AlertWebUI:
    """告警系统Web界面"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.app = FastAPI(title="HarborAI 告警系统", version="1.0.0")
        self.db_path = db_path
        
        # 初始化组件
        self.alert_manager = AlertManager()
        self.notification_service = NotificationService()
        self.suppression_manager = SuppressionManager()
        self.alert_history = AlertHistory(db_path=db_path)
        
        # 设置模板和静态文件
        self.templates = Jinja2Templates(directory="templates")
        
        # 注册路由
        self._register_routes()
        
    async def initialize(self):
        """初始化组件"""
        await self.alert_manager.initialize()
        await self.notification_service.initialize()
        await self.suppression_manager.initialize()
        await self.alert_history.initialize()
        
        # 设置依赖关系
        self.alert_manager.set_notification_service(self.notification_service)
        self.alert_manager.set_suppression_service(self.suppression_manager)
        self.alert_manager.history_service = self.alert_history
        
        # 加载默认配置
        config = get_default_config()
        
        # 加载默认规则
        for rule_config in config.get("alert_rules", []):
            rule = AlertRule(
                id=rule_config["id"],
                name=rule_config["name"],
                description=rule_config["description"],
                severity=AlertSeverity(rule_config["severity"]),
                condition=AlertCondition(rule_config["condition"]),
                metric=rule_config["metric"],
                threshold=rule_config["threshold"],
                duration=timedelta(seconds=rule_config["duration"]),
                labels=rule_config.get("labels", {}),
                annotations=rule_config.get("annotations", {})
            )
            await self.alert_manager.add_rule(rule)
            
    def _register_routes(self):
        """注册路由"""
        
        # 主页
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """仪表板页面"""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "title": "告警仪表板"
            })
            
        # 告警列表页面
        @self.app.get("/alerts", response_class=HTMLResponse)
        async def alerts_page(request: Request):
            """告警列表页面"""
            return self.templates.TemplateResponse("alerts.html", {
                "request": request,
                "title": "告警管理"
            })
            
        # 规则管理页面
        @self.app.get("/rules", response_class=HTMLResponse)
        async def rules_page(request: Request):
            """规则管理页面"""
            return self.templates.TemplateResponse("rules.html", {
                "request": request,
                "title": "规则管理"
            })
            
        # 抑制管理页面
        @self.app.get("/suppressions", response_class=HTMLResponse)
        async def suppressions_page(request: Request):
            """抑制管理页面"""
            return self.templates.TemplateResponse("suppressions.html", {
                "request": request,
                "title": "抑制管理"
            })
            
        # 历史查询页面
        @self.app.get("/history", response_class=HTMLResponse)
        async def history_page(request: Request):
            """历史查询页面"""
            return self.templates.TemplateResponse("history.html", {
                "request": request,
                "title": "告警历史"
            })
            
        # API 路由
        
        # 获取仪表板数据
        @self.app.get("/api/dashboard")
        async def get_dashboard_data():
            """获取仪表板数据"""
            try:
                # 获取活跃告警
                active_alerts = await self.alert_manager.get_active_alerts()
                
                # 获取统计信息
                stats = await self.alert_manager.get_statistics()
                
                # 获取最近24小时的告警趋势
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)
                trends = await self.alert_history.get_alert_trends(
                    start_time=start_time,
                    end_time=end_time,
                    interval="hour"
                )
                
                # 按严重级别分组告警
                severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
                for alert in active_alerts:
                    severity_counts[alert.severity.value] += 1
                    
                return {
                    "active_alerts_count": len(active_alerts),
                    "severity_distribution": severity_counts,
                    "statistics": stats,
                    "trends": trends,
                    "recent_alerts": [
                        {
                            "id": alert.id,
                            "rule_name": alert.rule_name,
                            "severity": alert.severity.value,
                            "status": alert.status.value,
                            "message": alert.message,
                            "started_at": alert.started_at.isoformat() if alert.started_at else None
                        }
                        for alert in active_alerts[:10]  # 最近10个告警
                    ]
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # 获取告警列表
        @self.app.get("/api/alerts")
        async def get_alerts(
            severity: Optional[str] = Query(None),
            status: Optional[str] = Query(None),
            limit: int = Query(50, ge=1, le=1000),
            offset: int = Query(0, ge=0)
        ):
            """获取告警列表"""
            try:
                if severity:
                    alerts = await self.alert_manager.get_alerts_by_severity(
                        AlertSeverity(severity)
                    )
                else:
                    alerts = await self.alert_manager.get_active_alerts()
                    
                # 应用分页
                total = len(alerts)
                alerts = alerts[offset:offset + limit]
                
                return {
                    "alerts": [
                        {
                            "id": alert.id,
                            "rule_id": alert.rule_id,
                            "rule_name": alert.rule_name,
                            "severity": alert.severity.value,
                            "status": alert.status.value,
                            "message": alert.message,
                            "metric_value": alert.metric_value,
                            "threshold": alert.threshold,
                            "started_at": alert.started_at.isoformat() if alert.started_at else None,
                            "acknowledged_by": alert.acknowledged_by,
                            "suppressed": alert.suppressed,
                            "labels": alert.labels,
                            "annotations": alert.annotations
                        }
                        for alert in alerts
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # 确认告警
        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str, user: str = Form(...)):
            """确认告警"""
            try:
                await self.alert_manager.acknowledge_alert(alert_id, user)
                return {"message": "告警已确认"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # 获取规则列表
        @self.app.get("/api/rules")
        async def get_rules():
            """获取规则列表"""
            try:
                rules = await self.alert_manager.get_rules()
                
                return {
                    "rules": [
                        {
                            "id": rule.id,
                            "name": rule.name,
                            "description": rule.description,
                            "severity": rule.severity.value,
                            "condition": rule.condition.value,
                            "metric": rule.metric,
                            "threshold": rule.threshold,
                            "duration": rule.duration.total_seconds(),
                            "enabled": rule.enabled,
                            "labels": rule.labels,
                            "annotations": rule.annotations
                        }
                        for rule in rules
                    ]
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # 添加规则
        @self.app.post("/api/rules")
        async def add_rule(
            id: str = Form(...),
            name: str = Form(...),
            description: str = Form(""),
            severity: str = Form(...),
            condition: str = Form(...),
            metric: str = Form(...),
            threshold: float = Form(...),
            duration: int = Form(60),
            labels: str = Form("{}"),
            annotations: str = Form("{}")
        ):
            """添加规则"""
            try:
                rule = AlertRule(
                    id=id,
                    name=name,
                    description=description or name,
                    severity=AlertSeverity(severity),
                    condition=AlertCondition(condition),
                    metric=metric,
                    threshold=threshold,
                    duration=timedelta(seconds=duration),
                    labels=json.loads(labels),
                    annotations=json.loads(annotations)
                )
                
                await self.alert_manager.add_rule(rule)
                return {"message": "规则添加成功"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # 删除规则
        @self.app.delete("/api/rules/{rule_id}")
        async def delete_rule(rule_id: str):
            """删除规则"""
            try:
                await self.alert_manager.remove_rule(rule_id)
                return {"message": "规则删除成功"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # 获取抑制规则列表
        @self.app.get("/api/suppressions")
        async def get_suppressions():
            """获取抑制规则列表"""
            try:
                rules = await self.suppression_manager.get_suppression_rules()
                
                return {
                    "suppressions": [
                        {
                            "id": rule.id,
                            "name": rule.name,
                            "type": rule.type.value,
                            "target_labels": rule.target_labels,
                            "start_time": rule.start_time.isoformat() if rule.start_time else None,
                            "end_time": rule.end_time.isoformat() if rule.end_time else None,
                            "reason": rule.reason,
                            "enabled": rule.enabled
                        }
                        for rule in rules
                    ]
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # 添加抑制规则
        @self.app.post("/api/suppressions")
        async def add_suppression(
            id: str = Form(...),
            name: str = Form(...),
            type: str = Form(...),
            target_labels: str = Form("{}"),
            start_time: str = Form(""),
            end_time: str = Form(""),
            reason: str = Form("")
        ):
            """添加抑制规则"""
            try:
                # 解析时间
                start_dt = datetime.fromisoformat(start_time) if start_time else datetime.now()
                end_dt = datetime.fromisoformat(end_time) if end_time else start_dt + timedelta(hours=1)
                
                rule = SuppressionRule(
                    id=id,
                    name=name,
                    type=SuppressionType(type),
                    target_labels=json.loads(target_labels),
                    start_time=start_dt,
                    end_time=end_dt,
                    reason=reason or "Web界面添加"
                )
                
                await self.suppression_manager.add_suppression_rule(rule)
                return {"message": "抑制规则添加成功"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # 获取告警历史
        @self.app.get("/api/history")
        async def get_history(
            rule_id: Optional[str] = Query(None),
            severity: Optional[str] = Query(None),
            start_time: Optional[str] = Query(None),
            end_time: Optional[str] = Query(None),
            limit: int = Query(50, ge=1, le=1000),
            offset: int = Query(0, ge=0)
        ):
            """获取告警历史"""
            try:
                # 解析时间
                start_dt = datetime.fromisoformat(start_time) if start_time else None
                end_dt = datetime.fromisoformat(end_time) if end_time else None
                
                records = await self.alert_history.get_alert_history(
                    rule_id=rule_id,
                    severity=severity,
                    start_time=start_dt,
                    end_time=end_dt,
                    limit=limit,
                    offset=offset
                )
                
                return {
                    "history": [
                        {
                            "alert_id": record.alert_id,
                            "rule_id": record.rule_id,
                            "rule_name": record.rule_name,
                            "severity": record.severity,
                            "status": record.status,
                            "message": record.message,
                            "metric_value": record.metric_value,
                            "threshold": record.threshold,
                            "started_at": record.started_at.isoformat() if record.started_at else None,
                            "resolved_at": record.resolved_at.isoformat() if record.resolved_at else None,
                            "acknowledged_by": record.acknowledged_by,
                            "labels": record.labels,
                            "annotations": record.annotations
                        }
                        for record in records
                    ]
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # 获取统计信息
        @self.app.get("/api/statistics")
        async def get_statistics(
            start_time: Optional[str] = Query(None),
            end_time: Optional[str] = Query(None)
        ):
            """获取统计信息"""
            try:
                # 解析时间
                start_dt = datetime.fromisoformat(start_time) if start_time else None
                end_dt = datetime.fromisoformat(end_time) if end_time else None
                
                stats = await self.alert_history.get_statistics(
                    start_time=start_dt,
                    end_time=end_dt
                )
                
                return stats
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


# HTML 模板内容
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
        .nav { margin: 20px 0; }
        .nav a { margin-right: 20px; text-decoration: none; color: #3498db; }
        .nav a:hover { text-decoration: underline; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .stat-item { text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; }
        .severity-low { color: #27ae60; }
        .severity-medium { color: #f39c12; }
        .severity-high { color: #e67e22; }
        .severity-critical { color: #e74c3c; }
        .alert-item { border-left: 4px solid #3498db; padding: 10px; margin: 10px 0; background: #ecf0f1; }
        .alert-item.critical { border-left-color: #e74c3c; }
        .alert-item.high { border-left-color: #e67e22; }
        .alert-item.medium { border-left-color: #f39c12; }
        .alert-item.low { border-left-color: #27ae60; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
    </div>
    
    <div class="nav">
        <a href="/">仪表板</a>
        <a href="/alerts">告警管理</a>
        <a href="/rules">规则管理</a>
        <a href="/suppressions">抑制管理</a>
        <a href="/history">告警历史</a>
    </div>
    
    <div class="card">
        <h2>系统概览</h2>
        <div class="stats" id="stats">
            <div class="stat-item">
                <div class="stat-number" id="active-alerts">-</div>
                <div class="stat-label">活跃告警</div>
            </div>
            <div class="stat-item">
                <div class="stat-number severity-critical" id="critical-alerts">-</div>
                <div class="stat-label">严重告警</div>
            </div>
            <div class="stat-item">
                <div class="stat-number severity-high" id="high-alerts">-</div>
                <div class="stat-label">高级告警</div>
            </div>
            <div class="stat-item">
                <div class="stat-number severity-medium" id="medium-alerts">-</div>
                <div class="stat-label">中级告警</div>
            </div>
            <div class="stat-item">
                <div class="stat-number severity-low" id="low-alerts">-</div>
                <div class="stat-label">低级告警</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>最近告警</h2>
        <div id="recent-alerts">
            <p>加载中...</p>
        </div>
    </div>
    
    <script>
        async function loadDashboard() {
            try {
                const response = await fetch('/api/dashboard');
                const data = await response.json();
                
                // 更新统计数据
                document.getElementById('active-alerts').textContent = data.active_alerts_count;
                document.getElementById('critical-alerts').textContent = data.severity_distribution.critical;
                document.getElementById('high-alerts').textContent = data.severity_distribution.high;
                document.getElementById('medium-alerts').textContent = data.severity_distribution.medium;
                document.getElementById('low-alerts').textContent = data.severity_distribution.low;
                
                // 更新最近告警
                const recentAlertsDiv = document.getElementById('recent-alerts');
                if (data.recent_alerts.length === 0) {
                    recentAlertsDiv.innerHTML = '<p>暂无活跃告警</p>';
                } else {
                    recentAlertsDiv.innerHTML = data.recent_alerts.map(alert => `
                        <div class="alert-item ${alert.severity}">
                            <strong>${alert.rule_name}</strong> - ${alert.severity}
                            <br>
                            <small>${alert.message}</small>
                            <br>
                            <small>开始时间: ${alert.started_at ? new Date(alert.started_at).toLocaleString() : 'N/A'}</small>
                        </div>
                    `).join('');
                }
                
            } catch (error) {
                console.error('加载仪表板数据失败:', error);
            }
        }
        
        // 页面加载时获取数据
        loadDashboard();
        
        // 每30秒刷新一次数据
        setInterval(loadDashboard, 30000);
    </script>
</body>
</html>
"""


def create_templates_directory():
    """创建模板目录和文件"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # 创建仪表板模板
    with open(templates_dir / "dashboard.html", "w", encoding="utf-8") as f:
        f.write(DASHBOARD_TEMPLATE)
        
    # 创建其他页面的基础模板
    for page in ["alerts", "rules", "suppressions", "history"]:
        template_content = DASHBOARD_TEMPLATE.replace(
            "最近告警", f"{page.title()} 管理"
        ).replace(
            "recent-alerts", f"{page}-content"
        )
        
        with open(templates_dir / f"{page}.html", "w", encoding="utf-8") as f:
            f.write(template_content)


async def create_app(db_path: str = ":memory:") -> FastAPI:
    """创建Web应用"""
    # 创建模板目录
    create_templates_directory()
    
    # 创建Web UI实例
    web_ui = AlertWebUI(db_path=db_path)
    await web_ui.initialize()
    
    return web_ui.app


if __name__ == "__main__":
    import uvicorn
    
    async def main():
        app = await create_app()
        config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        
    asyncio.run(main())