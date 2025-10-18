#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警系统配置

定义告警规则、通知配置和抑制规则的默认配置
"""

from datetime import timedelta
from typing import Dict, List, Any
from .alert_manager import AlertSeverity, AlertCondition
from .notification_service import NotificationChannel, NotificationPriority
from .suppression_manager import SuppressionType


# 系统监控告警阈值配置
SYSTEM_MONITORING_THRESHOLDS = {
    # CPU使用率阈值配置
    "cpu": {
        "warning": {
            "threshold": 70.0,  # 70%
            "duration": timedelta(minutes=10),
            "severity": AlertSeverity.MEDIUM,
            "description": "CPU使用率持续较高，建议关注"
        },
        "high": {
            "threshold": 80.0,  # 80%
            "duration": timedelta(minutes=5),
            "severity": AlertSeverity.HIGH,
            "description": "CPU使用率过高，可能影响服务性能"
        },
        "critical": {
            "threshold": 95.0,  # 95%
            "duration": timedelta(minutes=2),
            "severity": AlertSeverity.CRITICAL,
            "description": "CPU使用率严重过高，服务可能不可用"
        }
    },
    
    # 内存使用率阈值配置
    "memory": {
        "warning": {
            "threshold": 75.0,  # 75%
            "duration": timedelta(minutes=15),
            "severity": AlertSeverity.MEDIUM,
            "description": "内存使用率较高，建议监控"
        },
        "high": {
            "threshold": 85.0,  # 85%
            "duration": timedelta(minutes=10),
            "severity": AlertSeverity.HIGH,
            "description": "内存使用率过高，可能影响性能"
        },
        "critical": {
            "threshold": 95.0,  # 95%
            "duration": timedelta(minutes=3),
            "severity": AlertSeverity.CRITICAL,
            "description": "内存使用率严重过高，可能导致OOM"
        }
    },
    
    # 磁盘使用率阈值配置
    "disk": {
        "warning": {
            "threshold": 80.0,  # 80%
            "duration": timedelta(minutes=30),
            "severity": AlertSeverity.MEDIUM,
            "description": "磁盘使用率较高，建议清理"
        },
        "high": {
            "threshold": 90.0,  # 90%
            "duration": timedelta(minutes=5),
            "severity": AlertSeverity.HIGH,
            "description": "磁盘空间不足，需要立即清理"
        },
        "critical": {
            "threshold": 95.0,  # 95%
            "duration": timedelta(minutes=1),
            "severity": AlertSeverity.CRITICAL,
            "description": "磁盘空间严重不足，可能影响服务"
        }
    },
    
    # 网络I/O阈值配置
    "network": {
        "bandwidth_utilization": {
            "warning": {
                "threshold": 70.0,  # 70%
                "duration": timedelta(minutes=10),
                "severity": AlertSeverity.MEDIUM,
                "description": "网络带宽使用率较高"
            },
            "high": {
                "threshold": 85.0,  # 85%
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.HIGH,
                "description": "网络带宽使用率过高"
            },
            "critical": {
                "threshold": 95.0,  # 95%
                "duration": timedelta(minutes=2),
                "severity": AlertSeverity.CRITICAL,
                "description": "网络带宽接近饱和"
            }
        },
        "packet_loss": {
            "warning": {
                "threshold": 1.0,  # 1%
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "网络丢包率较高"
            },
            "high": {
                "threshold": 3.0,  # 3%
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "网络丢包率过高"
            },
            "critical": {
                "threshold": 10.0,  # 10%
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "网络丢包率严重过高"
            }
        },
        "connection_errors": {
            "warning": {
                "threshold": 5,  # 5个/分钟
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "网络连接错误较多"
            },
            "high": {
                "threshold": 20,  # 20个/分钟
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "网络连接错误过多"
            },
            "critical": {
                "threshold": 50,  # 50个/分钟
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "网络连接错误严重过多"
            }
        }
    },
    
    # API性能阈值配置
    "api_performance": {
        "response_time": {
            "warning": {
                "threshold": 1.0,  # 1秒
                "duration": timedelta(minutes=10),
                "severity": AlertSeverity.MEDIUM,
                "description": "API响应时间较慢"
            },
            "high": {
                "threshold": 2.0,  # 2秒
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.HIGH,
                "description": "API响应时间过慢，可能影响用户体验"
            },
            "critical": {
                "threshold": 5.0,  # 5秒
                "duration": timedelta(minutes=2),
                "severity": AlertSeverity.CRITICAL,
                "description": "API响应时间严重过慢"
            }
        },
        "error_rate": {
            "warning": {
                "threshold": 1.0,  # 1%
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "API错误率较高"
            },
            "high": {
                "threshold": 5.0,  # 5%
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "API错误率过高，需要立即检查"
            },
            "critical": {
                "threshold": 20.0,  # 20%
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "API错误率严重过高，服务可能不可用"
            }
        },
        "timeout_rate": {
            "warning": {
                "threshold": 0.5,  # 0.5%
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "API超时率较高"
            },
            "high": {
                "threshold": 2.0,  # 2%
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "API超时率过高，可能影响用户体验"
            },
            "critical": {
                "threshold": 10.0,  # 10%
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "API超时率严重过高"
            }
        },
        "throughput": {
            "low_warning": {
                "threshold": 100,  # 100 RPS
                "duration": timedelta(minutes=10),
                "severity": AlertSeverity.MEDIUM,
                "description": "API吞吐量较低，可能存在性能问题"
            },
            "anomaly_high": {
                "threshold": 3.0,  # 3倍标准差
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.HIGH,
                "description": "API请求量异常增长，可能是攻击或异常流量"
            }
        }
    },
    
    # 数据库性能阈值配置
    "database_performance": {
        "connection_pool": {
            "warning": {
                "threshold": 70.0,  # 70%
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "数据库连接池使用率较高"
            },
            "high": {
                "threshold": 85.0,  # 85%
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "数据库连接池使用率过高"
            },
            "critical": {
                "threshold": 95.0,  # 95%
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "数据库连接池接近耗尽"
            }
        },
        "query_duration": {
            "warning": {
                "threshold": 2.0,  # 2秒
                "duration": timedelta(minutes=10),
                "severity": AlertSeverity.MEDIUM,
                "description": "数据库查询时间较长"
            },
            "high": {
                "threshold": 5.0,  # 5秒
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "数据库查询时间过长，需要优化"
            },
            "critical": {
                "threshold": 15.0,  # 15秒
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "数据库查询时间严重过长"
            }
        },
        "connection_errors": {
            "warning": {
                "threshold": 1,  # 1个错误
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.HIGH,
                "description": "数据库连接失败"
            },
            "critical": {
                "threshold": 5,  # 5个错误
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "数据库连接频繁失败"
            }
        },
        "deadlocks": {
            "warning": {
                "threshold": 1,  # 1个死锁
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.MEDIUM,
                "description": "数据库发生死锁"
            },
            "high": {
                "threshold": 3,  # 3个死锁
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.HIGH,
                "description": "数据库死锁频繁发生"
            }
        },
        "replication_lag": {
            "warning": {
                "threshold": 30.0,  # 30秒
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "数据库复制延迟较高"
            },
            "high": {
                "threshold": 120.0,  # 2分钟
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "数据库复制延迟过高"
            },
            "critical": {
                "threshold": 600.0,  # 10分钟
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "数据库复制延迟严重过高"
            }
        }
    },
    
    # Token使用量阈值配置
    "token_usage": {
        "hourly_usage": {
            "warning": {
                "threshold": 10000,  # 10K tokens/小时
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "Token使用量较高"
            },
            "high": {
                "threshold": 50000,  # 50K tokens/小时
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "Token使用量过高"
            },
            "critical": {
                "threshold": 100000,  # 100K tokens/小时
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "Token使用量严重过高"
            }
        },
        "quota_usage": {
            "warning": {
                "threshold": 80.0,  # 80%
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "Token配额使用率较高"
            },
            "high": {
                "threshold": 90.0,  # 90%
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "Token配额使用率过高"
            },
            "critical": {
                "threshold": 95.0,  # 95%
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "Token配额接近耗尽"
            }
        },
        "anomaly_detection": {
            "threshold": 2.5,  # 2.5倍标准差
            "duration": timedelta(minutes=15),
            "severity": AlertSeverity.MEDIUM,
            "description": "Token使用量出现异常波动"
        }
    },
    
    # 成本控制阈值配置
    "cost_control": {
        "hourly_cost": {
            "warning": {
                "threshold": 50.0,  # $50/小时
                "duration": timedelta(minutes=10),
                "severity": AlertSeverity.MEDIUM,
                "description": "小时成本较高，建议关注"
            },
            "high": {
                "threshold": 100.0,  # $100/小时
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.HIGH,
                "description": "小时成本过高，需要检查使用情况"
            },
            "critical": {
                "threshold": 200.0,  # $200/小时
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "小时成本严重过高，需要立即控制"
            }
        },
        "daily_budget": {
            "warning": {
                "threshold": 500.0,  # $500/天
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "日成本接近预算"
            },
            "high": {
                "threshold": 800.0,  # $800/天
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "日成本超过预算80%"
            },
            "critical": {
                "threshold": 1000.0,  # $1000/天
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "日成本超过预算，需要立即控制"
            }
        },
        "monthly_budget": {
            "warning": {
                "threshold": 15000.0,  # $15K/月
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "月成本接近预算"
            },
            "high": {
                "threshold": 25000.0,  # $25K/月
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "月成本超过预算80%"
            },
            "critical": {
                "threshold": 30000.0,  # $30K/月
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "月成本超过预算，需要立即控制"
            }
        },
        "cost_per_user": {
            "warning": {
                "threshold": 10.0,  # $10/用户/天
                "duration": timedelta(minutes=30),
                "severity": AlertSeverity.MEDIUM,
                "description": "单用户成本较高"
            },
            "high": {
                "threshold": 20.0,  # $20/用户/天
                "duration": timedelta(minutes=10),
                "severity": AlertSeverity.HIGH,
                "description": "单用户成本过高"
            },
            "critical": {
                "threshold": 50.0,  # $50/用户/天
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.CRITICAL,
                "description": "单用户成本严重过高"
            }
        }
    },
    
    # 业务指标阈值配置
    "business_metrics": {
        "user_activity": {
            "low_activity": {
                "threshold": 0.5,  # 50%正常水平
                "duration": timedelta(minutes=30),
                "severity": AlertSeverity.MEDIUM,
                "description": "用户活跃度较低"
            },
            "anomaly_detection": {
                "threshold": 2.0,  # 2倍标准差
                "duration": timedelta(minutes=10),
                "severity": AlertSeverity.MEDIUM,
                "description": "用户活动模式异常"
            }
        },
        "conversion_rate": {
            "low_conversion": {
                "threshold": 0.7,  # 70%正常水平
                "duration": timedelta(minutes=60),
                "severity": AlertSeverity.MEDIUM,
                "description": "转化率较低"
            }
        },
        "data_consistency": {
            "errors": {
                "threshold": 1,  # 1个错误
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.HIGH,
                "description": "数据一致性错误"
            }
        }
    },
    
    # 安全相关阈值配置
    "security": {
        "failed_login_attempts": {
            "warning": {
                "threshold": 10,  # 10次/分钟
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "登录失败次数较多"
            },
            "high": {
                "threshold": 50,  # 50次/分钟
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "登录失败次数过多，可能存在攻击"
            },
            "critical": {
                "threshold": 100,  # 100次/分钟
                "duration": timedelta(minutes=1),
                "severity": AlertSeverity.CRITICAL,
                "description": "登录失败次数严重过多，可能正在遭受攻击"
            }
        },
        "suspicious_activity": {
            "threshold": 1,  # 1个可疑活动
            "duration": timedelta(minutes=1),
            "severity": AlertSeverity.HIGH,
            "description": "检测到可疑活动"
        },
        "rate_limit_violations": {
            "warning": {
                "threshold": 10,  # 10次/分钟
                "duration": timedelta(minutes=5),
                "severity": AlertSeverity.MEDIUM,
                "description": "频率限制违规较多"
            },
            "high": {
                "threshold": 50,  # 50次/分钟
                "duration": timedelta(minutes=3),
                "severity": AlertSeverity.HIGH,
                "description": "频率限制违规过多"
            }
        }
    }
}

# 环境特定阈值配置
ENVIRONMENT_SPECIFIC_THRESHOLDS = {
    "development": {
        # 开发环境阈值相对宽松
        "cpu_multiplier": 1.2,
        "memory_multiplier": 1.2,
        "cost_multiplier": 0.5,
        "response_time_multiplier": 2.0
    },
    "staging": {
        # 测试环境阈值接近生产
        "cpu_multiplier": 1.1,
        "memory_multiplier": 1.1,
        "cost_multiplier": 0.8,
        "response_time_multiplier": 1.5
    },
    "production": {
        # 生产环境使用标准阈值
        "cpu_multiplier": 1.0,
        "memory_multiplier": 1.0,
        "cost_multiplier": 1.0,
        "response_time_multiplier": 1.0
    }
}

# 动态阈值配置
DYNAMIC_THRESHOLD_CONFIG = {
    "enabled": True,
    "learning_period": timedelta(days=7),  # 学习周期
    "adjustment_factor": 0.1,  # 调整因子
    "min_data_points": 100,  # 最少数据点
    "confidence_level": 0.95,  # 置信水平
    "metrics": [
        "api_response_time_avg",
        "api_request_rate",
        "token_usage_rate",
        "cost_per_hour"
    ]
}


def generate_alert_rules_from_thresholds() -> List[Dict[str, Any]]:
    """
    基于系统监控阈值配置生成告警规则
    """
    alert_rules = []
    
    # CPU告警规则
    for level, config in SYSTEM_MONITORING_THRESHOLDS["cpu"].items():
        alert_rules.append({
            "id": f"cpu_usage_{level}",
            "name": f"CPU使用率{level}告警",
            "description": config["description"],
            "severity": config["severity"],
            "condition": AlertCondition.THRESHOLD,
            "metric": "cpu_usage_percent",
            "threshold": config["threshold"],
            "duration": int(config["duration"].total_seconds()),
            "labels": {"component": "system", "type": "resource", "level": level},
            "annotations": {
                "summary": f"CPU使用率{level}告警",
                "description": config["description"],
                "runbook": "检查高CPU进程和优化算法性能"
            }
        })
    
    # 内存告警规则
    for level, config in SYSTEM_MONITORING_THRESHOLDS["memory"].items():
        alert_rules.append({
            "id": f"memory_usage_{level}",
            "name": f"内存使用率{level}告警",
            "description": config["description"],
            "severity": config["severity"],
            "condition": AlertCondition.THRESHOLD,
            "metric": "memory_usage_percent",
            "threshold": config["threshold"],
            "duration": int(config["duration"].total_seconds()),
            "labels": {"component": "system", "type": "resource", "level": level},
            "annotations": {
                "summary": f"内存使用率{level}告警",
                "description": config["description"],
                "runbook": "检查内存泄漏和优化内存使用"
            }
        })
    
    # 磁盘告警规则
    for level, config in SYSTEM_MONITORING_THRESHOLDS["disk"].items():
        alert_rules.append({
            "id": f"disk_usage_{level}",
            "name": f"磁盘使用率{level}告警",
            "description": config["description"],
            "severity": config["severity"],
            "condition": AlertCondition.THRESHOLD,
            "metric": "disk_usage_percent",
            "threshold": config["threshold"],
            "duration": int(config["duration"].total_seconds()),
            "labels": {"component": "system", "type": "storage", "level": level},
            "annotations": {
                "summary": f"磁盘使用率{level}告警",
                "description": config["description"],
                "runbook": "清理日志文件和临时文件"
            }
        })
    
    # API性能告警规则
    for metric_type, metric_config in SYSTEM_MONITORING_THRESHOLDS["api_performance"].items():
        if metric_type == "throughput":
            # 特殊处理吞吐量告警
            for sub_type, config in metric_config.items():
                condition = AlertCondition.ANOMALY if "anomaly" in sub_type else AlertCondition.THRESHOLD
                alert_rules.append({
                    "id": f"api_{metric_type}_{sub_type}",
                    "name": f"API{metric_type}{sub_type}告警",
                    "description": config["description"],
                    "severity": config["severity"],
                    "condition": condition,
                    "metric": f"api_{metric_type}",
                    "threshold": config["threshold"],
                    "duration": int(config["duration"].total_seconds()),
                    "labels": {"component": "api", "type": "performance", "metric": metric_type},
                    "annotations": {
                        "summary": f"API{metric_type}{sub_type}告警",
                        "description": config["description"],
                        "runbook": "检查API性能和流量模式"
                    }
                })
        else:
            for level, config in metric_config.items():
                metric_name_map = {
                    "response_time": "api_response_time_avg",
                    "error_rate": "api_error_rate",
                    "timeout_rate": "api_timeout_rate"
                }
                alert_rules.append({
                    "id": f"api_{metric_type}_{level}",
                    "name": f"API{metric_type}{level}告警",
                    "description": config["description"],
                    "severity": config["severity"],
                    "condition": AlertCondition.THRESHOLD,
                    "metric": metric_name_map.get(metric_type, f"api_{metric_type}"),
                    "threshold": config["threshold"],
                    "duration": int(config["duration"].total_seconds()),
                    "labels": {"component": "api", "type": "performance", "level": level},
                    "annotations": {
                        "summary": f"API{metric_type}{level}告警",
                        "description": config["description"],
                        "runbook": "检查API性能和数据库查询优化"
                    }
                })
    
    # 数据库性能告警规则
    for metric_type, metric_config in SYSTEM_MONITORING_THRESHOLDS["database_performance"].items():
        for level, config in metric_config.items():
            metric_name_map = {
                "connection_pool": "database_connection_pool_usage",
                "query_duration": "database_query_duration_avg",
                "connection_errors": "database_connection_errors",
                "deadlocks": "database_deadlocks",
                "replication_lag": "database_replication_lag"
            }
            alert_rules.append({
                "id": f"database_{metric_type}_{level}",
                "name": f"数据库{metric_type}{level}告警",
                "description": config["description"],
                "severity": config["severity"],
                "condition": AlertCondition.THRESHOLD,
                "metric": metric_name_map.get(metric_type, f"database_{metric_type}"),
                "threshold": config["threshold"],
                "duration": int(config["duration"].total_seconds()),
                "labels": {"component": "database", "type": "performance", "level": level},
                "annotations": {
                    "summary": f"数据库{metric_type}{level}告警",
                    "description": config["description"],
                    "runbook": "检查数据库状态和性能优化"
                }
            })
    
    # Token使用量告警规则
    for metric_type, metric_config in SYSTEM_MONITORING_THRESHOLDS["token_usage"].items():
        if metric_type == "anomaly_detection":
            alert_rules.append({
                "id": "token_usage_anomaly",
                "name": "Token使用异常告警",
                "description": metric_config["description"],
                "severity": metric_config["severity"],
                "condition": AlertCondition.ANOMALY,
                "metric": "token_usage_rate",
                "threshold": metric_config["threshold"],
                "duration": int(metric_config["duration"].total_seconds()),
                "labels": {"component": "business", "type": "usage"},
                "annotations": {
                    "summary": "Token使用异常告警",
                    "description": metric_config["description"],
                    "runbook": "检查API调用日志和用户行为"
                }
            })
        else:
            for level, config in metric_config.items():
                metric_name_map = {
                    "hourly_usage": "token_usage_hourly",
                    "quota_usage": "token_quota_usage_percent"
                }
                alert_rules.append({
                    "id": f"token_{metric_type}_{level}",
                    "name": f"Token{metric_type}{level}告警",
                    "description": config["description"],
                    "severity": config["severity"],
                    "condition": AlertCondition.THRESHOLD,
                    "metric": metric_name_map.get(metric_type, f"token_{metric_type}"),
                    "threshold": config["threshold"],
                    "duration": int(config["duration"].total_seconds()),
                    "labels": {"component": "business", "type": "usage", "level": level},
                    "annotations": {
                        "summary": f"Token{metric_type}{level}告警",
                        "description": config["description"],
                        "runbook": "检查Token使用情况和配额管理"
                    }
                })
    
    # 成本控制告警规则
    for metric_type, metric_config in SYSTEM_MONITORING_THRESHOLDS["cost_control"].items():
        for level, config in metric_config.items():
            metric_name_map = {
                "hourly_cost": "cost_per_hour",
                "daily_budget": "daily_cost",
                "monthly_budget": "monthly_cost",
                "cost_per_user": "cost_per_user_daily"
            }
            alert_rules.append({
                "id": f"cost_{metric_type}_{level}",
                "name": f"成本{metric_type}{level}告警",
                "description": config["description"],
                "severity": config["severity"],
                "condition": AlertCondition.THRESHOLD,
                "metric": metric_name_map.get(metric_type, f"cost_{metric_type}"),
                "threshold": config["threshold"],
                "duration": int(config["duration"].total_seconds()),
                "labels": {"component": "business", "type": "cost", "level": level},
                "annotations": {
                    "summary": f"成本{metric_type}{level}告警",
                    "description": config["description"],
                    "runbook": "检查成本使用情况和控制措施"
                }
            })
    
    # 安全告警规则
    for metric_type, metric_config in SYSTEM_MONITORING_THRESHOLDS["security"].items():
        if metric_type == "suspicious_activity":
            alert_rules.append({
                "id": "security_suspicious_activity",
                "name": "可疑活动告警",
                "description": metric_config["description"],
                "severity": metric_config["severity"],
                "condition": AlertCondition.THRESHOLD,
                "metric": "security_suspicious_activity",
                "threshold": metric_config["threshold"],
                "duration": int(metric_config["duration"].total_seconds()),
                "labels": {"component": "security", "type": "activity"},
                "annotations": {
                    "summary": "可疑活动告警",
                    "description": metric_config["description"],
                    "runbook": "立即检查安全日志和用户行为"
                }
            })
        else:
            for level, config in metric_config.items():
                metric_name_map = {
                    "failed_login_attempts": "security_failed_logins",
                    "rate_limit_violations": "security_rate_limit_violations"
                }
                alert_rules.append({
                    "id": f"security_{metric_type}_{level}",
                    "name": f"安全{metric_type}{level}告警",
                    "description": config["description"],
                    "severity": config["severity"],
                    "condition": AlertCondition.THRESHOLD,
                    "metric": metric_name_map.get(metric_type, f"security_{metric_type}"),
                    "threshold": config["threshold"],
                    "duration": int(config["duration"].total_seconds()),
                    "labels": {"component": "security", "type": "threat", "level": level},
                    "annotations": {
                        "summary": f"安全{metric_type}{level}告警",
                        "description": config["description"],
                        "runbook": "检查安全日志和防护措施"
                    }
                })
    
    return alert_rules


# 默认告警规则配置
DEFAULT_ALERT_RULES = generate_alert_rules_from_thresholds()


# 默认通知配置
DEFAULT_NOTIFICATION_CONFIG = {
    "channels": [
        {
            "name": "console",
            "type": "console",
            "enabled": True,
            "config": {}
        },
        {
            "name": "email_admin",
            "type": "email",
            "enabled": False,
            "config": {
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "alerts@example.com",
                "password": "password",
                "from_email": "alerts@example.com",
                "to_emails": ["admin@example.com"],
                "use_tls": True,
                "use_ssl": False
            }
        },
        {
            "name": "webhook_slack",
            "type": "webhook",
            "enabled": False,
            "config": {
                "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "headers": {"Content-Type": "application/json"},
                "timeout": 10,
                "retry_attempts": 3
            }
        },
        {
            "name": "dingtalk_ops",
            "type": "dingtalk",
            "enabled": False,
            "config": {
                "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN",
                "secret": "YOUR_SECRET",
                "timeout": 10
            }
        }
    ],
    "routing": {
        # 严重程度路由规则
        "rules": [
            {
                "match": {"severity": "critical"},
                "channels": ["console", "email_admin", "dingtalk_ops", "webhook_slack"],
                "priority": "high"
            },
            {
                "match": {"severity": "high"},
                "channels": ["console", "email_admin", "dingtalk_ops"],
                "priority": "normal"
            },
            {
                "match": {"severity": "medium"},
                "channels": ["console", "email_admin"],
                "priority": "normal"
            },
            {
                "match": {"severity": "low"},
                "channels": ["console"],
                "priority": "low"
            },
            {
                "match": {"component": "security"},
                "channels": ["console", "email_admin", "dingtalk_ops"],
                "priority": "high"
            },
            {
                "match": {"component": "database"},
                "channels": ["console", "email_admin", "dingtalk_ops"],
                "priority": "high"
            }
        ]
    },
    "rate_limits": {
        "enabled": True,
        "max_notifications_per_minute": 10,
        "channels": {
            "console": {"max_per_minute": 100, "burst": 10},
            "email_admin": {"max_per_minute": 5, "burst": 3},
            "webhook_slack": {"max_per_minute": 20, "burst": 5},
            "dingtalk_ops": {"max_per_minute": 20, "burst": 5}
        }
    },
    "retry": {
        "max_attempts": 3,
        "backoff_factor": 2,
        "initial_delay": 1,
        "max_delay": 60
    },
    "templates": {
        "email": {
            "subject": "【{severity}告警】{alert_name}",
            "body": """
告警名称: {alert_name}
告警级别: {severity}
触发时间: {timestamp}
持续时间: {duration}
告警描述: {description}
当前值: {current_value}
阈值: {threshold}
组件: {component}
标签: {labels}

处理建议: {runbook}
确认链接: {ack_url}
            """
        },
        "slack": {
            "text": "【{severity}告警】{alert_name}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*告警名称:* {alert_name}\n*级别:* {severity}\n*描述:* {description}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": "*当前值:*\n{current_value}"},
                        {"type": "mrkdwn", "text": "*阈值:*\n{threshold}"},
                        {"type": "mrkdwn", "text": "*组件:*\n{component}"},
                        {"type": "mrkdwn", "text": "*持续时间:*\n{duration}"}
                    ]
                }
            ]
        },
        "dingtalk": {
            "msgtype": "markdown",
            "markdown": {
                "title": "【{severity}告警】{alert_name}",
                "text": """
### 【{severity}告警】{alert_name}

**告警级别:** {severity}  
**触发时间:** {timestamp}  
**持续时间:** {duration}  
**告警描述:** {description}  
**当前值:** {current_value}  
**阈值:** {threshold}  
**组件:** {component}  

**处理建议:** {runbook}
                """
            }
        }
    }
}


# 默认抑制规则
DEFAULT_SUPPRESSION_RULES = [
    {
        "id": "maintenance_window",
        "name": "维护窗口抑制",
        "type": SuppressionType.TIME_BASED,
        "enabled": False,
        "config": {
            "start_time": "02:00",  # 凌晨2点
            "end_time": "04:00",    # 凌晨4点
            "timezone": "Asia/Shanghai",
            "days": ["sunday"]      # 每周日
        },
        "labels": {},
        "description": "维护窗口期间抑制所有告警"
    },
    {
        "id": "duplicate_alerts",
        "name": "重复告警抑制",
        "type": SuppressionType.LABEL_BASED,
        "enabled": True,
        "config": {
            "match_labels": ["rule_id", "component"],
            "duration": timedelta(minutes=30),
            "max_suppressed": 10
        },
        "labels": {},
        "description": "抑制30分钟内的重复告警"
    },
    {
        "id": "database_cascade",
        "name": "数据库级联抑制",
        "type": SuppressionType.DEPENDENCY,
        "enabled": True,
        "config": {
            "parent_rule": "database_connection_errors_critical",
            "child_rules": [
                "api_response_time_high",
                "api_response_time_critical",
                "api_error_rate_high",
                "api_error_rate_critical"
            ],
            "duration": timedelta(minutes=15)
        },
        "labels": {},
        "description": "数据库故障时抑制相关告警"
    },
    {
        "id": "system_resource_cascade",
        "name": "系统资源级联抑制",
        "type": SuppressionType.DEPENDENCY,
        "enabled": True,
        "config": {
            "parent_rule": "cpu_usage_critical",
            "child_rules": [
                "api_response_time_warning",
                "api_response_time_high",
                "memory_usage_high"
            ],
            "duration": timedelta(minutes=10)
        },
        "labels": {},
        "description": "CPU严重过高时抑制相关性能告警"
    },
    {
        "id": "rate_limit_protection",
        "name": "速率限制保护",
        "type": SuppressionType.RATE_LIMIT,
        "enabled": True,
        "config": {
            "max_alerts": 10,
            "time_window": timedelta(minutes=5),
            "action": "suppress_new"
        },
        "labels": {},
        "description": "防止告警风暴，5分钟内最多10个告警"
    },
    {
        "id": "low_priority_night",
        "name": "夜间低优先级抑制",
        "type": SuppressionType.PATTERN_BASED,
        "enabled": True,
        "config": {
            "patterns": [
                {
                    "severity": ["low", "medium"],
                    "time_range": {"start": "22:00", "end": "08:00"},
                    "timezone": "Asia/Shanghai",
                    "exclude_components": ["security", "database"]
                }
            ]
        },
        "labels": {},
        "description": "夜间抑制低优先级告警（安全和数据库除外）"
    },
    {
        "id": "cost_alert_grouping",
        "name": "成本告警分组抑制",
        "type": SuppressionType.LABEL_BASED,
        "enabled": True,
        "config": {
            "match_labels": ["component"],
            "match_values": {"component": "business", "type": "cost"},
            "duration": timedelta(minutes=60),
            "max_suppressed": 5
        },
        "labels": {},
        "description": "成本相关告警1小时内最多5个"
    }
]


# 升级配置
ESCALATION_CONFIG = {
    "enabled": True,
    "global_settings": {
        "escalation_timeout": timedelta(hours=2),  # 全局升级超时时间
        "auto_resolve_timeout": timedelta(hours=24),  # 自动解决超时时间
        "escalation_cooldown": timedelta(minutes=10),  # 升级冷却时间
        "max_total_escalations": 5,  # 单个告警最大升级次数
        "business_hours": {
            "enabled": True,
            "start_time": "09:00",
            "end_time": "18:00",
            "timezone": "Asia/Shanghai",
            "weekdays": [0, 1, 2, 3, 4]  # 周一到周五
        }
    },
    "rules": [
        {
            "id": "critical_escalation",
            "name": "严重告警升级",
            "severity": AlertSeverity.CRITICAL,
            "enabled": True,
            "escalation_steps": [
                {
                    "step": 1,
                    "delay": timedelta(minutes=5),  # 5分钟后第一次升级
                    "channels": ["email_admin", "dingtalk_ops"],
                    "message_template": "【严重告警升级】{alert_name} 已持续 {duration}，需要立即处理",
                    "conditions": {
                        "business_hours_only": False,
                        "require_ack": False
                    }
                },
                {
                    "step": 2,
                    "delay": timedelta(minutes=15),  # 15分钟后第二次升级
                    "channels": ["email_admin", "dingtalk_ops", "webhook_slack"],
                    "message_template": "【严重告警升级-2】{alert_name} 已持续 {duration}，仍未解决，请立即响应",
                    "conditions": {
                        "business_hours_only": False,
                        "require_ack": True,  # 需要确认
                        "escalate_to": ["team_lead", "on_call_engineer"]
                    }
                },
                {
                    "step": 3,
                    "delay": timedelta(minutes=30),  # 30分钟后第三次升级
                    "channels": ["email_admin", "dingtalk_ops", "webhook_slack"],
                    "message_template": "【严重告警升级-3】{alert_name} 已持续 {duration}，升级至管理层",
                    "conditions": {
                        "business_hours_only": False,
                        "require_ack": True,
                        "escalate_to": ["manager", "director"],
                        "auto_create_incident": True  # 自动创建事故单
                    }
                }
            ],
            "auto_actions": {
                "create_incident": True,
                "page_on_call": True,
                "trigger_runbook": True
            }
        },
        {
            "id": "high_escalation",
            "name": "高级告警升级",
            "severity": AlertSeverity.HIGH,
            "enabled": True,
            "escalation_steps": [
                {
                    "step": 1,
                    "delay": timedelta(minutes=15),  # 15分钟后第一次升级
                    "channels": ["email_admin"],
                    "message_template": "【高级告警升级】{alert_name} 已持续 {duration}，请关注",
                    "conditions": {
                        "business_hours_only": True,  # 仅工作时间升级
                        "require_ack": False
                    }
                },
                {
                    "step": 2,
                    "delay": timedelta(minutes=45),  # 45分钟后第二次升级
                    "channels": ["email_admin", "dingtalk_ops"],
                    "message_template": "【高级告警升级-2】{alert_name} 已持续 {duration}，需要处理",
                    "conditions": {
                        "business_hours_only": True,
                        "require_ack": True,
                        "escalate_to": ["team_lead"]
                    }
                }
            ],
            "auto_actions": {
                "create_incident": False,
                "page_on_call": False,
                "trigger_runbook": False
            }
        },
        {
            "id": "medium_escalation",
            "name": "中级告警升级",
            "severity": AlertSeverity.MEDIUM,
            "enabled": True,
            "escalation_steps": [
                {
                    "step": 1,
                    "delay": timedelta(hours=1),  # 1小时后升级
                    "channels": ["email_admin"],
                    "message_template": "【中级告警升级】{alert_name} 已持续 {duration}，建议关注",
                    "conditions": {
                        "business_hours_only": True,
                        "require_ack": False
                    }
                }
            ],
            "auto_actions": {
                "create_incident": False,
                "page_on_call": False,
                "trigger_runbook": False
            }
        },
        {
            "id": "security_escalation",
            "name": "安全告警升级",
            "component": "security",
            "enabled": True,
            "escalation_steps": [
                {
                    "step": 1,
                    "delay": timedelta(minutes=2),  # 2分钟后立即升级
                    "channels": ["email_admin", "dingtalk_ops", "webhook_slack"],
                    "message_template": "【安全告警升级】{alert_name} 检测到安全威胁，需要立即处理",
                    "conditions": {
                        "business_hours_only": False,
                        "require_ack": True,
                        "escalate_to": ["security_team", "team_lead"]
                    }
                }
            ],
            "auto_actions": {
                "create_incident": True,
                "page_on_call": True,
                "trigger_runbook": True
            }
        }
    ],
    "notification_templates": {
        "escalation": {
            "subject": "【告警升级】{alert_name} - 第{step}次升级",
            "body": """
告警名称: {alert_name}
告警级别: {severity}
持续时间: {duration}
升级步骤: 第{step}次升级
升级原因: {escalation_reason}
告警详情: {alert_description}
当前值: {current_value}
阈值: {threshold}
处理建议: {runbook_url}
确认链接: {ack_url}
            """
        },
        "auto_resolve": {
            "subject": "【告警自动解决】{alert_name}",
            "body": """
告警名称: {alert_name}
解决时间: {resolve_time}
持续时间: {total_duration}
解决原因: 超时自动解决
            """
        }
    },
    "escalation_policies": {
        "default": {
            "name": "默认升级策略",
            "description": "适用于大部分告警的标准升级流程",
            "on_call_schedule": {
                "primary": ["engineer_1", "engineer_2"],
                "secondary": ["team_lead"],
                "manager": ["manager"]
            }
        },
        "critical_system": {
            "name": "关键系统升级策略",
            "description": "关键系统告警的快速升级流程",
            "on_call_schedule": {
                "primary": ["senior_engineer_1", "senior_engineer_2"],
                "secondary": ["architect", "team_lead"],
                "manager": ["director"]
            }
        },
        "security": {
            "name": "安全告警升级策略",
            "description": "安全相关告警的专门升级流程",
            "on_call_schedule": {
                "primary": ["security_engineer_1", "security_engineer_2"],
                "secondary": ["security_lead"],
                "manager": ["ciso"]
            }
        }
    }
}


# 告警聚合配置
AGGREGATION_CONFIG = {
    "enabled": True,
    "rules": [
        {
            "name": "component_errors",
            "group_by": ["component"],
            "time_window": timedelta(minutes=5),
            "threshold": 3,
            "message_template": "组件 {component} 出现 {count} 个告警",
            "severity_override": AlertSeverity.HIGH
        },
        {
            "name": "severity_burst",
            "group_by": ["severity"],
            "time_window": timedelta(minutes=2),
            "threshold": 5,
            "message_template": "{severity} 级别告警激增: {count} 个",
            "severity_override": AlertSeverity.CRITICAL
        },
        {
            "name": "system_resource_issues",
            "group_by": ["component", "type"],
            "match_labels": {"component": "system", "type": "resource"},
            "time_window": timedelta(minutes=10),
            "threshold": 2,
            "message_template": "系统资源告警聚合: {count} 个资源问题",
            "severity_override": AlertSeverity.HIGH
        },
        {
            "name": "api_performance_degradation",
            "group_by": ["component"],
            "match_labels": {"component": "api"},
            "time_window": timedelta(minutes=5),
            "threshold": 3,
            "message_template": "API性能告警聚合: {count} 个性能问题",
            "severity_override": AlertSeverity.HIGH
        },
        {
            "name": "security_incidents",
            "group_by": ["component"],
            "match_labels": {"component": "security"},
            "time_window": timedelta(minutes=1),
            "threshold": 2,
            "message_template": "安全告警聚合: {count} 个安全事件",
            "severity_override": AlertSeverity.CRITICAL
        }
    ]
}


# 指标收集配置
METRICS_CONFIG = {
    "collection_interval": 30,  # 秒
    "retention_period": timedelta(days=30),
    "providers": [
        {
            "name": "system_metrics",
            "type": "system",
            "enabled": True,
            "config": {
                "collect_cpu": True,
                "collect_memory": True,
                "collect_disk": True,
                "collect_network": True,
                "collection_interval": 30
            }
        },
        {
            "name": "database_metrics",
            "type": "database",
            "enabled": True,
            "config": {
                "connection_pool_size": True,
                "query_performance": True,
                "error_rates": True,
                "deadlock_detection": True,
                "replication_lag": True,
                "collection_interval": 60
            }
        },
        {
            "name": "api_metrics",
            "type": "api",
            "enabled": True,
            "config": {
                "response_times": True,
                "error_rates": True,
                "request_counts": True,
                "status_codes": True,
                "timeout_rates": True,
                "collection_interval": 30
            }
        },
        {
            "name": "business_metrics",
            "type": "business",
            "enabled": True,
            "config": {
                "token_usage": True,
                "cost_tracking": True,
                "user_activity": True,
                "conversion_rates": True,
                "collection_interval": 300  # 5分钟
            }
        },
        {
            "name": "security_metrics",
            "type": "security",
            "enabled": True,
            "config": {
                "failed_logins": True,
                "suspicious_activity": True,
                "rate_limit_violations": True,
                "collection_interval": 60
            }
        }
    ]
}


# 健康检查配置
HEALTH_CHECK_CONFIG = {
    "interval": timedelta(seconds=30),
    "timeout": timedelta(seconds=10),
    "checks": [
        {
            "name": "database",
            "type": "database",
            "enabled": True,
            "config": {
                "query": "SELECT 1",
                "timeout": 5,
                "critical_threshold": 3,  # 连续失败3次为严重
                "warning_threshold": 1   # 失败1次为警告
            }
        },
        {
            "name": "redis",
            "type": "redis",
            "enabled": False,
            "config": {
                "command": "PING",
                "timeout": 3,
                "critical_threshold": 3,
                "warning_threshold": 1
            }
        },
        {
            "name": "external_api",
            "type": "http",
            "enabled": False,
            "config": {
                "url": "https://api.example.com/health",
                "method": "GET",
                "timeout": 10,
                "expected_status": 200,
                "critical_threshold": 3,
                "warning_threshold": 1
            }
        },
        {
            "name": "disk_space",
            "type": "system",
            "enabled": True,
            "config": {
                "check_type": "disk_usage",
                "path": "/",
                "critical_threshold": 95,  # 95%使用率为严重
                "warning_threshold": 85    # 85%使用率为警告
            }
        },
        {
            "name": "memory_usage",
            "type": "system",
            "enabled": True,
            "config": {
                "check_type": "memory_usage",
                "critical_threshold": 95,
                "warning_threshold": 85
            }
        }
    ]
}


def apply_environment_thresholds(config: Dict[str, Any], environment: str = "production") -> Dict[str, Any]:
    """
    根据环境应用特定的阈值调整
    """
    if environment not in ENVIRONMENT_SPECIFIC_THRESHOLDS:
        return config
    
    multipliers = ENVIRONMENT_SPECIFIC_THRESHOLDS[environment]
    
    # 调整告警规则阈值
    for rule in config.get("alert_rules", []):
        metric = rule.get("metric", "")
        
        # CPU相关阈值调整
        if "cpu" in metric:
            rule["threshold"] *= multipliers["cpu_multiplier"]
        
        # 内存相关阈值调整
        elif "memory" in metric:
            rule["threshold"] *= multipliers["memory_multiplier"]
        
        # 响应时间相关阈值调整
        elif "response_time" in metric:
            rule["threshold"] *= multipliers["response_time_multiplier"]
        
        # 成本相关阈值调整
        elif "cost" in metric:
            rule["threshold"] *= multipliers["cost_multiplier"]
    
    return config


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "alert_rules": DEFAULT_ALERT_RULES,
        "notification": DEFAULT_NOTIFICATION_CONFIG,
        "suppression_rules": DEFAULT_SUPPRESSION_RULES,
        "escalation": ESCALATION_CONFIG,
        "aggregation": AGGREGATION_CONFIG,
        "metrics": METRICS_CONFIG,
        "health_check": HEALTH_CHECK_CONFIG,
        "thresholds": SYSTEM_MONITORING_THRESHOLDS,
        "dynamic_thresholds": DYNAMIC_THRESHOLD_CONFIG
    }


def get_production_config() -> Dict[str, Any]:
    """获取生产环境配置"""
    config = get_default_config()
    
    # 生产环境特定配置
    config["notification"]["channels"][1]["enabled"] = True  # 启用邮件通知
    config["notification"]["channels"][3]["enabled"] = True  # 启用钉钉通知
    config["suppression_rules"][0]["enabled"] = True  # 启用维护窗口
    
    # 应用生产环境阈值
    config = apply_environment_thresholds(config, "production")
    
    return config


def get_development_config() -> Dict[str, Any]:
    """获取开发环境配置"""
    config = get_default_config()
    
    # 开发环境特定配置
    config["notification"]["channels"] = [
        config["notification"]["channels"][0]  # 只保留控制台通知
    ]
    
    # 应用开发环境阈值
    config = apply_environment_thresholds(config, "development")
    
    return config


def get_staging_config() -> Dict[str, Any]:
    """获取测试环境配置"""
    config = get_default_config()
    
    # 测试环境特定配置
    config["notification"]["channels"][1]["enabled"] = True  # 启用邮件通知
    config["notification"]["channels"][2]["enabled"] = True  # 启用Slack通知
    
    # 应用测试环境阈值
    config = apply_environment_thresholds(config, "staging")
    
    return config