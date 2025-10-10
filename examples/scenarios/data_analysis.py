#!/usr/bin/env python3
"""
HarborAI 数据分析助手

场景描述:
构建智能数据分析平台，支持自动化数据处理、统计分析、可视化生成和洞察发现。
适用于业务分析、市场研究、运营优化等多种数据驱动的决策场景。

应用价值:
- 自动化数据清洗和预处理
- 智能统计分析和模式识别
- 自然语言查询和报告生成
- 可视化图表自动生成
- 业务洞察和建议提供

核心功能:
1. 自然语言数据查询和分析
2. 自动化统计分析和假设检验
3. 数据可视化和图表生成
4. 异常检测和趋势分析
5. 智能报告生成和洞察发现
"""

import asyncio
import json
import time
import uuid
import logging
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import io
import base64
from collections import defaultdict
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AnalysisType(Enum):
    """分析类型"""
    DESCRIPTIVE = "descriptive"           # 描述性分析
    DIAGNOSTIC = "diagnostic"             # 诊断性分析
    PREDICTIVE = "predictive"             # 预测性分析
    PRESCRIPTIVE = "prescriptive"         # 规范性分析
    EXPLORATORY = "exploratory"           # 探索性分析

class DataType(Enum):
    """数据类型"""
    NUMERICAL = "numerical"               # 数值型
    CATEGORICAL = "categorical"           # 分类型
    TEMPORAL = "temporal"                 # 时间序列
    TEXT = "text"                        # 文本型
    MIXED = "mixed"                      # 混合型

class ChartType(Enum):
    """图表类型"""
    LINE = "line"                        # 折线图
    BAR = "bar"                          # 柱状图
    SCATTER = "scatter"                  # 散点图
    HISTOGRAM = "histogram"              # 直方图
    BOX = "box"                          # 箱线图
    HEATMAP = "heatmap"                  # 热力图
    PIE = "pie"                          # 饼图
    VIOLIN = "violin"                    # 小提琴图

class InsightLevel(Enum):
    """洞察级别"""
    CRITICAL = "critical"                # 关键
    IMPORTANT = "important"              # 重要
    MODERATE = "moderate"                # 中等
    MINOR = "minor"                      # 次要

@dataclass
class DataSource:
    """数据源"""
    id: str
    name: str
    description: str
    data_type: DataType
    file_path: Optional[str] = None
    connection_string: Optional[str] = None
    query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: Optional[datetime] = None

@dataclass
class AnalysisRequest:
    """分析请求"""
    id: str
    query: str                           # 自然语言查询
    data_source_id: str
    analysis_type: AnalysisType
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 3                    # 1-5, 5最高

@dataclass
class StatisticalResult:
    """统计结果"""
    metric: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    significance: Optional[bool] = None
    interpretation: str = ""

@dataclass
class Insight:
    """数据洞察"""
    id: str
    title: str
    description: str
    level: InsightLevel
    confidence: float                    # 0-1
    supporting_data: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AnalysisResult:
    """分析结果"""
    id: str
    request: AnalysisRequest
    summary: str
    statistical_results: List[StatisticalResult]
    insights: List[Insight]
    visualizations: List[str] = field(default_factory=list)  # 图表文件路径
    raw_data: Optional[pd.DataFrame] = None
    execution_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_data(self, data_source: DataSource) -> pd.DataFrame:
        """加载数据"""
        try:
            if data_source.id in self.data_cache:
                return self.data_cache[data_source.id]
            
            if data_source.file_path:
                # 从文件加载
                if data_source.file_path.endswith('.csv'):
                    df = pd.read_csv(data_source.file_path)
                elif data_source.file_path.endswith('.xlsx'):
                    df = pd.read_excel(data_source.file_path)
                elif data_source.file_path.endswith('.json'):
                    df = pd.read_json(data_source.file_path)
                else:
                    raise ValueError(f"不支持的文件格式: {data_source.file_path}")
            
            elif data_source.connection_string and data_source.query:
                # 从数据库加载
                conn = sqlite3.connect(data_source.connection_string)
                df = pd.read_sql_query(data_source.query, conn)
                conn.close()
            
            else:
                # 生成示例数据
                df = self._generate_sample_data(data_source.data_type)
            
            # 缓存数据
            self.data_cache[data_source.id] = df
            
            logger.info(f"数据加载完成: {data_source.name}, 形状: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise e
    
    def _generate_sample_data(self, data_type: DataType) -> pd.DataFrame:
        """生成示例数据"""
        np.random.seed(42)
        
        if data_type == DataType.NUMERICAL:
            # 生成数值型数据
            n = 1000
            data = {
                'id': range(1, n+1),
                'value1': np.random.normal(100, 15, n),
                'value2': np.random.exponential(50, n),
                'value3': np.random.uniform(0, 100, n),
                'category': np.random.choice(['A', 'B', 'C', 'D'], n),
                'date': pd.date_range('2023-01-01', periods=n, freq='D')
            }
            
        elif data_type == DataType.TEMPORAL:
            # 生成时间序列数据
            dates = pd.date_range('2023-01-01', periods=365, freq='D')
            trend = np.linspace(100, 200, len(dates))
            seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
            noise = np.random.normal(0, 5, len(dates))
            
            data = {
                'date': dates,
                'value': trend + seasonal + noise,
                'category': np.random.choice(['产品A', '产品B', '产品C'], len(dates))
            }
            
        elif data_type == DataType.CATEGORICAL:
            # 生成分类数据
            n = 500
            data = {
                'customer_id': range(1, n+1),
                'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n),
                'gender': np.random.choice(['男', '女'], n),
                'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], n),
                'purchase_amount': np.random.lognormal(4, 1, n),
                'satisfaction': np.random.choice(['很满意', '满意', '一般', '不满意'], n, p=[0.3, 0.4, 0.2, 0.1])
            }
            
        else:
            # 默认混合数据
            n = 800
            data = {
                'id': range(1, n+1),
                'sales': np.random.lognormal(5, 0.5, n),
                'profit_margin': np.random.normal(0.15, 0.05, n),
                'region': np.random.choice(['华北', '华东', '华南', '西南'], n),
                'product_type': np.random.choice(['电子产品', '服装', '食品', '家具'], n),
                'quarter': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], n),
                'customer_rating': np.random.uniform(1, 5, n)
            }
        
        return pd.DataFrame(data)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 处理缺失值
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # 数值型用均值填充
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                # 分类型用众数填充
                df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # 处理异常值（使用IQR方法）
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 将异常值替换为边界值
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据摘要"""
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numerical_summary": {},
            "categorical_summary": {}
        }
        
        # 数值型列摘要
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            summary["numerical_summary"] = df[numerical_cols].describe().to_dict()
        
        # 分类型列摘要
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary["categorical_summary"][col] = {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head().to_dict()
            }
        
        return summary

class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self):
        pass
    
    def descriptive_analysis(self, df: pd.DataFrame, columns: List[str] = None) -> List[StatisticalResult]:
        """描述性统计分析"""
        results = []
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in df.columns and df[col].dtype in [np.number]:
                data = df[col].dropna()
                
                # 基本统计量
                results.extend([
                    StatisticalResult("均值", data.mean(), interpretation=f"{col}的平均值"),
                    StatisticalResult("中位数", data.median(), interpretation=f"{col}的中位数"),
                    StatisticalResult("标准差", data.std(), interpretation=f"{col}的标准差"),
                    StatisticalResult("偏度", stats.skew(data), interpretation=f"{col}的偏度"),
                    StatisticalResult("峰度", stats.kurtosis(data), interpretation=f"{col}的峰度")
                ])
                
                # 置信区间
                confidence_interval = stats.t.interval(0.95, len(data)-1, 
                                                     loc=data.mean(), 
                                                     scale=stats.sem(data))
                results.append(
                    StatisticalResult("95%置信区间", data.mean(), 
                                    confidence_interval=confidence_interval,
                                    interpretation=f"{col}均值的95%置信区间")
                )
        
        return results
    
    def correlation_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[StatisticalResult]]:
        """相关性分析"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return pd.DataFrame(), []
        
        # 计算相关系数矩阵
        corr_matrix = df[numerical_cols].corr()
        
        # 找出强相关关系
        results = []
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                col1, col2 = numerical_cols[i], numerical_cols[j]
                corr_value = corr_matrix.loc[col1, col2]
                
                # 计算显著性
                n = len(df[[col1, col2]].dropna())
                if n > 2:
                    t_stat = corr_value * np.sqrt((n-2) / (1-corr_value**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                    
                    interpretation = f"{col1}与{col2}的相关系数"
                    if abs(corr_value) > 0.7:
                        interpretation += " (强相关)"
                    elif abs(corr_value) > 0.3:
                        interpretation += " (中等相关)"
                    else:
                        interpretation += " (弱相关)"
                    
                    results.append(
                        StatisticalResult(
                            f"{col1} vs {col2}",
                            corr_value,
                            p_value=p_value,
                            significance=p_value < 0.05,
                            interpretation=interpretation
                        )
                    )
        
        return corr_matrix, results
    
    def hypothesis_testing(self, df: pd.DataFrame, group_col: str, value_col: str) -> List[StatisticalResult]:
        """假设检验"""
        results = []
        
        if group_col not in df.columns or value_col not in df.columns:
            return results
        
        groups = df.groupby(group_col)[value_col].apply(list)
        group_names = list(groups.index)
        
        if len(group_names) == 2:
            # 两组t检验
            group1, group2 = groups.iloc[0], groups.iloc[1]
            t_stat, p_value = stats.ttest_ind(group1, group2)
            
            results.append(
                StatisticalResult(
                    "两样本t检验",
                    t_stat,
                    p_value=p_value,
                    significance=p_value < 0.05,
                    interpretation=f"{group_names[0]}与{group_names[1]}在{value_col}上的差异检验"
                )
            )
            
        elif len(group_names) > 2:
            # 方差分析
            group_data = [groups.iloc[i] for i in range(len(group_names))]
            f_stat, p_value = stats.f_oneway(*group_data)
            
            results.append(
                StatisticalResult(
                    "单因素方差分析",
                    f_stat,
                    p_value=p_value,
                    significance=p_value < 0.05,
                    interpretation=f"各组在{value_col}上的差异检验"
                )
            )
        
        return results
    
    def anomaly_detection(self, df: pd.DataFrame, columns: List[str] = None) -> Tuple[pd.DataFrame, List[StatisticalResult]]:
        """异常检测"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        # 使用Isolation Forest进行异常检测
        data = df[columns].dropna()
        
        if len(data) < 10:
            return pd.DataFrame(), []
        
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 异常检测
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(data_scaled)
        
        # 添加异常标签
        anomaly_df = data.copy()
        anomaly_df['is_anomaly'] = anomaly_labels == -1
        
        # 统计结果
        anomaly_count = sum(anomaly_labels == -1)
        anomaly_rate = anomaly_count / len(data)
        
        results = [
            StatisticalResult(
                "异常点数量",
                anomaly_count,
                interpretation=f"检测到{anomaly_count}个异常点"
            ),
            StatisticalResult(
                "异常率",
                anomaly_rate,
                interpretation=f"异常率为{anomaly_rate:.2%}"
            )
        ]
        
        return anomaly_df, results

class VisualizationGenerator:
    """可视化生成器"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_chart(self, df: pd.DataFrame, chart_type: ChartType, 
                      x_col: str = None, y_col: str = None, 
                      group_col: str = None, title: str = None) -> str:
        """生成图表"""
        plt.figure(figsize=(10, 6))
        
        if chart_type == ChartType.LINE:
            if x_col and y_col:
                if group_col:
                    for group in df[group_col].unique():
                        group_data = df[df[group_col] == group]
                        plt.plot(group_data[x_col], group_data[y_col], label=group, marker='o')
                    plt.legend()
                else:
                    plt.plot(df[x_col], df[y_col], marker='o')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
        
        elif chart_type == ChartType.BAR:
            if x_col and y_col:
                if group_col:
                    df_pivot = df.pivot_table(values=y_col, index=x_col, columns=group_col, aggfunc='mean')
                    df_pivot.plot(kind='bar', ax=plt.gca())
                else:
                    df.groupby(x_col)[y_col].mean().plot(kind='bar')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.xticks(rotation=45)
        
        elif chart_type == ChartType.SCATTER:
            if x_col and y_col:
                if group_col:
                    for group in df[group_col].unique():
                        group_data = df[df[group_col] == group]
                        plt.scatter(group_data[x_col], group_data[y_col], label=group, alpha=0.6)
                    plt.legend()
                else:
                    plt.scatter(df[x_col], df[y_col], alpha=0.6)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
        
        elif chart_type == ChartType.HISTOGRAM:
            if x_col:
                if group_col:
                    for group in df[group_col].unique():
                        group_data = df[df[group_col] == group]
                        plt.hist(group_data[x_col], alpha=0.6, label=group, bins=20)
                    plt.legend()
                else:
                    plt.hist(df[x_col], bins=20, alpha=0.7)
                plt.xlabel(x_col)
                plt.ylabel('频次')
        
        elif chart_type == ChartType.BOX:
            if y_col:
                if group_col:
                    df.boxplot(column=y_col, by=group_col, ax=plt.gca())
                else:
                    df.boxplot(column=y_col, ax=plt.gca())
        
        elif chart_type == ChartType.HEATMAP:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        
        elif chart_type == ChartType.PIE:
            if x_col:
                value_counts = df[x_col].value_counts()
                plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        
        if title:
            plt.title(title)
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"{chart_type.value}_{int(time.time())}.png"
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_dashboard(self, df: pd.DataFrame, title: str = "数据分析仪表板") -> str:
        """生成综合仪表板"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # 1. 数值型变量分布
        if len(numerical_cols) > 0:
            col = numerical_cols[0]
            axes[0, 0].hist(df[col].dropna(), bins=20, alpha=0.7)
            axes[0, 0].set_title(f'{col} 分布')
            axes[0, 0].set_xlabel(col)
            axes[0, 0].set_ylabel('频次')
        
        # 2. 相关性热力图
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            axes[0, 1].set_title('相关性矩阵')
            axes[0, 1].set_xticks(range(len(numerical_cols)))
            axes[0, 1].set_yticks(range(len(numerical_cols)))
            axes[0, 1].set_xticklabels(numerical_cols, rotation=45)
            axes[0, 1].set_yticklabels(numerical_cols)
            plt.colorbar(im, ax=axes[0, 1])
        
        # 3. 分类变量分布
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            axes[1, 0].bar(range(len(value_counts)), value_counts.values)
            axes[1, 0].set_title(f'{col} 分布')
            axes[1, 0].set_xlabel(col)
            axes[1, 0].set_ylabel('计数')
            axes[1, 0].set_xticks(range(len(value_counts)))
            axes[1, 0].set_xticklabels(value_counts.index, rotation=45)
        
        # 4. 时间序列（如果有日期列）
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numerical_cols) > 0:
            date_col = date_cols[0]
            value_col = numerical_cols[0]
            df_sorted = df.sort_values(date_col)
            axes[1, 1].plot(df_sorted[date_col], df_sorted[value_col])
            axes[1, 1].set_title(f'{value_col} 时间趋势')
            axes[1, 1].set_xlabel(date_col)
            axes[1, 1].set_ylabel(value_col)
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            # 箱线图
            if len(numerical_cols) > 0:
                col = numerical_cols[0]
                axes[1, 1].boxplot(df[col].dropna())
                axes[1, 1].set_title(f'{col} 箱线图')
                axes[1, 1].set_ylabel(col)
        
        plt.tight_layout()
        
        # 保存仪表板
        filename = f"dashboard_{int(time.time())}.png"
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

class InsightGenerator:
    """洞察生成器"""
    
    def __init__(self, client: HarborAI):
        self.client = client
    
    async def generate_insights(self, df: pd.DataFrame, 
                              statistical_results: List[StatisticalResult],
                              analysis_context: str = "") -> List[Insight]:
        """生成数据洞察"""
        try:
            # 准备数据摘要
            data_summary = self._prepare_data_summary(df)
            stats_summary = self._prepare_stats_summary(statistical_results)
            
            insight_prompt = f"""
作为数据分析专家，请基于以下数据和统计分析结果，生成深入的业务洞察：

分析背景: {analysis_context}

数据摘要:
{data_summary}

统计分析结果:
{stats_summary}

请生成3-5个关键洞察，每个洞察包含：
1. 洞察标题（简洁明了）
2. 详细描述（包含具体数据支撑）
3. 重要性级别（critical/important/moderate/minor）
4. 置信度（0-1）
5. 业务建议（具体可执行的建议）

请以JSON格式返回：
{{
    "insights": [
        {{
            "title": "洞察标题",
            "description": "详细描述",
            "level": "important",
            "confidence": 0.85,
            "recommendations": ["建议1", "建议2"],
            "supporting_data": {{"key": "value"}}
        }}
    ]
}}
"""
            
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的数据分析师，擅长从数据中发现有价值的业务洞察。"},
                    {"role": "user", "content": insight_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            
            # 解析JSON结果
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                insights = []
                
                for insight_data in result_data.get("insights", []):
                    insight = Insight(
                        id=str(uuid.uuid4()),
                        title=insight_data.get("title", ""),
                        description=insight_data.get("description", ""),
                        level=InsightLevel(insight_data.get("level", "moderate")),
                        confidence=insight_data.get("confidence", 0.5),
                        supporting_data=insight_data.get("supporting_data", {}),
                        recommendations=insight_data.get("recommendations", [])
                    )
                    insights.append(insight)
                
                return insights
            
        except Exception as e:
            logger.error(f"洞察生成失败: {str(e)}")
        
        # 返回默认洞察
        return [
            Insight(
                id=str(uuid.uuid4()),
                title="数据概览",
                description=f"数据集包含{df.shape[0]}行{df.shape[1]}列，需要进一步分析以发现模式。",
                level=InsightLevel.MODERATE,
                confidence=0.7,
                supporting_data={"rows": df.shape[0], "columns": df.shape[1]},
                recommendations=["进行更深入的探索性数据分析", "检查数据质量和完整性"]
            )
        ]
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """准备数据摘要"""
        summary_parts = [
            f"数据形状: {df.shape[0]}行 x {df.shape[1]}列",
            f"数值型列: {len(df.select_dtypes(include=[np.number]).columns)}个",
            f"分类型列: {len(df.select_dtypes(include=['object']).columns)}个"
        ]
        
        # 缺失值情况
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            summary_parts.append(f"缺失值: {missing_info.sum()}个")
        
        # 数值型列的基本统计
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            for col in numerical_cols[:3]:  # 只显示前3列
                mean_val = df[col].mean()
                std_val = df[col].std()
                summary_parts.append(f"{col}: 均值={mean_val:.2f}, 标准差={std_val:.2f}")
        
        return "\n".join(summary_parts)
    
    def _prepare_stats_summary(self, statistical_results: List[StatisticalResult]) -> str:
        """准备统计摘要"""
        if not statistical_results:
            return "暂无统计分析结果"
        
        summary_parts = []
        for result in statistical_results[:10]:  # 只显示前10个结果
            summary_parts.append(f"{result.metric}: {result.value:.4f}")
            if result.p_value is not None:
                summary_parts.append(f"  p值: {result.p_value:.4f}")
            if result.interpretation:
                summary_parts.append(f"  解释: {result.interpretation}")
        
        return "\n".join(summary_parts)

class QueryProcessor:
    """查询处理器"""
    
    def __init__(self, client: HarborAI):
        self.client = client
    
    async def parse_query(self, query: str, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """解析自然语言查询"""
        try:
            parse_prompt = f"""
请解析以下自然语言数据分析查询，并返回结构化的分析参数：

用户查询: {query}

可用数据列: {', '.join(data_summary.get('columns', []))}
数据类型: {data_summary.get('dtypes', {})}

请分析用户想要进行什么类型的分析，并返回JSON格式的参数：
{{
    "analysis_type": "descriptive/diagnostic/predictive/exploratory",
    "target_columns": ["列名1", "列名2"],
    "group_by_column": "分组列名或null",
    "filter_conditions": {{"列名": "条件"}},
    "chart_type": "line/bar/scatter/histogram/box/heatmap/pie",
    "specific_operations": ["操作1", "操作2"],
    "interpretation": "查询意图的解释"
}}
"""
            
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个数据分析查询解析专家。"},
                    {"role": "user", "content": parse_prompt}
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
            logger.error(f"查询解析失败: {str(e)}")
        
        # 返回默认解析结果
        return {
            "analysis_type": "exploratory",
            "target_columns": [],
            "group_by_column": None,
            "filter_conditions": {},
            "chart_type": "histogram",
            "specific_operations": ["descriptive_statistics"],
            "interpretation": "进行探索性数据分析"
        }

class DataAnalysisAssistant:
    """数据分析助手"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.deepseek.com/v1",
                 db_path: str = "analysis_assistant.db"):
        
        # 初始化组件
        self.client = HarborAI(api_key=api_key, base_url=base_url)
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualization_generator = VisualizationGenerator()
        self.insight_generator = InsightGenerator(self.client)
        self.query_processor = QueryProcessor(self.client)
        
        # 数据源管理
        self.data_sources: Dict[str, DataSource] = {}
        
        # 分析历史
        self.analysis_history: List[AnalysisResult] = []
        
        # 初始化数据库
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建分析结果表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                data_source_id TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                summary TEXT,
                execution_time REAL,
                created_at TIMESTAMP NOT NULL,
                result_data TEXT
            )
        """)
        
        # 创建洞察表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                level TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analysis_results (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_data_source(self, data_source: DataSource):
        """注册数据源"""
        self.data_sources[data_source.id] = data_source
        logger.info(f"数据源已注册: {data_source.name}")
    
    async def analyze(self, query: str, data_source_id: str) -> AnalysisResult:
        """执行数据分析"""
        start_time = time.time()
        
        try:
            # 获取数据源
            if data_source_id not in self.data_sources:
                raise ValueError(f"数据源不存在: {data_source_id}")
            
            data_source = self.data_sources[data_source_id]
            
            # 加载和清洗数据
            df = self.data_processor.load_data(data_source)
            df_clean = self.data_processor.clean_data(df)
            
            # 获取数据摘要
            data_summary = self.data_processor.get_data_summary(df_clean)
            
            # 解析查询
            query_params = await self.query_processor.parse_query(query, data_summary)
            
            # 创建分析请求
            request = AnalysisRequest(
                id=str(uuid.uuid4()),
                query=query,
                data_source_id=data_source_id,
                analysis_type=AnalysisType(query_params.get("analysis_type", "exploratory")),
                parameters=query_params
            )
            
            # 执行统计分析
            statistical_results = []
            
            # 描述性统计
            if query_params.get("target_columns"):
                desc_results = self.statistical_analyzer.descriptive_analysis(
                    df_clean, query_params["target_columns"]
                )
                statistical_results.extend(desc_results)
            
            # 相关性分析
            corr_matrix, corr_results = self.statistical_analyzer.correlation_analysis(df_clean)
            statistical_results.extend(corr_results)
            
            # 假设检验（如果有分组列）
            if query_params.get("group_by_column") and query_params.get("target_columns"):
                group_col = query_params["group_by_column"]
                value_col = query_params["target_columns"][0]
                if group_col in df_clean.columns and value_col in df_clean.columns:
                    hyp_results = self.statistical_analyzer.hypothesis_testing(
                        df_clean, group_col, value_col
                    )
                    statistical_results.extend(hyp_results)
            
            # 异常检测
            anomaly_df, anomaly_results = self.statistical_analyzer.anomaly_detection(df_clean)
            statistical_results.extend(anomaly_results)
            
            # 生成可视化
            visualizations = []
            chart_type = ChartType(query_params.get("chart_type", "histogram"))
            
            if query_params.get("target_columns"):
                target_cols = query_params["target_columns"]
                group_col = query_params.get("group_by_column")
                
                if len(target_cols) >= 1:
                    chart_path = self.visualization_generator.generate_chart(
                        df_clean,
                        chart_type,
                        x_col=target_cols[0] if len(target_cols) > 0 else None,
                        y_col=target_cols[1] if len(target_cols) > 1 else target_cols[0],
                        group_col=group_col,
                        title=f"{query} - {chart_type.value}图"
                    )
                    visualizations.append(chart_path)
            
            # 生成综合仪表板
            dashboard_path = self.visualization_generator.generate_dashboard(
                df_clean, f"数据分析仪表板 - {query}"
            )
            visualizations.append(dashboard_path)
            
            # 生成洞察
            insights = await self.insight_generator.generate_insights(
                df_clean, statistical_results, query
            )
            
            # 生成分析摘要
            summary = await self._generate_analysis_summary(
                query, statistical_results, insights
            )
            
            # 创建分析结果
            execution_time = time.time() - start_time
            result = AnalysisResult(
                id=str(uuid.uuid4()),
                request=request,
                summary=summary,
                statistical_results=statistical_results,
                insights=insights,
                visualizations=visualizations,
                raw_data=df_clean,
                execution_time=execution_time
            )
            
            # 保存结果
            self._save_analysis_result(result)
            self.analysis_history.append(result)
            
            logger.info(f"分析完成: {result.id} (耗时: {execution_time:.3f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"分析失败: {str(e)}")
            raise e
    
    async def _generate_analysis_summary(self, 
                                       query: str,
                                       statistical_results: List[StatisticalResult],
                                       insights: List[Insight]) -> str:
        """生成分析摘要"""
        try:
            # 准备统计结果摘要
            stats_summary = []
            for result in statistical_results[:5]:  # 只取前5个结果
                stats_summary.append(f"- {result.metric}: {result.value:.4f}")
                if result.interpretation:
                    stats_summary.append(f"  {result.interpretation}")
            
            # 准备洞察摘要
            insights_summary = []
            for insight in insights[:3]:  # 只取前3个洞察
                insights_summary.append(f"- {insight.title}: {insight.description[:100]}...")
            
            summary_prompt = f"""
请基于以下分析结果，生成一个简洁明了的分析摘要：

用户查询: {query}

统计分析结果:
{chr(10).join(stats_summary)}

关键洞察:
{chr(10).join(insights_summary)}

请生成一个200字以内的分析摘要，包含：
1. 数据的主要特征
2. 关键发现
3. 重要结论
4. 建议的后续行动

摘要应该简洁、专业且易于理解。
"""
            
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的数据分析师，擅长总结分析结果。"},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"摘要生成失败: {str(e)}")
            return f"针对查询'{query}'的数据分析已完成，发现了{len(statistical_results)}个统计结果和{len(insights)}个关键洞察。"
    
    def _save_analysis_result(self, result: AnalysisResult):
        """保存分析结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 保存分析结果
        cursor.execute("""
            INSERT INTO analysis_results 
            (id, query, data_source_id, analysis_type, summary, execution_time, created_at, result_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.id,
            result.request.query,
            result.request.data_source_id,
            result.request.analysis_type.value,
            result.summary,
            result.execution_time,
            result.created_at,
            json.dumps({
                "statistical_results": [asdict(sr) for sr in result.statistical_results],
                "visualizations": result.visualizations
            })
        ))
        
        # 保存洞察
        for insight in result.insights:
            cursor.execute("""
                INSERT INTO insights 
                (id, analysis_id, title, description, level, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                insight.id,
                result.id,
                insight.title,
                insight.description,
                insight.level.value,
                insight.confidence,
                insight.created_at
            ))
        
        conn.commit()
        conn.close()
    
    def get_analysis_history(self, limit: int = 10) -> List[AnalysisResult]:
        """获取分析历史"""
        return self.analysis_history[-limit:]
    
    def get_data_sources(self) -> List[DataSource]:
        """获取数据源列表"""
        return list(self.data_sources.values())

# 演示函数
async def demo_basic_analysis():
    """演示基础数据分析"""
    print("\n📊 基础数据分析演示")
    print("=" * 50)
    
    # 创建数据分析助手
    assistant = DataAnalysisAssistant(api_key="your-deepseek-key")
    
    # 注册数据源
    data_source = DataSource(
        id="sales_data",
        name="销售数据",
        description="公司销售业绩数据",
        data_type=DataType.MIXED
    )
    assistant.register_data_source(data_source)
    
    print(f"📋 数据源已注册: {data_source.name}")
    
    # 执行分析
    query = "分析销售数据的分布情况和各地区的业绩差异"
    
    print(f"🔍 分析查询: {query}")
    print(f"🔄 正在执行分析...")
    
    start_time = time.time()
    result = await assistant.analyze(query, data_source.id)
    analysis_time = time.time() - start_time
    
    print(f"✅ 分析完成!")
    print(f"   分析ID: {result.id}")
    print(f"   执行时间: {analysis_time:.3f}s")
    print(f"   统计结果数: {len(result.statistical_results)}")
    print(f"   洞察数: {len(result.insights)}")
    print(f"   可视化图表数: {len(result.visualizations)}")
    
    # 显示分析摘要
    print(f"\n📄 分析摘要:")
    print(f"   {result.summary}")
    
    # 显示关键统计结果
    print(f"\n📈 关键统计结果:")
    for i, stat in enumerate(result.statistical_results[:5]):
        print(f"   {i+1}. {stat.metric}: {stat.value:.4f}")
        if stat.interpretation:
            print(f"      {stat.interpretation}")
    
    # 显示洞察
    print(f"\n💡 关键洞察:")
    for i, insight in enumerate(result.insights):
        print(f"   {i+1}. {insight.title} ({insight.level.value})")
        print(f"      {insight.description[:100]}...")
        if insight.recommendations:
            print(f"      建议: {', '.join(insight.recommendations[:2])}")
    
    # 显示可视化文件
    print(f"\n📊 生成的可视化:")
    for i, viz_path in enumerate(result.visualizations):
        print(f"   {i+1}. {viz_path}")
    
    return assistant, result

async def demo_natural_language_query():
    """演示自然语言查询"""
    print("\n🗣️ 自然语言查询演示")
    print("=" * 50)
    
    assistant = DataAnalysisAssistant(api_key="your-deepseek-key")
    
    # 注册时间序列数据源
    data_source = DataSource(
        id="time_series_data",
        name="时间序列数据",
        description="产品销量时间序列数据",
        data_type=DataType.TEMPORAL
    )
    assistant.register_data_source(data_source)
    
    # 多个自然语言查询
    queries = [
        "显示销量的时间趋势，并找出季节性模式",
        "比较不同产品类别的销售表现",
        "检测销量数据中的异常值",
        "分析销量与其他变量的相关性"
    ]
    
    results = []
    
    for i, query in enumerate(queries):
        print(f"\n🔍 查询 {i+1}: {query}")
        print(f"🔄 正在分析...")
        
        result = await assistant.analyze(query, data_source.id)
        results.append(result)
        
        print(f"✅ 分析完成 (耗时: {result.execution_time:.3f}s)")
        print(f"   摘要: {result.summary[:100]}...")
        
        # 显示最重要的洞察
        if result.insights:
            top_insight = max(result.insights, key=lambda x: x.confidence)
            print(f"   关键洞察: {top_insight.title}")
    
    # 显示分析历史
    print(f"\n📚 分析历史:")
    history = assistant.get_analysis_history()
    for i, result in enumerate(history):
        print(f"   {i+1}. {result.request.query[:50]}... (耗时: {result.execution_time:.3f}s)")
    
    return results

async def demo_advanced_analytics():
    """演示高级分析功能"""
    print("\n🔬 高级分析功能演示")
    print("=" * 50)
    
    assistant = DataAnalysisAssistant(api_key="your-deepseek-key")
    
    # 注册复杂数据源
    data_source = DataSource(
        id="complex_data",
        name="复杂业务数据",
        description="包含多维度的复杂业务数据",
        data_type=DataType.MIXED
    )
    assistant.register_data_source(data_source)
    
    # 高级分析查询
    advanced_queries = [
        "进行客户细分分析，识别不同的客户群体",
        "执行A/B测试分析，比较两个版本的效果差异",
        "预测未来3个月的销售趋势",
        "进行异常检测，找出可能的数据质量问题"
    ]
    
    print(f"📋 高级分析任务:")
    for i, query in enumerate(advanced_queries):
        print(f"   {i+1}. {query}")
    
    # 批量执行分析
    print(f"\n🔄 正在执行批量分析...")
    start_time = time.time()
    
    tasks = [assistant.analyze(query, data_source.id) for query in advanced_queries]
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    print(f"✅ 批量分析完成!")
    print(f"   总耗时: {total_time:.3f}s")
    print(f"   平均耗时: {total_time/len(results):.3f}s/个")
    
    # 分析结果汇总
    total_insights = sum(len(result.insights) for result in results)
    total_stats = sum(len(result.statistical_results) for result in results)
    total_visualizations = sum(len(result.visualizations) for result in results)
    
    print(f"\n📊 分析结果汇总:")
    print(f"   总洞察数: {total_insights}")
    print(f"   总统计结果数: {total_stats}")
    print(f"   总可视化数: {total_visualizations}")
    
    # 显示每个分析的关键发现
    print(f"\n🔍 关键发现:")
    for i, result in enumerate(results):
        print(f"   分析 {i+1}: {advanced_queries[i][:30]}...")
        if result.insights:
            top_insight = max(result.insights, key=lambda x: x.confidence)
            print(f"     关键洞察: {top_insight.title}")
            print(f"     置信度: {top_insight.confidence:.2f}")
    
    return results

async def demo_interactive_analysis():
    """演示交互式分析"""
    print("\n🎯 交互式分析演示")
    print("=" * 50)
    
    assistant = DataAnalysisAssistant(api_key="your-deepseek-key")
    
    # 注册数据源
    data_source = DataSource(
        id="interactive_data",
        name="交互式分析数据",
        description="用于交互式分析的示例数据",
        data_type=DataType.MIXED
    )
    assistant.register_data_source(data_source)
    
    # 模拟交互式分析会话
    conversation = [
        "首先给我一个数据的整体概览",
        "重点分析销售额最高的几个地区",
        "这些地区有什么共同特征？",
        "预测这些地区下个季度的表现",
        "给出具体的业务建议"
    ]
    
    print(f"💬 交互式分析会话:")
    
    context_results = []
    
    for i, query in enumerate(conversation):
        print(f"\n👤 用户: {query}")
        print(f"🤖 助手: 正在分析...")
        
        # 在查询中加入上下文
        if context_results:
            contextual_query = f"基于之前的分析结果，{query}"
        else:
            contextual_query = query
        
        result = await assistant.analyze(contextual_query, data_source.id)
        context_results.append(result)
        
        print(f"🤖 助手: {result.summary}")
        
        # 显示关键洞察
        if result.insights:
            top_insight = result.insights[0]
            print(f"💡 关键发现: {top_insight.title}")
            if top_insight.recommendations:
                print(f"📋 建议: {top_insight.recommendations[0]}")
    
    # 生成最终报告
    print(f"\n📄 生成最终分析报告...")
    
    final_summary = await assistant._generate_analysis_summary(
        "交互式分析会话总结",
        [stat for result in context_results for stat in result.statistical_results],
        [insight for result in context_results for insight in result.insights]
    )
    
    print(f"📊 最终分析报告:")
    print(f"   {final_summary}")
    
    return context_results

async def main():
    """主演示函数"""
    print("📊 HarborAI 数据分析助手演示")
    print("=" * 60)
    
    try:
        # 基础数据分析演示
        assistant, basic_result = await demo_basic_analysis()
        
        # 自然语言查询演示
        await demo_natural_language_query()
        
        # 高级分析功能演示
        await demo_advanced_analytics()
        
        # 交互式分析演示
        await demo_interactive_analysis()
        
        print("\n📈 系统性能统计:")
        history = assistant.get_analysis_history()
        if history:
            avg_time = sum(r.execution_time for r in history) / len(history)
            total_insights = sum(len(r.insights) for r in history)
            total_visualizations = sum(len(r.visualizations) for r in history)
            
            print(f"   总分析次数: {len(history)}")
            print(f"   平均分析时间: {avg_time:.3f}s")
            print(f"   总生成洞察: {total_insights}")
            print(f"   总生成图表: {total_visualizations}")
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境部署建议:")
        print("   1. 集成更多数据源（数据库、API、文件系统）")
        print("   2. 实现实时数据流分析和监控")
        print("   3. 添加更多高级分析算法（机器学习、深度学习）")
        print("   4. 构建交互式仪表板和报告系统")
        print("   5. 实现分析结果的自动化分发和告警")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())