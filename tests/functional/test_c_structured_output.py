# -*- coding: utf-8 -*-
"""
HarborAI 结构化输出测试模块

测试目标：
- 验证JSON Schema约束的结构化输出
- 测试Pydantic模型的数据验证
- 验证复杂嵌套结构的正确性
- 测试结构化输出的性能和准确性
"""

import pytest
import json
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime, date
from enum import Enum

from harborai import HarborAI
from harborai.core.exceptions import HarborAIError


# 测试用的Pydantic模型定义
class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(str, Enum):
    """优先级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Address(BaseModel):
    """地址模型"""
    street: str = Field(..., description="街道地址")
    city: str = Field(..., description="城市")
    state: str = Field(..., description="省/州")
    zip_code: str = Field(..., pattern=r"^\d{5,6}(-\d{4})?$", description="邮政编码")
    country: str = Field(default="China", description="国家")


class Person(BaseModel):
    """人员模型"""
    name: str = Field(..., min_length=1, max_length=100, description="姓名")
    age: int = Field(..., ge=0, le=150, description="年龄")
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", description="邮箱")
    phone: Optional[str] = Field(None, pattern=r"^\+?[1-9][\d\-]{1,14}$", description="电话号码")
    address: Optional[Address] = Field(None, description="地址信息")
    is_active: bool = Field(default=True, description="是否激活")


class Task(BaseModel):
    """任务模型"""
    id: str = Field(..., description="任务ID")
    title: str = Field(..., min_length=1, max_length=200, description="任务标题")
    description: Optional[str] = Field(None, max_length=1000, description="任务描述")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    priority: Priority = Field(default=Priority.MEDIUM, description="优先级")
    assignee: Optional[Person] = Field(None, description="负责人")
    due_date: Optional[date] = Field(None, description="截止日期")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class ProjectSummary(BaseModel):
    """项目摘要模型"""
    project_name: str = Field(..., description="项目名称")
    total_tasks: int = Field(..., ge=0, description="总任务数")
    completed_tasks: int = Field(..., ge=0, description="已完成任务数")
    completion_rate: float = Field(..., ge=0.0, le=1.0, description="完成率")
    team_members: List[Person] = Field(..., description="团队成员")
    active_tasks: List[Task] = Field(..., description="活跃任务")
    project_metadata: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict, description="项目元数据"
    )


class TestBasicStructuredOutput:
    """基础结构化输出测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.structured_output
    def test_simple_json_schema_output(self, mock_harborai_client):
        """测试简单JSON Schema输出"""
        # 定义简单的JSON Schema
        simple_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "is_student": {"type": "boolean"}
            },
            "required": ["name", "age", "is_student"]
        }
        
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps({
                    "name": "张三",
                    "age": 25,
                    "is_student": False
                }),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70
        )
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行结构化输出请求
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请按照指定的JSON Schema格式返回数据"},
                {"role": "user", "content": "生成一个人员信息"}
            ],
            response_format={
                "type": "json_object",
                "schema": simple_schema
            }
        )
        
        # 验证响应
        assert response is not None
        content = response.choices[0].message.content
        
        # 验证JSON格式
        parsed_data = json.loads(content)
        assert isinstance(parsed_data, dict)
        
        # 验证Schema约束
        assert "name" in parsed_data
        assert "age" in parsed_data
        assert "is_student" in parsed_data
        
        assert isinstance(parsed_data["name"], str)
        assert isinstance(parsed_data["age"], int)
        assert isinstance(parsed_data["is_student"], bool)
        assert parsed_data["age"] >= 0
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.structured_output
    def test_pydantic_model_output(self, mock_harborai_client):
        """测试Pydantic模型输出"""
        # 配置符合Person模型的mock响应
        person_data = {
            "name": "李四",
            "age": 30,
            "email": "lisi@example.com",
            "phone": "+86-13800138000",
            "address": {
                "street": "中关村大街1号",
                "city": "北京",
                "state": "北京市",
                "zip_code": "100080",
                "country": "China"
            },
            "is_active": True
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps(person_data),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行结构化输出请求
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请按照Person模型格式返回人员数据"},
                {"role": "user", "content": "生成一个完整的人员信息"}
            ],
            response_format={
                "type": "json_object",
                "schema": Person.model_json_schema()
            }
        )
        
        # 验证响应并解析为Pydantic模型
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        # 验证Pydantic模型
        person = Person(**parsed_data)
        assert person.name == "李四"
        assert person.age == 30
        assert person.email == "lisi@example.com"
        assert person.phone == "+86-13800138000"
        assert person.is_active is True
        
        # 验证嵌套地址模型
        assert person.address is not None
        assert person.address.street == "中关村大街1号"
        assert person.address.city == "北京"
        assert person.address.zip_code == "100080"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.structured_output
    def test_enum_validation(self, mock_harborai_client):
        """测试枚举值验证"""
        # 配置包含枚举的Task模型响应
        task_data = {
            "id": "task-001",
            "title": "完成项目文档",
            "description": "编写项目的技术文档和用户手册",
            "status": "in_progress",
            "priority": "high",
            "due_date": "2024-12-31",
            "created_at": "2024-01-15T10:30:00",
            "tags": ["文档", "技术", "用户手册"],
            "metadata": {
                "estimated_hours": 40,
                "complexity": "medium",
                "requires_review": True
            }
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps(task_data),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行结构化输出请求
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请按照Task模型格式返回任务数据"},
                {"role": "user", "content": "创建一个高优先级的进行中任务"}
            ],
            response_format={
                "type": "json_object",
                "schema": Task.model_json_schema()
            }
        )
        
        # 验证响应并解析为Task模型
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        # 验证Task模型和枚举值
        task = Task(**parsed_data)
        assert task.id == "task-001"
        assert task.title == "完成项目文档"
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.priority == Priority.HIGH
        
        # 验证列表和字典字段
        assert "文档" in task.tags
        assert "技术" in task.tags
        assert task.metadata["estimated_hours"] == 40
        assert task.metadata["requires_review"] is True


class TestComplexStructuredOutput:
    """复杂结构化输出测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.structured_output
    def test_nested_model_output(self, mock_harborai_client):
        """测试嵌套模型输出"""
        # 配置复杂嵌套的ProjectSummary响应
        project_data = {
            "project_name": "HarborAI测试项目",
            "total_tasks": 15,
            "completed_tasks": 8,
            "completion_rate": 0.533,
            "team_members": [
                {
                    "name": "张三",
                    "age": 28,
                    "email": "zhangsan@company.com",
                    "phone": "+86-13800138001",
                    "is_active": True
                },
                {
                    "name": "李四",
                    "age": 32,
                    "email": "lisi@company.com",
                    "phone": "+86-13800138002",
                    "is_active": True
                }
            ],
            "active_tasks": [
                {
                    "id": "task-001",
                    "title": "API设计",
                    "status": "in_progress",
                    "priority": "high",
                    "created_at": "2024-01-10T09:00:00",
                    "tags": ["API", "设计"],
                    "metadata": {}
                },
                {
                    "id": "task-002",
                    "title": "单元测试",
                    "status": "pending",
                    "priority": "medium",
                    "created_at": "2024-01-12T14:30:00",
                    "tags": ["测试", "质量保证"],
                    "metadata": {}
                }
            ],
            "project_metadata": {
                "start_date": "2024-01-01",
                "budget": 100000,
                "is_critical": True,
                "progress_percentage": 53.3
            }
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps(project_data),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行复杂结构化输出请求
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请按照ProjectSummary模型格式返回项目摘要数据"},
                {"role": "user", "content": "生成HarborAI测试项目的详细摘要"}
            ],
            response_format={
                "type": "json_object",
                "schema": ProjectSummary.model_json_schema()
            }
        )
        
        # 验证响应并解析为复杂模型
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        # 验证ProjectSummary模型
        project = ProjectSummary(**parsed_data)
        assert project.project_name == "HarborAI测试项目"
        assert project.total_tasks == 15
        assert project.completed_tasks == 8
        assert abs(project.completion_rate - 0.533) < 0.001
        
        # 验证嵌套的团队成员列表
        assert len(project.team_members) == 2
        assert project.team_members[0].name == "张三"
        assert project.team_members[1].name == "李四"
        
        # 验证嵌套的任务列表
        assert len(project.active_tasks) == 2
        assert project.active_tasks[0].title == "API设计"
        assert project.active_tasks[0].status == TaskStatus.IN_PROGRESS
        assert project.active_tasks[1].title == "单元测试"
        assert project.active_tasks[1].status == TaskStatus.PENDING
        
        # 验证项目元数据
        assert project.project_metadata["budget"] == 100000
        assert project.project_metadata["is_critical"] is True
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.structured_output
    def test_array_of_objects_output(self, mock_harborai_client):
        """测试对象数组输出"""
        # 配置对象数组响应
        tasks_data = [
            {
                "id": "task-001",
                "title": "前端开发",
                "status": "completed",
                "priority": "high",
                "created_at": "2024-01-01T10:00:00",
                "tags": ["前端", "React"],
                "metadata": {"hours_spent": 40}
            },
            {
                "id": "task-002",
                "title": "后端API",
                "status": "in_progress",
                "priority": "medium",
                "created_at": "2024-01-05T14:00:00",
                "tags": ["后端", "API"],
                "metadata": {"hours_spent": 25}
            },
            {
                "id": "task-003",
                "title": "数据库设计",
                "status": "pending",
                "priority": "low",
                "created_at": "2024-01-10T09:00:00",
                "tags": ["数据库", "设计"],
                "metadata": {"hours_spent": 0}
            }
        ]
        
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps(tasks_data),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行数组输出请求
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请返回Task对象的数组"},
                {"role": "user", "content": "生成3个不同状态的任务"}
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "array",
                    "items": Task.model_json_schema()
                }
            }
        )
        
        # 验证响应并解析为Task列表
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        # 验证数组结构
        assert isinstance(parsed_data, list)
        assert len(parsed_data) == 3
        
        # 验证每个Task对象
        tasks = [Task(**task_data) for task_data in parsed_data]
        
        assert tasks[0].title == "前端开发"
        assert tasks[0].status == TaskStatus.COMPLETED
        assert tasks[1].title == "后端API"
        assert tasks[1].status == TaskStatus.IN_PROGRESS
        assert tasks[2].title == "数据库设计"
        assert tasks[2].status == TaskStatus.PENDING
        
        # 验证不同优先级
        priorities = [task.priority for task in tasks]
        assert Priority.HIGH in priorities
        assert Priority.MEDIUM in priorities
        assert Priority.LOW in priorities


class TestStructuredOutputValidation:
    """结构化输出验证测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.structured_output
    @pytest.mark.error_handling
    def test_invalid_json_handling(self, mock_harborai_client):
        """测试无效JSON处理"""
        # 配置无效JSON响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="{invalid json content}",  # 无效JSON
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行请求
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请返回有效的JSON格式数据"},
                {"role": "user", "content": "生成人员信息"}
            ],
            response_format={"type": "json_object"}
        )
        
        # 验证JSON解析错误
        content = response.choices[0].message.content
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.structured_output
    @pytest.mark.error_handling
    def test_schema_validation_failure(self, mock_harborai_client):
        """测试Schema验证失败"""
        # 配置不符合Schema的响应
        invalid_person_data = {
            "name": "",  # 违反min_length约束
            "age": -5,   # 违反ge约束
            "email": "invalid-email",  # 违反pattern约束
            "phone": "123",  # 违反pattern约束
            "is_active": "yes"  # 类型错误
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps(invalid_person_data),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行请求
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请按照Person模型格式返回数据"},
                {"role": "user", "content": "生成人员信息"}
            ],
            response_format={
                "type": "json_object",
                "schema": Person.model_json_schema()
            }
        )
        
        # 验证Pydantic验证错误
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        with pytest.raises(ValidationError) as exc_info:
            Person(**parsed_data)
        
        # 验证具体的验证错误
        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        
        assert 'name' in error_fields  # 空字符串错误
        assert 'age' in error_fields   # 负数错误
        assert 'email' in error_fields # 格式错误
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.structured_output
    def test_optional_fields_handling(self, mock_harborai_client):
        """测试可选字段处理"""
        # 配置只包含必需字段的响应
        minimal_person_data = {
            "name": "王五",
            "age": 25,
            "email": "wangwu@example.com"
            # 省略所有可选字段
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps(minimal_person_data),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行请求
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请返回最小化的Person数据"},
                {"role": "user", "content": "只生成必需的人员字段"}
            ],
            response_format={
                "type": "json_object",
                "schema": Person.model_json_schema()
            }
        )
        
        # 验证最小化数据的解析
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        # 验证Person模型（应该成功，因为可选字段有默认值）
        person = Person(**parsed_data)
        assert person.name == "王五"
        assert person.age == 25
        assert person.email == "wangwu@example.com"
        assert person.phone is None  # 可选字段默认为None
        assert person.address is None  # 可选字段默认为None
        assert person.is_active is True  # 有默认值的字段
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.structured_output
    def test_default_values_handling(self, mock_harborai_client):
        """测试默认值处理"""
        # 配置部分字段缺失的Task数据
        partial_task_data = {
            "id": "task-default-test",
            "title": "测试默认值"
            # 省略status, priority, created_at等有默认值的字段
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps(partial_task_data),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行请求
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请返回部分Task数据"},
                {"role": "user", "content": "生成最简单的任务"}
            ],
            response_format={
                "type": "json_object",
                "schema": Task.model_json_schema()
            }
        )
        
        # 验证默认值的应用
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        # 验证Task模型和默认值
        task = Task(**parsed_data)
        assert task.id == "task-default-test"
        assert task.title == "测试默认值"
        assert task.status == TaskStatus.PENDING  # 默认值
        assert task.priority == Priority.MEDIUM   # 默认值
        assert task.tags == []  # 默认空列表
        assert task.metadata == {}  # 默认空字典
        assert isinstance(task.created_at, datetime)  # 默认工厂函数


class TestStructuredOutputPerformance:
    """结构化输出性能测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.structured_output
    @pytest.mark.performance
    def test_large_structured_output(self, mock_harborai_client):
        """测试大型结构化输出性能"""
        # 配置大型数据结构响应
        large_project_data = {
            "project_name": "大型项目测试",
            "total_tasks": 100,
            "completed_tasks": 60,
            "completion_rate": 0.6,
            "team_members": [
                {
                    "name": f"成员{i}",
                    "age": 25 + (i % 15),
                    "email": f"member{i}@company.com",
                    "is_active": i % 3 != 0
                }
                for i in range(20)  # 20个团队成员
            ],
            "active_tasks": [
                {
                    "id": f"task-{i:03d}",
                    "title": f"任务{i}",
                    "status": ["pending", "in_progress", "completed"][i % 3],
                    "priority": ["low", "medium", "high"][i % 3],
                    "created_at": "2024-01-01T10:00:00",
                    "tags": [f"tag{j}" for j in range(i % 5 + 1)],
                    "metadata": {f"key{j}": f"value{j}" for j in range(i % 3 + 1)}
                }
                for i in range(50)  # 50个活跃任务
            ],
            "project_metadata": {
                f"meta_key_{i}": f"meta_value_{i}" for i in range(20)
            }
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps(large_project_data),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行大型结构化输出请求
        import time
        start_time = time.time()
        
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请返回大型ProjectSummary数据"},
                {"role": "user", "content": "生成包含大量数据的项目摘要"}
            ],
            response_format={
                "type": "json_object",
                "schema": ProjectSummary.model_json_schema()
            }
        )
        
        # 验证和解析大型数据
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        # 验证Pydantic模型解析性能
        parse_start_time = time.time()
        project = ProjectSummary(**parsed_data)
        parse_end_time = time.time()
        
        end_time = time.time()
        
        # 验证大型数据结构
        assert project.project_name == "大型项目测试"
        assert len(project.team_members) == 20
        assert len(project.active_tasks) == 50
        assert len(project.project_metadata) == 20
        
        # 验证性能指标
        total_time = end_time - start_time
        parse_time = parse_end_time - parse_start_time
        
        # 解析时间应该合理（通常小于1秒）
        assert parse_time < 1.0, f"Parsing time too high: {parse_time}s"
        assert total_time < 2.0, f"Total time too high: {total_time}s"
        
        # 验证数据完整性
        assert all(member.name.startswith("成员") for member in project.team_members)
        assert all(task.title.startswith("任务") for task in project.active_tasks)
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.structured_output
    @pytest.mark.performance
    def test_schema_complexity_performance(self, mock_harborai_client):
        """测试复杂Schema性能"""
        # 定义复杂的嵌套Schema
        complex_schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "level3": {
                                        "type": "object",
                                        "properties": {
                                            "data": {"type": "string"},
                                            "nested_array": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "value": {"type": "number"},
                                                        "metadata": {
                                                            "type": "object",
                                                            "additionalProperties": True
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # 配置符合复杂Schema的响应
        complex_data = {
            "level1": {
                "level2": [
                    {
                        "level3": {
                            "data": f"data_{i}",
                            "nested_array": [
                                {
                                    "value": j * 1.5,
                                    "metadata": {f"key_{k}": f"value_{k}" for k in range(3)}
                                }
                                for j in range(5)
                            ]
                        }
                    }
                    for i in range(10)
                ]
            }
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=json.dumps(complex_data),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行复杂Schema请求
        import time
        start_time = time.time()
        
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请按照复杂Schema返回数据"},
                {"role": "user", "content": "生成复杂嵌套结构数据"}
            ],
            response_format={
                "type": "json_object",
                "schema": complex_schema
            }
        )
        
        # 验证复杂数据解析
        content = response.choices[0].message.content
        parsed_data = json.loads(content)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证复杂结构
        assert "level1" in parsed_data
        assert "level2" in parsed_data["level1"]
        assert len(parsed_data["level1"]["level2"]) == 10
        
        # 验证深层嵌套
        first_item = parsed_data["level1"]["level2"][0]
        assert "level3" in first_item
        assert "nested_array" in first_item["level3"]
        assert len(first_item["level3"]["nested_array"]) == 5
        
        # 验证性能
        assert processing_time < 1.0, f"Complex schema processing time too high: {processing_time}s"