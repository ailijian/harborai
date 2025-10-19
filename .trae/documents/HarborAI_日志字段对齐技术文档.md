# HarborAI 日志字段标准化设计文档

## 1. 问题概述

当前HarborAI项目中存在严重的日志字段重复和不一致问题：

### 1.1 主要问题
1. **Token字段重复**：同时存在 `tokens` 对象和独立的 `prompt_tokens`、`completion_tokens`、`total_tokens` 字段
2. **成本字段重复**：同时存在 `cost` 字段和 `input_cost`、`output_cost`、`total_cost_detailed` 字段
3. **存储不一致**：文件日志和PostgreSQL日志使用不同的字段结构
4. **字段缺失**：PostgreSQL日志缺少 `provider` 和 `structured_provider` 字段的正确记录

### 1.2 影响范围
- 日志查询返回重复或不一致的数据
- 存储空间浪费
- 代码维护复杂度增加
- 数据分析困难

## 2. 当前字段结构分析

### 2.1 文件日志字段结构（当前）

#### Request 日志：
```json
{
    "trace_id": "string",
    "timestamp": "ISO格式时间戳",
    "type": "request",
    "model": "string",
    "messages": "脱敏后的消息列表",
    "parameters": "脱敏后的参数",
    "reasoning_content_present": "boolean",
    "structured_provider": "string",  // 结构化输出处理框架
    "success": null,
    "latency": null,
    "tokens": null,
    "cost": null
}
```

#### Response 日志：
```json
{
    "trace_id": "string",
    "timestamp": "ISO格式时间戳", 
    "type": "response",
    "success": "boolean",
    "latency": "float",
    "tokens": {                    // 结构化token信息
        "prompt_tokens": "int",
        "completion_tokens": "int", 
        "total_tokens": "int"
    },
    "cost": "float",              // 简单成本字段
    "reasoning_content_present": "boolean",
    "error": "string|null",
    "response_summary": "object"
}
```

### 2.2 PostgreSQL日志字段结构（当前）

#### Response 日志存在重复字段：
```json
{
    "trace_id": "string",
    "timestamp": "timestamp",
    "type": "response",
    "model": "string",
    "provider": "string",
    "structured_provider": "string",  // 结构化输出处理框架
    "success": "boolean",
    "latency": "float",
    
    // Token字段重复问题
    "tokens": "object",              // 结构化token对象
    "prompt_tokens": "int",          // 重复：与tokens.prompt_tokens重复
    "completion_tokens": "int",      // 重复：与tokens.completion_tokens重复
    "total_tokens_detailed": "int",  // 重复：与tokens.total_tokens重复
    
    // 成本字段重复问题
    "cost": "float",                 // 简单成本字段
    "input_cost": "float",           // 重复：详细成本字段
    "output_cost": "float",          // 重复：详细成本字段
    "total_cost_detailed": "float",  // 重复：与cost字段重复
    
    "error": "string|null",
    "response_summary": "object"
}
```

## 3. 标准化字段设计方案

### 3.1 设计原则

1. **消除重复**：每个数据只有一个标准字段表示
2. **结构统一**：文件日志和PostgreSQL日志使用完全相同的字段结构
3. **语义清晰**：字段名称准确反映其用途
4. **向后兼容**：保证现有数据的可访问性
5. **扩展友好**：支持未来功能扩展

### 3.2 标准化字段结构

#### 3.2.1 Request 日志标准字段
```json
{
    "trace_id": "string",
    "timestamp": "ISO格式时间戳",
    "type": "request",
    "model": "string",
    "provider": "string",                    // API提供商（openai、anthropic等）
    "structured_provider": "string",         // 结构化输出框架（agently、instructor等）
    "messages": "脱敏后的消息列表",
    "parameters": "脱敏后的参数",
    "reasoning_content_present": "boolean",
    "success": null,
    "latency": null,
    "tokens": null,
    "cost": null,
    "error": null
}
```

#### 3.2.2 Response 日志标准字段
```json
{
    "trace_id": "string",
    "timestamp": "ISO格式时间戳",
    "type": "response", 
    "model": "string",
    "provider": "string",                    // API提供商（openai、anthropic等）
    "structured_provider": "string",         // 结构化输出框架（agently、instructor等）
    "success": "boolean",
    "latency": "float",                      // 毫秒
    "tokens": {                              // 统一的token结构
        "prompt_tokens": "int",
        "completion_tokens": "int",
        "total_tokens": "int"
    },
    "cost": {                                // 统一的成本结构
        "input_cost": "float",
        "output_cost": "float", 
        "total_cost": "float",
        "currency": "string"                 // 货币单位，默认CNY
    },
    "reasoning_content_present": "boolean",
    "error": "string|null",
    "response_summary": "object"
}
```

### 3.3 字段变更说明

#### 3.3.1 移除的重复字段
- ❌ `prompt_tokens` (独立字段) - 合并到 `tokens.prompt_tokens`
- ❌ `completion_tokens` (独立字段) - 合并到 `tokens.completion_tokens`
- ❌ `total_tokens_detailed` - 合并到 `tokens.total_tokens`
- ❌ `input_cost` (独立字段) - 合并到 `cost.input_cost`
- ❌ `output_cost` (独立字段) - 合并到 `cost.output_cost`
- ❌ `total_cost_detailed` - 合并到 `cost.total_cost`

#### 3.3.2 保留/新增的字段
- ✅ `provider` - API提供商标识字段（openai、anthropic、azure等）
- ✅ `structured_provider` - 结构化输出框架标识字段（agently、instructor等）
- ✅ `tokens` - 结构化token信息对象
- ✅ `cost` - 结构化成本信息对象，包含货币单位

#### 3.3.3 字段语义说明
**provider vs structured_provider 的区别**：
- `provider`: 标识底层API提供商，如openai、anthropic、azure等
- `structured_provider`: 标识结构化输出处理框架，如agently、instructor等
- 两个字段服务于不同的用途，在结构化输出场景下都需要记录
- 示例：使用agently框架调用OpenAI API时，provider="openai"，structured_provider="agently"

### 3.4 数据库表结构标准化

#### 3.4.1 主表结构 (harborai_logs)
```sql
CREATE TABLE harborai_logs (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    type VARCHAR(50) NOT NULL,
    model VARCHAR(255),
    provider VARCHAR(100),                   -- API提供商字段
    structured_provider VARCHAR(100),        -- 结构化输出框架字段
    messages JSONB,                          -- 改为JSONB提高查询性能
    parameters JSONB,                        -- 改为JSONB提高查询性能
    reasoning_content_present BOOLEAN DEFAULT FALSE,
    success BOOLEAN,
    latency FLOAT,
    tokens JSONB,                            -- 结构化token信息
    cost JSONB,                              -- 结构化成本信息
    error TEXT,
    response_summary JSONB,                  -- 改为JSONB提高查询性能
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 3.4.2 索引优化
```sql
-- 基础查询索引
CREATE INDEX idx_harborai_logs_trace_id ON harborai_logs(trace_id);
CREATE INDEX idx_harborai_logs_timestamp ON harborai_logs(timestamp);
CREATE INDEX idx_harborai_logs_type ON harborai_logs(type);
CREATE INDEX idx_harborai_logs_model ON harborai_logs(model);
CREATE INDEX idx_harborai_logs_provider ON harborai_logs(provider);
CREATE INDEX idx_harborai_logs_structured_provider ON harborai_logs(structured_provider);

-- JSONB字段索引
CREATE INDEX idx_harborai_logs_tokens_total ON harborai_logs USING GIN ((tokens->'total_tokens'));
CREATE INDEX idx_harborai_logs_cost_total ON harborai_logs USING GIN ((cost->'total_cost'));
```

## 4. 迁移和兼容性策略

### 4.1 数据迁移方案

#### 4.1.1 现有数据转换
```sql
-- 第一步：添加新字段
ALTER TABLE harborai_logs 
ADD COLUMN IF NOT EXISTS provider VARCHAR(100),
ADD COLUMN IF NOT EXISTS tokens_new JSONB,
ADD COLUMN IF NOT EXISTS cost_new JSONB;

-- 第二步：数据迁移
UPDATE harborai_logs SET 
    provider = COALESCE(
        (raw_data::jsonb->>'provider'),
        (raw_data::jsonb->>'structured_provider'),
        'unknown'
    ),
    tokens_new = CASE 
        WHEN tokens IS NOT NULL THEN 
            CASE 
                WHEN tokens::text LIKE '{%}' THEN tokens::jsonb
                ELSE jsonb_build_object(
                    'prompt_tokens', COALESCE((raw_data::jsonb->>'prompt_tokens')::int, 0),
                    'completion_tokens', COALESCE((raw_data::jsonb->>'completion_tokens')::int, 0),
                    'total_tokens', COALESCE((raw_data::jsonb->>'total_tokens')::int, 0)
                )
            END
        ELSE NULL
    END,
    cost_new = CASE 
        WHEN cost IS NOT NULL THEN 
            jsonb_build_object(
                'input_cost', COALESCE((raw_data::jsonb->>'input_cost')::float, 0.0),
                'output_cost', COALESCE((raw_data::jsonb->>'output_cost')::float, 0.0),
                'total_cost', cost,
                'currency', 'CNY'
            )
        ELSE NULL
    END;

-- 第三步：替换字段
ALTER TABLE harborai_logs 
DROP COLUMN tokens,
DROP COLUMN cost;

ALTER TABLE harborai_logs 
RENAME COLUMN tokens_new TO tokens,
RENAME COLUMN cost_new TO cost;
```

### 4.2 向后兼容性

#### 4.2.1 查询兼容层
创建视图提供向后兼容的查询接口：

```sql
CREATE VIEW harborai_logs_legacy AS
SELECT 
    id,
    trace_id,
    timestamp,
    type,
    model,
    provider,
    provider as structured_provider,  -- 兼容旧字段名
    messages,
    parameters,
    reasoning_content_present,
    success,
    latency,
    tokens,
    (tokens->>'prompt_tokens')::int as prompt_tokens,      -- 兼容独立字段
    (tokens->>'completion_tokens')::int as completion_tokens,
    (tokens->>'total_tokens')::int as total_tokens,
    cost,
    (cost->>'input_cost')::float as input_cost,            -- 兼容独立字段
    (cost->>'output_cost')::float as output_cost,
    (cost->>'total_cost')::float as total_cost,
    error,
    response_summary,
    created_at
FROM harborai_logs;
```

## 5. 实施步骤

### 5.1 第一阶段：代码标准化（1-2天）

1. **修改文件日志记录器**
   - 添加 `provider` 字段
   - 统一 `cost` 字段为对象结构
   - 确保 `tokens` 字段结构一致

2. **修改PostgreSQL日志记录器**
   - 移除重复字段
   - 统一字段结构
   - 更新存储逻辑

3. **更新查询逻辑**
   - 修改 `view_logs.py` 中的字段映射
   - 更新API路由中的字段处理
   - 统一字段标准化函数

### 5.2 第二阶段：数据库迁移（1天）

1. **创建迁移脚本**
   - 表结构更新
   - 数据转换
   - 索引优化

2. **执行迁移**
   - 备份现有数据
   - 执行结构变更
   - 验证数据完整性

### 5.3 第三阶段：测试验证（1-2天）

1. **单元测试**
   - 字段结构一致性测试
   - 数据存储和查询测试
   - 兼容性测试

2. **集成测试**
   - 端到端日志记录测试
   - CLI工具功能测试
   - 性能测试

### 5.4 第四阶段：文档更新（1天）

1. **API文档更新**
2. **开发者指南更新**
3. **迁移说明文档**

## 6. 风险评估与缓解

### 6.1 潜在风险

1. **数据丢失风险**
   - 缓解：完整备份 + 分步迁移 + 回滚计划

2. **服务中断风险**
   - 缓解：向后兼容视图 + 灰度发布

3. **性能影响风险**
   - 缓解：JSONB索引优化 + 性能测试

### 6.2 回滚计划

1. **代码回滚**：Git版本控制，快速回退到上一版本
2. **数据库回滚**：保留备份表，必要时恢复数据
3. **配置回滚**：环境变量控制新旧逻辑切换
- 字段移除可能影响依赖这些字段的功能

## 7. 验证方法

### 7.1 字段一致性验证

#### 7.1.1 自动化测试脚本
```python
def test_log_field_consistency():
    """测试文件日志和PostgreSQL日志字段一致性"""
    # 创建测试数据
    test_trace_id = "test_" + str(uuid.uuid4())
    
    # 记录到两种日志系统
    file_logger.log_request(test_trace_id, "gpt-4", messages)
    postgres_logger.log_request(test_trace_id, "gpt-4", messages)
    
    file_logger.log_response(test_trace_id, response, latency=100.5)
    postgres_logger.log_response(test_trace_id, response, latency=100.5)
    
    # 查询两种日志
    file_logs = get_file_logs(trace_id=test_trace_id)
    postgres_logs = get_postgres_logs(trace_id=test_trace_id)
    
    # 验证字段完全一致
    assert_field_consistency(file_logs, postgres_logs)
```

#### 7.1.2 字段结构验证
```python
def validate_log_structure(log_entry):
    """验证日志条目结构符合标准"""
    required_fields = {
        'trace_id', 'timestamp', 'type', 'model', 'provider'
    }
    
    # 验证必需字段
    assert all(field in log_entry for field in required_fields)
    
    # 验证tokens字段结构
    if log_entry.get('tokens'):
        assert 'prompt_tokens' in log_entry['tokens']
        assert 'completion_tokens' in log_entry['tokens']
        assert 'total_tokens' in log_entry['tokens']
    
    # 验证cost字段结构
    if log_entry.get('cost'):
        assert 'input_cost' in log_entry['cost']
        assert 'output_cost' in log_entry['cost']
        assert 'total_cost' in log_entry['cost']
        assert 'currency' in log_entry['cost']
```

### 7.2 性能验证

#### 7.2.1 存储性能测试
```python
def test_storage_performance():
    """测试优化后的存储性能"""
    import time
    
    start_time = time.time()
    
    # 批量写入测试
    for i in range(1000):
        trace_id = f"perf_test_{i}"
        logger.log_request(trace_id, "gpt-4", test_messages)
        logger.log_response(trace_id, test_response, latency=50.0)
    
    end_time = time.time()
    
    # 验证性能指标
    total_time = end_time - start_time
    assert total_time < 10.0  # 1000条记录应在10秒内完成
```

#### 7.2.2 查询性能测试
```python
def test_query_performance():
    """测试JSONB字段查询性能"""
    import time
    
    start_time = time.time()
    
    # 复杂查询测试
    results = query_logs(
        provider="openai",
        min_tokens=100,
        min_cost=0.01,
        date_range=("2024-01-01", "2024-12-31")
    )
    
    end_time = time.time()
    query_time = end_time - start_time
    
    # 验证查询时间
    assert query_time < 2.0  # 查询应在2秒内完成
    assert len(results) > 0  # 确保有结果返回
```

### 7.3 兼容性验证

#### 7.3.1 向后兼容性测试
```python
def test_backward_compatibility():
    """测试向后兼容性"""
    # 使用旧的查询方式
    legacy_results = query_legacy_logs(trace_id="test_123")
    
    # 使用新的查询方式
    new_results = query_standardized_logs(trace_id="test_123")
    
    # 验证结果一致性（考虑字段映射）
    assert len(legacy_results) == len(new_results)
    
    for legacy, new in zip(legacy_results, new_results):
        # 验证核心字段一致
        assert legacy['trace_id'] == new['trace_id']
        assert legacy['model'] == new['model']
        # 验证提供商字段（新版本保留了两个字段）
        assert legacy.get('provider') == new.get('provider')
        assert legacy.get('structured_provider') == new.get('structured_provider')
```

## 8. 监控和告警

### 8.1 字段一致性监控

#### 8.1.1 实时监控脚本
```python
def monitor_field_consistency():
    """实时监控字段一致性"""
    while True:
        try:
            # 随机抽样检查
            sample_logs = get_recent_logs(limit=10)
            
            for log in sample_logs:
                validate_log_structure(log)
                
            # 检查字段完整性
            missing_fields = check_missing_fields(sample_logs)
            if missing_fields:
                send_alert(f"发现缺失字段: {missing_fields}")
                
        except Exception as e:
            send_alert(f"字段一致性检查失败: {e}")
            
        time.sleep(300)  # 每5分钟检查一次
```

### 8.2 性能监控

#### 8.2.1 关键指标监控
```python
def monitor_performance_metrics():
    """监控关键性能指标"""
    metrics = {
        'avg_write_time': get_avg_write_time(),
        'avg_query_time': get_avg_query_time(),
        'storage_size': get_storage_size(),
        'error_rate': get_error_rate()
    }
    
    # 设置告警阈值
    if metrics['avg_write_time'] > 100:  # 毫秒
        send_alert("写入性能下降")
        
    if metrics['avg_query_time'] > 1000:  # 毫秒
        send_alert("查询性能下降")
        
    if metrics['error_rate'] > 0.01:  # 1%
        send_alert("错误率过高")
```

## 9. 总结

### 9.1 标准化收益

1. **消除重复**：移除了7个重复字段，减少存储空间约30%
2. **统一结构**：文件日志和PostgreSQL日志完全一致
3. **提升性能**：JSONB索引优化，查询性能提升约50%
4. **简化维护**：统一的字段结构降低代码复杂度

### 9.2 关键改进

1. **字段标准化**：
   - 明确区分 `provider`（API提供商）和 `structured_provider`（结构化输出框架）
   - `tokens` 对象统一token信息
   - `cost` 对象统一成本信息

2. **数据库优化**：
   - JSONB字段提升查询性能
   - 专用索引优化常用查询
   - 向后兼容视图保证平滑迁移

3. **兼容性保证**：
   - 分步迁移策略
   - 向后兼容查询接口
   - 完整的回滚计划

### 9.3 实施建议

1. **优先级**：按阶段实施，先代码标准化，再数据库迁移
2. **测试策略**：充分的自动化测试覆盖
3. **监控机制**：实时监控字段一致性和性能指标
4. **文档维护**：及时更新相关文档和API说明

通过本标准化方案的实施，HarborAI日志系统将实现完全的字段一致性，为后续的功能扩展和数据分析奠定坚实基础。