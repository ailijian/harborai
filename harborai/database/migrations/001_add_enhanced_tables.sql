-- HarborAI 日志系统重构 - 数据库Schema增强
-- 添加成本细分表和追踪信息表
-- 根据重构设计方案实现

-- 设置时区
SET timezone = 'UTC';

-- 创建UUID扩展（如果不存在）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==================== Token使用量表重构 ====================

-- 添加新字段到现有token_usage表（如果存在）
-- 注意：这里假设可能存在token_usage表，如果不存在则跳过
DO $$ 
BEGIN
    -- 检查token_usage表是否存在，如果不存在则创建
    IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'token_usage') THEN
        CREATE TABLE token_usage (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            log_id INTEGER NOT NULL, -- 关联到api_logs表
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            parsing_method VARCHAR(50) DEFAULT 'direct_extraction',
            confidence FLOAT DEFAULT 1.0,
            raw_usage_data JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    ELSE
        -- 如果表存在，添加新字段
        ALTER TABLE token_usage ADD COLUMN IF NOT EXISTS parsing_method VARCHAR(50) DEFAULT 'direct_extraction';
        ALTER TABLE token_usage ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 1.0;
        ALTER TABLE token_usage ADD COLUMN IF NOT EXISTS raw_usage_data JSONB DEFAULT '{}';
        ALTER TABLE token_usage ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
    END IF;
    
    -- 添加约束（无论表是新创建还是已存在）
    ALTER TABLE token_usage DROP CONSTRAINT IF EXISTS token_usage_consistency;
    ALTER TABLE token_usage ADD CONSTRAINT token_usage_consistency 
        CHECK (total_tokens = prompt_tokens + completion_tokens);

    ALTER TABLE token_usage DROP CONSTRAINT IF EXISTS token_usage_confidence_range;
    ALTER TABLE token_usage ADD CONSTRAINT token_usage_confidence_range 
        CHECK (confidence >= 0.0 AND confidence <= 1.0);

    ALTER TABLE token_usage DROP CONSTRAINT IF EXISTS token_usage_positive_tokens;
    ALTER TABLE token_usage ADD CONSTRAINT token_usage_positive_tokens 
        CHECK (prompt_tokens >= 0 AND completion_tokens >= 0 AND total_tokens >= 0);
END $$;

-- ==================== 成本细分表 ====================

-- 创建成本信息表
CREATE TABLE IF NOT EXISTS cost_info (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    log_id INTEGER NOT NULL, -- 关联到api_logs表
    input_cost DECIMAL(10, 6) NOT NULL DEFAULT 0.0,
    output_cost DECIMAL(10, 6) NOT NULL DEFAULT 0.0,
    total_cost DECIMAL(10, 6) NOT NULL DEFAULT 0.0,
    currency VARCHAR(10) DEFAULT 'CNY',
    pricing_source VARCHAR(50) DEFAULT 'environment_variable',
    pricing_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    pricing_details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 约束
    CONSTRAINT cost_info_positive_costs CHECK (
        input_cost >= 0 AND output_cost >= 0 AND total_cost >= 0
    ),
    CONSTRAINT cost_info_consistency CHECK (
        total_cost = input_cost + output_cost
    ),
    CONSTRAINT cost_info_valid_currency CHECK (
        currency IN ('CNY', 'USD', 'EUR', 'JPY', 'GBP')
    ),
    CONSTRAINT cost_info_valid_pricing_source CHECK (
        pricing_source IN ('builtin', 'environment_variable', 'dynamic', 'not_found', 'error')
    )
);

-- ==================== 分布式追踪信息表 ====================

-- 创建追踪信息表
CREATE TABLE IF NOT EXISTS tracing_info (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    log_id INTEGER NOT NULL, -- 关联到api_logs表
    
    -- 双Trace ID策略
    hb_trace_id VARCHAR(100) NOT NULL,
    otel_trace_id VARCHAR(32) NOT NULL,
    span_id VARCHAR(16) NOT NULL,
    parent_span_id VARCHAR(16),
    
    operation_name VARCHAR(100) DEFAULT 'ai.chat.completion',
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    duration_ms INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'ok',
    
    -- OpenTelemetry标准字段
    trace_flags VARCHAR(2) DEFAULT '01',
    trace_state TEXT DEFAULT '',
    
    -- API响应标签（精简版）
    api_tags JSONB DEFAULT '{}',
    
    -- 内部追踪标签（完整版）
    internal_tags JSONB DEFAULT '{}',
    
    logs JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 约束
    CONSTRAINT tracing_info_valid_status CHECK (
        status IN ('ok', 'error', 'timeout', 'cancelled')
    ),
    CONSTRAINT tracing_info_positive_duration CHECK (
        duration_ms >= 0
    )
);

-- ==================== 索引创建 ====================

-- Token使用量表索引
CREATE INDEX IF NOT EXISTS idx_token_usage_log_id ON token_usage(log_id);
CREATE INDEX IF NOT EXISTS idx_token_usage_parsing_method ON token_usage(parsing_method);
CREATE INDEX IF NOT EXISTS idx_token_usage_confidence ON token_usage(confidence);
CREATE INDEX IF NOT EXISTS idx_token_usage_created_at ON token_usage(created_at);

-- 成本信息表索引
CREATE INDEX IF NOT EXISTS idx_cost_info_log_id ON cost_info(log_id);
CREATE INDEX IF NOT EXISTS idx_cost_info_currency ON cost_info(currency);
CREATE INDEX IF NOT EXISTS idx_cost_info_pricing_source ON cost_info(pricing_source);
CREATE INDEX IF NOT EXISTS idx_cost_info_total_cost ON cost_info(total_cost);
CREATE INDEX IF NOT EXISTS idx_cost_info_created_at ON cost_info(created_at);

-- 追踪信息表索引
CREATE INDEX IF NOT EXISTS idx_tracing_info_log_id ON tracing_info(log_id);
CREATE INDEX IF NOT EXISTS idx_tracing_info_hb_trace_id ON tracing_info(hb_trace_id);
CREATE INDEX IF NOT EXISTS idx_tracing_info_otel_trace_id ON tracing_info(otel_trace_id);
CREATE INDEX IF NOT EXISTS idx_tracing_info_span_id ON tracing_info(span_id);
CREATE INDEX IF NOT EXISTS idx_tracing_info_operation_name ON tracing_info(operation_name);
CREATE INDEX IF NOT EXISTS idx_tracing_info_start_time ON tracing_info(start_time);
CREATE INDEX IF NOT EXISTS idx_tracing_info_status ON tracing_info(status);

-- ==================== 视图创建 ====================

-- 创建日志摘要视图
CREATE OR REPLACE VIEW log_summary AS
SELECT 
    al.id as log_id,
    al.timestamp,
    al.provider,
    al.model,
    al.status_code,
    al.duration_ms,
    tu.prompt_tokens,
    tu.completion_tokens,
    tu.total_tokens,
    tu.parsing_method,
    tu.confidence as token_confidence,
    ci.input_cost,
    ci.output_cost,
    ci.total_cost,
    ci.currency,
    ci.pricing_source,
    ti.hb_trace_id,
    ti.otel_trace_id,
    ti.operation_name,
    ti.status as trace_status
FROM api_logs al
LEFT JOIN token_usage tu ON al.id = tu.log_id
LEFT JOIN cost_info ci ON al.id = ci.log_id
LEFT JOIN tracing_info ti ON al.id = ti.log_id;

-- 创建每日统计视图
CREATE OR REPLACE VIEW daily_stats AS
SELECT 
    DATE(al.timestamp) as date,
    al.provider,
    al.model,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN al.status_code = 200 THEN 1 END) as success_requests,
    COUNT(CASE WHEN al.status_code != 200 THEN 1 END) as error_requests,
    AVG(al.duration_ms) as avg_duration_ms,
    SUM(COALESCE(tu.prompt_tokens, 0)) as total_prompt_tokens,
    SUM(COALESCE(tu.completion_tokens, 0)) as total_completion_tokens,
    SUM(COALESCE(tu.total_tokens, 0)) as total_tokens,
    SUM(COALESCE(ci.total_cost, 0)) as total_cost,
    AVG(COALESCE(tu.confidence, 1.0)) as avg_token_confidence
FROM api_logs al
LEFT JOIN token_usage tu ON al.id = tu.log_id
LEFT JOIN cost_info ci ON al.id = ci.log_id
GROUP BY DATE(al.timestamp), al.provider, al.model
ORDER BY date DESC, total_requests DESC;

-- ==================== 函数创建 ====================

-- 创建根据任意trace_id查询日志的函数
CREATE OR REPLACE FUNCTION get_log_by_any_trace_id(trace_id_param TEXT)
RETURNS TABLE (
    log_id INTEGER,
    log_timestamp TIMESTAMP WITH TIME ZONE,
    provider VARCHAR(100),
    model VARCHAR(100),
    request_data TEXT,
    response_data TEXT,
    status_code INTEGER,
    error_message TEXT,
    duration_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    parsing_method VARCHAR(50),
    token_confidence FLOAT,
    input_cost DECIMAL(10, 6),
    output_cost DECIMAL(10, 6),
    total_cost DECIMAL(10, 6),
    currency VARCHAR(10),
    pricing_source VARCHAR(50),
    hb_trace_id VARCHAR(100),
    otel_trace_id VARCHAR(32),
    span_id VARCHAR(16),
    operation_name VARCHAR(100),
    trace_status VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ls.log_id,
        ls.timestamp as log_timestamp,
        ls.provider,
        ls.model,
        al.request_data,
        al.response_data,
        ls.status_code,
        al.error_message,
        ls.duration_ms,
        ls.prompt_tokens,
        ls.completion_tokens,
        ls.total_tokens,
        ls.parsing_method,
        ls.token_confidence,
        ls.input_cost,
        ls.output_cost,
        ls.total_cost,
        ls.currency,
        ls.pricing_source,
        ls.hb_trace_id,
        ls.otel_trace_id,
        ti.span_id,
        ls.operation_name,
        ls.trace_status
    FROM log_summary ls
    JOIN api_logs al ON ls.log_id = al.id
    LEFT JOIN tracing_info ti ON ls.log_id = ti.log_id
    WHERE ls.hb_trace_id = trace_id_param 
       OR ls.otel_trace_id = trace_id_param;
END;
$$ LANGUAGE plpgsql;

-- ==================== 触发器创建 ====================

-- 创建更新时间戳的触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为各表添加更新时间戳触发器
DROP TRIGGER IF EXISTS update_token_usage_updated_at ON token_usage;
CREATE TRIGGER update_token_usage_updated_at
    BEFORE UPDATE ON token_usage
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_cost_info_updated_at ON cost_info;
CREATE TRIGGER update_cost_info_updated_at
    BEFORE UPDATE ON cost_info
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_tracing_info_updated_at ON tracing_info;
CREATE TRIGGER update_tracing_info_updated_at
    BEFORE UPDATE ON tracing_info
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ==================== 初始数据插入 ====================

-- 插入默认模型价格（如果cost_info表为空）
DO $$
BEGIN
    -- 这里可以插入一些默认的价格配置示例
    -- 实际使用中，价格将通过环境变量或动态配置设置
    NULL;
END $$;

-- ==================== 权限设置 ====================

-- 设置表权限（假设存在harborai用户）
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'harborai') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO harborai;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO harborai;
        GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO harborai;
    END IF;
END $$;

-- 输出完成信息
SELECT 'HarborAI 数据库Schema增强完成!' as message;
SELECT 'Created tables: token_usage, cost_info, tracing_info' as info;
SELECT 'Created views: log_summary, daily_stats' as info;
SELECT 'Created function: get_log_by_any_trace_id' as info;
SELECT 'Created indexes and triggers' as info;