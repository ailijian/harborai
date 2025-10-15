-- HarborAI PostgreSQL Database Initialization Script
-- Create necessary tables and indexes

-- Set timezone
SET timezone = 'UTC';

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create API logs table (main logging table)
CREATE TABLE IF NOT EXISTS api_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    provider VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    request_data TEXT,
    response_data TEXT,
    status_code INTEGER,
    error_message TEXT,
    duration_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create trace logs table
CREATE TABLE IF NOT EXISTS trace_logs (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(100) NOT NULL,
    span_id VARCHAR(100) NOT NULL,
    parent_span_id VARCHAR(100),
    operation_name VARCHAR(255) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_ms REAL,
    tags TEXT,
    logs TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model usage statistics table
CREATE TABLE IF NOT EXISTS model_usage (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    provider VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, provider, model)
);

-- Create HarborAI logs table (for postgres_logger.py)
CREATE TABLE IF NOT EXISTS harborai_logs (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    type VARCHAR(50) NOT NULL,
    model VARCHAR(255),
    messages TEXT,
    parameters TEXT,
    reasoning_content_present BOOLEAN,
    structured_provider VARCHAR(50),
    success BOOLEAN,
    latency FLOAT,
    tokens TEXT,
    cost FLOAT,
    error TEXT,
    response_summary TEXT,
    raw_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance

-- API logs table indexes
CREATE INDEX IF NOT EXISTS idx_api_logs_timestamp ON api_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_logs_provider_model ON api_logs(provider, model);
CREATE INDEX IF NOT EXISTS idx_api_logs_status_code ON api_logs(status_code);

-- Trace logs table indexes
CREATE INDEX IF NOT EXISTS idx_trace_logs_trace_id ON trace_logs(trace_id);
CREATE INDEX IF NOT EXISTS idx_trace_logs_start_time ON trace_logs(start_time);
CREATE INDEX IF NOT EXISTS idx_trace_logs_operation ON trace_logs(operation_name);

-- Model usage table indexes
CREATE INDEX IF NOT EXISTS idx_model_usage_date ON model_usage(date);
CREATE INDEX IF NOT EXISTS idx_model_usage_provider_model ON model_usage(provider, model);

-- HarborAI logs table indexes
CREATE INDEX IF NOT EXISTS idx_harborai_logs_trace_id ON harborai_logs(trace_id);
CREATE INDEX IF NOT EXISTS idx_harborai_logs_timestamp ON harborai_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_harborai_logs_type ON harborai_logs(type);
CREATE INDEX IF NOT EXISTS idx_harborai_logs_model ON harborai_logs(model);
CREATE INDEX IF NOT EXISTS idx_harborai_logs_provider ON harborai_logs(structured_provider);

-- Create a view for recent API activity
CREATE OR REPLACE VIEW recent_api_activity AS
SELECT 
    provider,
    model,
    COUNT(*) as total_requests,
    AVG(duration_ms) as avg_duration,
    SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) as success_count,
    SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) as error_count
FROM api_logs
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY provider, model
ORDER BY total_requests DESC;

-- Insert initial test data (optional)
INSERT INTO api_logs (provider, model, status_code, duration_ms, request_data, response_data)
VALUES 
    ('openai', 'gpt-4', 200, 1200.5, '{"messages": [{"role": "user", "content": "Hello"}]}', '{"content": "Hi there!"}'),
    ('deepseek', 'deepseek-chat', 200, 800.3, '{"messages": [{"role": "user", "content": "Test"}]}', '{"content": "Test response"}')
ON CONFLICT DO NOTHING;

-- Set table permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO harborai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO harborai;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO harborai;

-- Output initialization completion message
SELECT 'HarborAI PostgreSQL database initialization completed!' as message;
SELECT 'Created tables: api_logs, trace_logs, model_usage, harborai_logs' as info;
SELECT 'Created indexes and views' as info;
SELECT 'Set user permissions' as info;