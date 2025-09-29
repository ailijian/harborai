以下为基于 e:\project\harborai\.trae\documents\HarborAI_PRD.md 与 e:\project\harborai\.trae\documents\HarborAI_TDD.md 制定的 HarborAI 功能测试清单（Markdown）。本清单按模块分类，覆盖 PRD/TDD 中列出的全部功能点（含基础与高级功能），并为每个测试项提供：测试ID、功能描述、前置条件、测试步骤、预期结果与优先级。关键场景以「[关键]」标注。为便于追溯实现，关键模块对应的源码文件以引用形式标注。

范围声明与覆盖目标
- 覆盖范围：PRD 4.1/4.2 核心与高级功能、TDD 架构与接口设计、推理模型支持、结构化输出、可观测性、成本统计、异常/重试/降级、持久化与生命周期（含规划项）。
- 覆盖比例：100%（包含已实现与规划中项；规划项以较低优先级标注，并注明待实现）。
- 运行环境：Windows 11，PowerShell，Python 3.9+。
- 参考实现入口
  - <mcfile name="client.py" path="e:\project\harborai\harborai\api\client.py"></mcfile>
  - <mcfile name="decorators.py" path="e:\project\harborai\harborai\api\decorators.py"></mcfile>
  - <mcfile name="structured.py" path="e:\project\harborai\harborai\api\structured.py"></mcfile>
  - <mcfile name="client_manager.py" path="e:\project\harborai\harborai\core\client_manager.py"></mcfile>
  - <mcfile name="base_plugin.py" path="e:\project\harborai\harborai\core\base_plugin.py"></mcfile>
  - <mcfile name="openai_plugin.py" path="e:\project\harborai\harborai\core\plugins\openai_plugin.py"></mcfile>
  - <mcfile name="deepseek_plugin.py" path="e:\project\harborai\harborai\core\plugins\deepseek_plugin.py"></mcfile>
  - <mcfile name="doubao_plugin.py" path="e:\project\harborai\harborai\core\plugins\doubao_plugin.py"></mcfile>
  - <mcfile name="wenxin_plugin.py" path="e:\project\harborai\harborai\core\plugins\wenxin_plugin.py"></mcfile>
  - <mcfile name="logger.py" path="e:\project\harborai\harborai\utils\logger.py"></mcfile>
  - <mcfile name="retry.py" path="e:\project\harborai\harborai\utils\retry.py"></mcfile>
  - <mcfile name="exceptions.py" path="e:\project\harborai\harborai\utils\exceptions.py"></mcfile>
  - <mcfile name="tracer.py" path="e:\project\harborai\harborai\utils\tracer.py"></mcfile>
  - <mcfile name="postgres_logger.py" path="e:\project\harborai\harborai\storage\postgres_logger.py"></mcfile>
  - <mcfile name="lifecycle.py" path="e:\project\harborai\harborai\storage\lifecycle.py"></mcfile>
  - <mcfile name="settings.py" path="e:\project\harborai\harborai\config\settings.py"></mcfile>
  - <mcfile name="main.py" path="e:\project\harborai\harborai\cli\main.py"></mcfile>

优先级定义
- P0：核心能力，不可或缺，影响基础可用性。[关键] 多标注于此
- P1：高价值能力，强烈建议上线周期内完成
- P2：增强能力，可延后
- P3：规划/长期能力，当前可能未实现

通用前置条件
- 已安装依赖并可在 Windows PowerShell 下运行。
- 环境变量或配置正确设置（如 OPENAI_API_KEY、OPENAI_BASE_URL 等）。
- 如涉及持久化与生命周期测试，需本地或容器化 PostgreSQL 可用。

模块A：统一 API 与兼容性（OpenAI 风格）
参考：<mcfile name="client.py" path="e:\project\harborai\harborai\api\client.py"></mcfile> <mcfile name="client_manager.py" path="e:\project\harborai\harborai\core\client_manager.py"></mcfile>

- A-001 [关键][P0]
  - 描述：HarborAI 构造函数与 OpenAI 一致（api_key, base_url）
  - 前置：设置 OPENAI_API_KEY/OPENAI_BASE_URL 或在构造时显式传入
  - 步骤：实例化 HarborAI(api_key, base_url)
  - 预期：实例创建成功，无异常；内部路由与配置加载正确

- A-002 [关键][P0]
  - 描述：统一入口 client.chat.completions.create 与 OpenAI 对齐
  - 前置：见 A-001
  - 步骤：调用 create(model, messages=[{role,user}]) 非流式
  - 预期：返回对象包含 choices/usage/model/id 等字段，字段语义对齐 OpenAI

- A-003 [P1]
  - 描述：参数透传与扩展参数兼容（response_format/structured_provider/extra_body/retry_policy/fallback/trace_id/cost_tracking）
  - 前置：见 A-001
  - 步骤：分别设置上述参数并调用
  - 预期：参数不冲突；扩展参数被识别并生效；无破坏 OpenAI 兼容性

模块B：同步、异步与流式调用
参考：<mcfile name="client.py" path="e:\project\harborai\harborai\api\client.py"></mcfile> <mcfile name="decorators.py" path="e:\project\harborai\harborai\api\decorators.py"></mcfile>

- B-001 [关键][P0]
  - 描述：同步非流式调用成功返回
  - 前置：A-001
  - 步骤：create(stream=False) 默认
  - 预期：一次性返回 ChatCompletion

- B-002 [关键][P0]
  - 描述：同步流式调用返回迭代器
  - 前置：A-001
  - 步骤：for chunk in create(stream=True)
  - 预期：获得 ChatCompletionChunk 序列，delta 字段语义对齐

- B-003 [P1]
  - 描述：异步非流式调用（await）
  - 前置：支持 asyncio 环境
  - 步骤：await create(...)
  - 预期：返回对象结构同 B-001

- B-004 [P1]
  - 描述：异步流式调用（async for）
  - 前置：同 B-003
  - 步骤：async for chunk in create(stream=True)
  - 预期：chunk 结构同 B-002

模块C：结构化输出（Agently 默认 + 原生 Native）
参考：<mcfile name="structured.py" path="e:\project\harborai\harborai\api\structured.py"></mcfile>

- C-001 [关键][P0]
  - 描述：response_format=JSON Schema（Agently 默认）非流式解析
  - 前置：A-001
  - 步骤：传入 response_format={"type":"json_schema", ...}；不传 structured_provider
  - 预期：resp.parsed 返回符合 Schema 的对象；content 字段可为 JSON 文本或空

- C-002 [关键][P1]
  - 描述：structured_provider="native" 非流式解析
  - 前置：同 C-001
  - 步骤：传入 structured_provider="native"
  - 预期：使用厂商原生 schema 解析，resp.parsed 合规

- C-003 [P1]
  - 描述：Agently 流式结构化输出
  - 前置：C-001
  - 步骤：stream=True，逐块收集结构化片段并组装
  - 预期：最终 parsed 合法；中途片段可增量解析（若实现）

- C-004 [P1]
  - 描述：Schema 严格模式 strict=True
  - 前置：C-001
  - 步骤：提供超出 schema 的字段
  - 预期：解析行为符合 strict 约束（拒绝/忽略额外字段，按实现而定）

- C-005 [P2]
  - 描述：Schema 无效/字段类型错误时的失败与异常
  - 前置：C-001
  - 步骤：构造违反类型/缺失 required 的输入
  - 预期：有明确错误提示或返回严格失败结果

模块D：推理模型支持（Reasoner Models）
参考：<mcfile name="deepseek_plugin.py" path="e:\project\harborai\harborai\core\plugins\deepseek_plugin.py"></mcfile> <mcfile name="base_plugin.py" path="e:\project\harborai\harborai\core\base_plugin.py"></mcfile>

- D-001 [关键][P0]
  - 描述：调用 deepseek-reasoner 非流式返回，自动检测 reasoning_content
  - 前置：A-001，配置 deepseek
  - 步骤：model="deepseek-reasoner"，create(...)
  - 预期：choices[0].message.content 存在；若模型返回思考过程，存在 reasoning_content

- D-002 [关键][P0]
  - 描述：推理模型流式返回中 delta.reasoning_content 与 delta.content 正确区分
  - 前置：D-001
  - 步骤：stream=True，打印思考与答案两类片段
  - 预期：两类流片段顺序与语义正确

- D-003 [P1]
  - 描述：推理模型 + 结构化输出（Agently）
  - 前置：D-001, C-001
  - 步骤：携带 response_format JSON Schema
  - 预期：parsed 合法；同时保留 reasoning_content（若有）

- D-004 [P2]
  - 描述：厂商内置思考开关兼容（如 Doubao extra_body.thinking）
  - 前置：厂商支持思考开关
  - 步骤：extra_body={"thinking":{"type":"enabled"}} / disabled
  - 预期：开启时有 reasoning；关闭时无 reasoning

模块E：插件化与多厂商适配
参考：<mcfile name="client_manager.py" path="e:\project\harborai\harborai\core\client_manager.py"></mcfile> 以及各插件文件

- E-001 [关键][P0]
  - 描述：根据 model 自动路由到正确插件
  - 前置：A-001
  - 步骤：不同厂商模型名调用（OpenAI/Doubao/Wenxin/DeepSeek）
  - 预期：各自插件被调用；无交叉/误路由

- E-002 [P1]
  - 描述：OpenAI 插件基本调用（非流/流式）
  - 前置：OpenAI 凭证可用
  - 步骤：非流式/流式各一次
  - 预期：返回结构对齐 OpenAI

- E-003 [P1]
  - 描述：Doubao 插件调用（含 thinking 开关）
  - 前置：Doubao 凭证可用
  - 步骤：非流式；extra_body 切换
  - 预期：响应正常；思考开关生效

- E-004 [P1]
  - 描述：Wenxin 插件调用（非流/可选流）
  - 前置：Wenxin 凭证可用
  - 步骤：同 E-002
  - 预期：响应正常

- E-005 [P2]
  - 描述：插件注册机制健壮性（动态扫描/重复注册防护）
  - 前置：模拟重复/缺失情况
  - 步骤：加载插件目录，故意加入重复定义
  - 预期：有明确告警/去重；不影响运行

模块F：异常标准化与重试机制
参考：<mcfile name="exceptions.py" path="e:\project\harborai\harborai\utils\exceptions.py"></mcfile> <mcfile name="retry.py" path="e:\project\harborai\harborai\utils\retry.py"></mcfile> <mcfile name="decorators.py" path="e:\project\harborai\harborai\api\decorators.py"></mcfile>

- F-001 [关键][P0]
  - 描述：认证失败 -> AuthError
  - 前置：故意设置错误 API Key
  - 步骤：调用 create(...)
  - 预期：抛出/返回 AuthError 标准异常

- F-002 [关键][P0]
  - 描述：限流 -> RateLimitError（且触发重试）
  - 前置：模拟/诱发限流
  - 步骤：调用 create(...)，观察重试日志
  - 预期：分类为 RateLimitError；按策略重试至成功或用尽

- F-003 [关键][P0]
  - 描述：超时 -> TimeoutError（可配置重试）
  - 前置：设置极低超时
  - 步骤：调用 create(...)
  - 预期：TimeoutError；若策略允许则重试

- F-004 [P1]
  - 描述：网络错误/5xx -> 可重试错误；4xx（除429/401）-> 不重试
  - 前置：注入不同错误类型
  - 步骤：调用并观察重试行为
  - 预期：符合重试白名单；4xx 直接失败

- F-005 [P1]
  - 描述：指数退避 + jitter 参数生效
  - 前置：设置 retry_policy（重试次数/基数/抖动）
  - 步骤：触发可重试错误
  - 预期：重试间隔递增并带随机抖动；次数符合配置

模块G：降级策略（Fallback）
参考：<mcfile name="client_manager.py" path="e:\project\harborai\harborai\core\client_manager.py"></mcfile>

- G-001 [关键][P1]
  - 描述：fallback 列表按顺序降级（模型/厂商）
  - 前置：配置 fallback=["primary","secondary",...]
  - 步骤：使 primary 失败；观察自动切换
  - 预期：按顺序尝试直至成功；最终结果返回成功模型名

- G-002 [P1]
  - 描述：降级过程 trace_id 一致性
  - 前置：同 G-001
  - 步骤：关联日志 trace_id
  - 预期：同一 trace_id 贯穿所有尝试

- G-003 [P2]
  - 描述：全部失败后的复合错误/报告
  - 前置：所有候选失败
  - 步骤：触发调用
  - 预期：返回聚合错误信息，含各失败原因摘要

模块H：可观测性、日志与脱敏
参考：<mcfile name="tracer.py" path="e:\project\harborai\harborai\utils\tracer.py"></mcfile> <mcfile name="logger.py" path="e:\project\harborai\harborai\utils\logger.py"></mcfile> <mcfile name="postgres_logger.py" path="e:\project\harborai\harborai\storage\postgres_logger.py"></mcfile>

- H-001 [关键][P0]
  - 描述：trace_id 自动生成与外部传入生效
  - 前置：A-001
  - 步骤：不传/传 trace_id 分别调用
  - 预期：未传自动生成；已传沿用

- H-002 [关键][P0]
  - 描述：异步日志不阻塞主调用
  - 前置：开启日志
  - 步骤：并发多次调用，统计主流程延迟
  - 预期：日志写入异步，主流程无明显阻塞

- H-003 [关键][P0]
  - 描述：日志字段完整性（trace_id/model/request/response/latency/tokens/cost/success/reasoning_present/structured_provider）
  - 前置：不同模式组合调用
  - 步骤：检查日志记录
  - 预期：字段齐全且取值正确

- H-004 [关键][P0]
  - 描述：日志脱敏（API Key、敏感数据）
  - 前置：开启脱敏
  - 步骤：调用后检查日志原文
  - 预期：敏感信息不可见

- H-005 [P2]
  - 描述：日志队列压力测试（丢失/乱序检测）
  - 前置：高并发压测
  - 步骤：1K+ 次调用
  - 预期：不丢失；顺序按 trace 分辨合理

模块I：成本统计与计费配置
参考：<mcfile name="pricing.py" path="e:\project\harborai\harborai\core\pricing.py"></mcfile>（存在） <mcfile name="settings.py" path="e:\project\harborai\harborai\config\settings.py"></mcfile>

- I-001 [关键][P0]
  - 描述：usage.token 统计与 cost 计算
  - 前置：开启 cost_tracking
  - 步骤：调用并读取 usage/cost
  - 预期：非空且单价计算合理（与配置匹配）

- I-002 [P1]
  - 描述：自定义价格配置覆盖默认
  - 前置：修改配置/注入价格
  - 步骤：再次调用
  - 预期：cost 按新单价计算

- I-003 [P2]
  - 描述：多厂商/多模型价格映射一致性
  - 前置：跨插件调用
  - 步骤：比较成本
  - 预期：映射正确，无缺失项

模块J：持久化存储与生命周期（规划/可选）
参考：<mcfile name="postgres_logger.py" path="e:\project\harborai\harborai\storage\postgres_logger.py"></mcfile> <mcfile name="lifecycle.py" path="e:\project\harborai\harborai\storage\lifecycle.py"></mcfile>

- J-001 [P3]
  - 描述：日志写入 PostgreSQL（Docker 化）
  - 前置：PostgreSQL/连接参数就绪
  - 步骤：调用后查询表记录
  - 预期：记录完整、索引可用

- J-002 [P3]
  - 描述：生命周期管理：7天自动清理（短期数据）
  - 前置：造数据标记过期
  - 步骤：执行生命周期任务
  - 预期：短期数据被清理，关键日志保留

- J-003 [P3]
  - 描述：永久保存分类
  - 前置：标记关键日志
  - 步骤：运行清理
  - 预期：关键数据不被删除

模块K：配置管理与环境变量
参考：<mcfile name="settings.py" path="e:\project\harborai\harborai\config\settings.py"></mcfile>

- K-001 [P1]
  - 描述：从环境变量读取 api_key/base_url
  - 前置：设置 env
  - 步骤：不在构造函数显式传参
  - 预期：读取成功，连接可用

- K-002 [P1]
  - 描述：构造函数参数覆盖环境变量
  - 前置：同时设置 env 与构造参数
  - 步骤：调用
  - 预期：以构造参数为准

- K-003 [P2]
  - 描述：非法/缺失配置的错误提示
  - 前置：删/错 env
  - 步骤：调用
  - 预期：有清晰错误消息

模块L：CLI（命令行）
参考：<mcfile name="main.py" path="e:\project\harborai\harborai\cli\main.py"></mcfile>

- L-001 [P2]
  - 描述：CLI 启动与基础命令
  - 前置：安装可执行入口
  - 步骤：powershell 下运行 harborai --help
  - 预期：显示帮助与可用子命令

- L-002 [P2]
  - 描述：CLI 调用一次推理/非推理请求
  - 前置：凭证就绪
  - 步骤：harborai run --model gpt-4 --message "hi"
  - 预期：输出结果，含 trace/latency 概览（若设计）

模块M：安全与合规
参考：<mcfile name="logger.py" path="e:\project\harborai\harborai\utils\logger.py"></mcfile>

- M-001 [关键][P0]
  - 描述：不记录/输出明文密钥
  - 前置：设置密钥
  - 步骤：全链路调用+检查日志与异常输出
  - 预期：key 被脱敏或不出现

- M-002 [P1]
  - 描述：可选禁用日志模式
  - 前置：关闭日志开关
  - 步骤：调用
  - 预期：无日志写入

模块N：兼容性与对齐验证（OpenAI）
参考：<mcfile name="client.py" path="e:\project\harborai\harborai\api\client.py"></mcfile>

- N-001 [关键][P0]
  - 描述：替换 OpenAI SDK 为 HarborAI，示例无需改动业务逻辑即可运行
  - 前置：PRD 示例代码
  - 步骤：将 OpenAI 客户端替换为 HarborAI
  - 预期：示例全部运行通过，输出一致

- N-002 [关键][P0]
  - 描述：ChatCompletion/ChatCompletionChunk 字段对齐
  - 前置：B-001/B-002
  - 步骤：对比字段与含义
  - 预期：一致或可兼容适配


模块P：错误用例与健壮性
参考：<mcfile name="exceptions.py" path="e:\project\harborai\harborai\utils\exceptions.py"></mcfile>

- P-001 [P1]
  - 描述：参数缺失/类型错误（messages、model 等）
  - 前置：无
  - 步骤：传入非法参数
  - 预期：友好错误提示，类型检查到位

- P-002 [P1]
  - 描述：response_format 非法值/缺字段
  - 前置：无
  - 步骤：构造不完整 schema
  - 预期：清晰报错

- P-003 [P2]
  - 描述：structured_provider 非法值
  - 前置：无
  - 步骤：structured_provider="x"
  - 预期：被拒绝/回退默认，并告知

模块Q：文档与示例一致性（非代码）
参考：PRD/TDD

- Q-001 [P2]
  - 描述：README/示例与实际 API 对齐
  - 前置：文档在库
  - 步骤：跑通 README 示例
  - 预期：与代码一致

- Q-002 [P3]
  - 描述：TDD/PRD 中所有功能点均可找到对应测试项（本清单）
  - 前置：无
  - 步骤：逐项对照
  - 预期：映射完整，无遗漏

功能点→测试项覆盖映射（摘要）
- 统一 API/命名空间：A-001/A-002/N-001/N-002
- 同步/异步/流式：B-001/002/003/004
- 结构化输出（Agently/Native）：C-001/002/003/004/005
- 推理模型与自动兼容：D-001/002/003/004
- 插件化与路由：E-001/002/003/004/005
- 异步日志/Trace/脱敏：H-001/002/003/004/005
- 成本统计：I-001/002/003
- 重试机制：F-002/003/004/005
- 异常标准化：F-001/002/003/004
- 降级策略：G-001/002/003
- 持久化/生命周期：J-001/002/003（规划）
- 配置管理：K-001/002/003
- CLI：L-001/002
- 安全：M-001/002
- 性能与稳定性：O-001/002/003
- 健壮性错误用例：P-001/002/003
- 文档一致性：Q-001/002

# 模块O：性能测试模块
参考：整体架构与核心组件

### O.1 测试场景描述

#### O.1.1 基础性能测试场景
- **单次调用性能**：测试 HarborAI 客户端单次 API 调用的响应时间和资源消耗
- **封装开销测试**：测量 HarborAI 封装层相对于原生 SDK 的额外开销
- **不同模型性能对比**：对比不同厂商模型（OpenAI、DeepSeek、Doubao、Wenxin）的调用性能
- **推理模型 vs 非推理模型**：对比推理模型（如 deepseek-reasoner）与普通模型的性能差异

#### O.1.2 并发性能测试场景
- **高并发调用**：测试系统在高并发请求下的表现
- **异步调用性能**：测试异步调用模式的性能优势
- **流式调用性能**：测试流式输出模式的延迟和吞吐量
- **结构化输出并发**：测试结构化输出（Agently/Native）在并发场景下的性能

#### O.1.3 压力测试场景
- **极限并发测试**：测试系统承受的最大并发数
- **资源耗尽测试**：测试在资源受限情况下的降级表现
- **网络异常压力**：测试网络不稳定情况下的重试和容错性能
- **长时间运行稳定性**：测试系统长期运行的稳定性和资源泄露情况

#### O.1.4 特殊场景性能测试
- **日志系统性能**：测试异步日志系统对主调用链路的影响
- **降级策略性能**：测试 fallback 机制的切换延迟
- **成本统计性能**：测试成本计算对整体性能的影响
- **插件路由性能**：测试插件选择和路由的开销

### O.2 性能指标定义

#### O.2.1 响应时间指标
- **平均响应时间（Average Response Time）**：所有请求的平均响应时间
- **P50 响应时间**：50% 请求的响应时间阈值
- **P95 响应时间**：95% 请求的响应时间阈值
- **P99 响应时间**：99% 请求的响应时间阈值
- **封装开销（Wrapper Overhead）**：HarborAI 相对于原生 SDK 的额外延迟，目标 < 1ms

#### O.2.2 吞吐量指标
- **QPS（Queries Per Second）**：每秒处理的请求数量
- **TPS（Transactions Per Second）**：每秒完成的事务数量
- **并发用户数（Concurrent Users）**：系统能同时支持的用户数
- **最大吞吐量（Peak Throughput）**：系统能达到的最大处理能力

#### O.2.3 可靠性指标
- **成功率（Success Rate）**：请求成功的百分比，目标 > 99.9%
- **错误率（Error Rate）**：请求失败的百分比，目标 < 0.1%
- **可用性（Availability）**：系统正常运行时间百分比
- **重试成功率**：重试机制的有效性

#### O.2.4 资源使用率指标
- **CPU 使用率**：处理器使用百分比
- **内存使用率**：内存占用情况和增长趋势
- **网络带宽使用率**：网络 I/O 使用情况
- **文件句柄数**：打开的文件句柄数量
- **线程/协程数量**：并发执行单元的数量

#### O.2.5 业务指标
- **Token 处理速度**：每秒处理的 Token 数量
- **结构化输出解析时间**：JSON Schema 解析的耗时
- **日志写入延迟**：异步日志系统的写入延迟
- **成本计算耗时**：成本统计功能的计算时间

### O.3 测试环境配置

#### O.3.1 硬件环境
- **CPU**：Intel i7-12700K 或同等性能（8核16线程，基频3.6GHz）
- **内存**：32GB DDR4-3200
- **存储**：NVMe SSD 1TB（读写速度 > 3000MB/s）
- **网络**：千兆以太网连接，延迟 < 10ms

#### O.3.2 软件环境
- **操作系统**：Windows 11 Professional
- **Python 版本**：Python 3.9.x - 3.12.x
- **依赖库版本**：
  - aiohttp >= 3.8.0
  - asyncio（内置）
  - psutil >= 5.9.0（资源监控）
  - pytest-benchmark >= 4.0.0（性能测试）
- **数据库**：PostgreSQL 15.x（用于日志存储测试）
- **容器环境**：Docker Desktop 4.x（可选）

#### O.3.3 网络环境
- **模拟网络延迟**：使用 tc（Linux）或 NetLimiter（Windows）模拟不同网络条件
- **带宽限制**：模拟不同带宽环境（1Mbps、10Mbps、100Mbps）
- **网络抖动**：模拟网络不稳定情况（丢包率 0.1%-5%）

#### O.3.4 测试数据配置
- **API 密钥**：配置各厂商的有效 API 密钥
- **测试消息**：准备不同长度的测试消息（短文本、长文本、多轮对话）
- **JSON Schema**：准备不同复杂度的结构化输出模板
- **并发配置**：设置不同的并发级别（1、10、50、100、500、1000）

### O.4 测试步骤说明

#### O.4.1 基础性能测试步骤

**O-001 [关键][P0] 封装开销测试**
- **前置条件**：
  - 安装 HarborAI 和对应的原生 SDK（如 openai）
  - 配置相同的 API 密钥和测试环境
  - 准备标准测试消息
- **测试步骤**：
  1. 预热阶段：执行 10 次调用预热 JIT 和网络连接
  2. 原生 SDK 基准测试：使用 OpenAI SDK 执行 1000 次相同调用，记录时间
  3. HarborAI 测试：使用 HarborAI 执行相同的 1000 次调用，记录时间
  4. 计算封装开销：HarborAI 平均时间 - 原生 SDK 平均时间
  5. 重复测试 5 轮，取平均值
- **性能指标**：封装开销 < 1ms
- **测试代码示例**：
```python
import time
import statistics
from openai import OpenAI
from harborai import HarborAI

def benchmark_wrapper_overhead():
    # 配置
    api_key = "your-api-key"
    base_url = "your-base-url"
    test_message = [{"role": "user", "content": "Hello"}]
    
    # 原生 SDK
    openai_client = OpenAI(api_key=api_key, base_url=base_url)
    harbor_client = HarborAI(api_key=api_key, base_url=base_url)
    
    # 预热
    for _ in range(10):
        openai_client.chat.completions.create(model="ernie-3.5-8k", messages=test_message)
        harbor_client.chat.completions.create(model="ernie-3.5-8k", messages=test_message)
    
    # 基准测试
    openai_times = []
    harbor_times = []
    
    for _ in range(1000):
        # 测试原生 SDK
        start = time.perf_counter()
        openai_client.chat.completions.create(model="ernie-3.5-8k", messages=test_message)
        openai_times.append(time.perf_counter() - start)
        
        # 测试 HarborAI
        start = time.perf_counter()
        harbor_client.chat.completions.create(model="ernie-3.5-8k", messages=test_message)
        harbor_times.append(time.perf_counter() - start)
    
    # 计算开销
    openai_avg = statistics.mean(openai_times)
    harbor_avg = statistics.mean(harbor_times)
    overhead = harbor_avg - openai_avg
    
    print(f"OpenAI 平均时间: {openai_avg*1000:.2f}ms")
    print(f"HarborAI 平均时间: {harbor_avg*1000:.2f}ms")
    print(f"封装开销: {overhead*1000:.2f}ms")
    
    assert overhead < 0.001, f"封装开销 {overhead*1000:.2f}ms 超过 1ms 阈值"
```

**O-002 [关键][P0] 高并发成功率测试**
- **前置条件**：
  - 配置有效的 API 密钥
  - 确保网络连接稳定
  - 安装并发测试工具
- **测试步骤**：
  1. 设置并发级别：10、50、100、500、1000
  2. 每个并发级别执行 1000 次请求
  3. 记录成功、失败、超时的请求数量
  4. 计算成功率：成功请求数 / 总请求数
  5. 分析失败原因分布
- **性能指标**：成功率 > 99.9%
- **测试代码示例**：
```python
import asyncio
import aiohttp
from harborai import HarborAI
from concurrent.futures import ThreadPoolExecutor

async def concurrent_test(concurrency_level, total_requests):
    client = HarborAI(api_key="your-api-key")
    test_message = [{"role": "user", "content": "测试消息"}]
    
    success_count = 0
    error_count = 0
    timeout_count = 0
    
    async def single_request():
        nonlocal success_count, error_count, timeout_count
        try:
            response = await client.chat.completions.create(
                model="ernie-3.5-8k",
                messages=test_message,
                timeout=30
            )
            success_count += 1
            return response
        except asyncio.TimeoutError:
            timeout_count += 1
        except Exception as e:
            error_count += 1
            print(f"请求失败: {e}")
    
    # 创建并发任务
    semaphore = asyncio.Semaphore(concurrency_level)
    
    async def bounded_request():
        async with semaphore:
            return await single_request()
    
    # 执行并发测试
    tasks = [bounded_request() for _ in range(total_requests)]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # 计算结果
    success_rate = success_count / total_requests
    print(f"并发级别: {concurrency_level}")
    print(f"总请求数: {total_requests}")
    print(f"成功: {success_count}, 失败: {error_count}, 超时: {timeout_count}")
    print(f"成功率: {success_rate:.4f} ({success_rate*100:.2f}%)")
    
    assert success_rate > 0.999, f"成功率 {success_rate:.4f} 低于 99.9% 阈值"

# 运行测试
for concurrency in [10, 50, 100, 500, 1000]:
    asyncio.run(concurrent_test(concurrency, 1000))
```

#### O.4.2 资源监控测试步骤

**O-003 [P1] 内存和句柄泄露检查**
- **前置条件**：
  - 安装 psutil 库用于资源监控
  - 准备长时间运行的测试脚本
- **测试步骤**：
  1. 记录初始资源状态（内存、句柄数）
  2. 执行 10000 次 API 调用
  3. 每 1000 次调用记录一次资源使用情况
  4. 分析资源使用趋势
  5. 检查是否存在内存泄露或句柄泄露
- **性能指标**：内存增长 < 10MB，句柄数稳定

#### O.4.3 流式性能测试步骤

**O-004 [P1] 流式调用延迟测试**
- **测试步骤**：
  1. 发起流式调用请求
  2. 记录首个 chunk 到达时间（TTFB - Time To First Byte）
  3. 记录每个 chunk 的间隔时间
  4. 计算总体流式传输效率
- **性能指标**：TTFB < 500ms，chunk 间隔 < 100ms

### O.5 预期结果和实际结果对比

#### O.5.1 性能基准对比表

| 测试项目 | 预期结果 | 实际结果 | 达标状态 | 备注 |
|---------|---------|---------|----------|------|
| 封装开销 | < 1ms | _待测试_ | ⏳ | 核心性能指标 |
| 高并发成功率 | > 99.9% | _待测试_ | ⏳ | 生产可用性指标 |
| P95 响应时间 | < 2s | _待测试_ | ⏳ | 用户体验指标 |
| 最大 QPS | > 100 | _待测试_ | ⏳ | 吞吐量指标 |
| 内存使用 | < 100MB | _待测试_ | ⏳ | 资源效率指标 |
| TTFB（流式） | < 500ms | _待测试_ | ⏳ | 流式体验指标 |

#### O.5.2 不同场景性能对比

| 场景 | 响应时间 (P95) | QPS | 成功率 | 内存使用 |
|------|---------------|-----|--------|----------|
| 单线程调用 | _待测试_ | _待测试_ | _待测试_ | _待测试_ |
| 10 并发 | _待测试_ | _待测试_ | _待测试_ | _待测试_ |
| 100 并发 | _待测试_ | _待测试_ | _待测试_ | _待测试_ |
| 1000 并发 | _待测试_ | _待测试_ | _待测试_ | _待测试_ |
| 流式调用 | _待测试_ | _待测试_ | _待测试_ | _待测试_ |
| 结构化输出 | _待测试_ | _待测试_ | _待测试_ | _待测试_ |

#### O.5.3 厂商插件性能对比

| 插件 | 平均响应时间 | 封装开销 | 特殊功能 | 性能评级 |
|------|-------------|----------|----------|----------|
| OpenAI | _待测试_ | _待测试_ | 原生兼容 | ⭐⭐⭐⭐⭐ |
| DeepSeek | _待测试_ | _待测试_ | 推理模型 | _待评估_ |
| Doubao | _待测试_ | _待测试_ | 思考开关 | _待评估_ |
| Wenxin | _待测试_ | _待测试_ | 标准调用 | _待评估_ |

### O.6 性能瓶颈分析

#### O.6.1 潜在瓶颈点识别

**网络 I/O 瓶颈**
- **症状**：高并发下响应时间显著增加
- **原因**：网络连接池不足、DNS 解析延迟、TCP 连接复用不当
- **检测方法**：监控网络连接数、DNS 查询时间、TCP 握手时间

**CPU 计算瓶颈**
- **症状**：CPU 使用率持续高位、响应时间线性增长
- **原因**：JSON 解析、加密解密、正则表达式匹配
- **检测方法**：使用 cProfile 分析热点函数

**内存瓶颈**
- **症状**：内存使用持续增长、GC 频繁触发
- **原因**：大对象缓存、循环引用、内存泄露
- **检测方法**：使用 memory_profiler 监控内存分配

**异步任务瓶颈**
- **症状**：异步调用性能不如预期、任务队列积压
- **原因**：事件循环阻塞、协程调度不当、锁竞争
- **检测方法**：监控事件循环延迟、协程数量

#### O.6.2 瓶颈分析工具

**性能分析工具**
```python
# CPU 性能分析
import cProfile
import pstats

def profile_performance():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 执行测试代码
    run_performance_test()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 显示前20个热点函数

# 内存分析
from memory_profiler import profile

@profile
def memory_test():
    # 执行内存密集型操作
    pass

# 异步性能分析
import asyncio
import time

async def monitor_event_loop():
    while True:
        start = time.perf_counter()
        await asyncio.sleep(0)
        delay = time.perf_counter() - start
        if delay > 0.01:  # 10ms 阈值
            print(f"事件循环延迟: {delay*1000:.2f}ms")
        await asyncio.sleep(1)
```

#### O.6.3 性能监控指标

**实时监控指标**
- **响应时间分布**：P50、P95、P99 响应时间趋势
- **错误率趋势**：按时间窗口统计的错误率变化
- **资源使用率**：CPU、内存、网络 I/O 使用情况
- **并发连接数**：活跃连接数和连接池状态

### O.7 优化建议

#### O.7.1 网络优化建议

**连接池优化**
```python
# 优化连接池配置
import aiohttp

connector = aiohttp.TCPConnector(
    limit=100,              # 总连接数限制
    limit_per_host=30,      # 每个主机连接数限制
    ttl_dns_cache=300,      # DNS 缓存时间
    use_dns_cache=True,     # 启用 DNS 缓存
    keepalive_timeout=30,   # 保持连接时间
    enable_cleanup_closed=True  # 自动清理关闭的连接
)
```

**请求重试优化**
```python
# 智能重试策略
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def optimized_request(client, **kwargs):
    return await client.chat.completions.create(**kwargs)
```

#### O.7.2 内存优化建议

**对象池模式**
```python
# 实现对象池减少 GC 压力
class ResponsePool:
    def __init__(self, max_size=1000):
        self._pool = []
        self._max_size = max_size
    
    def get_response(self):
        if self._pool:
            return self._pool.pop()
        return ChatCompletion()
    
    def return_response(self, response):
        if len(self._pool) < self._max_size:
            response.reset()  # 重置对象状态
            self._pool.append(response)
```

**流式处理优化**
```python
# 使用生成器减少内存占用
async def stream_with_backpressure(stream, buffer_size=1024):
    buffer = []
    async for chunk in stream:
        buffer.append(chunk)
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer
```

#### O.7.3 并发优化建议

**信号量控制**
```python
# 使用信号量控制并发数
class ConcurrencyController:
    def __init__(self, max_concurrent=100):
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute(self, coro):
        async with self._semaphore:
            return await coro
```

**批处理优化**
```python
# 批量处理请求
async def batch_process(requests, batch_size=10):
    results = []
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)
    return results
```

#### O.7.4 监控和告警优化

**性能监控集成**
```python
# 集成 Prometheus 监控
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('harborai_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('harborai_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('harborai_active_connections', 'Active connections')

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def record_request(self, method, endpoint, duration):
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        REQUEST_DURATION.observe(duration)
    
    def update_connections(self, count):
        ACTIVE_CONNECTIONS.set(count)
```

**自动化性能测试**
```python
# 持续集成中的性能测试
import pytest

@pytest.mark.performance
def test_performance_regression():
    """性能回归测试"""
    baseline_time = 1.0  # 基准时间（秒）
    
    start = time.time()
    # 执行性能测试
    run_performance_benchmark()
    actual_time = time.time() - start
    
    # 允许 10% 的性能波动
    assert actual_time < baseline_time * 1.1, f"性能回归：{actual_time}s > {baseline_time * 1.1}s"
```

#### O.7.5 配置优化建议

**动态配置调优**
```python
# 根据系统负载动态调整配置
class AdaptiveConfig:
    def __init__(self):
        self.base_timeout = 30
        self.base_concurrency = 100
    
    def adjust_for_load(self, current_load):
        if current_load > 0.8:  # 高负载
            return {
                'timeout': self.base_timeout * 1.5,
                'concurrency': self.base_concurrency * 0.7
            }
        elif current_load < 0.3:  # 低负载
            return {
                'timeout': self.base_timeout * 0.8,
                'concurrency': self.base_concurrency * 1.3
            }
        return {
            'timeout': self.base_timeout,
            'concurrency': self.base_concurrency
        }
```

通过以上详细的性能测试模块，可以全面评估 HarborAI 的性能表现，及时发现瓶颈并进行优化，确保系统在生产环境中的稳定性和高性能。


备注与说明
- 标注为 [关键] 的测试项代表对外体验与稳定性最关键的路径，建议优先自动化与回归覆盖。
- 标注为 P3 的持久化与生命周期测试项当前为规划能力，若尚未实现，请在执行阶段标记为 Blocked，并在后续版本启用。
- 为确保“几乎无感迁移”，建议将 N-001/N-002 纳入每次发布的回归基线。
- 建议对 F/G/H/I 模块的组合场景设计端到端 E2E 用例，例如“限流→重试→失败→降级→成功→日志与成本落表”。

如需，我可以基于上述清单为你生成可执行的自动化测试样例（pytest/pytest-asyncio + requests_mock/aiostream 等），并按模块落盘到 tests/ 目录。
        