
# 测试报告摘要

## 测试概览

| 指标 | 数值 |
|------|------|
| 总测试数 | {{ summary.total_tests }} |
| 通过测试 | {{ summary.passed_tests }} |
| 失败测试 | {{ summary.failed_tests }} |
| 跳过测试 | {{ summary.skipped_tests }} |
| 错误测试 | {{ summary.error_tests }} |
| 成功率 | {{ "%.2f" | format(summary.success_rate) }}% |
| 执行时间 | {{ "%.2f" | format(summary.execution_time) }}秒 |
| 代码覆盖率 | {{ "%.2f" | format(summary.coverage_percentage) }}% |

## 测试状态

{% if summary.success_rate >= 95 %}
✅ **测试状态：优秀** - 成功率达到{{ "%.2f" | format(summary.success_rate) }}%
{% elif summary.success_rate >= 80 %}
⚠️ **测试状态：良好** - 成功率为{{ "%.2f" | format(summary.success_rate) }}%，建议关注失败用例
{% else %}
❌ **测试状态：需要改进** - 成功率仅为{{ "%.2f" | format(summary.success_rate) }}%，请及时修复问题
{% endif %}

## 详细报告

- [性能测试报告](performance_report.html)
- [安全测试报告](security_report.html)

---
生成时间：{{ timestamp }}
环境：{{ summary.environment }}
