import re

# 读取原文件
with open('tests/functional/test_a_api_compatibility.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复常见的编码问题
fixes = [
    ('测试可选参?', '测试可选参数'),
    ('验证消息格式验?', '验证消息格式验证'),
    ('测试有效的消息格?', '测试有效的消息格式'),
    ('测试无效的消息格?', '测试无效的消息格式'),
    ('空消息列?', '空消息列表'),
    ('测试响应格式一致性"', '测试响应格式一致性'),
    ('֤óɹ', '验证成功'),
    ('传?', '传递'),
    ('组?', '组合'),
    ('设?', '设置'),
    ('响?', '响应'),
    ('错?', '错误'),
    ('处?', '处理'),
    ('兼容?', '兼容性'),
    ('一致?', '一致性'),
    ('不为?', '不为空'),
    ('temperature?', 'temperature值'),
    ('max_tokens?', 'max_tokens值'),
]

# 应用修复
for old, new in fixes:
    content = content.replace(old, new)

# 移除可能的问题字符
content = re.sub(r'[\ufffd\u200b\u2060\ufeff]', '', content)

# 重新写入文件
with open('tests/functional/test_a_api_compatibility.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('文件编码问题已全面修复')