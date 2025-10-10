#!/usr/bin/env python3
import re

# 测试修复后的正则表达式
fixed_regex = r'^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$'
test_version = 'v1.0.0-beta.1'

print(f'修复后的正则: {fixed_regex}')
print(f'测试版本: {test_version}')
print(f'匹配结果: {bool(re.match(fixed_regex, test_version))}')

# 测试变量传递
version_num = '1.0.0-beta.1'
print(f'\n测试变量传递:')
print(f'version_num = "{version_num}"')
print(f'格式化字符串: version = "{version_num}"')