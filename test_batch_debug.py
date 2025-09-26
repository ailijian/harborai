#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import os
import json
import shutil
from harborai.cli.main import cli
from click.testing import CliRunner

# 设置测试环境
os.environ['PYTEST_CURRENT_TEST'] = 'test'

# 创建临时目录和文件
temp_dir = tempfile.mkdtemp()
input_file = os.path.join(temp_dir, 'test_input.txt')
output_file = os.path.join(temp_dir, 'test_output.json')

# 创建输入文件
with open(input_file, 'w', encoding='utf-8') as f:
    f.write('消息1\n消息2\n')

# 运行命令
runner = CliRunner()
result = runner.invoke(cli, [
    '--quiet',
    'batch-process',
    '--input-file', input_file,
    '--output-file', output_file,
    '--provider', 'deepseek',
    '--model', 'deepseek-chat'
])

print(f'Exit code: {result.exit_code}')
print(f'Output: {result.output}')
print(f'Exception: {result.exception}')
print(f'Output file exists: {os.path.exists(output_file)}')

if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Results count: {len(data)}')
    print(f'First result: {data[0] if data else "None"}')

# 清理
shutil.rmtree(temp_dir)