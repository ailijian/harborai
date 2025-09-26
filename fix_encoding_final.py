#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import codecs

def fix_file_encoding():
    file_path = r'E:\project\harborai\tests\functional\test_a_api_compatibility.py'
    
    # 尝试读取文件
    content = None
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except Exception as e:
            print(f"使用 {encoding} 编码失败: {e}")
            continue
    
    if content is None:
        print("无法读取文件")
        return
    
    # 修复常见的编码问题
    replacements = [
        # 常见乱码修复
        ('楠岃瘉', '验证'),
        ('鍝嶅簲', '响应'),
        ('鍙傛暟', '参数'),
        ('娴嬭瘯', '测试'),
        ('鍒楄〃', '列表'),
        ('缁勫悎', '组合'),
        ('鍏煎', '兼容'),
        ('鎺ュ彛', '接口'),
        ('娴佸紡', '流式'),
        ('鍘傚晢', '厂商'),
        ('涓€鑷存€?', '一致性'),
        ('鍙傛暟', '参数'),
        ('鍒楄〃', '列表'),
        ('缁勫悎', '组合'),
        ('鍏煎', '兼容'),
        ('鎺ュ彛', '接口'),
        ('娴佸紡', '流式'),
        ('鍘傚晢', '厂商'),
        ('涓€鑷存€?', '一致性'),
        # 修复参数名问题
        ('temperature值', 'temperature'),
        ('max_tokens值', 'max_tokens'),
        ('temperature值0.7', 'temperature=0.7'),
        ('temperature值0.8', 'temperature=0.8'),
        ('max_tokens值100', 'max_tokens=100'),
        ('max_tokens值150', 'max_tokens=150'),
        ('max_tokens值-1', 'max_tokens=-1'),
        # 修复其他问题
        ('传?', '传递'),
        ('组?', '组合'),
        ('参?', '参数'),
        ('列?', '列表'),
        ('验?', '验证'),
        ('响?', '响应'),
    ]
    
    # 应用替换
    for old, new in replacements:
        content = content.replace(old, new)
    
    # 移除有问题的Unicode字符
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', content)
    
    # 确保文件以正确的方式结束
    if not content.endswith('\n'):
        content += '\n'
    
    # 检查并修复未闭合的docstring
    lines = content.split('\n')
    in_triple_quote = False
    quote_type = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if '"""' in stripped:
            count = stripped.count('"""')
            if count % 2 == 1:  # 奇数个三引号
                in_triple_quote = not in_triple_quote
                if in_triple_quote:
                    quote_type = '"""'
                else:
                    quote_type = None
    
    # 如果最后还有未闭合的三引号，添加闭合
    if in_triple_quote and quote_type:
        content += quote_type + '\n'
    
    # 写入修复后的文件
    try:
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        print("文件编码问题已彻底修复")
    except Exception as e:
        print(f"写入文件失败: {e}")

if __name__ == '__main__':
    fix_file_encoding()