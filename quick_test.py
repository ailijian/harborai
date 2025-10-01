#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from harborai import HarborAI

client = HarborAI()

# 测试DeepSeek Native
try:
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=[
            {'role': 'system', 'content': '你是一个情感分析专家。请分析用户输入的文本情感，并以JSON格式返回结果。'},
            {'role': 'user', 'content': '分析这句话的情感：今天天气真好'}
        ],
        response_format={
            'type': 'json_schema',
            'json_schema': {
                'name': 'SentimentAnalysis',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'sentiment': {'type': 'string', 'enum': ['positive', 'negative', 'neutral']},
                        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}
                    },
                    'required': ['sentiment', 'confidence']
                }
            }
        },
        structured_provider='native'
    )
    print('DeepSeek Native 成功')
    print(f'Response type: {type(response)}')
    if hasattr(response, 'choices') and len(response.choices) > 0:
        message = response.choices[0].message
        print(f'Message type: {type(message)}')
        has_parsed = hasattr(message, 'parsed')
        print(f'Has parsed: {has_parsed}')
        if has_parsed:
            print(f'Parsed: {message.parsed}')
        if hasattr(message, 'content'):
            print(f'Content: {message.content}')
except Exception as e:
    print(f'DeepSeek Native 失败: {e}')
    import traceback
    traceback.print_exc()