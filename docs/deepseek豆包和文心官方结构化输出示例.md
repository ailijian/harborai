# deepseek官方示例（文档链接：https://api-docs.deepseek.com/zh-cn/guides/json_mode）
```
import json
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

EXAMPLE INPUT: 
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

user_prompt = "Which is the longest river in the world? The Nile River."

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
)

print(json.loads(response.choices[0].message.content))

# 请求返回结果：
{
    "question": "Which is the longest river in the world?",
    "answer": "The Nile River"
}
```

# 百度文心大模型官方示例（文档链接：https://ai.baidu.com/ai-doc/AISTUDIO/rm344erns#55-%E7%BB%93%E6%9E%84%E5%8C%96%E8%BE%93%E5%87%BA）
```

{
  "model": "ernie-3.5-8k",
  "messages": [
    {
      "role": "user",
      "content": "今天上海天气"
    }
  ],
  "response_format": {
    "type": "json_schema"
  }
}

```

# 豆包官方示例（文档连接：https://www.volcengine.com/docs/82379/1568221）
```
from openai import OpenAI
import os
from pydantic import BaseModel

client = OpenAI(
    # 从环境变量中获取方舟 API Key
    api_key=os.environ.get("ARK_API_KEY"),
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)

class Step(BaseModel):
    explanation: str
    output: str
class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str
    
completion = client.beta.chat.completions.parse(
    model = "doubao-seed-1-6-250615",  # 替换为您需要使用的模型
    messages = [
        {"role": "system", "content": "你是一位数学辅导老师。"},
        {"role": "user", "content": "使用中文解题: 8x + 9 = 32 and x + y = 1"},
    ],
    response_format=MathResponse,
    extra_body={
         "thinking": {
             "type": "disabled" # 不使用深度思考能力
             # "type": "enabled" # 使用深度思考能力
         }
     }
)
resp = completion.choices[0].message.parsed
# 打印 JSON 格式结果
print(resp.model_dump_json(indent=2))

# 请求返回结果：
{
  "steps": [
    {
      "explanation": "解第一个方程8x + 9 = 32，先将等式两边同时减去9，得到8x = 32 - 9",
      "output": "8x = 23"
    },
    {
      "explanation": "然后等式两边同时除以8，求出x的值",
      "output": "x = 23/8"
    },
    {
      "explanation": "将x = 23/8代入第二个方程x + y = 1，求解y，即y = 1 - x",
      "output": "y = 1 - 23/8"
    },
    {
      "explanation": "计算1 - 23/8，通分后得到(8 - 23)/8",
      "output": "y = -15/8"
    }
  ],
  "final_answer": "x = 23/8，y = -15/8"
}
```