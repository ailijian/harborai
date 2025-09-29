from openai import OpenAI
import os

# Please install OpenAI SDK first: `pip3 install openai`


client = OpenAI(api_key="sk-d996b310528f44ffb1d7bf5b23b5313b", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)