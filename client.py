from openai import OpenAI

from llm_agent.config import MODEL_ID

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)
print("request")
response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": "Explain what FastAPI is in one sentence.",
        }
    ],
    max_completion_tokens=100,
)

print(response.choices[0].message.content)
print("Usage:", response.usage)
