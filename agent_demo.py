from openai import OpenAI

from llm_agent.agent import run_agent
from llm_agent.tools import ADD_ENTRY_TOOL, add_entry_tool

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)

answer = run_agent(
    client=client,
    messages=[
        {
            "role": "user",
            "content": "Add 50 calories: kitkat",
        },
    ],
    tools=[ADD_ENTRY_TOOL],
    handlers={"add_entry": add_entry_tool},
)

print(answer)
