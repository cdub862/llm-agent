from datetime import datetime
from zoneinfo import ZoneInfo

from openai import OpenAI

from llm_agent.agent import run_agent
from llm_agent.tools import HANDLERS, TOOLS

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)

today = datetime.now(ZoneInfo("America/Santo_Domingo")).date().isoformat()

user_message = input("You: ")

answer = run_agent(
    client=client,
    messages=[
        {
            "role": "system",
            "content": (
                f"Today's date is {today}. "
                "Resolve relative dates using this date. "
                "Use tools for calorie records and calculations. "
                "Find existing entry IDs using list_entries. "
                "Do not invent IDs or claim an operation succeeded "
                "unless the tool confirms it."
            ),
        },
        {
            "role": "user",
            "content": user_message,
        },
    ],
    tools=TOOLS,
    handlers=HANDLERS,
    max_steps=5,
)

print("\nAssistant:", answer)
