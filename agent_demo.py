from openai import OpenAI

from llm_agent.agent import run_agent
from llm_agent.tools import (
    ADD_ENTRY_TOOL,
    LIST_ENTRIES_TOOL,
    add_entry_tool,
    list_entries_tool,
)

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)

answer = run_agent(
    client=client,
    messages=[
        {
            "role": "user",
            "content": "Show me everything I ate today.",
        },
    ],
    tools=[ADD_ENTRY_TOOL, LIST_ENTRIES_TOOL],
    handlers={
        "add_entry": add_entry_tool,
        "list_entries": list_entries_tool,
    },
)

print(answer)
