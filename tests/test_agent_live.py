import os
from datetime import date
from unittest.mock import Mock

import pytest
from openai import OpenAI

from llm_agent.agent import run_agent
from llm_agent.schemas import EntryInput
from llm_agent.storage import add_entry, list_entries
from llm_agent.tools import HANDLERS, TOOLS


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_AGENT") != "1",
    reason="Live LLM evaluation is opt-in",
)
def test_live_qwen_add_and_summarize(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="local",
        timeout=180.0,
    )
    add_entry(
        EntryInput(
            date=date(2026, 10, 11),
            description="Toast",
            calories=200,
            category="grains",
            tags=["breakfast"],
        )
    )

    # Real handlers wrapped with mocks to record which tools Qwen uses.
    handlers = {name: Mock(wraps=handler) for name, handler in HANDLERS.items()}

    answer = run_agent(
        client=client,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use tools to record food and calculate totals. "
                    "Whenever the user requests a daily total, you must call "
                    "get_calorie_summary, even after adding an entry. "
                    "Never assume there are no existing entries."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Perform these operations in order:\n"
                    "1. Call add_entry to log 300 calories of oatmeal "
                    "on 2026-10-11, category grains, tag breakfast.\n"
                    "2. After add_entry succeeds, call get_calorie_summary "
                    "for 2026-10-11.\n"
                    "3. Report the total returned by get_calorie_summary. "
                    "Do not calculate the total yourself."
                ),
            },
        ],
        tools=TOOLS,
        handlers=handlers,
        max_steps=5,
    )

    print("Answer:", answer)
    print(
        "Tool calls:", {name: handler.call_count for name, handler in handlers.items()}
    )

    entries = list_entries(
        start_date=date(2026, 10, 11),
        end_date=date(2026, 10, 11),
    )

    assert len(entries) == 2
    assert handlers["add_entry"].call_count == 1
    assert handlers["get_calorie_summary"].call_count >= 1
    assert "500" in answer
