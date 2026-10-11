from llm_agent.schemas import EntryInput
from llm_agent.storage import add_entry

ADD_ENTRY_TOOL = {
    "type": "function",
    "function": {
        "name": "add_entry",
        "description": "Record a food entry in the calorie tracker.",
        "parameters": EntryInput.model_json_schema(),
    },
}


def add_entry_tool(**kwargs) -> dict:
    entry = EntryInput.model_validate(kwargs)
    return add_entry(entry)
