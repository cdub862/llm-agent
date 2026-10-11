from llm_agent.schemas import EntryInput, ListEntriesInput
from llm_agent.storage import add_entry, list_entries

ADD_ENTRY_TOOL = {
    "type": "function",
    "function": {
        "name": "add_entry",
        "description": "Record a food entry in the calorie tracker.",
        "parameters": EntryInput.model_json_schema(),
    },
}


LIST_ENTRIES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_entries",
        "description": (
            "Retrieve logged food entries, optionally filtered by date. "
            "For a single day, use the same start_date and end_date."
        ),
        "parameters": ListEntriesInput.model_json_schema(),
    },
}


def add_entry_tool(**kwargs) -> dict:
    entry = EntryInput.model_validate(kwargs)
    return add_entry(entry)


def list_entries_tool(**kwargs) -> list[dict]:
    filters = ListEntriesInput.model_validate(kwargs)

    return list_entries(
        start_date=filters.start_date,
        end_date=filters.end_date,
    )
