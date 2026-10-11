from llm_agent.schemas import (
    CalorieSummaryInput,
    DeleteEntryToolInput,
    EntryInput,
    EntryUpdate,
    ListEntriesInput,
    UpdateEntryToolInput,
)
from llm_agent.storage import (
    add_entry,
    delete_entry,
    get_calorie_summary,
    list_entries,
    update_entry,
)

ADD_ENTRY_TOOL = {
    "type": "function",
    "function": {
        "name": "add_entry",
        "description": "Record a food entry with its calories.",
        "parameters": EntryInput.model_json_schema(),
    },
}

LIST_ENTRIES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_entries",
        "description": (
            "Retrieve logged food entries with optional date filters. "
            "Also use this to find entry IDs before updating or deleting."
        ),
        "parameters": ListEntriesInput.model_json_schema(),
    },
}

UPDATE_ENTRY_TOOL = {
    "type": "function",
    "function": {
        "name": "update_entry",
        "description": (
            "Update an existing food entry using its ID. "
            "Only provide the fields that need changing."
        ),
        "parameters": UpdateEntryToolInput.model_json_schema(),
    },
}

DELETE_ENTRY_TOOL = {
    "type": "function",
    "function": {
        "name": "delete_entry",
        "description": "Delete a food entry using its ID.",
        "parameters": DeleteEntryToolInput.model_json_schema(),
    },
}

CALORIE_SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_calorie_summary",
        "description": (
            "Calculate logged calorie totals, daily averages, "
            "and category breakdowns for an inclusive date range. "
            "For the last 7 days, include today and the previous 6 days."
        ),
        "parameters": CalorieSummaryInput.model_json_schema(),
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


def update_entry_tool(**kwargs) -> dict:
    arguments = UpdateEntryToolInput.model_validate(kwargs)

    changes = EntryUpdate.model_validate(
        arguments.model_dump(
            exclude={"entry_id"},
            exclude_unset=True,
        )
    )

    return update_entry(arguments.entry_id, changes)


def delete_entry_tool(**kwargs) -> dict:
    arguments = DeleteEntryToolInput.model_validate(kwargs)
    return delete_entry(arguments.entry_id)


def get_calorie_summary_tool(**kwargs) -> dict:
    arguments = CalorieSummaryInput.model_validate(kwargs)

    return get_calorie_summary(
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )


TOOLS = [
    ADD_ENTRY_TOOL,
    LIST_ENTRIES_TOOL,
    UPDATE_ENTRY_TOOL,
    DELETE_ENTRY_TOOL,
    CALORIE_SUMMARY_TOOL,
]

HANDLERS = {
    "add_entry": add_entry_tool,
    "list_entries": list_entries_tool,
    "update_entry": update_entry_tool,
    "delete_entry": delete_entry_tool,
    "get_calorie_summary": get_calorie_summary_tool,
}
