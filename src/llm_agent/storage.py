import json
from datetime import date as Date
from pathlib import Path
from uuid import uuid4

from llm_agent.schemas import EntryInput


def add_entry(
    entry: EntryInput,
    *,
    data_dir: Path = Path("data"),
) -> dict:
    path = data_dir / f"{entry.date:%Y-%m}.json"

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"entries": []}

    record = {
        "id": str(uuid4()),
        **entry.model_dump(mode="json"),
    }

    data["entries"].append(record)

    data_dir.mkdir(parents=True, exist_ok=True)

    temp_path = data_dir / f".{path.name}.{uuid4().hex}.tmp"

    try:
        temp_path.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)

    return record


def list_entries(
    *,
    start_date: Date | None = None,
    end_date: Date | None = None,
    data_dir: Path = Path("data"),
) -> list[dict]:
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")

    entries = []

    for path in sorted(data_dir.glob("????-??.json")):
        month = path.stem

        if start_date and month < start_date.strftime("%Y-%m"):
            continue

        if end_date and month > end_date.strftime("%Y-%m"):
            continue

        data = json.loads(path.read_text(encoding="utf-8"))

        for entry in data["entries"]:
            entry_date = Date.fromisoformat(entry["date"])

            if start_date and entry_date < start_date:
                continue

            if end_date and entry_date > end_date:
                continue

            entries.append(entry)

    return sorted(entries, key=lambda entry: entry["date"])
