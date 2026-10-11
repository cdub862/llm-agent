import json
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
