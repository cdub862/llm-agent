import json
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from llm_agent.schemas import EntryInput, EntryUpdate


def _write_month(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        temp_path.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


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

    _write_month(path, data)
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


def update_entry(
    entry_id: str,
    changes: EntryUpdate,
    *,
    data_dir: Path = Path("data"),
) -> dict:
    updates = changes.model_dump(mode="json", exclude_unset=True)

    if not updates:
        raise ValueError("No fields provided for update")

    if any(value is None for value in updates.values()):
        raise ValueError("Entry fields cannot be null")

    for source_path in sorted(data_dir.glob("????-??.json")):
        data = json.loads(source_path.read_text(encoding="utf-8"))

        for index, record in enumerate(data["entries"]):
            if record["id"] != entry_id:
                continue

            updated = EntryInput.model_validate(
                {
                    **{key: value for key, value in record.items() if key != "id"},
                    **updates,
                }
            )

            updated_record = {
                "id": entry_id,
                **updated.model_dump(mode="json"),
            }

            destination_path = data_dir / f"{updated.date:%Y-%m}.json"

            if source_path == destination_path:
                data["entries"][index] = updated_record
                _write_month(source_path, data)
            else:
                if destination_path.exists():
                    destination_data = json.loads(
                        destination_path.read_text(encoding="utf-8")
                    )
                else:
                    destination_data = {"entries": []}

                destination_data["entries"].append(updated_record)

                # Write the destination first to avoid losing the entry.
                _write_month(destination_path, destination_data)

                data["entries"].pop(index)
                _write_month(source_path, data)

            return updated_record

    raise KeyError(f"Entry not found: {entry_id}")


def delete_entry(
    entry_id: str,
    *,
    data_dir: Path = Path("data"),
) -> dict:
    for path in sorted(data_dir.glob("????-??.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        for index, record in enumerate(data["entries"]):
            if record["id"] != entry_id:
                continue

            deleted = data["entries"].pop(index)

            _write_month(path, data)

            return deleted

    raise KeyError(f"Entry not found: {entry_id}")


def get_calorie_summary(
    *,
    start_date: Date,
    end_date: Date,
    data_dir: Path = Path("data"),
) -> dict:
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")

    days = (end_date - start_date).days + 1

    daily_totals = {}

    for offset in range(days):
        day = start_date + timedelta(days=offset)
        daily_totals[day.isoformat()] = 0

    by_category = {}

    entries = list_entries(
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
    )

    for entry in entries:
        calories = entry["calories"]
        category = entry["category"]

        daily_totals[entry["date"]] += calories

        by_category[category] = by_category.get(category, 0) + calories

    total_calories = sum(daily_totals.values())

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_calories": total_calories,
        "average_daily_calories": round(total_calories / days, 2),
        "daily_totals": daily_totals,
        "by_category": dict(sorted(by_category.items())),
    }
