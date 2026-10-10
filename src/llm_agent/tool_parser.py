import json
import re


def parse_tool_call(content: str) -> dict | None:
    match = re.fullmatch(
        r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*",
        content,
        flags=re.DOTALL,
    )

    if not match:
        return None

    data = json.loads(match.group(1))

    if not isinstance(data, dict):
        raise TypeError("Tool call must be a JSON object")

    if not isinstance(data.get("name"), str):
        raise TypeError("Tool call must contain a function name")

    if not data["name"].strip():
        raise ValueError("Tool function name cannot be empty")

    if not isinstance(data.get("arguments"), dict):
        raise TypeError("Tool arguments must be an object")

    return data
