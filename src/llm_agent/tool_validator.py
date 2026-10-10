from jsonschema import Draft202012Validator, FormatChecker


def validate_tool_call(
    tool_call: dict,
    tools: list[dict],
) -> dict:
    name = tool_call["name"]
    arguments = tool_call["arguments"]

    tool = next(
        (
            tool
            for tool in tools
            if tool.get("type") == "function"
            and tool.get("function", {}).get("name") == name
        ),
        None,
    )

    if tool is None:
        raise ValueError(f"Unknown tool: {name}")

    schema = tool["function"]["parameters"]

    Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    ).validate(arguments)

    return tool_call
