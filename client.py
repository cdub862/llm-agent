from openai import OpenAI

from llm_agent.config import MODEL_ID

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)


add_entry_tool = {
    "type": "function",
    "function": {
        "name": "add_entry",
        "description": "Log a food entry with its calories.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date in YYYY-MM-DD format.",
                },
                "description": {
                    "type": "string",
                    "minLength": 1,
                },
                "calories": {
                    "type": "integer",
                    "minimum": 0,
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "protein",
                        "grains",
                        "vegetables",
                        "fruits",
                        "dairy",
                        "fast_food",
                        "snacks",
                        "sweets",
                        "beverages",
                        "other",
                    ],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "date",
                "description",
                "calories",
                "category",
                "tags",
            ],
            "additionalProperties": False,
        },
    },
}


response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": "Add 300 calories: tortilla cheese salami",
        }
    ],
    tools=[add_entry_tool],
    max_completion_tokens=200,
)

print("Finish reason:", response.choices[0].finish_reason)
print("Tool calls:", response.choices[0].message.tool_calls)
