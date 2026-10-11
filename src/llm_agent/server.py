import time
import uuid
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from jsonschema import ValidationError
from pydantic import BaseModel, Field

from llm_agent.config import MODEL_ID
from llm_agent.inference import generate
from llm_agent.tool_parser import parse_tool_call, to_openai_tool_call
from llm_agent.tool_validator import validate_tool_call

app = FastAPI()


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    max_completion_tokens: int = Field(default=100, ge=1, le=512)
    tools: list[dict] | None = None


@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    if request.model != MODEL_ID:
        raise HTTPException(status_code=404, detail="Model not found")

    today = datetime.now(ZoneInfo("America/Santo_Domingo")).date().isoformat()

    messages = [
        {
            "role": "system",
            "content": (
                f"Current date: {today}. "
                "When logging food, use the current date unless "
                "the user specifies another date. "
                "Resolve relative dates such as yesterday."
            ),
        },
        *[message.model_dump(exclude_unset=True) for message in request.messages],
    ]

    result = generate(
        messages=messages,
        max_new_tokens=request.max_completion_tokens,
        tools=request.tools,
    )

    parsed = None

    if request.tools and result["finish_reason"] == "stop":
        parsed = parse_tool_call(result["content"])

    if parsed is not None:
        try:
            validate_tool_call(parsed, request.tools)
        except (ValueError, ValidationError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Invalid model-generated tool call: {exc.message if isinstance(exc, ValidationError) else str(exc)}",
            ) from exc

        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [to_openai_tool_call(parsed)],
        }
        finish_reason = "tool_calls"
    else:
        message = {
            "role": "assistant",
            "content": result["content"],
        }
        finish_reason = result["finish_reason"]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": (result["prompt_tokens"] + result["completion_tokens"]),
        },
    }
