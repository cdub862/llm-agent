import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_agent.config import MODEL_ID
from llm_agent.inference import generate

app = FastAPI()


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    max_completion_tokens: int = Field(default=100, ge=1, le=512)


@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    if request.model != MODEL_ID:
        raise HTTPException(status_code=404, detail="Model not found")

    result = generate(
        messages=[message.model_dump() for message in request.messages],
        max_new_tokens=request.max_completion_tokens,
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["content"],
                },
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": (result["prompt_tokens"] + result["completion_tokens"]),
        },
    }
