# LLM Agent

A minimal, fully local LLM inference and agent experiment built with Python, FastAPI, and an open-weight language model.

The project explores implementing an OpenAI-compatible API and server-side agent with tool calling and CRUD operations on structured data, without agent frameworks or hosted inference services.

## Tech Stack

- Python
- PyTorch and Hugging Face Transformers
- Qwen3-4B-Instruct (4-bit quantization)
- FastAPI and Pydantic
- OpenAI Python SDK
- uv

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support
- Sufficient VRAM for a 4-bit quantized model (developed on an RTX 3080 with 8 GB VRAM)
