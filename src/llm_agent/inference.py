import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from llm_agent.config import MODEL_ID

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)

model.eval()


def generate(messages: list[dict], max_new_tokens: int = 100) -> dict:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][prompt_tokens:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    finished = bool(
        len(generated_tokens) and eos_ids and generated_tokens[-1].item() in eos_ids
    )

    return {
        "content": response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(generated_tokens),
        "finish_reason": "stop" if finished else "length",
    }
