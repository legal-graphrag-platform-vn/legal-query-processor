"""Load the local LoRA adapter and test one query interactively."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT_DIR / "models"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = """Bạn là bộ xử lý truy vấn cho hệ thống Legal GraphRAG tiếng Việt.
Hãy dùng lịch sử hội thoại và câu hỏi mới nhất để trả về đúng một JSON object, không kèm giải thích hay Markdown.

JSON phải có đúng các trường:
- status: "ready" hoặc "needs_clarification".
- standalone_query: string hoặc null.
- plan_type: "single", "parallel", "comparison", "multi_hop" hoặc null.
- subqueries: danh sách object gồm id, query, intent, depends_on.
- clarification_question: string hoặc null.

Mỗi intent của subquery chỉ được là "factual", "definition", "validity" hoặc "hierarchy".
depends_on chỉ chứa id của các subquery đứng trước.
Nếu status="needs_clarification": standalone_query và plan_type phải là null, subqueries phải rỗng, clarification_question phải là một câu hỏi làm rõ.
Nếu status="ready": standalone_query và plan_type phải khác null, subqueries phải có ít nhất một phần tử, clarification_question phải là null."""


def load_model():
    """Load Qwen 4-bit and attach the adapter stored in models/."""
    assert ADAPTER_DIR.is_dir(), f"Không tìm thấy: {ADAPTER_DIR}"
    assert (ADAPTER_DIR / "adapter_config.json").exists(), (
        f"Không tìm thấy adapter_config.json trong {ADAPTER_DIR}"
    )
    assert torch.cuda.is_available(), (
        "Script dùng bitsandbytes 4-bit và cần GPU CUDA."
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )
    model.set_adapter("default")
    model.eval()

    print("Load model và LoRA adapter thành công")
    print("Adapter:", ADAPTER_DIR)
    print("Device:", model.device)
    return model, tokenizer


def process_query(
    current_query: str,
    model,
    tokenizer,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append(
        {
            "role": "user",
            "content": current_query.strip(),
        }
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(
        outputs[0, input_length:],
        skip_special_tokens=True,
    ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "parse_error": True,
            "raw_output": text,
        }


def main() -> None:
    model, tokenizer = load_model()
    query = input("Nhập câu hỏi: ")
    result = process_query(query, model, tokenizer)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
