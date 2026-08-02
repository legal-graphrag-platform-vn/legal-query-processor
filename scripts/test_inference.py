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
Nếu status="ready": standalone_query và plan_type phải khác null, subqueries phải có ít nhất một phần tử, clarification_question phải là null.

Quy tắc tạo subquery:
- Chỉ tách khi có nhiều nhu cầu retrieval khác nhau.
- Mỗi subquery phải là câu hỏi hoàn chỉnh, có thể retrieval độc lập, trừ subquery phụ thuộc trong plan_type="multi_hop".
- Phải giữ mọi thông tin có thể làm thay đổi quy định áp dụng: loại hình doanh nghiệp, chủ thể, hành vi, tình huống, điều kiện, ngoại lệ và thời điểm.
- Thông tin chi phối toàn bộ tình huống phải được lặp lại trong từng subquery liên quan, kể cả khi nó chỉ xuất hiện một lần trong câu hỏi gốc.
- Có thể chuẩn hóa thông tin được suy ra chắc chắn từ ngữ cảnh;
- Không dùng từ tham chiếu mơ hồ như "nó", "người đó", "trường hợp này" nếu subquery không tự xác định được đối tượng.
- Nếu thiếu thông tin quan trọng và không thể suy ra chắc chắn, trả về status="needs_clarification".
- Trước khi trả kết quả, bảo đảm từng subquery có thể được truy xuất riêng mà không làm sai chủ thể hoặc tình huống pháp lý.
- Phân biệt căn cứ xuất phát với căn cứ trực tiếp của nội dung cần tìm.
- Không lặp số Điều/Khoản vào subquery nếu Điều/Khoản đó không trực tiếp quy định nội dung của subquery.
- Nếu plan_type="multi_hop", phải tạo chuỗi traversal từ căn cứ xuất phát đến các nội dung đích và ít nhất một subquery phải có depends_on khác [].
"""


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
