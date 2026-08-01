"""Generate query-processing candidates with DeepSeek V4 Flash only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
SYSTEM_PROMPT = """Bạn tạo dữ liệu huấn luyện cho bộ phân tích truy vấn pháp luật Việt Nam.
Chỉ trả về một JSON object có key "samples". Không markdown, không giải thích.

Mỗi sample phải có schema:
{
  "input": {
    "conversation_history": [{"role":"user|assistant","content":"..."}],
    "current_query": "..."
  },
  "output": {
    "status": "ready|needs_clarification",
    "standalone_query": "..." | null,
    "plan_type": "single|parallel|comparison|multi_hop" | null,
    "subqueries": [
      {
        "id": "q1",
        "query": "...",
        "intent": "factual|definition|validity|hierarchy",
        "depends_on": []
      }
    ],
    "clarification_question": "..." | null
  }
}

Quy tắc:
- Rewrite bằng lịch sử nhưng không thêm dữ kiện không có.
- Thiếu đối tượng/thời điểm/văn bản đến mức không thể truy xuất: needs_clarification.
- ready: clarification_question=null.
- needs_clarification: standalone_query=null, plan_type=null, subqueries=[].
- single: đúng 1 subquery, depends_on=[].
- parallel/comparison: >=2 subquery độc lập, mọi depends_on=[].
- multi_hop: >=2 subquery; subquery sau phụ thuộc subquery trước.
- comparison và multi_hop chỉ là plan_type, không phải intent.
- Mỗi subquery chỉ có một intent.
- Gán leaf intent theo NGHĨA, không chỉ theo từ khóa:
  * definition: chỉ hỏi định nghĩa/ý nghĩa của một thuật ngữ pháp lý, ví dụ
    "Vốn điều lệ là gì?".
  * factual: hỏi quy tắc, quyền, nghĩa vụ, điều kiện, thủ tục, số lượng, tỷ lệ,
    thời hạn hoặc một hành vi có được phép hay không.
  * validity: chỉ hỏi hiệu lực, thời điểm áp dụng, sửa đổi, bãi bỏ hoặc thay thế
    của văn bản/quy định. "Có được thực hiện hành vi X không?" là factual,
    không phải validity.
  * hierarchy: hỏi Điều/Khoản/Điểm thuộc cấu trúc nào, văn bản hướng dẫn nào,
    hoặc một quy định dẫn chiếu tới đâu.
- "Điều kiện/quyền/nghĩa vụ/thủ tục ... là gì?" là factual, không phải definition.
- "Văn bản/quy định nào bị thay thế hoặc thay thế văn bản nào?" là validity.
- Chỉ dùng multi_hop khi không thể thực thi subquery sau nếu chưa có kết quả
  của subquery trước. Nếu các subquery đã tự chứa đủ đối tượng để chạy độc lập,
  phải dùng parallel hoặc comparison và depends_on=[].
- Query của subquery có depends_on phải nhắc rõ kết quả trước bằng q1/q2 hoặc
  cụm như "điều đó", "đối tượng được xác định ở q1", "kết quả q1".
- Không tạo depends_on chỉ vì các câu cùng chủ đề.
- Dependency chỉ được trỏ tới ID đứng trước.
- Viết tiếng Việt tự nhiên, đa dạng, ưu tiên lĩnh vực doanh nghiệp nhưng không cần
  trích dẫn hay trả lời nội dung luật.
"""


def _load_validator(root: Path):
    path = root / "scripts" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("query_dataset_contract", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.validate_samples


def _normalized_key(sample: dict[str, Any]) -> str:
    history = sample["input"]["conversation_history"]
    text = " ".join(item["content"] for item in history)
    text += " " + sample["input"]["current_query"]
    return re.sub(r"\W+", " ", text.lower()).strip()


def _request_prompt(batch_size: int, batch_index: int) -> str:
    topics = (
        "thành lập và đăng ký doanh nghiệp",
        "công ty trách nhiệm hữu hạn",
        "công ty cổ phần và cổ đông",
        "công ty hợp danh và doanh nghiệp tư nhân",
        "vốn, góp vốn và chuyển nhượng phần vốn",
        "quản trị, họp và biểu quyết",
        "người đại diện, chi nhánh và văn phòng đại diện",
        "tổ chức lại, sáp nhập, chia tách và chuyển đổi",
        "giải thể, phá sản và chấm dứt hoạt động",
        "doanh nghiệp nhà nước và doanh nghiệp xã hội",
        "trái phiếu doanh nghiệp",
        "hiệu lực, sửa đổi và hệ thống văn bản",
    )
    styles = (
        "ngôn ngữ pháp lý trang trọng",
        "cách hỏi đời thường",
        "hội thoại follow-up ngắn",
        "câu dài có nhiều chi tiết nhiễu",
        "câu viết tắt hoặc thiếu dấu nhẹ nhưng vẫn hiểu được",
        "yêu cầu đối chiếu hai đối tượng hoặc hai thời điểm",
    )
    topic = topics[(batch_index - 1) % len(topics)]
    style = styles[((batch_index - 1) // len(topics)) % len(styles)]
    quota = ""
    if batch_size >= 10:
        quota = (
            " Trong mỗi 10 mẫu: 4 single, 2 parallel, 2 comparison, "
            "2 needs_clarification. Trong các subquery ready phải "
            "có ít nhất 2 definition, 2 validity, 2 hierarchy; phần còn lại "
            "factual. Có ít nhất 4 mẫu chứa conversation_history."
        )
    return (
        f"Tạo đúng {batch_size} samples mới. Batch seed {batch_index}. "
        "CHỈ được dùng single, parallel, comparison hoặc needs_clarification. "
        "CẤM tạo plan_type=multi_hop và CẤM mọi depends_on khác []. "
        "Có cả câu độc lập lẫn hội thoại 2-5 lượt. "
        f"Chủ đề ưu tiên của batch: {topic}. Phong cách ưu tiên: {style}. "
        f"Không lặp lại câu trong cùng batch.{quota}"
    )


def _call(
    api_key: str, batch_size: int, batch_index: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=120.0,
        max_retries=2,
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _request_prompt(batch_size, batch_index)},
        ],
        stream=False,
        temperature=0.9,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content or ""
    payload = json.loads(content)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Response does not contain a samples list")
    usage = response.usage
    return samples, {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def generate(
    root: Path,
    *,
    target_count: int,
    batch_size: int,
    max_calls: int,
    workers: int,
    run_id: str | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    if target_count < 1 or target_count > 10_000:
        raise ValueError("target_count must be between 1 and 10000")
    if batch_size < 1 or batch_size > 20:
        raise ValueError("batch_size must be between 1 and 20")
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")

    synthetic_root = root / "synthetic"
    synthetic_root.mkdir(parents=True, exist_ok=True)
    lock_path = synthetic_root / ".generation.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another generator owns lock: {lock_path}. "
            "Do not start a concurrent API run."
        ) from exc
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)

    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", resolved_run_id):
        lock_path.unlink(missing_ok=True)
        raise ValueError("run_id may contain only letters, digits, underscore, hyphen")
    output = synthetic_root / "runs" / resolved_run_id
    if output.exists():
        lock_path.unlink(missing_ok=True)
        raise FileExistsError(f"Run directory already exists: {output}")
    output.mkdir(parents=True)
    accepted_path = output / "accepted.jsonl"
    rejected_path = output / "rejected.jsonl"
    raw_path = output / "raw_batches.jsonl"
    for path in (accepted_path, rejected_path, raw_path):
        path.write_text("", encoding="utf-8")

    validate_samples = _load_validator(root)
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    calls = 0
    rejected = 0
    api_error_count = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    try:
        while len(accepted) < target_count and calls < max_calls:
            remaining_call_budget = max_calls - calls
            remaining_samples = target_count - len(accepted)
            estimated_calls_needed = max(
                1, (remaining_samples + batch_size - 1) // batch_size
            )
            round_size = min(
                workers, remaining_call_budget, estimated_calls_needed
            )
            batch_indexes = list(range(calls + 1, calls + round_size + 1))
            calls += round_size
            results: list[
                tuple[
                    int,
                    list[dict[str, Any]] | None,
                    dict[str, int] | None,
                    Exception | None,
                ]
            ] = []
            with ThreadPoolExecutor(max_workers=round_size) as executor:
                futures = {
                    executor.submit(_call, api_key, batch_size, batch_index): batch_index
                    for batch_index in batch_indexes
                }
                for future in as_completed(futures):
                    batch_index = futures[future]
                    try:
                        rows, call_usage = future.result()
                        results.append((batch_index, rows, call_usage, None))
                    except Exception as exc:
                        results.append((batch_index, None, None, exc))

            for batch_index, rows, call_usage, error in sorted(results):
                if error is not None:
                    api_error_count += 1
                    _append_jsonl(
                        rejected_path,
                        {
                            "batch": batch_index,
                            "reason": "api_or_parse_error",
                            "error": str(error),
                        },
                    )
                    continue
                assert rows is not None and call_usage is not None
                usage["prompt_tokens"] += call_usage["prompt_tokens"]
                usage["completion_tokens"] += call_usage["completion_tokens"]
                _append_jsonl(
                    raw_path,
                    {
                        "batch": batch_index,
                        "model": MODEL,
                        "samples": rows,
                        "usage": call_usage,
                    },
                )

                for row_index, row in enumerate(rows):
                    sample = {
                        "sample_id": (
                            f"{resolved_run_id}_ds4f_{batch_index:05d}_{row_index:02d}"
                        ),
                        **row,
                        "metadata": {
                            "generator_model": MODEL,
                            "thinking": "disabled",
                            "review_status": "generated_review_required",
                            "batch": batch_index,
                            "run_id": resolved_run_id,
                        },
                    }
                    try:
                        validate_samples([sample])
                        sample_output = sample["output"]
                        if sample_output["plan_type"] == "multi_hop":
                            raise ValueError("multi_hop_excluded_from_this_run")
                        if any(
                            subquery["depends_on"]
                            for subquery in sample_output["subqueries"]
                        ):
                            raise ValueError("dependencies_excluded_from_this_run")
                        key = _normalized_key(sample)
                        if key in seen:
                            raise ValueError("duplicate_input")
                        seen.add(key)
                    except Exception as exc:
                        rejected += 1
                        _append_jsonl(
                            rejected_path,
                            {"sample": sample, "reason": str(exc)},
                        )
                        continue
                    accepted.append(sample)
                    _append_jsonl(accepted_path, sample)
                    if len(accepted) >= target_count:
                        break
                if len(accepted) >= target_count:
                    break

        summary = {
            "schema_version": "query-processing-synthetic-v1",
            "run_id": resolved_run_id,
            "run_directory": str(output),
            "model": MODEL,
            "thinking": "disabled",
            "target_count": target_count,
            "accepted_count": len(accepted),
            "rejected_count": rejected,
            "api_error_count": api_error_count,
            "api_call_count": calls,
            "workers": workers,
            "usage": usage,
            "review_status": "generated_review_required",
        }
        (output / "generation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-calls", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    summary = generate(
        args.root.resolve(),
        target_count=args.target_count,
        batch_size=args.batch_size,
        max_calls=args.max_calls,
        workers=args.workers,
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["accepted_count"] == summary["target_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
