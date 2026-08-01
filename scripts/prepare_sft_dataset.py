"""Convert the final query-processing dataset to conversational SFT JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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

OUTPUT_KEYS = (
    "status",
    "standalone_query",
    "plan_type",
    "subqueries",
    "clarification_question",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def convert_sample(sample: dict[str, Any]) -> dict[str, Any]:
    sample_input = sample["input"]
    history = sample_input["conversation_history"]
    current_query = sample_input["current_query"].strip()
    if not current_query:
        raise ValueError(f"Empty current_query: {sample['sample_id']}")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for message in history:
        role = message["role"]
        content = message["content"].strip()
        if role not in {"user", "assistant"} or not content:
            raise ValueError(f"Invalid history: {sample['sample_id']}")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "system" and messages[-1]["role"] != "assistant":
        raise ValueError(f"History must end with assistant: {sample['sample_id']}")
    messages.append({"role": "user", "content": current_query})

    output = sample["output"]
    target = {key: output[key] for key in OUTPUT_KEYS}
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                target, ensure_ascii=False, separators=(",", ":")
            ),
        }
    )
    return {"sample_id": sample["sample_id"], "messages": messages}


def prepare(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    history_turn_counts: Counter[int] = Counter()
    for split in ("train", "validation", "test"):
        samples = read_jsonl(source_dir / f"{split}.jsonl")
        converted = []
        for sample in samples:
            row = convert_sample(sample)
            converted.append(row)
            history_turn_counts[len(row["messages"]) - 3] += 1
        write_jsonl(output_dir / f"{split}.jsonl", converted)
        split_counts[split] = len(converted)

    summary = {
        "schema_version": "query-processing-conversational-sft-v1",
        "source_schema_version": "query-processing-final-dataset-v1",
        "split_counts": split_counts,
        "sample_count": sum(split_counts.values()),
        "history_turn_counts": {
            str(key): value for key, value in sorted(history_turn_counts.items())
        },
        "training_fields": ["messages"],
        "traceability_fields_not_used_for_loss": ["sample_id"],
        "target": "assistant JSON only",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=root / "final")
    parser.add_argument("--output-dir", type=Path, default=root / "sft")
    args = parser.parse_args()
    summary = prepare(args.source_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
