"""Build query-processing SFT data from the reviewed retrieval query set."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


LEAF_INTENTS = {
    "factual",
    "validity",
    "hierarchy",
    "definition",
}
PLAN_TYPES = {"single", "parallel", "comparison", "multi_hop"}

COMPLEX_DECOMPOSITIONS: dict[str, list[dict[str, str]]] = {
    "multi_hop_01": [
        {"query": "Điều 38 dẫn chiếu đến nội dung nào tại Điều 41?", "intent": "hierarchy", "depends_on": []},
        {"query": "Khoản nào trong điều được xác định ở q1 quy định tên gây nhầm lẫn?", "intent": "hierarchy", "depends_on": ["q1"]},
    ],
    "multi_hop_02": [
        {"query": "Khoản 3 Điều 145 quy định gì về cuộc họp lần thứ ba?", "intent": "factual", "depends_on": []},
        {"query": "Điều kiện tiến hành cuộc họp lần thứ hai được dẫn chiếu từ kết quả q1 là gì?", "intent": "factual", "depends_on": ["q1"]},
        {"query": "Điều kiện tiến hành cuộc họp lần thứ nhất được dẫn chiếu từ kết quả q2 là gì?", "intent": "factual", "depends_on": ["q2"]},
    ],
    "multi_hop_03": [
        {"query": "Điều 57 dẫn chiếu quyền yêu cầu của thành viên tại Điều nào?", "intent": "hierarchy", "depends_on": []},
        {"query": "Khoản nào của điều được xác định ở q1 quy định quyền của nhóm thành viên đủ điều kiện?", "intent": "hierarchy", "depends_on": ["q1"]},
    ],
    "multi_hop_04": [
        {"query": "Khoản 2 Điều 68 dẫn chiếu điều nào về chuyển nhượng phần vốn góp?", "intent": "hierarchy", "depends_on": []},
        {"query": "Khoản nào của điều được xác định ở q1 quy định trình tự chào bán phần vốn góp?", "intent": "hierarchy", "depends_on": ["q1"]},
    ],
    "multi_hop_05": [
        {"query": "Khoản 1 Điều 52 dẫn chiếu những trường hợp nào tại Điều 53?", "intent": "hierarchy", "depends_on": []},
        {"query": "Điều được xác định ở q1 quy định cách xử lý phần vốn góp trong từng trường hợp như thế nào?", "intent": "factual", "depends_on": ["q1"]},
    ],
    "comparison_01": [
        {"query": "Quy định về quyền thành lập doanh nghiệp tại năm 2020 là gì?", "intent": "factual", "depends_on": []},
        {"query": "Quy định về quyền thành lập doanh nghiệp tại năm 2021 là gì?", "intent": "factual", "depends_on": []},
    ],
    "comparison_02": [
        {"query": "Quy định về vốn điều lệ trước năm 2021 là gì?", "intent": "factual", "depends_on": []},
        {"query": "Quy định về vốn điều lệ sau năm 2021 là gì?", "intent": "factual", "depends_on": []},
    ],
    "comparison_03": [
        {"query": "Điều kiện giải thể doanh nghiệp tại năm 2021 là gì?", "intent": "factual", "depends_on": []},
        {"query": "Điều kiện giải thể doanh nghiệp tại năm 2022 là gì?", "intent": "factual", "depends_on": []},
    ],
    "comparison_04": [
        {"query": "Quy định về công ty trách nhiệm hữu hạn trước năm 2021 là gì?", "intent": "factual", "depends_on": []},
        {"query": "Quy định về công ty trách nhiệm hữu hạn sau năm 2021 là gì?", "intent": "factual", "depends_on": []},
    ],
    "comparison_05": [
        {"query": "Thủ tục đăng ký doanh nghiệp tại năm 2021 là gì?", "intent": "factual", "depends_on": []},
        {"query": "Thủ tục đăng ký doanh nghiệp tại năm 2023 là gì?", "intent": "factual", "depends_on": []},
    ],
    "hierarchy_04": [
        {"query": "Khoản quy định nghĩa vụ đăng ký thay đổi thuộc Điều nào?", "intent": "hierarchy", "depends_on": []},
        {"query": "Khoản quy định trường hợp thay đổi theo quyết định của Tòa án thuộc Điều nào?", "intent": "hierarchy", "depends_on": []},
    ],
}

PLAN_OVERRIDES = {"hierarchy_04": "parallel"}


def scoped_query(query: str) -> str:
    if "Luật Doanh nghiệp 2020" in query:
        return query
    return f"Theo Luật Doanh nghiệp 2020, {query[0].lower()}{query[1:]}"


def make_samples(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for case in cases:
        query_id = case["query_id"]
        query = case["query"]
        intent = case["intent"]
        if intent not in LEAF_INTENTS | {"comparison", "multi_hop"}:
            raise ValueError(f"Unsupported intent: {intent}")
        subqueries = COMPLEX_DECOMPOSITIONS.get(
            query_id, [{"query": query, "intent": intent, "depends_on": []}]
        )
        plan_type = PLAN_OVERRIDES.get(
            query_id,
            intent if intent in {"comparison", "multi_hop"} else "single",
        )

        samples.append(
            _sample(
                sample_id=f"{query_id}_direct",
                base_query_id=query_id,
                history=[],
                current_query=query,
                standalone_query=query,
                subqueries=subqueries,
                plan_type=plan_type,
                seed_query_intent=intent,
                variant="direct",
            )
        )
        standalone = scoped_query(query)
        samples.append(
            _sample(
                sample_id=f"{query_id}_history",
                base_query_id=query_id,
                history=[
                    {"role": "user", "content": query},
                    {
                        "role": "assistant",
                        "content": "Bạn muốn hỏi theo Luật Doanh nghiệp 2020 đúng không?",
                    },
                ],
                current_query="Đúng, theo luật đó.",
                standalone_query=standalone,
                subqueries=[
                    {**item, "query": scoped_query(item["query"])}
                    for item in subqueries
                ],
                plan_type=plan_type,
                seed_query_intent=intent,
                variant="history_resolution",
            )
        )
    validate_samples(samples)
    return samples


def _sample(
    *,
    sample_id: str,
    base_query_id: str,
    history: list[dict[str, str]],
    current_query: str,
    standalone_query: str,
    subqueries: list[dict[str, Any]],
    plan_type: str,
    seed_query_intent: str,
    variant: str,
) -> dict[str, Any]:
    numbered_subqueries = [
        {"id": f"q{index}", **subquery}
        for index, subquery in enumerate(subqueries, start=1)
    ]
    return {
        "sample_id": sample_id,
        "input": {
            "conversation_history": history,
            "current_query": current_query,
        },
        "output": {
            "status": "ready",
            "standalone_query": standalone_query,
            "plan_type": plan_type,
            "subqueries": numbered_subqueries,
            "clarification_question": None,
        },
        "metadata": {
            "base_query_id": base_query_id,
            "variant": variant,
            "seed_query_intent": seed_query_intent,
            "label_source": "approved_retrieval_pilot_plus_curated_transformation",
            "review_status": "generated_review_required",
        },
    }


def split_samples(
    samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    intent_by_base = {
        sample["metadata"]["base_query_id"]: sample["metadata"]["seed_query_intent"]
        for sample in samples
    }
    base_ids = sorted(intent_by_base)
    if len(base_ids) != 30:
        raise ValueError(f"Expected 30 base queries, got {len(base_ids)}")
    split_ids = {"train": set(), "validation": set(), "test": set()}
    holdout_index = 0
    for intent in sorted(set(intent_by_base.values())):
        intent_ids = [
            base_id for base_id in base_ids if intent_by_base[base_id] == intent
        ]
        train_count = int(len(intent_ids) * 0.8)
        split_ids["train"].update(intent_ids[:train_count])
        for base_id in intent_ids[train_count:]:
            split = "validation" if holdout_index % 2 == 0 else "test"
            split_ids[split].add(base_id)
            holdout_index += 1
    return {
        name: [
            sample
            for sample in samples
            if sample["metadata"]["base_query_id"] in ids
        ]
        for name, ids in split_ids.items()
    }


def validate_samples(samples: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id in ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        ids.add(sample_id)
        output = sample["output"]
        status = output["status"]
        if status not in {"ready", "needs_clarification"}:
            raise ValueError(f"Invalid status in {sample_id}")
        if status == "needs_clarification":
            if (
                output["standalone_query"] is not None
                or output["plan_type"] is not None
                or output["subqueries"]
                or not output["clarification_question"]
            ):
                raise ValueError(f"Invalid clarification output in {sample_id}")
            continue
        if output["clarification_question"] is not None:
            raise ValueError(f"Unexpected clarification question in {sample_id}")
        if not output["standalone_query"].strip():
            raise ValueError(f"Blank standalone query in {sample_id}")
        if not output["subqueries"]:
            raise ValueError(f"No subqueries in {sample_id}")
        for subquery in output["subqueries"]:
            if not subquery["query"].strip():
                raise ValueError(f"Blank subquery in {sample_id}")
            if subquery["intent"] not in LEAF_INTENTS:
                raise ValueError(f"Invalid subquery intent in {sample_id}")
        plan_type = output["plan_type"]
        if plan_type not in PLAN_TYPES:
            raise ValueError(f"Invalid plan type in {sample_id}")
        subquery_ids = [item["id"] for item in output["subqueries"]]
        if len(subquery_ids) != len(set(subquery_ids)):
            raise ValueError(f"Duplicate subquery ID in {sample_id}")
        seen: set[str] = set()
        for subquery in output["subqueries"]:
            dependencies = subquery["depends_on"]
            if any(dependency not in seen for dependency in dependencies):
                raise ValueError(f"Forward or unknown dependency in {sample_id}")
            if dependencies and not re.search(
                r"\b(q\d+|đó|này|được xác định|được tìm thấy|kết quả)\b",
                subquery["query"],
                re.IGNORECASE,
            ):
                raise ValueError(f"Dependency has no result reference in {sample_id}")
            seen.add(subquery["id"])
        if plan_type == "single" and (
            len(output["subqueries"]) != 1 or output["subqueries"][0]["depends_on"]
        ):
            raise ValueError(f"Invalid single plan in {sample_id}")
        if plan_type in {"parallel", "comparison"} and (
            len(output["subqueries"]) < 2
            or any(item["depends_on"] for item in output["subqueries"])
        ):
            raise ValueError(f"Invalid independent plan in {sample_id}")
        if plan_type == "multi_hop" and (
            len(output["subqueries"]) < 2
            or not any(item["depends_on"] for item in output["subqueries"])
        ):
            raise ValueError(f"Invalid multi-hop plan in {sample_id}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def build(root: Path, seed_path: Path) -> dict[str, Any]:
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    cases = seed["cases"]
    samples = make_samples(cases)
    splits = split_samples(samples)
    output = root / "data"
    staging = root / ".dataset-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for name, rows in splits.items():
            _write_jsonl(staging / f"{name}.jsonl", rows)
        subquery_intent_counts = {
            intent: sum(
                subquery["intent"] == intent
                for sample in samples
                for subquery in sample["output"]["subqueries"]
            )
            for intent in sorted(LEAF_INTENTS)
        }
        plan_type_counts = {
            plan_type: sum(
                sample["output"]["plan_type"] == plan_type for sample in samples
            )
            for plan_type in sorted(PLAN_TYPES)
        }
        summary = {
            "schema_version": "query-processing-sft-v1",
            "sample_count": len(samples),
            "base_query_count": len(cases),
            "split_counts": {
                name: len(rows) for name, rows in splits.items()
            },
            "subquery_intent_counts": subquery_intent_counts,
            "plan_type_counts": plan_type_counts,
            "split_seed_query_intent_counts": {
                name: {
                    intent: sum(
                        sample["metadata"]["seed_query_intent"] == intent
                        for sample in rows
                    )
                    for intent in sorted(LEAF_INTENTS | {"comparison", "multi_hop"})
                }
                for name, rows in splits.items()
            },
            "history_sample_count": sum(
                bool(sample["input"]["conversation_history"])
                for sample in samples
            ),
            "decomposed_sample_count": sum(
                sample["output"]["plan_type"] != "single"
                for sample in samples
            ),
            "review_status": "generated_review_required",
        }
        _write_json(staging / "dataset_summary.json", summary)
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("configs/evaluation/retrieval_pilot_l59_2020.json"),
    )
    args = parser.parse_args()
    summary = build(args.root.resolve(), args.seed.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
