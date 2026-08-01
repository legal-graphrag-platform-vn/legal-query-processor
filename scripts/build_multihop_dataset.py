"""Build deterministic multi-hop query-planning samples without API calls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


TOPICS: tuple[dict[str, str], ...] = (
    {"topic": "đăng ký thành lập công ty cổ phần", "actor": "người thành lập", "action": "nộp hồ sơ đăng ký công ty cổ phần"},
    {"topic": "đăng ký công ty trách nhiệm hữu hạn một thành viên", "actor": "chủ sở hữu công ty", "action": "đăng ký công ty trách nhiệm hữu hạn một thành viên"},
    {"topic": "thay đổi người đại diện theo pháp luật", "actor": "doanh nghiệp", "action": "đăng ký thay đổi người đại diện theo pháp luật"},
    {"topic": "thay đổi vốn điều lệ", "actor": "doanh nghiệp", "action": "đăng ký thay đổi vốn điều lệ"},
    {"topic": "chuyển nhượng phần vốn góp", "actor": "thành viên công ty", "action": "chuyển nhượng phần vốn góp"},
    {"topic": "chào bán cổ phần", "actor": "công ty cổ phần", "action": "chào bán cổ phần"},
    {"topic": "triệu tập họp Đại hội đồng cổ đông", "actor": "Hội đồng quản trị", "action": "triệu tập họp Đại hội đồng cổ đông"},
    {"topic": "thông qua nghị quyết Hội đồng thành viên", "actor": "Hội đồng thành viên", "action": "thông qua nghị quyết"},
    {"topic": "thành lập chi nhánh", "actor": "doanh nghiệp", "action": "đăng ký hoạt động chi nhánh"},
    {"topic": "tạm ngừng kinh doanh", "actor": "doanh nghiệp", "action": "thông báo tạm ngừng kinh doanh"},
    {"topic": "giải thể doanh nghiệp", "actor": "doanh nghiệp", "action": "thực hiện thủ tục giải thể"},
    {"topic": "sáp nhập doanh nghiệp", "actor": "các công ty tham gia sáp nhập", "action": "thực hiện sáp nhập"},
    {"topic": "chia công ty", "actor": "công ty bị chia", "action": "thực hiện thủ tục chia công ty"},
    {"topic": "chuyển đổi loại hình doanh nghiệp", "actor": "doanh nghiệp", "action": "chuyển đổi loại hình"},
    {"topic": "góp vốn bằng tài sản", "actor": "thành viên góp vốn", "action": "góp vốn bằng tài sản"},
    {"topic": "định giá tài sản góp vốn", "actor": "các thành viên sáng lập", "action": "định giá tài sản góp vốn"},
    {"topic": "mua lại phần vốn góp", "actor": "công ty", "action": "mua lại phần vốn góp"},
    {"topic": "phát hành trái phiếu riêng lẻ", "actor": "doanh nghiệp phát hành", "action": "phát hành trái phiếu riêng lẻ"},
    {"topic": "công bố thông tin doanh nghiệp", "actor": "doanh nghiệp", "action": "công bố thông tin"},
    {"topic": "lưu giữ tài liệu doanh nghiệp", "actor": "doanh nghiệp", "action": "lưu giữ tài liệu"},
    {"topic": "thực hiện quyền của cổ đông", "actor": "cổ đông", "action": "thực hiện quyền của cổ đông"},
    {"topic": "yêu cầu triệu tập họp", "actor": "nhóm thành viên", "action": "yêu cầu triệu tập họp"},
    {"topic": "xử lý phần vốn góp khi thành viên chết", "actor": "công ty và người thừa kế", "action": "xử lý phần vốn góp"},
    {"topic": "chấm dứt hoạt động chi nhánh", "actor": "doanh nghiệp", "action": "chấm dứt hoạt động chi nhánh"},
)


def _q(
    query: str, intent: str, depends_on: list[str] | None = None
) -> dict[str, Any]:
    return {"query": query, "intent": intent, "depends_on": depends_on or []}


def _guidance(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Văn bản nào hướng dẫn {topic['topic']}, và trong văn bản đó "
        f"{topic['actor']} phải {topic['action']} theo thủ tục nào?"
    )
    return original, [
        _q(f"Văn bản nào hướng dẫn {topic['topic']}?", "hierarchy"),
        _q(
            f"Trong văn bản được xác định ở q1, thủ tục để "
            f"{topic['actor']} {topic['action']} là gì?",
            "factual",
            ["q1"],
        ),
    ]


def _authority(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Cơ quan nào tiếp nhận việc {topic['action']}, và cơ quan đó phải "
        "xử lý yêu cầu trong thời hạn nào?"
    )
    return original, [
        _q(f"Cơ quan nào tiếp nhận việc {topic['action']}?", "factual"),
        _q(
            "Cơ quan được xác định ở q1 phải xử lý yêu cầu trong thời hạn nào?",
            "factual",
            ["q1"],
        ),
    ]


def _responsible_subject(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Chủ thể nào chịu trách nhiệm chính về {topic['topic']}, và chủ thể đó "
        "phải thực hiện nghĩa vụ cụ thể nào?"
    )
    return original, [
        _q(f"Chủ thể nào chịu trách nhiệm chính về {topic['topic']}?", "factual"),
        _q(
            "Chủ thể được xác định ở q1 phải thực hiện nghĩa vụ cụ thể nào?",
            "factual",
            ["q1"],
        ),
    ]


def _applicable_document(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Văn bản nào điều chỉnh {topic['topic']}, và quy định trong văn bản đó "
        f"yêu cầu {topic['actor']} làm gì?"
    )
    return original, [
        _q(f"Văn bản nào điều chỉnh {topic['topic']}?", "hierarchy"),
        _q(
            f"Quy định trong văn bản được xác định ở q1 yêu cầu "
            f"{topic['actor']} làm gì?",
            "factual",
            ["q1"],
        ),
    ]


def _replacement(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Văn bản hiện hành nào thay thế quy định cũ về {topic['topic']}, "
        "và văn bản đó có hiệu lực từ thời điểm nào?"
    )
    return original, [
        _q(
            f"Văn bản hiện hành nào thay thế quy định cũ về {topic['topic']}?",
            "validity",
        ),
        _q(
            "Văn bản được xác định ở q1 có hiệu lực từ thời điểm nào?",
            "validity",
            ["q1"],
        ),
    ]


def _classification(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Việc {topic['action']} thuộc loại thủ tục pháp lý nào, và thủ tục đó "
        "cần những thành phần hồ sơ gì?"
    )
    return original, [
        _q(f"Việc {topic['action']} thuộc loại thủ tục pháp lý nào?", "definition"),
        _q(
            "Thủ tục được xác định ở q1 cần những thành phần hồ sơ gì?",
            "factual",
            ["q1"],
        ),
    ]


def _document_article_rule(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Hãy xác định văn bản điều chỉnh {topic['topic']}, tìm điều khoản liên "
        f"quan trong văn bản đó, rồi cho biết {topic['actor']} phải làm gì."
    )
    return original, [
        _q(f"Văn bản nào điều chỉnh {topic['topic']}?", "hierarchy"),
        _q(
            "Điều khoản nào trong văn bản được xác định ở q1 điều chỉnh vấn đề này?",
            "hierarchy",
            ["q1"],
        ),
        _q(
            f"Điều khoản được xác định ở q2 yêu cầu {topic['actor']} làm gì?",
            "factual",
            ["q2"],
        ),
    ]


def _authority_document_deadline(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Đối với việc {topic['action']}, hãy xác định cơ quan có thẩm quyền, "
        "văn bản mà cơ quan đó áp dụng và thời hạn xử lý theo văn bản ấy."
    )
    return original, [
        _q(f"Cơ quan nào có thẩm quyền xử lý việc {topic['action']}?", "factual"),
        _q(
            "Văn bản nào được cơ quan xác định ở q1 áp dụng cho thủ tục này?",
            "hierarchy",
            ["q1"],
        ),
        _q(
            "Văn bản được xác định ở q2 quy định thời hạn xử lý là bao lâu?",
            "factual",
            ["q2"],
        ),
    ]


def _condition_consequence(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Điều kiện nào cho phép {topic['actor']} {topic['action']}, và nếu không "
        "đáp ứng điều kiện đó thì hậu quả pháp lý là gì?"
    )
    return original, [
        _q(
            f"Điều kiện nào cho phép {topic['actor']} {topic['action']}?",
            "factual",
        ),
        _q(
            "Nếu điều kiện được xác định ở q1 không được đáp ứng thì hậu quả pháp lý là gì?",
            "factual",
            ["q1"],
        ),
    ]


def _definition_rule_exception(topic: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    original = (
        f"Khái niệm pháp lý nào bao quát {topic['topic']}, quy tắc áp dụng cho "
        "khái niệm đó là gì và có ngoại lệ nào không?"
    )
    return original, [
        _q(f"Khái niệm pháp lý nào bao quát {topic['topic']}?", "definition"),
        _q(
            "Quy tắc nào áp dụng cho khái niệm được xác định ở q1?",
            "factual",
            ["q1"],
        ),
        _q(
            "Quy tắc được xác định ở q2 có ngoại lệ nào không?",
            "factual",
            ["q2"],
        ),
    ]


BLUEPRINTS: tuple[
    tuple[str, Callable[[dict[str, str]], tuple[str, list[dict[str, Any]]]]],
    ...,
] = (
    ("guidance_to_procedure", _guidance),
    ("authority_to_deadline", _authority),
    ("subject_to_obligation", _responsible_subject),
    ("document_to_rule", _applicable_document),
    ("replacement_to_effective_date", _replacement),
    ("classification_to_documents", _classification),
    ("document_article_rule", _document_article_rule),
    ("authority_document_deadline", _authority_document_deadline),
    ("condition_to_consequence", _condition_consequence),
    ("definition_rule_exception", _definition_rule_exception),
)


def _variant(
    original: str, topic: dict[str, str], variant_index: int
) -> tuple[list[dict[str, str]], str]:
    if variant_index == 0:
        return [], original
    if variant_index == 1:
        return [
            {"role": "user", "content": f"Tôi đang tìm hiểu về {topic['topic']}."},
            {
                "role": "assistant",
                "content": f"Bạn muốn hỏi đầy đủ như sau phải không: {original}",
            },
        ], "Đúng, đó là câu hỏi của tôi."
    if variant_index == 2:
        return [
            {"role": "user", "content": f"Trường hợp của tôi liên quan đến {topic['topic']}."},
            {
                "role": "assistant",
                "content": (
                    f"Với chủ thể là {topic['actor']}, bạn có muốn hỏi: {original}"
                ),
            },
        ], "Đúng. Hãy giữ nguyên đầy đủ câu hỏi đó."
    if variant_index == 3:
        return [
            {"role": "user", "content": original},
            {"role": "assistant", "content": "Bạn muốn tách câu hỏi này thành các bước phụ thuộc nhau?"},
        ], "Đúng, bước sau phải dùng kết quả của bước trước."
    return [
        {"role": "user", "content": f"Tôi cần xử lý việc {topic['action']}."},
        {"role": "assistant", "content": "Bạn đã biết căn cứ hoặc chủ thể có thẩm quyền chưa?"},
        {"role": "user", "content": "Chưa, tôi muốn xác định từ đầu."},
        {
            "role": "assistant",
            "content": f"Bạn muốn hỏi đầy đủ như sau phải không: {original}",
        },
    ], "Đúng, đó là câu hỏi của tôi."


def build_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for blueprint_name, builder in BLUEPRINTS:
        for topic_index, topic in enumerate(TOPICS):
            original, raw_subqueries = builder(topic)
            base_case_id = f"{blueprint_name}_{topic_index:02d}"
            for variant_index in range(5):
                history, current_query = _variant(original, topic, variant_index)
                subqueries = [
                    {"id": f"q{index}", **subquery}
                    for index, subquery in enumerate(raw_subqueries, start=1)
                ]
                samples.append(
                    {
                        "sample_id": f"mh_{base_case_id}_v{variant_index}",
                        "input": {
                            "conversation_history": history,
                            "current_query": current_query,
                        },
                        "output": {
                            "status": "ready",
                            "standalone_query": original,
                            "plan_type": "multi_hop",
                            "subqueries": subqueries,
                            "clarification_question": None,
                        },
                        "metadata": {
                            "base_case_id": base_case_id,
                            "blueprint": blueprint_name,
                            "topic": topic["topic"],
                            "variant": variant_index,
                            "generator": "deterministic_dependency_blueprint_v1",
                            "review_status": "generated_review_required",
                        },
                    }
                )
    return samples


def split_samples(
    samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    split_by_base: dict[str, str] = {}
    for blueprint_index, (blueprint_name, _) in enumerate(BLUEPRINTS):
        base_ids = sorted(
            {
                sample["metadata"]["base_case_id"]
                for sample in samples
                if sample["metadata"]["blueprint"] == blueprint_name
            }
        )
        train_count = 20 if blueprint_index < 2 else 19
        split_by_base.update({base_id: "train" for base_id in base_ids[:train_count]})
        for offset, base_id in enumerate(base_ids[train_count:]):
            split_by_base[base_id] = (
                "validation" if (offset + blueprint_index) % 2 == 0 else "test"
            )
    return {
        split: [
            sample
            for sample in samples
            if split_by_base[sample["metadata"]["base_case_id"]] == split
        ]
        for split in ("train", "validation", "test")
    }


def _load_validator(root: Path):
    path = root / "scripts" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("query_dataset_contract_mh", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.validate_samples


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def build(root: Path) -> dict[str, Any]:
    samples = build_samples()
    _load_validator(root)(samples)
    splits = split_samples(samples)
    output = root / "multihop"
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_jsonl(output / f"{split}.jsonl", rows)
    _write_jsonl(output / "all.jsonl", samples)

    blueprint_counts = Counter(
        sample["metadata"]["blueprint"] for sample in samples
    )
    hop_counts = Counter(
        len(sample["output"]["subqueries"]) for sample in samples
    )
    summary = {
        "schema_version": "query-processing-multihop-v1",
        "sample_count": len(samples),
        "base_case_count": len(
            {sample["metadata"]["base_case_id"] for sample in samples}
        ),
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "blueprint_counts": dict(sorted(blueprint_counts.items())),
        "hop_counts": {str(hops): count for hops, count in sorted(hop_counts.items())},
        "api_calls": 0,
        "review_status": "generated_review_required",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
