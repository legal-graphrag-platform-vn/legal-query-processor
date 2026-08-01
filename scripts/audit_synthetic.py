"""Audit generated non-multi-hop candidates without mutating raw accepted data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFINITION_CONFLICT_RE = re.compile(
    r"điều kiện|thủ tục|quyền|nghĩa vụ|bao nhiêu|tỷ lệ|thời hạn|có được",
    re.IGNORECASE,
)
VALIDITY_SIGNAL_RE = re.compile(
    r"hiệu lực|sửa đổi|bãi bỏ|thay thế|áp dụng|ban hành|ngày|năm|thời điểm",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def audit(run_dir: Path) -> dict[str, Any]:
    samples = _read_jsonl(run_dir / "accepted.jsonl")
    flagged: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()

    for sample in samples:
        reasons: set[str] = set()
        output = sample["output"]
        if output["plan_type"] == "multi_hop":
            reasons.add("MULTI_HOP_NOT_ALLOWED")
        for subquery in output["subqueries"]:
            if subquery["depends_on"]:
                reasons.add("DEPENDENCY_NOT_ALLOWED")
            query = subquery["query"]
            if (
                subquery["intent"] == "definition"
                and DEFINITION_CONFLICT_RE.search(query)
            ):
                reasons.add("DEFINITION_INTENT_CONFLICT")
            if (
                subquery["intent"] == "validity"
                and not VALIDITY_SIGNAL_RE.search(query)
            ):
                reasons.add("VALIDITY_INTENT_WITHOUT_SIGNAL")
        if reasons:
            for reason in reasons:
                flag_counts[reason] += 1
            flagged.append({"sample": sample, "reasons": sorted(reasons)})
        else:
            clean.append(sample)

    plans = Counter(sample["output"]["plan_type"] for sample in samples)
    statuses = Counter(sample["output"]["status"] for sample in samples)
    intents = Counter(
        subquery["intent"]
        for sample in samples
        for subquery in sample["output"]["subqueries"]
    )
    summary = {
        "schema_version": "query-processing-synthetic-audit-v1",
        "source_sample_count": len(samples),
        "auto_clean_candidate_count": len(clean),
        "flagged_review_count": len(flagged),
        "flag_counts": dict(sorted(flag_counts.items())),
        "status_counts": dict(sorted(statuses.items(), key=lambda item: str(item[0]))),
        "plan_type_counts": dict(
            sorted(plans.items(), key=lambda item: str(item[0]))
        ),
        "subquery_intent_counts": dict(sorted(intents.items())),
        "multi_hop_count": plans.get("multi_hop", 0),
        "dependency_count": sum(
            bool(subquery["depends_on"])
            for sample in samples
            for subquery in sample["output"]["subqueries"]
        ),
        "review_status": "human_review_still_required",
    }
    _write_jsonl(run_dir / "auto_clean_candidates.jsonl", clean)
    _write_jsonl(run_dir / "flagged_review.jsonl", flagged)
    (run_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
