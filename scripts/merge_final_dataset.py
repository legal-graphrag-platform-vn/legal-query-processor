"""Merge selected non-multi-hop and multi-hop samples without split leakage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", value).lower()
    return re.sub(r"\W+", " ", value).strip()


def _group_id(sample: dict[str, Any], source: str) -> str:
    if source == "multihop":
        raw = f"mh:{sample['metadata']['base_case_id']}"
    elif sample["output"]["standalone_query"]:
        raw = f"nm:{_normalize(sample['output']['standalone_query'])}"
    else:
        raw = "nm-clarify:" + _normalize(
            json.dumps(sample["input"], ensure_ascii=False, sort_keys=True)
        )
    return "grp_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _stratum(sample: dict[str, Any]) -> str:
    return sample["output"]["plan_type"] or "needs_clarification"


def normalize_conversation_history(sample: dict[str, Any]) -> bool:
    history = sample["input"]["conversation_history"]
    if not history:
        sample["metadata"]["history_normalization"] = "unchanged_empty"
        return True
    if history[0].get("role") != "user":
        return False
    for index, message in enumerate(history):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected or not message.get("content", "").strip():
            return False
    if history[-1]["role"] == "user":
        trailing_user = history.pop()["content"].strip()
        current_query = sample["input"]["current_query"].strip()
        sample["input"]["current_query"] = (
            f"{trailing_user}\n{current_query}" if current_query else trailing_user
        )
        sample["metadata"]["history_normalization"] = (
            "moved_trailing_user_to_current_query"
        )
    else:
        sample["metadata"]["history_normalization"] = "unchanged_valid"
    return not history or history[-1]["role"] == "assistant"


def assign_splits(
    samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample["metadata"]["split_group_id"]].append(sample)

    groups_by_stratum: dict[str, list[tuple[str, list[dict[str, Any]]]]] = (
        defaultdict(list)
    )
    for group_id, rows in groups.items():
        strata = {_stratum(row) for row in rows}
        if len(strata) != 1:
            raise ValueError(f"Split group spans strata: {group_id}")
        groups_by_stratum[next(iter(strata))].append((group_id, rows))

    assigned = {"train": [], "validation": [], "test": []}
    split_order = ("train", "validation", "test")
    for stratum, stratum_groups in sorted(groups_by_stratum.items()):
        total = sum(len(rows) for _, rows in stratum_groups)
        targets = {
            "train": round(total * 0.8),
            "validation": round(total * 0.1),
        }
        targets["test"] = total - targets["train"] - targets["validation"]
        counts = Counter()
        ordered = sorted(
            stratum_groups,
            key=lambda item: hashlib.sha256(
                f"query-processing-final-v1:{item[0]}".encode("utf-8")
            ).hexdigest(),
        )
        for _, rows in ordered:
            split = max(
                split_order,
                key=lambda name: (
                    targets[name] - counts[name],
                    -split_order.index(name),
                ),
            )
            assigned[split].extend(rows)
            counts[split] += len(rows)
    return assigned


def _load_validator(root: Path):
    path = root / "scripts" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("query_dataset_contract_final", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.validate_samples


def merge(root: Path, non_multihop_path: Path, multihop_path: Path) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    excluded_invalid_history_count = 0
    normalized_trailing_user_count = 0
    for source, path in (
        ("deepseek_v4_flash_non_multihop", non_multihop_path),
        ("multihop", multihop_path),
    ):
        for original in _read_jsonl(path):
            sample = json.loads(json.dumps(original, ensure_ascii=False))
            sample["metadata"]["dataset_source"] = source
            if not normalize_conversation_history(sample):
                excluded_invalid_history_count += 1
                continue
            if (
                sample["metadata"]["history_normalization"]
                == "moved_trailing_user_to_current_query"
            ):
                normalized_trailing_user_count += 1
            sample["metadata"]["split_group_id"] = _group_id(
                sample, "multihop" if source == "multihop" else "non_multihop"
            )
            sample["metadata"]["selection_status"] = "selected_after_auto_audit"
            selected.append(sample)

    ids = [sample["sample_id"] for sample in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id across merged sources")
    strata_by_group: dict[str, set[str]] = defaultdict(set)
    for sample in selected:
        strata_by_group[sample["metadata"]["split_group_id"]].add(_stratum(sample))
    conflicting_groups = {
        group_id
        for group_id, strata in strata_by_group.items()
        if len(strata) > 1
    }
    conflicting_sample_count = sum(
        sample["metadata"]["split_group_id"] in conflicting_groups
        for sample in selected
    )
    selected = [
        sample
        for sample in selected
        if sample["metadata"]["split_group_id"] not in conflicting_groups
    ]
    _load_validator(root)(selected)
    splits = assign_splits(selected)

    owner_by_group: dict[str, str] = {}
    for split, rows in splits.items():
        for sample in rows:
            group_id = sample["metadata"]["split_group_id"]
            owner = owner_by_group.setdefault(group_id, split)
            if owner != split:
                raise ValueError(f"Split leakage for {group_id}")

    output = root / "final"
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_jsonl(output / f"{split}.jsonl", rows)
    _write_jsonl(output / "all.jsonl", selected)

    plans = Counter(_stratum(sample) for sample in selected)
    intents = Counter(
        subquery["intent"]
        for sample in selected
        for subquery in sample["output"]["subqueries"]
    )
    sources = Counter(
        sample["metadata"]["dataset_source"] for sample in selected
    )
    summary = {
        "schema_version": "query-processing-final-dataset-v1",
        "sample_count": len(selected),
        "excluded_flagged_non_multihop_count": 81,
        "excluded_conflicting_group_count": len(conflicting_groups),
        "excluded_conflicting_sample_count": conflicting_sample_count,
        "excluded_invalid_history_count": excluded_invalid_history_count,
        "normalized_trailing_user_count": normalized_trailing_user_count,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "split_group_count": len(owner_by_group),
        "source_counts": dict(sorted(sources.items())),
        "plan_type_counts": dict(sorted(plans.items())),
        "subquery_intent_counts": dict(sorted(intents.items())),
        "split_leakage_count": 0,
        "selection_status": "auto_audited_not_human_gold",
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
    parser.add_argument(
        "--non-multihop",
        type=Path,
        default=Path(
            "training/query_processing/synthetic/runs/"
            "non_multihop_6000_v1/auto_clean_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--multihop",
        type=Path,
        default=Path("training/query_processing/multihop/all.jsonl"),
    )
    args = parser.parse_args()
    summary = merge(
        args.root.resolve(),
        args.non_multihop.resolve(),
        args.multihop.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
