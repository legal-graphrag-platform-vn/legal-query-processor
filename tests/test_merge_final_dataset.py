from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "merge_final_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("merge_final_dataset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FinalDatasetMergeTests(unittest.TestCase):
    def test_moves_trailing_user_turn_into_current_query(self) -> None:
        sample = {
            "input": {
                "conversation_history": [
                    {"role": "user", "content": "Tôi hỏi về vốn."},
                    {"role": "assistant", "content": "Bạn hỏi loại vốn nào?"},
                    {"role": "user", "content": "Vốn điều lệ."},
                ],
                "current_query": "Nó là gì?",
            },
            "metadata": {},
        }

        valid = MODULE.normalize_conversation_history(sample)

        self.assertTrue(valid)
        self.assertEqual(
            [row["role"] for row in sample["input"]["conversation_history"]],
            ["user", "assistant"],
        )
        self.assertEqual(
            sample["input"]["current_query"],
            "Vốn điều lệ.\nNó là gì?",
        )

    def test_rejects_history_starting_with_assistant(self) -> None:
        sample = {
            "input": {
                "conversation_history": [
                    {"role": "assistant", "content": "Bạn hỏi gì?"}
                ],
                "current_query": "Vốn điều lệ.",
            },
            "metadata": {},
        }

        self.assertFalse(MODULE.normalize_conversation_history(sample))

    def test_group_never_leaks_across_splits(self) -> None:
        samples = []
        for group_index in range(30):
            for variant in range(2):
                samples.append(
                    {
                        "sample_id": f"s{group_index}_{variant}",
                        "output": {"plan_type": "single"},
                        "metadata": {
                            "split_group_id": f"group_{group_index}"
                        },
                    }
                )

        splits = MODULE.assign_splits(samples)
        owners = {}
        for split, rows in splits.items():
            for row in rows:
                group_id = row["metadata"]["split_group_id"]
                self.assertEqual(owners.setdefault(group_id, split), split)

        self.assertEqual(sum(len(rows) for rows in splits.values()), 60)


if __name__ == "__main__":
    unittest.main()
