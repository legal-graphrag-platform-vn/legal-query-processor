from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_sft_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_sft_dataset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrepareSftDatasetTests(unittest.TestCase):
    def test_converts_history_to_real_chat_turns(self) -> None:
        sample = {
            "sample_id": "sample-1",
            "input": {
                "conversation_history": [
                    {"role": "user", "content": "Tôi hỏi về công ty."},
                    {"role": "assistant", "content": "Bạn muốn hỏi gì?"},
                ],
                "current_query": "Điều kiện thành lập là gì?",
            },
            "output": {
                "status": "ready",
                "standalone_query": "Điều kiện thành lập công ty là gì?",
                "plan_type": "single",
                "subqueries": [
                    {
                        "id": "q1",
                        "query": "Điều kiện thành lập công ty là gì?",
                        "intent": "factual",
                        "depends_on": [],
                    }
                ],
                "clarification_question": None,
            },
            "metadata": {"must_not_be_trained": True},
        }

        row = MODULE.convert_sample(sample)

        self.assertEqual(
            [message["role"] for message in row["messages"]],
            ["system", "user", "assistant", "user", "assistant"],
        )
        target = json.loads(row["messages"][-1]["content"])
        self.assertEqual(target, sample["output"])
        self.assertNotIn("metadata", row)
        self.assertEqual(row["sample_id"], "sample-1")

    def test_rejects_history_ending_with_user(self) -> None:
        sample = {
            "sample_id": "sample-2",
            "input": {
                "conversation_history": [
                    {"role": "user", "content": "Câu trước"}
                ],
                "current_query": "Câu mới",
            },
            "output": {key: None for key in MODULE.OUTPUT_KEYS},
        }

        with self.assertRaisesRegex(ValueError, "end with assistant"):
            MODULE.convert_sample(sample)


if __name__ == "__main__":
    unittest.main()
