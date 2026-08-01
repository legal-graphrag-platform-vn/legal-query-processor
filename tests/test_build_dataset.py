from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_query_dataset", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QueryDatasetTests(unittest.TestCase):
    def test_history_variant_resolves_scope(self) -> None:
        cases = [
            {
                "query_id": "definition_01",
                "query": "Vốn điều lệ là gì?",
                "intent": "definition",
            }
        ]

        samples = MODULE.make_samples(cases)

        history = samples[1]
        self.assertEqual(history["input"]["current_query"], "Đúng, theo luật đó.")
        self.assertNotIn("intent", history["output"])
        self.assertEqual(history["output"]["subqueries"][0]["intent"], "definition")
        self.assertEqual(history["output"]["plan_type"], "single")
        self.assertIn(
            "Luật Doanh nghiệp 2020",
            history["output"]["standalone_query"],
        )

    def test_complex_query_has_curated_decomposition(self) -> None:
        cases = [
            {
                "query_id": "multi_hop_01",
                "query": "Điều 38 dẫn chiếu Điều 41 và khoản nào tại Điều 41?",
                "intent": "multi_hop",
            }
        ]

        sample = MODULE.make_samples(cases)[0]

        self.assertEqual(sample["output"]["plan_type"], "multi_hop")
        self.assertEqual(len(sample["output"]["subqueries"]), 2)
        self.assertEqual(
            [item["intent"] for item in sample["output"]["subqueries"]],
            ["hierarchy", "hierarchy"],
        )
        self.assertEqual(sample["output"]["subqueries"][1]["depends_on"], ["q1"])

    def test_split_keeps_variants_of_base_query_together(self) -> None:
        samples = []
        for index in range(30):
            for variant in ("direct", "history"):
                samples.append(
                    {
                        "sample_id": f"q{index}_{variant}",
                        "input": {
                            "conversation_history": [],
                            "current_query": "Q",
                        },
                        "output": {
                            "status": "ready",
                            "standalone_query": "Q",
                            "plan_type": "single",
                            "subqueries": [
                                {
                                    "id": "q1",
                                    "query": "Q",
                                    "intent": "factual",
                                    "depends_on": [],
                                }
                            ],
                            "clarification_question": None,
                        },
                        "metadata": {
                            "base_query_id": f"q{index:02d}",
                            "seed_query_intent": "factual",
                        },
                    }
                )

        splits = MODULE.split_samples(samples)
        memberships = {
            sample["metadata"]["base_query_id"]: split
            for split, rows in splits.items()
            for sample in rows
        }

        self.assertEqual([len(rows) for rows in splits.values()], [48, 6, 6])
        self.assertEqual(len(memberships), 30)

    def test_rejects_dependency_without_reference_to_prior_result(self) -> None:
        sample = {
            "sample_id": "bad_dependency",
            "input": {"conversation_history": [], "current_query": "Q"},
            "output": {
                "status": "ready",
                "standalone_query": "Q",
                "plan_type": "multi_hop",
                "subqueries": [
                    {
                        "id": "q1",
                        "query": "Tìm điều được dẫn chiếu.",
                        "intent": "hierarchy",
                        "depends_on": [],
                    },
                    {
                        "id": "q2",
                        "query": "Quy định về tỷ lệ biểu quyết là gì?",
                        "intent": "factual",
                        "depends_on": ["q1"],
                    },
                ],
                "clarification_question": None,
            },
            "metadata": {"base_query_id": "bad"},
        }

        with self.assertRaisesRegex(ValueError, "no result reference"):
            MODULE.validate_samples([sample])


if __name__ == "__main__":
    unittest.main()
