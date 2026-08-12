from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "threshold_scan"
    / "04_scan_path_threshold_grid.py"
)

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prodive-matplotlib-test")
)

if importlib.util.find_spec("tqdm") is None:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub


def load_threshold_module():
    spec = importlib.util.spec_from_file_location("prodive_threshold_scan", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PathStatisticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_threshold_module()

    def test_one_to_one_statistics_keep_distinct_fields(self):
        path = [(1, 2), (2, 3), (3, 4)]
        data = {1: {2: 10.0}, 2: {3: 20.0}, 3: {4: 30.0}}

        result = self.module._process_path_as_one_to_one(path, data)
        self.assertIsNotNone(result)
        stats = result["stats"]
        self.assertEqual(stats["main_segment_len"], 8)
        self.assertEqual(stats["num_sub_segments"], 1)
        self.assertEqual(stats["avg_sub_segment_len"], 8.0)
        self.assertEqual(stats["structural_fit_iou"], 1.0)
        self.assertEqual(float(stats["avg_similarity"]), 20.0)

        records = self.module.process_analysis_results(
            [result],
            file_name="example.npz",
            hmm_main="PF00001",
            hmm_sub="PF00002",
            top_n_to_save=None,
            first_threshold=2.0,
            second_threshold=13.0,
        )
        self.assertEqual(records[0]["Main_Segment_Len"], 8)
        self.assertEqual(records[0]["Num_Sub_Segments"], 1)
        self.assertEqual(records[0]["Avg_Sub_Segment_Len"], 8.0)
        self.assertEqual(float(records[0]["Avg_Similarity"]), 20.0)


if __name__ == "__main__":
    unittest.main()
