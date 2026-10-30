"""Tests for the per-window decode plans (impl.driver: parse_plan_windows, window_index, check_plan_windows,
StreamedPlannedRun.plan_for): window bounds, the consistency check over the plans' runtime, and the plan a class
compiles with.

  .venv/bin/python -m pytest python/tests/test_plan_windows.py
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from perseus.impl.driver import StreamedPlannedRun, check_plan_windows, parse_plan_windows, window_index

RUNTIME = "SM_CHEB_GS=1 FIDESLIB_BTS_SPARSE_B_K=24 FIDESLIB_BTS_RAISE_DROPS=1,3,5 FIDESLIB_BTS_RAISE_DROPS=1,2,3"


class _Rec:
    n_layers = 2
    first = 1


class PlanWindows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="perseus_plan_windows_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def plan(self, name, runtime=RUNTIME, configs="configs/x/configs.json", stamp=None, rot=None, blocks=(0, 1, 2)):
        d = os.path.join(self.tmp, name)
        os.makedirs(d)
        with open(os.path.join(d, "PLAN_CMD.txt"), "w") as f:
            f.write(f"graph=graphs/g\nrecipe=PLAN_CF_MAX=20\nruntime={runtime}\nconfigs={configs}\n")
        for b in blocks:
            with open(os.path.join(d, f"block_{b}_placement.json"), "w") as f:
                json.dump({"version": 2, "config": stamp or {"ckks": {"n": 16}}, "placements": [{"site": name}]}, f)
        if rot is not None:
            with open(os.path.join(d, "plan_rot_steps.json"), "w") as f:
                json.dump(rot, f)
        return d

    def test_parse_sorts_and_rejects(self):
        self.assertEqual(parse_plan_windows("256:b, 128:a,512:d,384:c"),
                         [(128, "a"), (256, "b"), (384, "c"), (512, "d")])
        for bad in ("128", "128:", "x:a", "0:a", "128:a,128:b", ""):
            with self.assertRaises(ValueError):
                parse_plan_windows(bad)

    def test_window_bounds(self):
        w = parse_plan_windows("128:a,256:b,384:c,512:d")
        self.assertEqual([window_index(w, p) for p in (0, 1, 127, 128, 255, 256, 383, 384, 511)],
                         [0, 0, 0, 1, 1, 2, 2, 3, 3])
        with self.assertRaises(ValueError):
            window_index(w, 512)

    def test_check_equal_runtime_and_rot_union(self):
        a = self.plan("a", rot=[1, 2, -4])
        b = self.plan("b", runtime=" ".join(reversed(RUNTIME.split()[:2])) + " " + " ".join(RUNTIME.split()[2:]),
                      rot=[2, 8])
        c = self.plan("c")
        self.assertEqual(check_plan_windows([a, b, c]), [-4, 1, 2, 8])
        self.assertIsNone(check_plan_windows([c, c]))

    def test_check_names_the_difference(self):
        a = self.plan("a")
        cases = {"runtime switches SM_CHEB_GS": self.plan("r", runtime=RUNTIME.replace("SM_CHEB_GS=1", "SM_CHEB_GS=0")),
                 "runtime switches FIDESLIB_BTS_RAISE_DROPS": self.plan("d", runtime=RUNTIME + " FIDESLIB_BTS_RAISE_DROPS=1"),
                 "configs": self.plan("g", configs="configs/y/configs.json"),
                 "stamp": self.plan("s", stamp={"ckks": {"n": 17}}),
                 "blocks": self.plan("k", blocks=(0, 1))}
        for what, d in cases.items():
            with self.assertRaisesRegex(ValueError, what):
                check_plan_windows([a, d])

    def test_plan_for(self):
        a, b = self.plan("a"), self.plan("b")
        run = StreamedPlannedRun(_Rec(), a, None, [], None, work_dir=self.tmp, streams={}, stream_rows=None,
                                 plan_windows=parse_plan_windows(f"128:{a},512:{b}"))
        self.assertEqual([run.plan_for(p) for p in (1, 127, 128, 511)], [(0, a), (0, a), (1, b), (1, b)])
        single = StreamedPlannedRun(_Rec(), a, None, [], None, work_dir=self.tmp, streams={}, stream_rows=None)
        self.assertEqual(single.plan_for(500), (None, a))
        with self.assertRaises(ValueError):
            StreamedPlannedRun(_Rec(), a, None, [], None, work_dir=self.tmp, streams={}, stream_rows=None,
                               plan_windows=[(128, a), (512, self.plan("x", configs="configs/z.json"))])


if __name__ == "__main__":
    unittest.main()
