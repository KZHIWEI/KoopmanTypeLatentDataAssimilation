import ast
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.eval.aggregate_benchmark import aggregate


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for seed, value in [(1, 1.0), (2, 3.0), (3, 5.0)]:
            for kind in ("result", "curve"):
                folder = self.root / f"seed{seed}" / "demo" / kind
                folder.mkdir(parents=True)
                (folder / "_meta.json").write_text(json.dumps({"dataset": "demo", "seed": str(seed), "protocol": {"starts": [[0, 0], [0, 2]], "run_length": 2}}))
                record = {"forecast": {"rrmse": value, "rrmse_post_spinup": value, "rrmse_final_step": value}, "da": {"obs30": {"rrmse": value}}} if kind == "result" else {"forecast": [[value, value+2], [value+2, value+4]], "da": {"obs30": [[value, value+2], [value+2, value+4]]}}
                (folder / "KAE.json").write_text(json.dumps(record))
                if seed == 1:
                    (folder / "DMD.json").write_text(json.dumps(record))

    def run_summary(self):
        return aggregate(self.root, "demo", [1, 2, 3], ["KAE", "DMD"], [30])

    def test_equal_seed_weight_and_single_baseline(self):
        summary = self.run_summary()
        model = summary["models"]["KAE"]
        self.assertEqual(model["da"]["obs30"]["rrmse"], {"mean": 3.0, "std": 2.0, "n": 3, "values": [1.0, 3.0, 5.0]})
        self.assertEqual(model["curves"]["da"]["obs30"]["mean"], [4.0, 6.0])
        self.assertEqual(model["curves"]["da"]["obs30"]["std"], [2.0, 2.0])
        self.assertIsNone(summary["models"]["DMD"]["da"]["obs30"]["rrmse"]["std"])
        self.assertEqual(summary["models"]["DMD"]["seeds"], [1])

    def test_cached_failure_is_error(self):
        p = self.root / "seed2/demo/result/KAE.json"
        data = json.loads(p.read_text()); data["_last_run_error"] = {"error": "latest run failed"}
        p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.run_summary()

    def test_seed_paths_and_checkpoint_selection(self):
        tree = ast.parse(Path("src/eval/benchmark.py").read_text())
        names = {"EVAL_SEED", "OUT"}
        nodes = [n for n in tree.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name == "_seed_swap")]
        paths = []
        for seed in [1, 2, 3]:
            from types import SimpleNamespace
            ns = {"os": SimpleNamespace(environ={"EVAL_SEED": str(seed), "EVAL_OUTPUT_ROOT": "custom-results"}, path=os.path)}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "benchmark seed setup", "exec"), ns)
            paths.append(ns["OUT"])
            self.assertEqual(ns["_seed_swap"](["ckpt_release/demo/kae/seed1/kae.pt"]), [f"ckpt_release/demo/kae/seed{seed}/kae.pt"])
            self.assertEqual(ns["OUT"], os.path.join("custom-results", f"seed{seed}"))
        self.assertEqual(len(set(paths)), 3)

    def run_entrypoint(self, fail_seed=""):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash is not installed")
        wrapper = self.root / "fake-python.sh"
        wrapper.write_text('#!/usr/bin/env bash\nprintf "%s|%s\\n" "${EVAL_SEED:-}" "$*" >> "$CALL_LOG"\nif [[ -n "${FAIL_SEED:-}" && "${EVAL_SEED:-}" == "$FAIL_SEED" ]]; then exit 3; fi\n')
        wrapper.chmod(0o755)
        log = self.root / "calls.log"
        env = dict(os.environ, PY=wrapper.as_posix(), CALL_LOG=log.as_posix(), DATASETS="demo", SEEDS="1 2 3", FAIL_SEED=fail_seed)
        env.pop("EVAL_SEED", None)
        run = subprocess.run([bash, "experiments/main/benchmark/run.sh"], env=env, capture_output=True, text=True)
        return run, log.read_text().splitlines()

    def test_entrypoint_runs_seeds_then_summary(self):
        run, lines = self.run_entrypoint()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(lines), 8)
        self.assertEqual(lines[0], "1|-m src.eval.sigma_selection demo dmd")
        self.assertEqual([line.split("|")[0] for line in lines[1:7]], ["1", "1", "2", "2", "3", "3"])
        self.assertTrue(all(",dmd" in line for line in lines[1:3]))
        self.assertTrue(all(",dmd" not in line for line in lines[3:7]))
        self.assertIn("src.eval.aggregate_benchmark --dataset demo --seeds 1 2 3", lines[-1])

    def test_entrypoint_stops_before_summary_on_failure(self):
        run, lines = self.run_entrypoint(fail_seed="2")
        self.assertEqual(run.returncode, 3, run.stderr)
        self.assertEqual(len(lines), 4)
        self.assertFalse(any("aggregate_benchmark" in line for line in lines))

    def test_missing_seed_is_error(self):
        (self.root / "seed3/demo/result/KAE.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_summary()

    def test_failed_result_is_error(self):
        (self.root / "seed2/demo/result/KAE.json").write_text(json.dumps({"error": "checkpoint missing"}))
        with self.assertRaises(ValueError):
            self.run_summary()

    def test_protocol_mismatch_is_error(self):
        (self.root / "seed2/demo/curve/_meta.json").write_text(json.dumps({"dataset": "demo", "seed": "2", "protocol": {"run_length": 99}}))
        with self.assertRaises(ValueError):
            self.run_summary()

    def test_nonfinite_is_error(self):
        p = self.root / "seed2/demo/result/KAE.json"
        data = json.loads(p.read_text()); data["da"]["obs30"]["rrmse"] = float("nan")
        p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.run_summary()

    def test_curve_shape_mismatch_is_error(self):
        p = self.root / "seed2/demo/curve/KAE.json"
        data = json.loads(p.read_text()); data["da"]["obs30"] = [[1.0]]
        p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.run_summary()


if __name__ == "__main__":
    unittest.main()
