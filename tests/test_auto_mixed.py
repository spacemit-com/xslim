"""Synthetic real-ORT evidence; these fixtures make no deployment accuracy claim."""

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from xslim.candidate_runner import CandidateRunner, config_to_dict, effective_plan_key, plan_from_dict, score_outputs
from xslim.xslim_pipeline import quantize_onnx_model
from xslim.xslim_setting import XSlimSettingFactory


def fixture(root, *, mode="auto_mixed", **strategy):
    rng = np.random.default_rng(71)
    arrays = [rng.normal(size=(8, 8)).astype(np.float32) for _ in range(3)]
    arrays[1][:, 0] *= 20
    nodes = [helper.make_node("MatMul", ["input", "w0"], ["a"], name="locked"),
             helper.make_node("Sigmoid", ["a"], ["s"], name="activation"),
             helper.make_node("MatMul", ["s", "w1"], ["b"], name="sensitive"),
             helper.make_node("MatMul", ["b", "w2"], ["output"], name="tail")]
    model = helper.make_model(helper.make_graph(nodes, "auto-mixed-fixture",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8])],
        [numpy_helper.from_array(array, f"w{i}") for i, array in enumerate(arrays)]),
        opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    paths = []
    for i in range(12):
        path = root / f"{i}.npy"
        np.save(path, rng.normal(size=(1, 8)).astype(np.float32))
        paths.append(str(path))
    (root / "data.list").write_text("\n".join(paths))
    config = {"model_parameters": {"skip_onnxsim": True, "opset": 18,
                                  "working_dir": str(root), "output_prefix": "result"},
        "calibration_parameters": {"calibration_step": 10, "calibration_device": "cpu",
            "input_parameters": [{"input_name": "input", "input_shape": [1, 8], "file_type": "npy",
                                   "data_list_path": str(root / "data.list")}]},
        "quantization_parameters": {"analysis_enable": False, "finetune_level": 0},
        "strategy_parameters": {"mode": mode, "default_precision": "int8", "score_steps": 4,
            "node_precisions": {"locked": "int8"}, "max_trials": 12, "max_adjust_groups": 1,
            "cosine_similarity_min": 1.0, **strategy}}
    return model, config


class AutoMixedTests(unittest.TestCase):
    def test_schema_roundtrip_and_invalid_boundaries(self):
        for field, values in {"max_time_s": [True, float("nan"), float("inf"), 0],
                              "cosine_similarity_min": [False, float("nan"), 1.1, -1.1],
                              "max_trials": [True, 0, 1.5], "score_steps": [False, 0],
                              "max_adjust_groups": [True, -1]}.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    XSlimSettingFactory.from_json({"strategy_parameters": {"mode": "auto_mixed", field: value}})
        setting = XSlimSettingFactory.from_json({"strategy_parameters": {"mode": "auto_mixed",
                                                                        "max_adjust_groups": 0}})
        self.assertEqual(XSlimSettingFactory.from_json({"strategy_parameters": config_to_dict(setting.strategy_parameters)}).strategy_parameters.mode,
                         "auto_mixed")
        self.assertEqual(setting.strategy_parameters.max_adjust_groups, 0)

    def test_invalid_metric_outputs_never_pass(self):
        for value in [0., float("inf"), float("nan")]:
            result = score_outputs(((np.ones(2),),), ((np.full(2, value),),))
            self.assertFalse(result["valid"])
            self.assertIsNone(result["cosine"])
        self.assertFalse(score_outputs(((np.zeros(2),),), ((np.zeros(2),),))["valid"])

    def test_runner_real_export_cache_locks_and_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root)
            original = model.SerializeToString()
            runner = CandidateRunner(model, XSlimSettingFactory.from_json(config), work_dir=root / "runs")
            baseline = runner.run(None, purpose="baseline")
            self.assertEqual(baseline.status, "ok", baseline.error)
            self.assertEqual(model.SerializeToString(), original)
            self.assertEqual(runner.full_conversion_count, 1)
            plan = plan_from_dict(baseline.plan.to_dict())
            self.assertEqual(plan.fingerprint, baseline.plan.fingerprint)
            group = next(g for g in plan.groups if "sensitive" in g["members"])
            promoted = plan.promote_group(group["id"])
            with self.assertRaisesRegex(ValueError, "locked"):
                plan.with_precisions({"locked": "fp32"})
            candidate = runner.run(promoted)
            self.assertEqual(candidate.status, "ok", candidate.error)
            repeated = runner.run(promoted.promote_group(group["id"]))
            self.assertTrue(repeated.cache_hit)
            self.assertEqual(runner.full_conversion_count, 2)
            changed_provenance = copy.deepcopy(promoted)
            changed_provenance.decisions["sensitive"].reasons.append("audit only")
            self.assertEqual(effective_plan_key(changed_provenance), effective_plan_key(promoted))
            cached = runner.run(changed_provenance)
            self.assertTrue(cached.cache_hit)
            candidate.model.graph.name = "caller mutation"
            candidate.outputs[0][0].fill(0)
            self.assertNotEqual(cached.model.graph.name, candidate.model.graph.name)
            final = runner.run(promoted, purpose="final", use_cache=False)
            self.assertEqual(final.status, "ok", final.error)
            self.assertEqual(runner.full_conversion_count, 3)
            for expected, actual in zip(cached.outputs, final.outputs):
                for a, b in zip(expected, actual):
                    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
            prepared = runner.prepare()
            ort_output = final.run(copy.deepcopy(prepared.tuning_batches[0]))
            np.testing.assert_allclose(ort_output[0], final.outputs[0][0], rtol=1e-5, atol=1e-6)
            prepared.tuning_batches[0]["input"].fill(999)
            self.assertFalse(np.all(runner.prepare().tuning_batches[0]["input"] == 999))

    def test_public_auto_real_improvement_keeps_int8(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root)
            quantize_onnx_model(config, model)
            report = json.loads((root / "result_auto_mixed_report.json").read_text())
            self.assertTrue(report["final_check"]["verified"], report)
            self.assertTrue(report["adjusted_groups"], report)
            self.assertLess(report["published"]["metrics"]["mse"], report["baseline"]["metrics"]["mse"])
            decisions = report["plan"]["decisions"]
            self.assertEqual(decisions["locked"]["precision"], "int8")
            self.assertTrue(decisions["locked"]["locked"])
            self.assertGreaterEqual(sum(d["precision"] == "int8" and not d["locked"]
                                        for d in decisions.values()), 1)
            self.assertEqual(len(report["adjusted_groups"]), 1)
            self.assertEqual(report["attempts"][-1]["purpose"], "final")

    def test_zero_adjustment_budget_and_missing_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root, max_adjust_groups=0)
            quantize_onnx_model(config, model)
            report = json.loads((root / "result_auto_mixed_report.json").read_text())
            self.assertEqual(report["full_conversion_count"], 2)
            self.assertEqual(report["adjusted_groups"], [])
            self.assertTrue(report["final_check"]["verified"])
            config["calibration_parameters"]["input_parameters"][0].pop("data_list_path")
            with self.assertRaisesRegex(ValueError, "actual user"):
                quantize_onnx_model(config, model)

    def test_preparation_deadline_prevents_new_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root)
            runner = CandidateRunner(model, XSlimSettingFactory.from_json(config), work_dir=root / "runs",
                                     started_at=time.monotonic() - 2, max_time_s=1)
            result = runner.run(None)
            self.assertEqual(result.status, "budget_exhausted")
            self.assertEqual(runner.full_conversion_count, 0)

    def test_resume_retains_count_cache_and_context_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root)
            setting = XSlimSettingFactory.from_json(config)
            runner = CandidateRunner(model, setting, work_dir=root / "runs", max_full_conversions=1)
            result = runner.run(None)
            self.assertEqual(result.status, "ok", result.error)
            resumed = CandidateRunner(model, setting, work_dir=root / "runs", max_full_conversions=1)
            hit = resumed.run(result.plan)
            self.assertTrue(hit.cache_hit)
            self.assertEqual(resumed.full_conversion_count, 1)
            self.assertIsNotNone(hit.session)
            self.assertGreaterEqual(resumed.elapsed_s, result.elapsed_s)
            next_plan = result.plan.promote_group(next(g["id"] for g in result.plan.groups if g["promotable"]))
            self.assertEqual(resumed.run(next_plan).status, "budget_exhausted")
            np.save(root / "0.npy", np.full((1, 8), 2, np.float32))
            changed = CandidateRunner(model, setting, work_dir=root / "runs")
            with self.assertRaisesRegex(ValueError, "context changed"):
                changed.prepare()

    def test_preprocessing_content_changes_identity_and_actual_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root)
            preprocess = root / "custom_stageb.py"
            template = "import torch\ndef load(paths, info):\n    return torch.full((1, 8), VALUE)\n"
            preprocess.write_text(template.replace("VALUE", "1.0"))
            config["calibration_parameters"]["input_parameters"][0]["preprocess_file"] = str(preprocess) + ":load"
            setting = XSlimSettingFactory.from_json(config)
            first = CandidateRunner(model, setting, work_dir=root / "first").prepare()
            preprocess.write_text(template.replace("VALUE", "2.0"))
            second = CandidateRunner(model, setting, work_dir=root / "second").prepare()
            self.assertNotEqual(first.config_fingerprint, second.config_fingerprint)
            self.assertNotEqual(first.data_fingerprint, second.data_fingerprint)
            self.assertTrue(np.all(second.tuning_batches[0]["input"] == 2))

    def test_real_invalid_zero_output_and_failed_attempt_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root)
            # Zero graph output remains an executed ONNX result, never a fake passing cosine.
            zero = np.zeros((8, 8), dtype=np.float32)
            model.graph.initializer[-1].CopyFrom(numpy_helper.from_array(zero, "w2"))
            runner = CandidateRunner(model, XSlimSettingFactory.from_json(config), work_dir=root / "zero")
            invalid = runner.run(None)
            self.assertEqual(invalid.status, "invalid_metrics", invalid.error)
            self.assertIn("zero-norm", invalid.metrics["invalid_reason"])
            model, config = fixture(root)
            def broken_hook(model, *, plan):
                raise RuntimeError("deliberate hook failure")
            broken = CandidateRunner(model, XSlimSettingFactory.from_json(config), work_dir=root / "broken",
                                     post_qdq_hook=broken_hook, hook_fingerprint="broken-v1")
            failed = broken.run(None)
            self.assertEqual(failed.status, "conversion_failed")
            self.assertEqual(broken.full_conversion_count, 1)
            self.assertTrue(broken.run(None).cache_hit)
            self.assertEqual(broken.full_conversion_count, 1)

    def test_budget_stop_reason_during_candidate_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root, max_trials=3, max_adjust_groups=2)
            quantize_onnx_model(config, model)
            report = json.loads((root / "result_auto_mixed_report.json").read_text())
            self.assertEqual(report["stop_reason"], "budget_exhausted")
            self.assertEqual(report["full_conversion_count"], 3)
            self.assertTrue(report["final_check"]["performed"])

    def test_public_path_input_preserves_float_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root, max_trials=2, max_adjust_groups=0)
            source = root / "original.onnx"
            onnx.save(model, source)
            before = source.read_bytes()
            quantize_onnx_model(config, str(source))
            self.assertEqual(source.read_bytes(), before)
            exported = root / "original.q.onnx"
            self.assertTrue(exported.exists())
            self.assertTrue(any(node.op_type == "QuantizeLinear" for node in onnx.load(exported).graph.node))
            report = json.loads((root / "original.q_final_check.json").read_text())
            self.assertTrue(report["verified"])
            with self.assertRaisesRegex(ValueError, "must differ"):
                quantize_onnx_model(config, str(source), str(source))
            self.assertEqual(source.read_bytes(), before)
            alias = root / "alias.onnx"
            alias.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "must differ"):
                quantize_onnx_model(config, str(source), str(alias))
            self.assertEqual(source.read_bytes(), before)
