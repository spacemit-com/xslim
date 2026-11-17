import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import xslim
from xslim.precision_strategy import build_precision_plan, record_quantized_layout, validate_export_layout
from xslim.ppq_decorator import OnnxParser
from xslim.quantizer.xslim import XSlimQuantizer
from xslim.xslim_setting import XSlimSettingFactory


def floating_model():
    nodes = [helper.make_node("MatMul", ["input", "weight"], ["a.out"], name="a"),
             helper.make_node("Sigmoid", ["a.out"], ["b.out"], name="b"),
             helper.make_node("Sigmoid", ["b.out"], ["output"], name="c"),
             helper.make_node("LeakyRelu", ["a.out"], ["side.out"], name="side")]
    graph = helper.make_graph(nodes, "plan", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4]),
                               helper.make_tensor_value_info("side.out", TensorProto.FLOAT, [1, 4])],
                              [numpy_helper.from_array(np.eye(4, dtype=np.float32), "weight")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)


class PrecisionPlanTests(unittest.TestCase):
    def plan(self, strategy, quant=None, model=None, graph=None):
        model = model or floating_model()
        graph = graph or OnnxParser().build(copy.deepcopy(model))
        setting = XSlimSettingFactory.from_json({"strategy_parameters": strategy,
                                                 "quantization_parameters": quant or {}})
        return build_precision_plan(model, graph, setting, XSlimQuantizer(graph).quant_operation_types)

    def test_precedence_and_subgraph_scope(self):
        plan = self.plan({"default_precision": "fp32", "op_type_precisions": {"Sigmoid": "int8", "LeakyRelu": "int8"},
                          "node_precisions": {"c": "int8"}, "rules": [
                              {"selector": "subgraph", "start_nodes": ["b"], "end_nodes": ["c"],
                               "precision": "fp32"}]})
        self.assertEqual({name: d.precision for name, d in plan.decisions.items()},
                         {"a": "fp32", "b": "fp32", "c": "int8", "side": "int8"})
        self.assertFalse(plan.decisions["a"].locked)
        self.assertTrue(plan.decisions["b"].locked)

    def test_same_priority_conflict_even_when_shadowed(self):
        with self.assertRaisesRegex(ValueError, "same-priority"):
            self.plan({"node_precisions": {"b": "int8"}, "rules": [
                {"selector": "type", "op_types": ["Sigmoid"], "precision": "int8"},
                {"selector": "type", "op_types": ["Sigmoid"], "precision": "fp32"}]})
        with self.assertRaisesRegex(ValueError, "same-priority"):
            self.plan({"node_precisions": {"b": "int8"}, "rules": [
                {"selector": "node", "names": ["b"], "precision": "fp32"}]})

    def test_legacy_last_wins_and_overrides_ignore(self):
        plan = self.plan({"op_type_precisions": {"MatMul": "fp32"}}, {
            "ignore_op_names": ["a"], "operator_quantization": [
                {"pattern": "a", "precision": "fp32"}, {"pattern": "a", "precision": "int8"}]})
        self.assertEqual(plan.decisions["a"].precision, "int8")
        with self.assertRaisesRegex(ValueError, "same-priority"):
            self.plan({"node_precisions": {"a": "fp32"}}, {"operator_quantization": [
                {"pattern": "a", "precision": "int8"}]})

    def test_mapping_exact_rename_and_lost_targets(self):
        model = floating_model()
        graph = OnnxParser().build(copy.deepcopy(model))
        operation = graph.operations.pop("a")
        operation._name = "renamed"
        graph.operations["renamed"] = operation
        plan = self.plan({"node_precisions": {"a": "int8"}}, model=model, graph=graph)
        self.assertEqual(plan.source_to_optimized["a"], "renamed")
        graph.operations.pop("renamed")
        with self.assertRaisesRegex(ValueError, "disappeared"):
            self.plan({"node_precisions": {"a": "int8"}}, model=model, graph=graph)

    def test_ambiguous_names_and_invalid_targets_fail(self):
        for names in [("", "b"), ("b", "b")]:
            model = floating_model()
            model.graph.node[0].name, model.graph.node[1].name = names
            # Build the valid graph first: ambiguity is in original identity only.
            with self.assertRaisesRegex(ValueError, "unique"):
                self.plan({"node_precisions": {"b": "fp32"}}, model=model,
                          graph=OnnxParser().build(floating_model()))
        with self.assertRaisesRegex(ValueError, "matched no operations"):
            self.plan({"node_precisions": {"absent": "fp32"}})
        with self.assertRaisesRegex(ValueError, "connected"):
            self.plan({"rules": [{"selector": "subgraph", "start_nodes": ["side"],
                                 "end_nodes": ["b"], "precision": "fp32"}]})
        with patch("xslim.precision_strategy.logger.warning") as warning:
            self.plan({"op_type_precisions": {"Nonexistent": "fp32"}})
        warning.assert_called_once()

    def test_schema_static_only_fp16_capability_and_empty_default(self):
        empty = XSlimSettingFactory.from_json({"strategy_parameters": {},
                                              "quantization_parameters": {"precision_level": 4}})
        self.assertFalse(empty.strategy_parameters.active)
        with self.assertRaisesRegex(ValueError, "static INT8"):
            XSlimSettingFactory.from_json({"strategy_parameters": {"default_precision": "fp32"},
                                          "quantization_parameters": {"precision_level": 3}})
        with self.assertRaisesRegex(NotImplementedError, "FP16"):
            self.plan({"allowed_precisions": ["fp16", "fp32"], "default_precision": "fp16"})
        with self.assertRaisesRegex(ValueError, "allowed"):
            self.plan({"allowed_precisions": ["fp32"], "node_precisions": {"a": "int8"}})
        with self.assertRaisesRegex(ValueError, "allowed_precisions"):
            self.plan({"allowed_precisions": ["fp32"]})

    def test_candidate_copy_locks_and_fused_group_atomicity(self):
        plan = self.plan({"default_precision": "int8", "node_precisions": {"side": "fp32"}})
        plan.groups = [{"id": "fused", "members": ["a", "b"], "precision": "int8", "locked": False,
                        "promotable": True, "reasons": ["shared output root"]}]
        before = plan.fingerprint
        promoted = plan.promote_group("fused")
        self.assertEqual(promoted.decisions["a"].precision, "fp32")
        self.assertEqual(promoted.source_to_optimized, plan.source_to_optimized)
        self.assertEqual(plan.fingerprint, before)
        self.assertNotEqual(promoted.fingerprint, before)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            plan.with_precisions({"a": "fp32"})
        with self.assertRaisesRegex(ValueError, "locked"):
            plan.with_precisions({"side": "int8"})


class PrecisionExportTests(unittest.TestCase):
    def test_real_mixed_export_metadata_report_and_tamper(self):
        model = floating_model()
        helper.set_model_props(model, {"parent-acceptance": "2026-11-17", "xslim_version": "source-version"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = np.array([[-1., 0., 1., 2.]], dtype=np.float32)
            np.save(root / "sample.npy", sample)
            (root / "data.list").write_text((str(root / "sample.npy") + "\n") * 10)
            output = root / "quantized.onnx"
            config = {
                "model_parameters": {"skip_onnxsim": True, "opset": 18},
                "calibration_parameters": {"calibration_step": 10, "calibration_device": "cpu",
                    "input_parameters": [{"input_name": "input", "input_shape": [1, 4], "file_type": "npy",
                                          "data_list_path": str(root / "data.list")}]},
                "quantization_parameters": {"analysis_enable": False, "finetune_level": 0},
                "strategy_parameters": {"default_precision": "fp32", "node_precisions": {"a": "int8"}},
            }
            recorded = []

            def record(plan, graph):
                key = plan.request_fingerprint
                record_quantized_layout(plan, graph)
                self.assertEqual(plan.request_fingerprint, key)
                recorded.append(copy.deepcopy(plan))

            with patch("xslim.xslim_pipeline.record_quantized_layout", side_effect=record):
                exported = xslim.quantize_onnx_model(config, model, str(output))
            onnx.checker.check_model(exported)
            props = {entry.key: entry.value for entry in exported.metadata_props}
            self.assertEqual(len(props), len(exported.metadata_props))
            self.assertEqual(props["parent-acceptance"], "2026-11-17")
            self.assertNotEqual(props["xslim_version"], "source-version")
            self.assertIn("xslim_export_time", props)
            report = json.loads((root / "quantized_precision_report.json").read_text())
            self.assertTrue(report["verified"])
            self.assertEqual(report["coverage"]["int8_nodes"], 1)
            self.assertEqual(report["coverage"]["optimized_nodes"], 4)
            self.assertTrue(report["layout"]["a"]["qdq_nodes"])
            self.assertTrue(report["plan"]["quantization_domains"])
            self.assertTrue(report["plan"]["decisions"]["a"]["quantized_activation_sockets"])
            # The unrelated branch keeps its own FP32 input/output sockets.
            side = next(node for node in exported.graph.node if node.name == "side")
            self.assertEqual(list(side.input), ["a.out"])
            self.assertEqual(list(side.output), ["side.out"])
            plan = recorded[0]
            for corruption in ("bypass", "reroute", "negative_scale", "nan_scale", "weight_axis"):
                with self.subTest(corruption=corruption):
                    damaged = copy.deepcopy(exported)
                    target = next(node for node in damaged.graph.node if node.name == "a")
                    producers = {tensor: node for node in damaged.graph.node for tensor in node.output}
                    dq = producers[target.input[0]]
                    q = producers[dq.input[0]]
                    if corruption == "bypass":
                        target.input[0] = q.input[0]
                    elif corruption == "reroute":
                        damaged.graph.input.append(helper.make_tensor_value_info("other", TensorProto.FLOAT, [1, 4]))
                        q.input[0] = "other"
                    elif corruption == "weight_axis":
                        weight_dq = producers[target.input[1]]
                        del weight_dq.attribute[:]
                        weight_dq.attribute.append(helper.make_attribute("axis", 9))
                        for name in weight_dq.input[1:]:
                            tensor = next(item for item in damaged.graph.initializer if item.name == name)
                            values = numpy_helper.to_array(tensor)
                            tensor.CopyFrom(numpy_helper.from_array(np.full(3, values.flat[0],
                                                                           dtype=values.dtype), name))
                    else:
                        scale = next(item for item in damaged.graph.initializer if item.name == q.input[1])
                        value = numpy_helper.to_array(scale).copy()
                        value[...] = -1 if corruption == "negative_scale" else np.nan
                        scale.CopyFrom(numpy_helper.from_array(value, scale.name))
                    with self.assertRaises((ValueError, onnx.checker.ValidationError,
                                            onnx.shape_inference.InferenceError)):
                        validate_export_layout(plan, damaged)
            reference = ort.InferenceSession(floating_model().SerializeToString(), providers=["CPUExecutionProvider"])
            actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            for expected, observed in zip(reference.run(None, {"input": sample}), actual.run(None, {"input": sample})):
                np.testing.assert_allclose(observed, expected, rtol=0, atol=0.02)
            # Reconstruct the reusable plan to independently reject a floating graph labelled INT8.
            plan = build_precision_plan(floating_model(), OnnxParser().build(floating_model()),
                                        XSlimSettingFactory.from_json({"strategy_parameters": config["strategy_parameters"]}),
                                        XSlimQuantizer(OnnxParser().build(floating_model())).quant_operation_types)
            with self.assertRaisesRegex(ValueError, "QDQ evidence"):
                validate_export_layout(plan, floating_model())


if __name__ == "__main__":
    unittest.main()
