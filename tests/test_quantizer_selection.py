import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

import xslim
from xslim.optimizer.refine import QuantizeConfigRefinePass
from xslim.ppq_decorator import BaseGraph, Operation, QuantableOperation, QuantizationStates, TargetPlatform, Variable
from xslim.quantizer.xslim import XSlimQuantizer
from xslim.xslim_setting import XSlimSetting, XSlimSettingFactory


class OperatorQuantizationTests(unittest.TestCase):
    def settings(self, rules, **parameters):
        return XSlimSettingFactory.from_json({
            "quantization_parameters": {"operator_quantization": rules, **parameters},
        })

    def dispatch(self, names, rules, op_type="Relu", **parameters):
        graph = BaseGraph(name="selection")
        for name in names:
            operation = Operation(name=name, op_type=op_type, attributes={}, platform=TargetPlatform.UNSPECIFIED)
            source = Variable(name=name + ".input", dest_ops=[operation])
            output = Variable(name=name + ".output", source_op=operation)
            operation.inputs.append(source)
            operation.outputs.append(output)
            graph.operations[name] = operation
            graph.variables.update({source.name: source, output.name: output})
        quantizer = XSlimQuantizer(graph)
        quantizer.build_prequant_pipeline = Mock(return_value=Mock())
        quantizer.build_quant_pipeline = Mock(return_value=Mock())
        quantizer.quantize(torch.ones(1), [], Mock(), self.settings(rules, **parameters))
        return graph

    def test_ordered_wildcards_override_exclusions_and_keep_case(self):
        graph = self.dispatch(
            ["encoder.0.relu", "encoder.1.relu", "encoder.2.relu", "Encoder.0.relu", "tail"],
            [{"pattern": "encoder.*", "precision": "fp32"},
             {"pattern": "encoder.?.relu", "precision": "int8"},
             {"pattern": "encoder.[12].relu", "precision": "fp32"}],
            ignore_op_types=["Relu"],
        )
        self.assertIsInstance(graph.operations["encoder.0.relu"], QuantableOperation)
        for name in ["encoder.1.relu", "encoder.2.relu", "Encoder.0.relu", "tail"]:
            self.assertEqual(graph.operations[name].platform, TargetPlatform.FP32)

    def test_exact_name_does_not_match_operator_type(self):
        with self.assertRaisesRegex(ValueError, "matched no operations"):
            self.dispatch(["encoder.relu"], [{"pattern": "Relu", "precision": "fp32"}])

    def test_explicit_int8_keeps_socket_constraints_and_overrides_precision_refine(self):
        graph = self.dispatch(["activation"], [{"pattern": "*", "precision": "int8", "calibration_type": "minmax"}],
                              precision_level=2)
        QuantizeConfigRefinePass(precision_level=3).optimize(graph)
        operation = graph.operations["activation"]
        for config, _ in operation.config_with_variable:
            self.assertEqual(config.state, QuantizationStates.INITIAL)
            self.assertEqual(config.observer_algorithm, "minmax")

    def test_unsupported_int8_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support static INT8"):
            self.dispatch(["shape"], [{"pattern": "*", "precision": "int8"}], op_type="Shape")

    def test_invalid_rules_fail_during_config_parsing(self):
        for rules in [None, {}, [1], [{"pattern": "", "precision": "int8"}],
                      [{"pattern": "*", "precision": "fp16"}],
                      [{"pattern": "*", "precision": "fp32", "calibration_type": "mse"}],
                      [{"pattern": "*", "precision": "int8", "calibration_type": "unknown"}]]:
            with self.subTest(rules=rules), self.assertRaises((ValueError, TypeError)):
                self.settings(rules)
        with self.assertRaisesRegex(ValueError, "only for static INT8"):
            self.settings([{"pattern": "*", "precision": "int8"}], precision_level=3)

    def test_defaults_leave_operator_selection_disabled(self):
        self.assertEqual(XSlimSetting().quantization_parameters.operator_quantization, [])


class OperatorQuantizationPipelineTests(unittest.TestCase):
    def test_public_api_exports_explicit_fp32_and_int8_and_executes(self):
        # Synthetic calibration proves dispatch/export/execution, not deployment accuracy.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = numpy_helper.from_array(np.eye(4, dtype=np.float32), "weight")
            graph = helper.make_graph([
                helper.make_node("MatMul", ["input", "weight"], ["hidden"], name="encoder.matmul"),
                helper.make_node("Sigmoid", ["hidden"], ["output"], name="head.sigmoid"),
            ], "selection", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])], [weights])
            model_path = root / "model.onnx"
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10), model_path)
            sample_path = root / "sample.npy"
            sample = np.array([[-1., 0., 1., 2.]], dtype=np.float32)
            np.save(sample_path, sample)
            list_path = root / "data.list"
            list_path.write_text((str(sample_path) + "\n") * 10)
            output_path = root / "quantized.onnx"
            result = xslim.quantize_onnx_model({
                "model_parameters": {"onnx_model": str(model_path), "skip_onnxsim": True, "opset": 18},
                "calibration_parameters": {"calibration_step": 10, "calibration_device": "cpu",
                    "calibration_type": "minmax", "input_parameters": [{"input_name": "input",
                        "input_shape": [1, 4], "file_type": "npy", "data_list_path": str(list_path)}]},
                "quantization_parameters": {"analysis_enable": False, "finetune_level": 0,
                    "ignore_op_names": ["encoder.matmul"], "operator_quantization": [
                        {"pattern": "encoder.*", "precision": "int8"},
                        {"pattern": "head.*", "precision": "fp32"}]},
            }, output_path=str(output_path))
            onnx.checker.check_model(result)
            nodes = {node.name: node for node in result.graph.node}
            self.assertEqual(nodes["head.sigmoid"].input[0], "hidden")
            self.assertTrue(any(node.op_type == "QuantizeLinear" for node in result.graph.node))
            session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            actual = session.run(None, {"input": sample})[0]
            # Predeclared tolerance for this identity-weight toy model.
            np.testing.assert_allclose(actual, 1. / (1. + np.exp(-sample)), rtol=0, atol=0.02)


if __name__ == "__main__":
    unittest.main()
