import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

import xslim
from xslim.optimizer.observer import TorchAutoObserver
from xslim.optimizer.refine import QuantizeConfigRefinePass
from xslim.ppq_decorator import BaseGraph, Operation, QuantableOperation, QuantizationStates, TargetPlatform, Variable
from xslim.ppq_decorator import QuantizeFusionPass, QuantizeSimplifyPass
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
        if op_type == "Gemm":
            for operation in graph.operations.values():
                weight = Variable(operation.name + ".weight", value=torch.eye(4), is_parameter=True,
                                  dest_ops=[operation])
                bias = Variable(operation.name + ".bias", value=torch.zeros(4), is_parameter=True,
                                dest_ops=[operation])
                operation.inputs.extend([weight, bias])
                graph.variables.update({weight.name: weight, bias.name: bias})
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

    def connected(self, rules, op_type):
        graph = self.dispatch(["producer", "consumer"], rules, op_type=op_type)
        first, second = graph.operations["producer"], graph.operations["consumer"]
        old = second.inputs[0]
        old.dest_ops.remove(second)
        second.inputs[0] = first.outputs[0]
        first.outputs[0].dest_ops.append(second)
        del graph.variables[old.name]
        QuantizeConfigRefinePass().optimize(graph)
        QuantizeFusionPass(fuse_activation=True, fuse_passive_op=True, fuse_relu_clip=True,
                           activation_type={"Relu", "Clip"}).optimize(graph)
        QuantizeSimplifyPass().optimize(graph, dataloader=[], executor=None)
        return graph

    def test_pinned_method_survives_fusion_and_simplify(self):
        for name, op_type in [("producer", "Relu"), ("consumer", "Sigmoid")]:
            with self.subTest(name=name):
                graph = self.connected([{"pattern": name, "precision": "int8", "calibration_type": "minmax"}],
                                       op_type=op_type)
                QuantizeConfigRefinePass.apply_operator_calibration(graph)
                operation = graph.operations[name]
                for config, variable in operation.config_with_variable:
                    if not variable.is_parameter:
                        self.assertEqual(config.dominated_by.observer_algorithm, "minmax")

    def test_conflicting_pins_on_shared_domain_fail(self):
        graph = self.connected([
            {"pattern": "producer", "precision": "int8", "calibration_type": "minmax"},
            {"pattern": "consumer", "precision": "int8", "calibration_type": "percentile"},
        ], op_type="Sigmoid")
        with self.assertRaisesRegex(ValueError, "Conflicting operator calibration"):
            QuantizeConfigRefinePass.apply_operator_calibration(graph)

    def test_int8_rule_preserves_shape_input_socket(self):
        graph = self.dispatch(["reshape"], [{"pattern": "*", "precision": "int8"}], op_type="Reshape")
        operation = graph.operations["reshape"]
        # Construct the real data + shape sockets through the public config initializer.
        shape = Variable("shape", value=torch.tensor([1]), is_parameter=True)
        operation.inputs.append(shape)
        config = XSlimQuantizer.create_default_quant_config(operation)
        self.assertEqual(config.input_quantization_config[0].state, QuantizationStates.INITIAL)
        self.assertEqual(config.input_quantization_config[1].state, QuantizationStates.FP32)

    def test_explicit_int8_preserves_fp32_bias(self):
        graph = self.dispatch(["dense"], [{"pattern": "*", "precision": "int8"}], op_type="Gemm")
        config = graph.operations["dense"].config
        self.assertEqual(config.input_quantization_config[0].state, QuantizationStates.INITIAL)
        self.assertEqual(config.input_quantization_config[1].state, QuantizationStates.INITIAL)
        self.assertEqual(config.input_quantization_config[2].state, QuantizationStates.FP32)

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
        self.convert_and_execute("minmax", None)

    def test_public_api_auto_and_explicit_calibration_overrides(self):
        for global_method, operator_method in [("auto", None), ("minmax", "auto"), ("auto", "minmax")]:
            with self.subTest(global_method=global_method, operator_method=operator_method):
                self.convert_and_execute(global_method, operator_method)

    def convert_and_execute(self, global_method, operator_method):
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
            original_render = TorchAutoObserver.render_quantization_config
            with patch.object(TorchAutoObserver, "render_quantization_config", autospec=True,
                              side_effect=original_render) as render:
                result = xslim.quantize_onnx_model({
                "model_parameters": {"onnx_model": str(model_path), "skip_onnxsim": True, "opset": 18},
                "calibration_parameters": {"calibration_step": 10, "calibration_device": "cpu",
                    "calibration_type": global_method, "input_parameters": [{"input_name": "input",
                        "input_shape": [1, 4], "file_type": "npy", "data_list_path": str(list_path)}]},
                "quantization_parameters": {"analysis_enable": False, "finetune_level": 0,
                    "ignore_op_names": ["encoder.matmul"], "operator_quantization": [
                        {"pattern": "encoder.*", "precision": "int8", "calibration_type": operator_method},
                        {"pattern": "head.*", "precision": "fp32"}]},
            }, output_path=str(output_path))
            expect_auto = operator_method == "auto" or (operator_method is None and global_method == "auto")
            self.assertEqual(bool(render.call_count), expect_auto)
            for call in render.call_args_list:
                report = call.args[0]._quant_cfg.detail["auto_calibration"]
                self.assertEqual(set(report["mse"]), {"minmax", "percentile", "mse", "kl"})
                self.assertEqual(report["mse"][report["method"]], min(report["mse"].values()))
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
