"""End-to-end regression tests for the public quantization pipeline."""

import os
import shutil
import sys
import tempfile
import unittest
import warnings

import onnx
import torch
import torchvision
from onnx import TensorProto
from onnx.external_data_helper import convert_model_to_external_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import xslim


class TestQuantizePipeline(unittest.TestCase):
    """Validate model-level FP16 and dynamic quantization flows."""

    input_shape = (1, 3, 64, 64)
    model_cache_dir = os.environ.get("XSLIM_TEST_MODEL_CACHE")
    _tempdirs = []

    @classmethod
    def _export_torchvision_model(cls, model, path):
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                model,
                torch.randn(*cls.input_shape),
                path,
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                dynamo=False,
            )

    @classmethod
    def _get_model_path(cls, model_name):
        if cls.model_cache_dir:
            os.makedirs(cls.model_cache_dir, exist_ok=True)
            return os.path.join(cls.model_cache_dir, "{}.onnx".format(model_name))
        return None

    @classmethod
    def _prepare_model(cls, model_name, model):
        cached_model_path = cls._get_model_path(model_name)
        if cached_model_path is not None:
            if not os.path.exists(cached_model_path):
                cls._export_torchvision_model(model, cached_model_path)
            return cached_model_path

        tempdir = tempfile.mkdtemp()
        cls._tempdirs.append(tempdir)
        model_path = os.path.join(tempdir, "{}.onnx".format(model_name))
        cls._export_torchvision_model(model, model_path)
        return model_path

    @classmethod
    def tearDownClass(cls):
        """Remove any temporary directories created during tests."""
        for tempdir in getattr(cls, "_tempdirs", []):
            shutil.rmtree(tempdir, ignore_errors=True)

    @classmethod
    def _build_config(cls, model_path, precision_level):
        return {
            "model_parameters": {
                "onnx_model": model_path,
                "skip_onnxsim": True,
            },
            "calibration_parameters": {
                "input_parametres": [
                    {
                        "input_name": "input",
                        "input_shape": list(cls.input_shape),
                        "dtype": "float32",
                    }
                ]
            },
            "quantization_parameters": {
                "precision_level": precision_level,
                "analysis_enable": False,
            },
        }

    def test_dynamic_quantize_pipeline_supports_external_data_resnet18(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source_model_path = self._prepare_model("resnet18", torchvision.models.resnet18(weights=None))
            model_path = os.path.join(tempdir, "resnet18_external.onnx")
            output_path = os.path.join(tempdir, "resnet18_external.dynq.onnx")

            shutil.copyfile(source_model_path, model_path)

            model_with_external_data = onnx.load(model_path)
            convert_model_to_external_data(
                model_with_external_data,
                all_tensors_to_one_file=True,
                location="resnet18_external.data",
                size_threshold=0,
                convert_attribute=False,
            )
            onnx.save_model(model_with_external_data, model_path)

            external_model = onnx.load(model_path, load_external_data=False)
            self.assertTrue(
                any(initializer.data_location == TensorProto.EXTERNAL for initializer in external_model.graph.initializer)
            )
            self.assertTrue(os.path.exists(os.path.join(tempdir, "resnet18_external.data")))

            quantized_model = xslim.quantize_onnx_model(
                self._build_config(model_path, precision_level=3), output_path=output_path
            )

            self.assertTrue(os.path.exists(output_path))
            onnx.checker.check_model(quantized_model)

            node_types = {node.op_type for node in quantized_model.graph.node}
            self.assertTrue(any("DynamicQuantize" in op_type for op_type in node_types))
            self.assertIn("DequantizeLinear", node_types)

    def test_fp16_pipeline_converts_mobilenet_v2_end_to_end(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source_model_path = self._prepare_model("mobilenet_v2", torchvision.models.mobilenet_v2(weights=None))
            model_path = os.path.join(tempdir, "mobilenetv2.onnx")
            output_path = os.path.join(tempdir, "mobilenetv2.fp16.onnx")

            shutil.copyfile(source_model_path, model_path)

            fp16_model = xslim.quantize_onnx_model(
                self._build_config(model_path, precision_level=4), output_path=output_path
            )

            self.assertTrue(os.path.exists(output_path))
            onnx.checker.check_model(fp16_model)
            self.assertEqual(fp16_model.graph.input[0].type.tensor_type.elem_type, TensorProto.FLOAT)
            self.assertIn(TensorProto.FLOAT16, {initializer.data_type for initializer in fp16_model.graph.initializer})


if __name__ == "__main__":
    unittest.main()
