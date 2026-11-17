import unittest

import torch

from xslim.optimizer.observer import TorchAutoObserver
from xslim.ppq_decorator import (
    PPQLinearQuantFunction,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TensorQuantizationConfig,
    Variable,
    ppq_common,
)
from xslim.xslim_setting import XSlimSetting, XSlimSettingFactory


class AutoCalibrationTests(unittest.TestCase):
    def config(self, methods=None, sample_limit=8192):
        return TensorQuantizationConfig(
            policy=QuantizationPolicy(QuantizationProperty.PER_TENSOR + QuantizationProperty.LINEAR
                                      + QuantizationProperty.ASYMMETRICAL + QuantizationProperty.POWER_OF_2),
            rounding=RoundingPolicy.ROUND_HALF_EVEN, num_of_bits=8, quant_min=-128, quant_max=127,
            observer_algorithm="auto", state=QuantizationStates.INITIAL,
            detail={"auto_calibration_methods": methods or ["minmax", "percentile", "mse", "kl"],
                    "auto_calibration_samples": sample_limit, ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE: .999},
        )

    def calibrate(self, values, methods=None, config=None):
        config = config if config is not None else self.config(methods)
        observer = TorchAutoObserver(Variable("activation"), config)
        observer.observe(values)
        observer.render_quantization_config()
        # First pass must not quantize the graph or mutate its shared scale.
        self.assertEqual(config.state, QuantizationStates.INITIAL)
        self.assertIsNone(config.scale)
        observer.observe(values)
        observer.render_quantization_config()
        self.assertEqual(config.state, QuantizationStates.ACTIVATED)
        return config

    def test_data_changes_selected_method_and_ties_follow_candidate_order(self):
        regular = torch.linspace(-.99, .99, 4096)
        outlier = torch.cat([torch.linspace(-.99, .99, 4095), torch.tensor([1.01])])
        first = self.calibrate(regular, ["minmax", "percentile"])
        second = self.calibrate(outlier, ["minmax", "percentile"])
        self.assertEqual(first.observer_algorithm, "minmax")
        self.assertEqual(second.observer_algorithm, "percentile")
        self.assertLess(second.detail["auto_calibration"]["mse"]["percentile"],
                        second.detail["auto_calibration"]["mse"]["minmax"])
        reversed_order = self.calibrate(regular, ["percentile", "minmax"])
        self.assertEqual(reversed_order.observer_algorithm, "percentile")

    def test_all_candidates_are_scored_with_actual_qdq(self):
        values = torch.linspace(-.99, 1.01, 256)
        config = self.calibrate(values)
        report = config.detail["auto_calibration"]
        self.assertEqual(set(report["mse"]), {"minmax", "percentile", "mse", "kl"})
        expected = (PPQLinearQuantFunction(values, config).double() - values.double()).square().mean().item()
        self.assertAlmostEqual(report["mse"][report["method"]], expected)
        self.assertEqual(report["mse"][report["method"]], min(report["mse"].values()))

    def test_sampling_is_bounded_and_includes_later_calibration_batches(self):
        observer = TorchAutoObserver(Variable("activation"), self.config(["minmax"], sample_limit=32))
        observer.observe(torch.zeros(1024))
        observer.observe(torch.ones(1024))
        self.assertEqual(observer._samples.numel(), 32)
        self.assertGreaterEqual(observer._samples.sum().item(), 8)
        self.assertLessEqual(observer._samples.sum().item(), 24)
        observer.render_quantization_config()
        observer.observe(torch.zeros(1024))
        observer.observe(torch.ones(1024))
        observer.render_quantization_config()
        self.assertEqual(observer._quant_cfg.detail["auto_calibration"]["sample_count"], 32)
        self.assertEqual(observer._samples.numel(), 0)

    def test_sampling_is_independent_of_batch_partition(self):
        values = torch.cat([torch.zeros(1024), torch.ones(1024)])
        samples = []
        for batches in [[values], list(values.split(1024)),
                        [values[:1024], *values[1024:].split(1)]]:
            observer = TorchAutoObserver(Variable("activation"), self.config(["minmax"], sample_limit=32))
            for batch in batches:
                observer.observe(batch)
            samples.append(observer._samples)
        self.assertTrue(torch.equal(samples[0], samples[1]))
        self.assertTrue(torch.equal(samples[0], samples[2]))
        self.assertGreaterEqual(samples[0].sum().item(), 8)
        self.assertLessEqual(samples[0].sum().item(), 24)

    def test_shared_quantization_domain_keeps_one_selected_scale(self):
        master = self.config(["minmax", "percentile"])
        shared = self.config()
        shared.dominated_by = master
        self.calibrate(torch.linspace(-.99, .99, 256), config=master)
        self.assertIs(shared.dominated_by, master)
        self.assertTrue(torch.equal(shared.scale, master.scale))
        self.assertEqual(shared.state, QuantizationStates.OVERLAPPED)

    def test_constant_input_and_empty_or_nonfinite_data(self):
        config = self.calibrate(torch.zeros(32))
        self.assertTrue(torch.isfinite(config.scale).all())
        for invalid in [torch.tensor([float("nan")]), torch.tensor([float("inf")])]:
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Non-finite"):
                TorchAutoObserver(Variable("activation"), self.config()).observe(invalid)
        observer = TorchAutoObserver(Variable("empty"), self.config())
        observer.observe(torch.empty(0))
        with self.assertRaisesRegex(ValueError, "No non-empty"):
            observer.render_quantization_config()

    def test_configuration_validation_and_defaults(self):
        self.assertEqual(XSlimSetting().calibration_parameters.calibration_type, "default")
        for field, value in [("auto_calibration_methods", []), ("auto_calibration_methods", ["auto"]),
                             ("auto_calibration_methods", ["minmax", "minmax"]),
                             ("auto_calibration_methods", [1]), ("auto_calibration_methods", None),
                             ("auto_calibration_samples", 0), ("auto_calibration_samples", True),
                             ("auto_calibration_samples", 1.5)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                XSlimSettingFactory.from_json({"calibration_parameters": {
                    "input_parameters": [{}], field: value}})
        for level in [3, 4, 100]:
            with self.subTest(level=level), self.assertRaisesRegex(ValueError, "only for static INT8"):
                XSlimSettingFactory.from_json({
                    "calibration_parameters": {"calibration_type": "auto", "input_parameters": [{}]},
                    "quantization_parameters": {"precision_level": level},
                })


if __name__ == "__main__":
    unittest.main()
