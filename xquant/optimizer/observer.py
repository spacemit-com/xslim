from typing import Iterable, List, Set, Union, Dict, Callable, Tuple
import torch
import functools
import numpy as np
from ppq.core import (
    OBSERVER_MSE_HIST_BINS,
    PASSIVE_OPERATIONS,
    TensorQuantizationConfig,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TargetPlatform,
    empty_ppq_cache,
    ppq_warning,
    ppq_quant_param_computing_function,
    common as ppq_common,
    convert_any_to_numpy,
)
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.quantization.observer import (
    CalibrationHook,
    OperationObserver,
    TensorObserverFactroy,
    TorchHistObserver,
    TorchMinMaxObserver,
    TorchMSEObserver,
    TorchPercentileObserver,
    range as ppq_range,
)
from ppq.quantization.measure import torch_KL_divergence


class TorchXQuantObserver(TorchHistObserver):
    def __init__(
        self,
        watch_on: Variable,
        quant_cfg: TensorQuantizationConfig,
        hist_bins: int = ppq_common.OBSERVER_FLOATING_MSE_FETCHES,
    ):
        super().__init__(watch_on, quant_cfg)
        if not ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE in quant_cfg.detail:
            self._percentile = ppq_common.OBSERVER_PERCENTILE
        else:
            self._percentile = quant_cfg.detail[ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE]

        if ppq_common.OBSERVER_MIN_SCALE_MANUL_OVERRIDE in quant_cfg.detail:
            self._scale_threshold = quant_cfg.detail[ppq_common.OBSERVER_MIN_SCALE_MANUL_OVERRIDE]
        else:
            self._scale_threshold = ppq_common.OBSERVER_MIN_SCALE

        self._hist_bins = hist_bins
        self._last_hist = torch.zeros([self._hist_bins], dtype=torch.float32)
        self._hist_history = []
        self._channel_min_max = []

        self._force_range_min = -(2**16)
        self._force_range_max = 2**16
        if watch_on.source_op is not None:
            self._force_range_min = (
                watch_on.source_op._detail.get("output_force_range", {})
                .get(watch_on.name, {})
                .get("min", self._force_range_min)
            )
            self._force_range_max = (
                watch_on.source_op._detail.get("output_force_range", {})
                .get(watch_on.name, {})
                .get("max", self._force_range_max)
            )
        self._hist_edge = 0.002
        self._percentile_collector = []
        self._percentile_maxs = []
        self._percentile_mins = []
        self._min_val_collector = []
        self._max_val_collector = []

    def observe(self, value: torch.Tensor):
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        if self._phase == "Detecting Minmax":
            assert value is not None, (
                "You are observing an Empty Tensor. "
                "(This Error is usually due to you have a wrong Quantizer configuration.)"
            )
            assert value.numel() > 0, f"You are observing an empty tensor({self._watch_on.name})."
            assert isinstance(value, torch.Tensor), "TorchMinMaxObserver can only deal with torch Tensor values"
            if self._quant_cfg.state == QuantizationStates.INITIAL:
                if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                    numel = value.numel()
                    min_idx, max_idx = int(numel * (1 - self._percentile)), int(numel * (self._percentile))
                    # torch.kthvalue needs index from 1 to numel ...
                    min_idx = max(0, min_idx) + 1
                    max_idx = min(max_idx, numel - 1) + 1
                    _min = torch.kthvalue(value.flatten(), k=min_idx, dim=0)[0].view(1, -1)
                    _max = torch.kthvalue(value.flatten(), k=max_idx, dim=0)[0].view(1, -1)
                    self._percentile_collector.append(torch.cat([_max, _min], dim=-1))
                    self._min_val_collector.append(value.min().reshape(1))
                    self._max_val_collector.append(value.max().reshape(1))
                elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                    raise PermissionError("Percentile observer can not deal with per channel quantization.")
                else:
                    raise TypeError("Min-max Observer only work with per-tensor or per-channel quantize policy.")

        elif self._phase == "Collating Hist":
            if self._hist is None:
                self._hist = torch.zeros(size=(self._hist_bins,), dtype=torch.int32, device=value.device)
                self._hist_min = 0
                self._hist_max = 0

            if self._quant_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                hist = torch.histc(value, self._hist_bins, min=self._percentile_min_val, max=self._percentile_max_val)
                self._hist += hist.int()
                self._hist_min += int(torch.sum(value < self._percentile_min_val))
                self._hist_max += int(torch.sum(value > self._percentile_max_val))

            elif self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRICAL):
                hist = torch.histc(torch.abs(value), self._hist_bins, min=0, max=self._hist_scale * self._hist_bins)
                self._hist += hist.int()

            else:
                raise TypeError(
                    "Quantization Property is invalid, " "expect either ASYMMETRICAL or SYMMETRICAL config here."
                )

    def render_quantization_config(self):
        # If TQC is not prepared for calibration, just skip this execution.
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        if not self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            raise ValueError("Hist observer can only apply with per-tensor quantization config.")

        if self._phase == "Detecting Minmax":
            self._abs_min_val = torch.min(torch.cat(self._min_val_collector, dim=0)).cpu().item()
            self._abs_max_val = torch.max(torch.cat(self._max_val_collector, dim=0)).cpu().item()
            self._percentile_collector = torch.cat(self._percentile_collector, dim=0).float().mean(dim=0).cpu()

            self._percentile_min_val = self._percentile_collector[1].item()
            self._percentile_max_val = self._percentile_collector[0].item()

            if self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRICAL):
                hist_range = float(max(abs(self._abs_max_val), abs(self._abs_min_val)))
            else:
                hist_range = self._abs_max_val - self._abs_min_val

            self._hist_scale = hist_range / self._hist_bins
            self._phase = "Collating Hist"
        elif self._phase == "Collating Hist":
            scale, offset = self.hist_to_scale_offset(
                histogram=self._hist, hist_bins=self._hist_bins, hist_scale=self._hist_scale, config=self._quant_cfg
            )
            device = self._hist.device
            self._quant_cfg.scale = torch.tensor([scale], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([offset], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED

    def compute_mse_loss(self, histogram: np.ndarray, start: int, step: int, end: int):
        if end > histogram.size:
            end = histogram.size

        idx_range = np.arange(0, histogram.size).astype(np.float32)

        mid_val = (step - 1) / 2
        idx_range[start:end] -= start
        idx_range[start:end] = idx_range[start:end] % step
        idx_range[start:end] -= mid_val
        idx_range[start:end] = mid_val - np.abs(idx_range[start:end]) + 0.5

        idx_range[:start] = start - idx_range[:start] - 0.5
        idx_range[end:] = idx_range[end:] - end + 0.5

        loss = (idx_range * idx_range * histogram).sum()

        return float(loss)

    @ppq_quant_param_computing_function
    def hist_to_scale_offset(
        self,
        histogram: torch.Tensor,
        hist_bins: int,
        hist_scale: float,
        config: TensorQuantizationConfig,
        scale_threshold: float = ppq_common.OBSERVER_MIN_SCALE,
    ) -> Tuple[float, int]:
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL) and config.policy.has_property(
            QuantizationProperty.PER_TENSOR
        ):
            histogram = histogram.to("cpu").float()
            num_of_elements = histogram.sum()

            full_scale, full_offset = ppq_range.minmax_to_scale_offset(
                max(self._force_range_min, self._abs_min_val),
                min(self._force_range_max, self._abs_max_val),
                config,
                self._scale_threshold,
            )

            percentile_scale, percentile_offset = ppq_range.minmax_to_scale_offset(
                max(self._force_range_min, self._percentile_min_val),
                min(self._force_range_max, self._percentile_max_val),
                config,
                self._scale_threshold,
            )

            num_of_quant_levels = (self._quant_cfg.quant_max - self._quant_cfg.quant_min) + 1
            losses = []

            histogram[: int(hist_bins * self._hist_edge)] = 0

            histogram[-int(hist_bins * self._hist_edge) :] = 0

            hist_sum = histogram.sum()
            for bin_start in range(0, hist_bins, num_of_quant_levels):
                for bin_range in range(
                    num_of_quant_levels,
                    hist_bins - bin_start + num_of_quant_levels - 1,
                    num_of_quant_levels,
                ):
                    p_hist = torch.zeros(size=(bin_range,), dtype=torch.float, device="cpu")
                    p_hist[:bin_range].copy_(histogram[bin_start : bin_start + bin_range])
                    p_hist[bin_range - 1] += torch.sum(histogram[bin_start + bin_range :])
                    p_hist[0] += torch.sum(histogram[:bin_start])

                    p_hist = p_hist / hist_sum

                    expand_ratio = int(bin_range / num_of_quant_levels)
                    q_hist = histogram[bin_start : bin_start + bin_range].clone()
                    q_hist = q_hist.reshape((num_of_quant_levels, expand_ratio))
                    positive_map = q_hist > 0
                    positive_cnt = positive_map.sum(axis=1, keepdim=True)
                    positive_cnt[positive_cnt == 0] = 1
                    q_hist = torch.div(q_hist.sum(axis=1, keepdim=True), positive_cnt)
                    q_hist = q_hist.repeat([1, expand_ratio])
                    q_hist = q_hist * positive_map
                    q_hist = q_hist / torch.sum(q_hist)
                    q_hist = q_hist.flatten()

                    losses.append(
                        {
                            "kl": torch_KL_divergence(p_hist, q_hist),
                            "bin_range": bin_range,
                            "bin_start": bin_start,
                        }
                    )

            losses = sorted(losses, key=lambda x: x["kl"])
            bin_start = losses[0]["bin_start"]
            bin_range = losses[0]["bin_range"]
            min_range_val = max(self._force_range_min, self._abs_min_val + bin_start * hist_scale)
            max_range_val = min(self._force_range_max, min_range_val + bin_range * hist_scale)
            scale, offset = ppq_range.minmax_to_scale_offset(
                min_range_val, max_range_val, config, self._scale_threshold
            )
            return scale, offset

        elif config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError("Torch Mse observer do not support PER_CHANNEL policy now, please wait.")

        elif config.policy.has_property(QuantizationProperty.SYMMETRICAL) and config.policy.has_property(
            QuantizationProperty.PER_TENSOR
        ):
            return super().hist_to_scale_offset(histogram, hist_bins, hist_scale, config, scale_threshold)

        raise Exception("Oops, there might be some mistakes.")
