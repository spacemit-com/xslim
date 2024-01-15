#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from typing import Iterable, List, Set, Union, Dict, Callable, Tuple
import torch
import functools
import math
import numpy as np
from ppq.core import (
    TensorQuantizationConfig,
    QuantizationProperty,
    QuantizationStates,
    ppq_quant_param_computing_function,
    common as ppq_common,
    convert_any_to_numpy,
)
from ppq.IR import Variable
from ppq.quantization.observer import (
    TorchHistObserver,
    range as ppq_range,
)
from ppq.quantization.measure import torch_KL_divergence


class TorchXQuantObserver(TorchHistObserver):
    def __init__(
        self,
        watch_on: Variable,
        quant_cfg: TensorQuantizationConfig,
        hist_bins: int = ppq_common.OBSERVER_FLOATING_MSE_FETCHES,
        single_alg: str = None,
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

        self._single_alg = single_alg
        self._hist_bins = hist_bins
        self._channel_min_max = []

        self._force_range_min = -(2**31)
        self._force_range_max = 2**31
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
        self._percentile_collector = []
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

            if self._quant_cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                hist = torch.histc(value, self._hist_bins, min=self._full_min_val, max=self._full_max_val)
                self._hist += hist.int()

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
            self._full_min_val = torch.min(torch.cat(self._min_val_collector, dim=0)).cpu().item()
            self._full_max_val = torch.max(torch.cat(self._max_val_collector, dim=0)).cpu().item()
            percentile_reduce = torch.cat(self._percentile_collector, dim=0).float().mean(dim=0).cpu()

            self._percentile_min_val = percentile_reduce[1].item()
            self._percentile_max_val = percentile_reduce[0].item()

            if self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRICAL):
                hist_range = float(max(abs(self._full_max_val), abs(self._full_min_val)))
            else:
                hist_range = self._full_max_val - self._full_min_val

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

    def compute_mse_loss(
        self, histogram: np.ndarray, bin_start: int, bin_range: int, num_of_quant_levels: int, hist_sum: float
    ) -> float:
        bin_end = bin_start + bin_range
        if bin_end > histogram.size:
            bin_end = histogram.size

        idx_range = np.arange(0, histogram.size).astype(np.float32)
        step = (bin_end - bin_start) / num_of_quant_levels
        if step < 1:
            step = 1

        mid_val = (step - 1) / 2
        idx_range[bin_start:bin_end] -= bin_start
        idx_range[bin_start:bin_end] = idx_range[bin_start:bin_end] % step
        idx_range[bin_start:bin_end] -= mid_val
        idx_range[bin_start:bin_end] = mid_val - np.abs(idx_range[bin_start:bin_end]) + 0.5

        idx_range[:bin_start] = bin_start - idx_range[:bin_start] - 0.5
        idx_range[bin_end:] = idx_range[bin_end:] - bin_end + 0.5

        histogram = histogram / hist_sum

        loss = (idx_range * idx_range * histogram).sum()

        return float(loss)

    def compute_kl_loss(
        self, histogram: torch.Tensor, bin_start: int, bin_range: int, num_of_quant_levels: int, hist_sum: float
    ) -> float:
        bin_end = bin_start + bin_range
        if bin_end > histogram.numel():
            bin_end = histogram.numel()
        bin_range = bin_end - bin_start

        p_hist = torch.zeros(size=(bin_range,), dtype=torch.float, device=histogram.device)
        p_hist[:bin_range].copy_(histogram[bin_start:bin_end])
        p_hist[bin_range - 1] += torch.sum(histogram[bin_end:])
        p_hist[0] += torch.sum(histogram[:bin_start])
        p_hist = p_hist / hist_sum

        expand_ratio = math.ceil(bin_range / num_of_quant_levels)
        zero_pad_num = expand_ratio * num_of_quant_levels - bin_range

        q_hist = torch.zeros(size=(bin_range + zero_pad_num,), dtype=torch.float, device=histogram.device)
        q_hist[:bin_range].copy_(histogram[bin_start:bin_end])
        q_hist = q_hist.reshape((num_of_quant_levels, expand_ratio))

        positive_map = q_hist > 0
        positive_cnt = positive_map.sum(axis=1, keepdim=True)
        positive_cnt[positive_cnt == 0] = 1
        q_hist = torch.div(q_hist.sum(axis=1, keepdim=True), positive_cnt)
        q_hist = q_hist.repeat([1, expand_ratio])
        q_hist = q_hist * positive_map
        q_hist = q_hist / torch.sum(q_hist)
        q_hist = q_hist.flatten()
        q_hist = q_hist[:bin_range]

        kl_loss = torch_KL_divergence(p_hist, q_hist)

        return float(kl_loss)

    @ppq_quant_param_computing_function
    def hist_to_scale_offset(
        self,
        histogram: torch.Tensor,
        hist_bins: int,
        hist_scale: float,
        config: TensorQuantizationConfig,
        scale_threshold: float = 2**-24,
    ) -> Tuple[float, int]:
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL) and config.policy.has_property(
            QuantizationProperty.PER_TENSOR
        ):
            histogram = histogram.float().cpu()
            histogram_np = histogram.numpy()

            losses = []
            num_of_quant_levels = (self._quant_cfg.quant_max - self._quant_cfg.quant_min) + 1
            offset_step = hist_bins // num_of_quant_levels * 2
            range_step = hist_bins // num_of_quant_levels * 2

            percentile_scale, percentile_offset = ppq_range.minmax_to_scale_offset(
                max(self._force_range_min, self._percentile_min_val),
                min(self._force_range_max, self._percentile_max_val),
                config,
                self._scale_threshold,
            )

            if num_of_quant_levels > 2**10 or hist_scale == 0:
                return percentile_scale, percentile_offset

            hist_sum = float(histogram.sum())

            percentile_bin_start = math.floor((self._percentile_min_val - self._full_min_val) / hist_scale)
            percentile_bin_start = percentile_bin_start if percentile_bin_start >= 0 else 0
            percentile_bin_range = math.ceil((self._percentile_max_val - self._percentile_min_val) / hist_scale)
            percentile_kl_loss = self.compute_kl_loss(
                histogram, percentile_bin_start, percentile_bin_range, num_of_quant_levels, hist_sum
            )
            percentile_mse_loss = self.compute_mse_loss(
                histogram_np,
                percentile_bin_start,
                percentile_bin_range,
                num_of_quant_levels,
                hist_sum,
            )

            losses.append(
                {
                    "mse": percentile_mse_loss,
                    "kl": percentile_kl_loss,
                    "bin_range": percentile_bin_range,
                    "bin_start": percentile_bin_start,
                    "scale": percentile_scale,
                    "offset": percentile_offset,
                    "percentile_loss": True,
                }
            )

            for bin_start in range(0, hist_bins, offset_step):
                for bin_range in range(
                    num_of_quant_levels,
                    hist_bins - bin_start + offset_step - 1,
                    range_step,
                ):
                    min_range_val = max(self._force_range_min, self._full_min_val + bin_start * hist_scale)
                    max_range_val = min(self._force_range_max, min_range_val + bin_range * hist_scale)
                    scale, offset = ppq_range.minmax_to_scale_offset(
                        min_range_val, max_range_val, config, self._scale_threshold
                    )

                    kl_loss = self.compute_kl_loss(histogram, bin_start, bin_range, num_of_quant_levels, hist_sum)
                    mse_loss = self.compute_mse_loss(
                        histogram_np,
                        bin_start,
                        bin_range,
                        num_of_quant_levels,
                        hist_sum,
                    )

                    losses.append(
                        {
                            "mse": mse_loss,
                            "kl": kl_loss,
                            "bin_range": bin_range,
                            "bin_start": bin_start,
                            "scale": scale,
                            "offset": offset,
                        }
                    )

            losses_kl = sorted(losses, key=lambda x: x["kl"])
            losses_mse = sorted(losses_kl, key=lambda x: x["mse"])

            if self._single_alg == "mse":
                scale = losses_mse[0]["scale"]
                offset = losses_mse[0]["offset"]
                return scale, offset
            elif self._single_alg == "kl":
                scale = losses_kl[0]["scale"]
                offset = losses_kl[0]["offset"]
                return scale, offset
            else:
                valid_topk = 10
                loss_scale = [
                    x["kl"] * x["kl"] / (x["mse"] * x["mse"])
                    for x, y in zip(losses_mse[:valid_topk], losses_mse[:valid_topk])
                ]
                loss_scale = math.sqrt(sum(loss_scale) / len(loss_scale))
                losses = sorted(losses, key=lambda x: x["kl"] + x["mse"] * loss_scale)
                scale = losses[0]["scale"]
                offset = losses[0]["offset"]

                return scale, offset

        elif config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError("XQuant observer do not support PER_CHANNEL policy now, please wait.")

        elif config.policy.has_property(QuantizationProperty.SYMMETRICAL) and config.policy.has_property(
            QuantizationProperty.PER_TENSOR
        ):
            return super().hist_to_scale_offset(histogram, hist_bins, hist_scale, config, scale_threshold)

        raise Exception("Oops, there might be some mistakes.")


class TorchXQuantMSEObserver(TorchXQuantObserver):
    def __init__(
        self,
        watch_on: Variable,
        quant_cfg: TensorQuantizationConfig,
        hist_bins: int = ppq_common.OBSERVER_FLOATING_MSE_FETCHES,
    ):
        super().__init__(watch_on, quant_cfg, hist_bins, "mse")


class TorchXQuantKLObserver(TorchXQuantObserver):
    def __init__(
        self,
        watch_on: Variable,
        quant_cfg: TensorQuantizationConfig,
        hist_bins: int = ppq_common.OBSERVER_FLOATING_MSE_FETCHES,
    ):
        super().__init__(watch_on, quant_cfg, hist_bins, "kl")
