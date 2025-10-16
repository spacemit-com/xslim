#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import functools
import math
import time
from typing import Callable, Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import torch
from xslim.logger import logger

from ..defs import OBSERVER_FLOATING_MSE_FETCHES, OBSERVER_MIN_SCALE_THRESHOLD, OBSERVER_PERCENTILE
from ..ppq_decorator import (
    BaseTensorObserver,
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
    Variable,
    minmax_to_scale_offset,
    ppq_common,
    ppq_observer,
    ppq_quant_param_computing_function,
    torch_KL_divergence,
)


class TorchXSlimObserver(BaseTensorObserver):
    def __init__(
        self,
        watch_on: Variable,
        quant_cfg: TensorQuantizationConfig,
        hist_bins: int = OBSERVER_FLOATING_MSE_FETCHES,
        single_alg: str = None,
    ):
        self._watch_on = watch_on
        self._phase = "Detecting Minmax"
        self._hist = None
        self._hist_scale = None
        self._min = None
        self._max = None
        self._hist_bins = hist_bins
        self._quant_cfg = quant_cfg
        if not ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE in quant_cfg.detail:
            self._percentile = OBSERVER_PERCENTILE
        else:
            self._percentile = quant_cfg.detail[ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE]

        if ppq_common.OBSERVER_MIN_SCALE_MANUL_OVERRIDE in quant_cfg.detail:
            self._scale_threshold = quant_cfg.detail[ppq_common.OBSERVER_MIN_SCALE_MANUL_OVERRIDE]
        else:
            self._scale_threshold = OBSERVER_MIN_SCALE_THRESHOLD

        self._single_alg = single_alg
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
        self._none_value_detected = False
        self._value_device = "cpu"
        self._num_of_quant_levels = (self._quant_cfg.quant_max - self._quant_cfg.quant_min) + 1
        self._offset_step = int(self._hist_bins // self._num_of_quant_levels * 2)
        self._range_step = int(self._hist_bins // self._num_of_quant_levels * 2)
        observe_batch_size = self._hist_bins // self._range_step
        self._hist_p_tensor = torch.zeros(
            size=(observe_batch_size, self._hist_bins), dtype=torch.float, device=self._value_device
        )
        self._hist_q_tensor = torch.zeros(
            size=(observe_batch_size, self._hist_bins), dtype=torch.float, device=self._value_device
        )

    def observe(self, value: torch.Tensor):
        if self._quant_cfg.state not in {QuantizationStates.INITIAL}:
            return

        if isinstance(value, torch.Tensor):
            self._value_device = value.device.type

        if value is None or value.numel() == 0:
            self._none_value_detected = True
            return

        assert isinstance(value, torch.Tensor), "XSlimObserver can only deal with torch Tensor values"
        if self._phase == "Detecting Minmax":
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

        if self._none_value_detected:
            logger.warning("None value detected at var {}".format(self._watch_on.name))
            self._quant_cfg.scale = torch.tensor([1.0], dtype=torch.float32, device=self._value_device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([0], dtype=torch.float32, device=self._value_device).squeeze(0)
            self._quant_cfg.state = QuantizationStates.ACTIVATED
            self._quant_cfg.detail["NONE_VALUE"] = True
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
            self._quant_cfg.scale = torch.tensor([scale], dtype=torch.float32, device=self._value_device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([offset], dtype=torch.float32, device=self._value_device).squeeze(0)
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

    def compute_mse_loss_batch(
        self,
        histogram: torch.Tensor,
        bin_start: int,
        bin_range: torch.Tensor,
        num_of_quant_levels: int,
        hist_sum: float,
    ) -> float:
        bin_end = bin_start + bin_range
        batch_size = bin_range.numel()
        idx_range = torch.range(0, histogram.numel() - 1).to(torch.int32).tile([batch_size, 1]).to(histogram.device)
        idx_range_cpy = (
            torch.range(0, histogram.numel() - 1).to(torch.float32).tile([batch_size, 1]).to(histogram.device)
        )
        step = (
            torch.clip(bin_range / num_of_quant_levels, 1, histogram.numel())
            .view(-1, 1)
            .to(torch.int32)
            .to(histogram.device)
        )
        mid_val = (step - 1) / 2

        idx_range[:, bin_start:] -= bin_start
        idx_range[:, bin_start:] = idx_range[:, bin_start:] % step
        idx_range = idx_range.to(torch.float32)
        idx_range[:, bin_start:] -= mid_val
        idx_range[:, bin_start:] = mid_val - torch.abs(idx_range[:, bin_start:]) + 0.5
        idx_range[:, :bin_start] = bin_start - idx_range[:, :bin_start] - 0.5

        for idx in range(batch_size):
            idx_range[idx, bin_end[idx] :].copy_(idx_range_cpy[idx, bin_end[idx] :] - bin_end[idx] + 0.5)

        histogram = histogram / hist_sum

        mse_loss = (idx_range * idx_range * histogram).sum(dim=-1)

        return mse_loss

    def compute_kl_loss_batch(
        self,
        histogram: torch.Tensor,
        bin_start: int,
        bin_range: torch.Tensor,
        num_of_quant_levels: int,
        hist_sum: float,
    ):
        batch_size = bin_range.numel()
        bin_end = bin_start + bin_range
        self._hist_p_tensor[:, : histogram.numel() - bin_start].copy_(histogram[bin_start:])
        self._hist_q_tensor[:, : histogram.numel() - bin_start].copy_(histogram[bin_start:])
        hist_start_sum = torch.sum(histogram[:bin_start])
        for idx in range(batch_size):
            self._hist_p_tensor[idx, bin_end[idx] - 1] += torch.sum(histogram[bin_end[idx] :])
            self._hist_p_tensor[idx, bin_end[idx] :] = 0
            self._hist_p_tensor[idx:, 0] += hist_start_sum
            self._hist_q_tensor[idx, bin_end[idx] :] = 0

        p_hist = self._hist_p_tensor[:batch_size] / hist_sum

        q_hist = self._hist_q_tensor[:batch_size].view(batch_size, num_of_quant_levels, -1)
        expand_size = q_hist.shape[-1]
        positive_map = q_hist > 0
        positive_cnt = positive_map.sum(axis=-1, keepdim=True)
        positive_cnt[positive_cnt == 0] = 1
        q_hist = torch.div(q_hist.sum(axis=-1, keepdim=True), positive_cnt)
        q_hist = q_hist.repeat([1, 1, expand_size])
        q_hist = q_hist * positive_map
        q_hist = q_hist.view(batch_size, -1)
        q_hist = q_hist / torch.sum(q_hist, dim=-1, keepdim=True)

        def torch_KL_divergence_batch(hist: torch.Tensor, ref_hist: torch.Tensor, eps=1e-30) -> float:
            if len(hist) != len(ref_hist):
                raise ValueError("Can not compute KL divergence, len(hist) != len(ref_hist")
            lhs = hist.double()
            batch_size = hist.shape[0]
            rhs = torch.log10(hist.double() + eps) - torch.log10(ref_hist.double() + eps)
            return torch.matmul(lhs.view(batch_size, 1, -1), rhs.view(batch_size, -1, 1)).view(-1)

        kl_loss = torch_KL_divergence_batch(p_hist, q_hist)

        return kl_loss

    @ppq_quant_param_computing_function
    def hist_to_scale_offset(
        self,
        histogram: torch.Tensor,
        hist_bins: int,
        hist_scale: float,
        config: TensorQuantizationConfig,
        scale_threshold: float = OBSERVER_MIN_SCALE_THRESHOLD,
    ) -> Tuple[float, int]:
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL) and config.policy.has_property(
            QuantizationProperty.PER_TENSOR
        ):
            t_start = time.time()

            self._hist_p_tensor = self._hist_p_tensor.to(histogram.device)
            self._hist_q_tensor = self._hist_q_tensor.to(histogram.device)
            histogram_cpu = histogram.float().cpu()
            histogram_np = histogram_cpu.numpy()

            losses = []

            percentile_scale, percentile_offset = minmax_to_scale_offset(
                max(self._force_range_min, self._percentile_min_val),
                min(self._force_range_max, self._percentile_max_val),
                config,
                self._scale_threshold,
            )

            full_scale, full_offset = minmax_to_scale_offset(
                max(self._force_range_min, self._full_min_val),
                min(self._force_range_max, self._full_max_val),
                config,
                self._scale_threshold,
            )

            if hist_scale == 0.0 or abs(self._percentile_min_val - self._percentile_max_val) < self._scale_threshold:
                logger.debug("observer render time cost {}".format(time.time() - t_start))
                return full_scale, full_offset

            percentile_bin_start = math.floor((self._percentile_min_val - self._full_min_val) / hist_scale)
            percentile_bin_start = max(percentile_bin_start, 0)
            percentile_bin_range = math.ceil((self._percentile_max_val - self._percentile_min_val) / hist_scale)
            if self._num_of_quant_levels > 2**10:
                logger.debug("observer render time cost {}".format(time.time() - t_start))
                return percentile_scale, percentile_offset

            hist_sum = float(histogram.sum())
            if percentile_bin_range + percentile_bin_start > hist_bins:
                percentile_bin_range = hist_bins - percentile_bin_start
            percentile_kl_loss = self.compute_kl_loss_batch(
                histogram,
                percentile_bin_start,
                torch.tensor([percentile_bin_range]),
                self._num_of_quant_levels,
                hist_sum,
            )[0].item()
            percentile_mse_loss = self.compute_mse_loss(
                histogram_np,
                percentile_bin_start,
                percentile_bin_range,
                self._num_of_quant_levels,
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

            for bin_start in range(0, hist_bins, self._offset_step):
                min_range_val_list = []
                max_range_val_list = []
                bin_range_list = []
                losses_start_idx = len(losses)
                for bin_range in range(
                    self._num_of_quant_levels,
                    hist_bins - bin_start + self._offset_step - 1,
                    self._range_step,
                ):
                    min_range_val = max(self._force_range_min, self._full_min_val + bin_start * hist_scale)
                    max_range_val = min(self._force_range_max, min_range_val + bin_range * hist_scale)
                    scale, offset = minmax_to_scale_offset(min_range_val, max_range_val, config, self._scale_threshold)
                    min_range_val_list.append(min_range_val)
                    max_range_val_list.append(max_range_val)
                    if bin_range + bin_start > hist_bins:
                        bin_range = hist_bins - bin_start
                    bin_range_list.append(bin_range)
                    mse_loss = self.compute_mse_loss(
                        histogram_np,
                        bin_start,
                        bin_range,
                        self._num_of_quant_levels,
                        hist_sum,
                    )
                    losses.append(
                        {
                            "mse": mse_loss,
                            "kl": 0,
                            "bin_range": bin_range,
                            "bin_start": bin_start,
                            "scale": scale,
                            "offset": offset,
                        }
                    )

                if len(bin_range_list) > 0:
                    kl_loss_tensor = self.compute_kl_loss_batch(
                        histogram,
                        bin_start,
                        torch.tensor(bin_range_list),
                        self._num_of_quant_levels,
                        hist_sum,
                    )

                    for idx, bin_range in enumerate(bin_range_list):
                        losses[losses_start_idx + idx]["kl"] = kl_loss_tensor[idx].item()

            losses_kl = sorted(losses, key=lambda x: x["kl"])
            losses_mse = sorted(losses_kl, key=lambda x: x["mse"])

            if self._single_alg == "mse":
                scale = losses_mse[0]["scale"]
                offset = losses_mse[0]["offset"]
            elif self._single_alg == "kl":
                scale = losses_kl[0]["scale"]
                offset = losses_kl[0]["offset"]
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

            logger.debug("observer render time cost {}".format(time.time() - t_start))

            return scale, offset

        raise Exception("Oops, there might be some mistakes.")


class TorchXSlimMSEObserver(TorchXSlimObserver):
    def __init__(
        self,
        watch_on: Variable,
        quant_cfg: TensorQuantizationConfig,
        hist_bins: int = ppq_common.OBSERVER_FLOATING_MSE_FETCHES,
    ):
        super().__init__(watch_on, quant_cfg, hist_bins, "mse")


class TorchXSlimKLObserver(TorchXSlimObserver):
    def __init__(
        self,
        watch_on: Variable,
        quant_cfg: TensorQuantizationConfig,
        hist_bins: int = ppq_common.OBSERVER_FLOATING_MSE_FETCHES,
    ):
        super().__init__(watch_on, quant_cfg, hist_bins, "kl")


ppq_observer.OBSERVER_TABLE["kl"] = TorchXSlimKLObserver
ppq_observer.OBSERVER_TABLE["mse"] = TorchXSlimMSEObserver
ppq_observer.OBSERVER_TABLE["xslim"] = TorchXSlimObserver
