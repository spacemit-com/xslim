"""Budgeted greedy group promotion, scored on real exported model outputs."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import onnx

from .candidate_runner import CandidateRunner


def _improves(candidate, current):
    if not candidate.metrics.get("valid"):
        return False
    if not current.metrics.get("valid"):
        return True
    # Do not trade a cosine regression for a smaller squared error.
    return (candidate.metrics["mse"] < current.metrics["mse"] and
            candidate.metrics["cosine"] >= current.metrics["cosine"])


def _same_outputs(left, right):
    return (len(left) == len(right) and all(len(a) == len(b) and all(
        x.shape == y.shape and np.allclose(x, y, rtol=1e-5, atol=1e-6, equal_nan=False)
        for x, y in zip(a, b)) for a, b in zip(left, right)))


def _evidence(result):
    if result is None:
        return None
    return {name: copy.deepcopy(getattr(result, name)) for name in
            ("status", "model_path", "effective_plan_key", "cache_key", "cache_hit", "metrics",
             "error", "full_conversion_count", "model_fingerprint", "elapsed_s")}


def run_auto_mixed(original_model, config, *, started_at=None):
    strategy = config.strategy_parameters
    started_at = time.monotonic() if started_at is None else started_at
    root = Path(config.model_parameters.working_dir)
    prefix = config.model_parameters.output_prefix
    runner = CandidateRunner(original_model, config, work_dir=root / f"{prefix}_candidates",
        started_at=started_at, max_full_conversions=strategy.max_trials, max_time_s=strategy.max_time_s)
    report_path = root / f"{prefix}_auto_mixed_report.json"
    report = {"mode": "auto_mixed", "verified": False, "final_check": {"performed": False, "verified": False},
              "tolerance": {"rtol": 1e-5, "atol": 1e-6}, "adjusted_groups": [], "stop_reason": None}
    selected = baseline = final = published = None
    try:
        prepared = runner.prepare()
        report["identity"] = {name: getattr(prepared, name) for name in
            ("source_fingerprint", "config_fingerprint", "data_fingerprint", "runtime_fingerprint",
             "context_fingerprint", "data_ids")}
        baseline = runner.run(None, purpose="baseline")
        if baseline.model is None:
            raise RuntimeError(f"auto_mixed baseline unavailable: {baseline.status}: {baseline.error}")
        selected = baseline
        promotions = []
        while True:
            if not selected.metrics.get("valid"):
                report["stop_reason"] = "invalid_metrics"
                break
            if selected.metrics["cosine"] >= strategy.cosine_similarity_min:
                report["stop_reason"] = "target_reached"
                break
            if len(promotions) >= strategy.max_adjust_groups:
                report["stop_reason"] = "max_adjust_groups"
                break
            # Reserve a complete independent final conversion.
            if not runner.budget_available(reserve=1):
                report["stop_reason"] = "budget_exhausted"
                break
            round_best = selected
            winning_group = None
            round_budget_exhausted = False
            for group in selected.plan.groups:
                if not group["promotable"]:
                    continue
                if not runner.budget_available(reserve=1):
                    round_budget_exhausted = True
                    break
                candidate_plan = selected.plan.promote_group(group["id"])
                candidate = runner.run(candidate_plan)
                if _improves(candidate, round_best):
                    round_best, winning_group = candidate, group["id"]
            if winning_group is None:
                report["stop_reason"] = "budget_exhausted" if round_budget_exhausted else "no_improvement"
                break
            selected = round_best
            promotions.append(winning_group)
            if round_budget_exhausted:
                report["stop_reason"] = "budget_exhausted"
                break
        report["adjusted_groups"] = promotions
        report["selected"] = _evidence(selected)
        # Every published changed plan must pass a fresh complete conversion.
        if runner.budget_available():
            final = runner.run(selected.plan, purpose="final", use_cache=False)
            outputs_match = final.model is not None and _same_outputs(selected.outputs, final.outputs)
            same_plan = (final.plan is not None and
                         final.effective_plan_key == selected.effective_plan_key)
            # baseline None uses an alias key; compare effective assignments for it.
            if selected is baseline and final.plan is not None:
                from .candidate_runner import effective_plan_key
                same_plan = effective_plan_key(final.plan) == effective_plan_key(selected.plan)
            final_ok = (outputs_match and same_plan and final.status == "ok" and
                        (selected is baseline or _improves(final, baseline)))
            report["final_check"] = {"performed": True, "verified": bool(final_ok),
                "outputs_match": bool(outputs_match), "same_plan": bool(same_plan),
                "result": _evidence(final)}
            if final_ok:
                published = final
            else:
                # The original legal baseline artifact is already complete; keep it.
                published = baseline
                report["fallback_reason"] = "final_check_failed"
        else:
            published = baseline
            report["fallback_reason"] = "final_budget_exhausted"
        report["verified"] = bool(report["final_check"]["verified"])
        report["target_met"] = bool(published.metrics.get("valid") and
                                     published.metrics["cosine"] >= strategy.cosine_similarity_min)
        report["published"] = _evidence(published)
        report["plan"] = published.plan.to_dict()
        report["layout"] = published.layout
        onnx.save(copy.deepcopy(published.model), root / f"{prefix}.onnx")
        (root / f"{prefix}_precision_report.json").write_text(
            json.dumps(published.layout, indent=2, ensure_ascii=False, allow_nan=False))
        return copy.deepcopy(published.model)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["baseline"] = _evidence(baseline)
        report["attempts"] = runner.attempts
        report["full_conversion_count"] = runner.full_conversion_count
        report["elapsed_s"] = runner.elapsed_s
        report["time_overrun"] = runner.elapsed_s > strategy.max_time_s
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        independent = {key: report.get(key) for key in
            ("mode", "verified", "identity", "tolerance", "baseline", "selected", "published",
             "final_check", "layout", "fallback_reason", "stop_reason", "full_conversion_count",
             "elapsed_s", "time_overrun", "error")}
        independent["plan_request_fingerprint"] = (published.plan.request_fingerprint
                                                   if published is not None else None)
        (root / f"{prefix}_final_check.json").write_text(
            json.dumps(independent, indent=2, ensure_ascii=False, allow_nan=False))
