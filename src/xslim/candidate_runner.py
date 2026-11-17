"""Real exported-QDQ candidate execution shared by precision strategies."""

import copy
import hashlib
import json
import importlib.metadata
import platform
import time
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .calibration_helper import CalibrationCollect, XSlimDataset
from .precision_strategy import PrecisionDecision, PrecisionPlan


def _optional_distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def config_to_dict(value):
    """Serialize effective settings locally without changing the legacy serializer."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: config_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_to_dict(item) for item in value]
    if hasattr(value, "__dict__"):
        return config_to_dict(vars(value))
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def model_fingerprint(model):
    # Active strategy validation uses in-memory protobuf. Fail closed at its limit.
    if model.ByteSize() >= 2_000_000_000:
        raise ValueError("Precision candidate execution does not support models >= 2GB.")
    canonical = copy.deepcopy(model)
    properties = [(entry.key, entry.value) for entry in canonical.metadata_props
                  if entry.key != "xslim_export_time"]
    del canonical.metadata_props[:]
    for key, value in properties:
        entry = canonical.metadata_props.add()
        entry.key, entry.value = key, value
    return hashlib.sha256(canonical.SerializeToString()).hexdigest()


def effective_plan_key(plan):
    if plan is None:
        return "baseline"
    return fingerprint({name: {"precision": d.precision, "calibration_type": d.calibration_type,
                               "force_int8": d.requested_precision == "int8"}
                        for name, d in sorted(plan.decisions.items())})


def plan_from_dict(value):
    """Restore ALL plan fields, including socket/domain/layout evidence."""
    data = copy.deepcopy(value)
    data["decisions"] = {name: PrecisionDecision(**d) for name, d in data["decisions"].items()}
    return PrecisionPlan(**data)


def make_session(model_or_path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model_or_path, options, providers=["CPUExecutionProvider"])


def score_outputs(reference, outputs):
    """Element-weighted MSE and minimum per-batch/output cosine; invalid never passes."""
    evidence = []
    error_sum = 0.0
    count = 0
    if len(reference) != len(outputs) or not reference:
        return {"valid": False, "mse": None, "cosine": None, "invalid_reason": "batch count mismatch"}
    for batch_index, (expected, actual) in enumerate(zip(reference, outputs)):
        if len(expected) != len(actual):
            return {"valid": False, "mse": None, "cosine": None, "invalid_reason": "output count mismatch"}
        for output_index, (left, right) in enumerate(zip(expected, actual)):
            left, right = np.asarray(left), np.asarray(right)
            if left.shape != right.shape or not left.size:
                reason = "empty output or shape mismatch"
            elif not np.isfinite(left).all() or not np.isfinite(right).all():
                reason = "nonfinite output"
            else:
                # Scale before cosine to avoid overflow/underflow in norm products.
                a, b = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
                a_scale, b_scale = np.max(np.abs(a)), np.max(np.abs(b))
                if a_scale == 0 or b_scale == 0:
                    reason = "zero-norm output"
                else:
                    a_unit, b_unit = a / a_scale, b / b_scale
                    cosine = float(np.clip(np.dot(a_unit, b_unit) /
                                           (np.linalg.norm(a_unit) * np.linalg.norm(b_unit)), -1, 1))
                    with np.errstate(over="ignore", invalid="ignore"):
                        squared = float(np.sum(np.square(a - b)))
                    if not np.isfinite(squared) or not np.isfinite(cosine):
                        reason = "nonfinite metric"
                    else:
                        evidence.append({"batch": batch_index, "output": output_index,
                                         "mse": squared / a.size, "cosine": cosine})
                        error_sum += squared
                        count += a.size
                        continue
            return {"valid": False, "mse": None, "cosine": None, "invalid_reason": reason,
                    "per_output": evidence}
    if not count or not np.isfinite(error_sum):
        return {"valid": False, "mse": None, "cosine": None, "invalid_reason": "invalid aggregate"}
    return {"valid": True, "mse": error_sum / count, "cosine": min(v["cosine"] for v in evidence),
            "per_output": evidence, "invalid_reason": None}


@dataclass
class PreparedEvidence:
    tuning_batches: tuple
    data_ids: tuple
    source_fingerprint: str
    config_fingerprint: str
    data_fingerprint: str
    runtime_fingerprint: str
    context_fingerprint: str
    float_outputs: tuple
    float_session: object = field(repr=False)

    def run_float(self, feeds):
        return self.float_session.run(None, copy.deepcopy(feeds))

    def copy_batches(self):
        return copy.deepcopy(self.tuning_batches)


@dataclass
class CandidateResult:
    status: str
    model: object = None
    model_path: object = None
    plan: object = None
    effective_plan_key: object = None
    cache_key: object = None
    cache_hit: bool = False
    layout: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    outputs: tuple = ()
    session: object = field(default=None, repr=False)
    error: object = None
    full_conversion_count: int = 0
    model_fingerprint: object = None
    elapsed_s: float = 0.0

    def run(self, feeds):
        return self.session.run(None, copy.deepcopy(feeds))


class CandidateRunner:
    def __init__(self, original_model, config, *, work_dir, started_at=None, max_full_conversions=8,
                 max_time_s=600, prior_full_conversion_count=0, prior_elapsed_s=0,
                 post_qdq_hook=None, hook_fingerprint=None):
        # Start before source loading and hashing: preparation is part of the budget.
        self.started_at = time.monotonic() if started_at is None else started_at
        self.prior_elapsed_s = prior_elapsed_s
        self.started_wall_time = time.time() - max(0, time.monotonic() - self.started_at)
        self._original_model = copy.deepcopy(onnx.load(original_model) if isinstance(original_model, (str, Path))
                                            else original_model)
        # Loading external data materializes bytes; hash those bytes, not a location/mtime.
        onnx.external_data_helper.convert_model_from_external_data(self._original_model)
        self._config = copy.deepcopy(config)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_full_conversions = max_full_conversions
        self.max_time_s = max_time_s
        self.full_conversion_count = prior_full_conversion_count
        self.post_qdq_hook = post_qdq_hook
        if post_qdq_hook is not None and not hook_fingerprint:
            raise ValueError("post_qdq_hook requires a stable hook_fingerprint.")
        self.hook_fingerprint = hook_fingerprint
        self.prepared = None
        self._cache = {}
        self._batches = ()
        self._reference = ()
        self.attempts = []
        self._state_path = self.work_dir / "candidate_state.json"

    @property
    def elapsed_s(self):
        return self.prior_elapsed_s + time.monotonic() - self.started_at

    def budget_available(self, reserve=0):
        return (self.elapsed_s < self.max_time_s and
                self.full_conversion_count + reserve < self.max_full_conversions)

    @property
    def original_model(self):
        return copy.deepcopy(self._original_model)

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def prepare(self):
        if self.prepared is not None:
            return self._public_prepared()
        from .xslim_pipeline import dispatch_graph, xslim_load_onnx_graph
        from .optimizer import GraphLegalized

        parameters = self._config.calibration_parameters
        if not parameters.input_parameters or any(not p.data_list_path for p in parameters.input_parameters):
            raise ValueError("Precision strategies require actual user calibration data_list_path for every input.")
        source_hash = model_fingerprint(self._original_model)
        graph, _, _, _ = xslim_load_onnx_graph(copy.deepcopy(self._original_model),
            not self._config.model_parameters.skip_onnxsim,
            self._config.quantization_parameters.truncate_var_names, self._config.model_parameters.opset)
        GraphLegalized(graph)()
        graph = dispatch_graph(graph, "conservative")
        parameters.check_input_parameters(graph)
        dataset = XSlimDataset(parameters)
        loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.auto_batch_size)
        collect = CalibrationCollect(parameters.input_parameters, parameters.calibration_device)
        # CalibrationCollect uses import_module, whose cache can retain old code after
        # a file edit. Compile custom source afresh, independently of .pyc timestamps.
        for item in parameters.input_parameters:
            specification = item.preprocess_file
            if isinstance(specification, str) and specification not in {"IMAGENET", "PT_IMAGENET"}:
                filename, function_name = specification.rsplit(":", 1)
                source_path = Path(filename).resolve()
                module = types.ModuleType(f"_xslim_preprocess_{hashlib.sha256(source_path.read_bytes()).hexdigest()}")
                module.__file__ = str(source_path)
                exec(compile(source_path.read_bytes(), str(source_path), "exec"), module.__dict__)
                collect.custom_transforms[item.input_name] = getattr(module, function_name)
        batches, data_hash = [], hashlib.sha256()
        # Hash raw inputs even if preprocessing maps different files to equal tensors.
        for item in parameters.input_parameters:
            listing = Path(item.data_list_path)
            data_hash.update(listing.read_bytes())
            for name in listing.read_text().splitlines():
                if not name.strip():
                    continue
                path = listing.parent / name.strip()
                data_hash.update(str(path.resolve()).encode())
                with path.open("rb") as sample:
                    for chunk in iter(lambda: sample.read(1024 * 1024), b""):
                        data_hash.update(chunk)
        limit = max(parameters.calibration_step, self._config.strategy_parameters.score_steps)
        for index, raw in enumerate(loader):
            if index >= limit:
                break
            batch = {name: tensor.detach().cpu().numpy().copy() for name, tensor in collect(raw).items()}
            for name, array in batch.items():
                data_hash.update(json.dumps([index, name, array.dtype.str, array.shape]).encode())
                data_hash.update(array.tobytes())
                array.setflags(write=False)
            batches.append(batch)
        if not batches:
            raise ValueError("No actual calibration batches are available.")
        self._batches = tuple(batches)
        float_session = make_session(self._original_model.SerializeToString())
        names = [v.name for v in float_session.get_inputs()]
        if any(set(batch) != set(names) for batch in batches):
            raise ValueError("Strategy scoring requires complete original model inputs (truncated inputs unsupported).")
        score_batches = self._batches[:self._config.strategy_parameters.score_steps]
        self._reference = tuple(tuple(float_session.run(None, copy.deepcopy(batch))) for batch in score_batches)
        runtime_hash = fingerprint({"ort": ort.__version__, "onnx": onnx.__version__, "torch": torch.__version__,
                                    "providers": float_session.get_providers(), "threads": 1,
                                    "hook": self.hook_fingerprint, "runner_version": 3,
                                    "onnxconverter_common": _optional_distribution_version("onnxconverter-common"),
                                    "python": platform.python_version(), "platform": platform.platform(),
                                    "numpy": np.__version__, "implementation": {
                                        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                        for name in ("candidate_runner.py", "xslim_pipeline.py", "precision_strategy.py", "local_fp16.py")}})
        preprocess_files = {}
        for item in parameters.input_parameters:
            specification = item.preprocess_file
            if isinstance(specification, str) and specification not in {"IMAGENET", "PT_IMAGENET"}:
                path = Path(specification.rsplit(":", 1)[0]).resolve()
                preprocess_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Include collector implementation as well as custom preprocessing file contents.
        preprocess_files["CalibrationCollect"] = hashlib.sha256(
            Path(__import__(CalibrationCollect.__module__, fromlist=["__file__"]).__file__).read_bytes()).hexdigest()
        config_hash = fingerprint({"setting": config_to_dict(self._config), "preprocessing": preprocess_files})
        context_hash = fingerprint([source_hash, config_hash, data_hash.hexdigest(), runtime_hash])
        self.prepared = PreparedEvidence(copy.deepcopy(score_batches),
            tuple(f"batch:{i}" for i in range(len(score_batches))), source_hash, config_hash,
            data_hash.hexdigest(), runtime_hash, context_hash, copy.deepcopy(self._reference), float_session)
        self._restore_state()
        return self._public_prepared()

    def _persist_state(self, *, converting=False):
        entries = {}
        for key, result in self._cache.items():
            entries[key] = {"status": result.status, "model_path": result.model_path,
                "plan": result.plan.to_dict() if result.plan else None,
                "effective_plan_key": result.effective_plan_key, "cache_key": result.cache_key,
                "layout": result.layout, "metrics": result.metrics, "error": result.error,
                "model_fingerprint": result.model_fingerprint}
        payload = {"context_fingerprint": self.prepared.context_fingerprint,
            "full_conversion_count": self.full_conversion_count, "elapsed_s": self.elapsed_s,
            "saved_wall_time": time.time(), "converting": converting,
            "attempts": self.attempts, "entries": entries}
        temporary = self._state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, allow_nan=False))
        temporary.replace(self._state_path)

    def _restore_state(self):
        if not self._state_path.exists():
            return
        state = json.loads(self._state_path.read_text())
        if state["context_fingerprint"] != self.prepared.context_fingerprint:
            raise ValueError("Candidate cache context changed; use a separate work_dir for new model/config/data.")
        self.full_conversion_count = max(self.full_conversion_count, state["full_conversion_count"])
        saved_elapsed = state["elapsed_s"]
        if state.get("converting"):
            # Recovery is anchored before this run: live preparation must not be counted twice.
            saved_elapsed += max(0, self.started_wall_time - state["saved_wall_time"])
        self.prior_elapsed_s = max(self.prior_elapsed_s, saved_elapsed)
        self.attempts = state.get("attempts", [])
        for key, value in state.get("entries", {}).items():
            data = copy.deepcopy(value)
            data["plan"] = plan_from_dict(data["plan"]) if data["plan"] else None
            if data["model_path"]:
                try:
                    model = onnx.load(data["model_path"])
                    if model_fingerprint(model) != data["model_fingerprint"]:
                        continue
                    data["model"] = model
                    data["session"] = make_session(data["model_path"])
                    data["outputs"] = tuple(tuple(data["session"].run(None, copy.deepcopy(batch)))
                                            for batch in self.prepared.tuning_batches)
                    data["metrics"] = score_outputs(self._reference, data["outputs"])
                except Exception:
                    continue
            self._cache[key] = CandidateResult(**data)

    def _public_prepared(self):
        evidence = copy.copy(self.prepared)
        evidence.tuning_batches = copy.deepcopy(self.prepared.tuning_batches)
        evidence.float_outputs = copy.deepcopy(self._reference)
        return evidence

    def _public_result(self, result, cache_hit=False):
        public = copy.copy(result)
        for name in ("model", "plan", "layout", "metrics", "outputs"):
            setattr(public, name, copy.deepcopy(getattr(result, name)))
        public.cache_hit = cache_hit
        public.full_conversion_count = self.full_conversion_count
        public.elapsed_s = self.elapsed_s
        return public

    def run(self, plan=None, *, purpose="candidate", use_cache=True):
        from .xslim_pipeline import _convert_precision_candidate

        self.prepare()
        plan_key = effective_plan_key(plan)
        cache_key = fingerprint([self.prepared.context_fingerprint, plan_key])
        if use_cache and cache_key in self._cache:
            return self._public_result(self._cache[cache_key], True)
        if not self.budget_available():
            return CandidateResult("budget_exhausted", effective_plan_key=plan_key, cache_key=cache_key,
                                   full_conversion_count=self.full_conversion_count, elapsed_s=self.elapsed_s)
        self.full_conversion_count += 1
        self._persist_state(converting=True)
        path = self.work_dir / f"{self.full_conversion_count:04d}-{purpose}.onnx"
        try:
            model, actual_plan, layout = _convert_precision_candidate(
                copy.deepcopy(self._original_model), copy.deepcopy(self._config), copy.deepcopy(plan), self._batches,
                str(path), post_qdq_hook=self.post_qdq_hook)
            session = make_session(str(path))
            outputs = tuple(tuple(session.run(None, copy.deepcopy(batch)))
                            for batch in self.prepared.tuning_batches)
            metrics = score_outputs(self._reference, outputs)
            result = CandidateResult("ok" if metrics["valid"] else "invalid_metrics", model, str(path), actual_plan,
                plan_key, cache_key, False, layout, metrics, outputs, session,
                full_conversion_count=self.full_conversion_count, model_fingerprint=model_fingerprint(model))
        except Exception as exc:
            result = CandidateResult("conversion_failed", effective_plan_key=plan_key, cache_key=cache_key,
                                     error=f"{type(exc).__name__}: {exc}",
                                     full_conversion_count=self.full_conversion_count)
        result.elapsed_s = self.elapsed_s
        self.attempts.append({"purpose": purpose, "cache_key": cache_key, "status": result.status,
                              "full_conversion_count": self.full_conversion_count, "elapsed_s": self.elapsed_s,
                              "error": result.error})
        self._cache[cache_key] = result
        if plan is None and result.plan is not None:
            alias = fingerprint([self.prepared.context_fingerprint, effective_plan_key(result.plan)])
            self._cache[alias] = result
        self._persist_state()
        return self._public_result(result)
