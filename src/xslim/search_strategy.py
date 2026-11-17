'Task-metric search on real exported artifacts, with explicit backend evidence.'

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import onnx

from .auto_mixed import _evidence, _same_outputs
from .candidate_runner import CandidateRunner, config_to_dict, effective_plan_key, fingerprint, model_fingerprint
from .xslim_setting import SearchParameterSetting, XSlimSettingFactory


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _setting(config):
    return XSlimSettingFactory.from_json(copy.deepcopy(config)) if isinstance(config, dict) else config


def _conversion_config(config):
    value = config_to_dict(_setting(config))
    value['strategy_parameters'].pop('search', None)
    # Artifact and plan already bind source/precision. Paths and search policies
    # do not change a numerical conversion; preprocessing source contents do.
    for key in ('working_dir', 'output_prefix', 'onnx_model'):
        value['model_parameters'].pop(key, None)
    for key in ('mode', 'max_trials', 'max_time_s', 'max_adjust_groups', 'cosine_similarity_min'):
        value['strategy_parameters'].pop(key, None)
    sources = {}
    for item in value['calibration_parameters']['input_parameters']:
        specification = item.get('preprocess_file')
        if isinstance(specification, str) and specification not in {'IMAGENET', 'PT_IMAGENET'}:
            filename, _ = specification.rsplit(':', 1)
            sources[specification] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return {'settings': value, 'preprocessing_sources': sources}


def build_measurement_request(model, plan, config):
    'Build the exact identity a deployment measurement must repeat.'
    config = _setting(config)
    config.strategy_parameters.search.check(config)
    backend = config.strategy_parameters.search.backend
    if backend is None:
        raise ValueError('strategy_parameters.search.backend is required for measurement requests.')
    if isinstance(plan, dict):
        from .candidate_runner import plan_from_dict
        plan = plan_from_dict(plan)
    return {'schema_version': 1, 'identity': {
        'model_fingerprint': model_fingerprint(model), 'plan_fingerprint': effective_plan_key(plan),
        'config_fingerprint': fingerprint(_conversion_config(config)), **copy.deepcopy(backend)}}


def _evaluator_identity(evaluator):
    if evaluator is None or not callable(getattr(evaluator, 'evaluate', None)):
        raise ValueError('mode=search requires evaluator.evaluate(model, *, split).')
    keys = ('metric_name', 'direction', 'preprocessing_version', 'postprocessing_version',
            'tuning_data_id', 'validation_data_id', 'fingerprint')
    identity = {key: getattr(evaluator, key, None) for key in keys}
    if any(not isinstance(v, str) or not v.strip() for v in identity.values()):
        raise ValueError('Evaluator identity fields must be nonempty strings: ' + ', '.join(keys))
    higher = getattr(evaluator, 'higher_is_better', None)
    if not isinstance(higher, bool) or identity['direction'] != ('maximize' if higher else 'minimize'):
        raise ValueError('Evaluator direction/higher_is_better must consistently declare maximize or minimize.')
    if identity['tuning_data_id'] == identity['validation_data_id']:
        raise ValueError('Evaluator tuning_data_id and validation_data_id must be distinct.')
    identity['higher_is_better'] = higher
    return identity


def _task_result(value):
    if not isinstance(value, dict) or not {'metric', 'sample_count'} <= set(value):
        raise ValueError('Evaluator result requires metric and sample_count.')
    metric, count = value['metric'], value['sample_count']
    if (isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric)
            or not 0 <= metric <= 1):
        raise ValueError('Evaluator metric must be finite numeric in [0,1].')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Evaluator sample_count must be a positive integer.')
    return {'metric': float(metric), 'sample_count': count}


def _record(value):
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'identity', 'metrics'}
            or type(value['schema_version']) is not int or value['schema_version'] != 1):
        raise ValueError('Measurement record requires schema_version=1, identity, metrics only.')
    identity = value['identity']
    if not isinstance(identity, dict) or set(identity) != {
            'model_fingerprint', 'plan_fingerprint', 'config_fingerprint', 'hardware', 'runtime', 'protocol'}:
        raise ValueError('Measurement identity requires model/plan/config/hardware/runtime/protocol.')
    for key in ('model_fingerprint', 'plan_fingerprint', 'config_fingerprint'):
        if not isinstance(identity[key], str) or not identity[key].strip():
            raise ValueError(f'Measurement identity.{key} must be nonempty.')
    temporary = SearchParameterSetting()
    temporary.backend = {key: copy.deepcopy(identity[key]) for key in ('hardware', 'runtime', 'protocol')}
    temporary.check(None)
    metrics = value['metrics']
    if not isinstance(metrics, dict) or not metrics or set(metrics) - {'latency_ms', 'peak_memory_bytes'}:
        raise ValueError('Measurement metrics accepts latency_ms/peak_memory_bytes only.')
    for key, metric in metrics.items():
        kinds = (int,) if key == 'peak_memory_bytes' else (int, float)
        if isinstance(metric, bool) or not isinstance(metric, kinds) or not math.isfinite(metric) or metric <= 0:
            raise ValueError(f'Measurement {key} must be finite positive (bytes: integer).')
    return copy.deepcopy(value)


def _load_measurements(path):
    if path is None:
        return []
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'records'}
            or type(value['schema_version']) is not int or value['schema_version'] != 1
            or not isinstance(value['records'], list)):
        raise ValueError('Measurement file requires schema_version=1 and records list only.')
    return [_record(v) for v in value['records']]


def _backend_status(request, records, settings):
    needed = set()
    if settings.objective == 'min_latency' or settings.constraints['max_latency_ms'] is not None:
        needed.add('latency_ms')
    if settings.objective == 'min_memory' or settings.constraints['max_memory_bytes'] is not None:
        needed.add('peak_memory_bytes')
    rejected, matches = [], []
    for record in records:
        # JSON booleans and numbers are distinct evidence identities (True != 1).
        mismatches = [key for key in request['identity']
                      if fingerprint(record['identity'][key]) != fingerprint(request['identity'][key])]
        if mismatches:
            rejected.append({'mismatched_fields': mismatches, 'identity': record['identity']})
        else:
            matches.append(record)
    if len({fingerprint(v['metrics']) for v in matches}) > 1:
        raise ValueError('Conflicting measurement records for the same exact identity.')
    metrics = matches[0]['metrics'] if matches else {}
    missing = sorted(needed - set(metrics))
    violations = []
    for limit, metric in [('max_latency_ms', 'latency_ms'), ('max_memory_bytes', 'peak_memory_bytes')]:
        if settings.constraints[limit] is not None and metric in metrics and metrics[metric] > settings.constraints[limit]:
            violations.append(limit)
    status = 'failed' if violations else ('verified' if matches and not missing else 'pending')
    return {'status': status, 'verified': status == 'verified', 'metrics': metrics,
            'request': request, 'missing_metrics': missing, 'rejected_records': rejected,
            'violated_constraints': violations, 'required': bool(needed),
            'evidence_scope': 'Caller-supplied deployment record; identity checked, board run not independently authenticated.'}


class _SearchState:
    def __init__(self, path, runner):
        self.path, self.runner = path, runner
        self.data = json.loads(path.read_text()) if path.exists() else {}
        self.data.setdefault('scores', {})
        self.data.setdefault('validation_runs', [])

    def checkpoint(self, in_progress=None):
        self.data.update(full_conversion_count=self.runner.full_conversion_count, elapsed_s=self.runner.elapsed_s,
                         in_progress=in_progress, saved_wall_time=time.time())
        _write(self.path, self.data)

    def call(self, label, callback):
        if self.runner.elapsed_s >= self.runner.max_time_s:
            raise RuntimeError(f"search max_time_s exhausted before {label}.")
        self.checkpoint(label)
        try:
            return callback()
        finally:
            self.checkpoint()


def _screen(plan, prepared, limit):
    # Cheap structural exposure proxy weighted by real frozen input magnitude.
    # It ranks candidates only: this is neither a task score nor board evidence.
    import numpy as np
    scale = sum(float(np.mean(np.abs(a.astype(np.float64)))) for b in prepared.tuning_batches for a in b.values())
    candidates = []
    for group in plan.groups:
        if group['locked'] or group['precision'] != 'int8':
            continue
        for precision in ('fp32', 'fp16'):
            if precision not in plan.allowed_precisions:
                continue
            candidate = plan.promote_group(group['id'], precision)
            score = scale * sum(len(plan.decisions[n].outputs) + len(plan.decisions[n].parameter_inputs)
                                for n in group['members'])
            candidates.append((score, group['id'], precision, candidate))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[:limit]


def run_search(original_model, config, *, started_at=None, evaluator=None,
               measurement_callback=None, measurement_file=None):
    settings = config.strategy_parameters.search
    settings.check(config)
    evaluator_id = _evaluator_identity(evaluator)
    if measurement_callback is not None and measurement_file is not None:
        raise ValueError('measurement_callback and measurement_file are mutually exclusive.')
    if measurement_callback is not None and not callable(measurement_callback):
        raise ValueError('measurement_callback must be callable.')
    if (measurement_callback is not None or measurement_file is not None) and settings.backend is None:
        raise ValueError('Measurement inputs require strategy_parameters.search.backend.')
    records = _load_measurements(measurement_file)
    started_at = time.monotonic() if started_at is None else started_at
    root, prefix = Path(config.model_parameters.working_dir), config.model_parameters.output_prefix
    state_path = root / f'{prefix}_search_state.json'
    restored = json.loads(state_path.read_text()) if state_path.exists() else {}
    elapsed = restored.get('elapsed_s', 0)
    if restored.get('in_progress'):
        # Count crash downtime up to this run start; preparation below is timed live.
        start_wall = time.time() - max(0, time.monotonic() - started_at)
        elapsed += max(0, start_wall - restored['saved_wall_time'])
    runner_config = copy.deepcopy(config)
    # Conversion reuse is independent of task score/search policy identity.
    runner_config.strategy_parameters.search = SearchParameterSetting()
    runner = CandidateRunner(original_model, runner_config, work_dir=root / f'{prefix}_search_candidates',
        started_at=started_at, max_full_conversions=settings.max_full_trials, max_time_s=settings.max_time_s,
        prior_full_conversion_count=max(settings.prior_full_conversion_count, restored.get('full_conversion_count', 0)),
        prior_elapsed_s=max(settings.prior_elapsed_s, elapsed))
    state = _SearchState(state_path, runner)
    report = {'schema_version': 1, 'mode': 'search', 'objective': settings.objective,
              'verified': False, 'target_met': False, 'status': 'pending',
              'final_check': {'performed': False, 'verified': False},
              'validation': {'performed': False, 'status': 'pending', 'verified': False},
              'backend': {'status': 'pending', 'verified': False, 'reason': 'No matched deployment measurement.'},
              'stop_reason': None, 'fallback_reason': None, 'evaluator': evaluator_id,
              'max_metric_drop': settings.max_metric_drop, 'screening': [], 'candidates': [],
              'task_cache_hits': 0, 'task_cache_misses': 0, 'deduplicated': 0,
              'tolerance': {'rtol': 1e-5, 'atol': 1e-6}}
    baseline = selected = published = None
    final = None
    scores, measured = {}, {}
    reference = None
    try:
        prepared = state.call('prepare', runner.prepare)
        report['identity'] = {name: getattr(prepared, name) for name in
            ('source_fingerprint', 'config_fingerprint', 'data_fingerprint', 'runtime_fingerprint',
             'context_fingerprint', 'data_ids')}
        report['identity']['evaluator'] = evaluator_id
        report['identity']['data_disjointness'] = 'Caller-declared split identities; evaluator owns preprocessing and split contents.'

        def evaluate(model, split, *, fresh=False):
            if _evaluator_identity(evaluator) != evaluator_id:
                raise ValueError('Evaluator identity changed during search.')
            key = fingerprint([prepared.context_fingerprint, model_fingerprint(model), evaluator_id, split])
            if not fresh and key in state.data['scores']:
                report['task_cache_hits'] += 1
                return _task_result(state.data['scores'][key])
            report['task_cache_misses'] += 1
            value = state.call('evaluate:' + split,
                               lambda: _task_result(evaluator.evaluate(copy.deepcopy(model), split=split)))
            if _evaluator_identity(evaluator) != evaluator_id:
                raise ValueError('Evaluator identity changed during evaluation.')
            state.data['scores'][key] = value
            state.checkpoint()
            return value

        def measure(result):
            if settings.backend is None:
                return {'status': 'pending', 'verified': False, 'required': False, 'metrics': {},
                        'reason': 'No backend descriptor/measurement; CPU timings and file sizes are not deployment evidence.'}
            request = build_measurement_request(result.model, result.plan, config)
            supplied = list(records)
            if measurement_callback is not None:
                value = state.call('measurement', lambda: measurement_callback(copy.deepcopy(result.model),
                                                                              request=copy.deepcopy(request)))
                if value is not None:
                    supplied.append(_record(value))
            return _backend_status(request, supplied, settings)

        def drop(value, ref):
            return ref['metric'] - value['metric'] if evaluator_id['higher_is_better'] else value['metric'] - ref['metric']

        def task_feasible(result):
            return result.status == 'ok' and drop(scores[id(result)], reference) <= settings.max_metric_drop

        def feasible(result):
            backend = measured[id(result)]
            return task_feasible(result) and (not backend.get('required') or backend['status'] == 'verified')

        def assess(result):
            if result.status != 'ok':
                return False
            scores[id(result)] = evaluate(result.model, 'tuning')
            if scores[id(result)]['sample_count'] != reference['sample_count']:
                raise ValueError('Tuning sample_count differs from the frozen float reference.')
            measured[id(result)] = measure(result)
            return True

        def rank(result):
            value = scores[id(result)]['metric']
            accuracy = -value if evaluator_id['higher_is_better'] else value
            if settings.objective == 'min_error':
                return (0 if feasible(result) else 1, accuracy)
            key = 'latency_ms' if settings.objective == 'min_latency' else 'peak_memory_bytes'
            backend = measured[id(result)]
            # Pending artifacts may be published as pending, never as a successful backend objective.
            metric = backend['metrics'].get(key, float('inf'))
            return (0 if feasible(result) else 1, metric, accuracy)

        if runner.elapsed_s >= settings.max_time_s:
            raise RuntimeError('search budget exhausted during preparation before baseline.')
        reference = evaluate(runner.original_model, 'tuning')
        report['float_reference'] = reference
        baseline = state.call('conversion:baseline', lambda: runner.run(None, purpose='baseline'))
        if not assess(baseline):
            raise RuntimeError(f'search baseline unavailable: {baseline.status}: {baseline.error}')
        selected = baseline
        seen = {effective_plan_key(baseline.plan)}
        screened_count = 0
        while True:
            if settings.stop_on_feasible and feasible(selected):
                report['stop_reason'] = 'stop_on_feasible'
                break
            if screened_count >= settings.max_candidates:
                report['stop_reason'] = 'max_candidates'
                break
            if not runner.budget_available(reserve=1):
                report['stop_reason'] = 'max_time_s' if runner.elapsed_s >= settings.max_time_s else 'max_full_trials'
                break
            proposals = _screen(selected.plan, prepared, settings.max_candidates - screened_count)
            round_best = selected
            for proxy, group, precision, plan in proposals:
                key = effective_plan_key(plan)
                if key in seen:
                    report['deduplicated'] += 1
                    continue
                seen.add(key)
                screened_count += 1
                report['screening'].append({'group': group, 'precision': precision, 'plan_fingerprint': key,
                    'score': proxy, 'method': 'Structural exposure weighted by frozen input magnitude; heuristic only.'})
                if not runner.budget_available(reserve=1):
                    break
                candidate = state.call('conversion:candidate', lambda plan=plan: runner.run(plan))
                valid = assess(candidate)
                evidence = _evidence(candidate)
                if valid:
                    evidence.update(task_metric=scores[id(candidate)], metric_drop=drop(scores[id(candidate)], reference),
                                    backend=measured[id(candidate)])
                    if rank(candidate) < rank(round_best):
                        round_best = candidate
                report['candidates'].append(evidence)
                if settings.stop_on_feasible and valid and feasible(candidate):
                    round_best = candidate
                    break
            if round_best is selected:
                report['stop_reason'] = ('max_time_s' if runner.elapsed_s >= settings.max_time_s else
                                         'max_full_trials' if not runner.budget_available(reserve=1) else 'no_improvement')
                break
            selected = round_best
        report['selected'] = {**_evidence(selected), 'task_metric': scores[id(selected)], 'backend': measured[id(selected)]}
        if runner.budget_available():
            final = state.call('conversion:final', lambda: runner.run(selected.plan, purpose='final', use_cache=False))
            outputs_match = final.model is not None and _same_outputs(selected.outputs, final.outputs)
            same_plan = final.plan is not None and effective_plan_key(final.plan) == effective_plan_key(selected.plan)
            final_ok = final.status == 'ok' and outputs_match and same_plan
            report['final_check'] = {'performed': True, 'verified': bool(final_ok), 'outputs_match': outputs_match,
                                     'same_plan': same_plan, 'result': _evidence(final)}
            if final_ok:
                published = final
                # Freeze the final numeric artifact before touching validation.
                scores[id(final)] = evaluate(final.model, 'tuning')
                if scores[id(final)]['sample_count'] != reference['sample_count']:
                    raise ValueError('Final tuning sample_count differs from the frozen float reference.')
                measured[id(final)] = measure(final)
            else:
                published = baseline
                report['fallback_reason'] = 'final_check_failed'
        else:
            published = baseline
            report['fallback_reason'] = 'final_budget_exhausted'
        report['published'] = {**_evidence(published), 'task_metric': scores[id(published)],
                               'metric_drop': drop(scores[id(published)], reference)}
        report['backend'] = measured[id(published)]
        report['plan'], report['layout'] = published.plan.to_dict(), published.layout
        onnx.save(copy.deepcopy(published.model), root / f'{prefix}.onnx')
        _write(root / f'{prefix}_precision_report.json', published.layout)
        frozen = {'plan_fingerprint': effective_plan_key(published.plan),
                  'model_fingerprint': model_fingerprint(published.model)}
        if report['final_check']['verified']:
            validation = {'performed': True, 'status': 'failed', 'verified': False, 'frozen': frozen,
                          'feeds_selection': False}
            report['validation'] = validation
            state.data['validation_runs'].append(frozen)
            state.checkpoint()
            try:
                left = evaluate(runner.original_model, 'validation', fresh=True)
                right = evaluate(published.model, 'validation', fresh=True)
                if left['sample_count'] != right['sample_count']:
                    raise ValueError('Validation sample_count differs between reference and frozen selected model.')
                validation.update(reference=left, selected=right, metric_drop=drop(right, left))
                validation['verified'] = drop(right, left) <= settings.max_metric_drop
                validation['status'] = 'verified' if validation['verified'] else 'failed'
            except Exception as error:
                validation['error'] = f'{type(error).__name__}: {error}'
            # Deliberately no selection, conversion or fallback below this point.
        task_verified = bool(report['final_check']['verified'] and report['validation']['verified'] and task_feasible(published))
        report['task_metric'] = {'verified': task_verified, 'status': 'verified' if task_verified else 'failed',
                                 'tuning': scores[id(published)], 'reference': reference}
        backend_ok = not report['backend'].get('required') or report['backend']['status'] == 'verified'
        report['verified'] = bool(task_verified and backend_ok and runner.elapsed_s <= settings.max_time_s)
        report['target_met'] = report['verified']
        report['status'] = ('verified' if report['verified'] else
                            'pending' if task_verified and report['backend']['status'] == 'pending' else 'failed')
        return copy.deepcopy(published.model)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['baseline'] = _evidence(baseline)
        if baseline is not None and id(baseline) in scores:
            report['baseline']['task_metric'] = scores[id(baseline)]
        if baseline is not None and id(baseline) in measured:
            report['baseline']['backend'] = measured[id(baseline)]
        report.update(attempts=runner.attempts, full_conversion_count=runner.full_conversion_count,
                      elapsed_s=runner.elapsed_s, time_overrun=runner.elapsed_s > settings.max_time_s)
        if report['time_overrun'] or 'max_time_s exhausted' in report.get('error', ''):
            report.update(stop_reason='max_time_s', verified=False, target_met=False, status='failed')
        report['budgets'] = {'max_full_trials': settings.max_full_trials, 'max_time_s': settings.max_time_s,
                             'max_candidates': settings.max_candidates, 'baseline_final_failures_counted': True}
        state.checkpoint()
        _write(root / f'{prefix}_search_report.json', report)
        _write(root / f'{prefix}_final_check.json', {key: report.get(key) for key in
            ('mode', 'verified', 'status', 'identity', 'evaluator', 'final_check', 'validation', 'backend',
             'baseline', 'selected', 'published', 'plan', 'layout', 'fallback_reason', 'stop_reason',
             'full_conversion_count', 'elapsed_s', 'time_overrun', 'error')})
