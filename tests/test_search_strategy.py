"""Synthetic task/measurement evidence and real local-FP16 safety regressions."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort

from test_auto_mixed import fixture
from test_precision_strategy import activation_model, quantized_activation
from xslim.candidate_runner import CandidateRunner, config_to_dict, effective_plan_key, model_fingerprint
from xslim.precision_strategy import record_quantized_layout, validate_export_layout
from xslim.local_fp16 import convert_local_fp16
from xslim.search_strategy import (
    _SearchState, _backend_status, _evaluator_identity, _record, _task_result,
    build_measurement_request,

)
from xslim.xslim_pipeline import _convert_precision_candidate, quantize_onnx_model
from xslim.xslim_setting import SearchParameterSetting, XSlimSettingFactory


class TaskEvaluator:
    metric_name = 'bounded_regression'
    higher_is_better = True
    direction = 'maximize'
    preprocessing_version = 'identity-v1'
    postprocessing_version = 'normalized-mae-v1'
    tuning_data_id = 'fixture71-0-7'
    validation_data_id = 'fixture71-8-11'
    fingerprint = 'seed71-v1'

    def __init__(self, model, root):
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = 1
        self.options.inter_op_num_threads = 1
        self.samples = [np.load(root / f'{i}.npy') for i in range(12)]
        self.source_fingerprint = model_fingerprint(model)
        session = self.session(model)
        self.reference = [session.run(None, {'input': x})[0] for x in self.samples]
        self.calls = []
        self.fail_validation = False

    def session(self, model):
        return ort.InferenceSession(model.SerializeToString(), self.options,
                                    providers=['CPUExecutionProvider'])

    def evaluate(self, model, *, split):
        self.calls.append(split)
        if self.fail_validation and split == 'validation' and model_fingerprint(model) != self.source_fingerprint:
            raise ValueError('validation-only failure')
        indices = range(8) if split == 'tuning' else range(8, 12)
        session = self.session(model)
        error = np.mean([np.mean(np.abs(session.run(None, {'input': self.samples[i]})[0] - self.reference[i]))
                         for i in indices])
        return {'metric': float(1 / (1 + error)), 'sample_count': len(indices)}


class SearchSchemaAndMeasurementTests(unittest.TestCase):
    def test_strict_schema_roundtrip(self):
        setting = XSlimSettingFactory.from_json({'strategy_parameters': {'mode': 'search', 'search': {
            'max_full_trials': 3, 'max_metric_drop': .2}}})
        self.assertEqual(setting.strategy_parameters.search.max_full_trials, 3)
        self.assertEqual(setting.strategy_parameters.search.max_time_s, 600)
        restored = XSlimSettingFactory.from_json({'strategy_parameters': config_to_dict(setting)['strategy_parameters']})
        self.assertEqual(vars(restored.strategy_parameters.search), vars(setting.strategy_parameters.search))
        invalid = {'unknown': [1], 'objective': ['bad', []], 'max_full_trials': [True, 0, 2.5],
                   'max_time_s': [True, 0, float('nan')], 'max_metric_drop': [-.1, 1.1, False],
                   'stop_on_feasible': [1], 'constraints': [{'bad': 2}], 'backend': [{}]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    XSlimSettingFactory.from_json({'strategy_parameters': {'mode': 'search', 'search': {key: value}}})
        for value in [None, {}, {'metric': True, 'sample_count': 2}, {'metric': 1.1, 'sample_count': 2},
                      {'metric': .5, 'sample_count': 0}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _task_result(value)
        with self.assertRaisesRegex(ValueError, 'requires evaluator'):
            _evaluator_identity(None)

    def test_measurement_identity_and_all_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = fixture(root, mode='search')
            settings = SearchParameterSetting()
            settings.objective = 'min_latency'
            settings.backend = {'hardware': {'id': 'board', 'version': 'v1'},
                                'runtime': {'name': 'ort', 'version': '1'},
                                'protocol': {'id': 'warm-single-thread', 'version': '2'}}
            settings.constraints['max_memory_bytes'] = 2048
            settings.check(None)
            config['strategy_parameters']['search'] = vars(settings)
            plan, graph = quantized_activation('Relu', {'default_precision': 'int8'})
            record_quantized_layout(plan, graph)
            request = build_measurement_request(source, plan, config)
            record = _record({'schema_version': 1, 'identity': request['identity'],
                              'metrics': {'latency_ms': 2.0, 'peak_memory_bytes': 1024}})
            self.assertEqual(_backend_status(request, [record], settings)['status'], 'verified')
            for key in request['identity']:
                bad = copy.deepcopy(record)
                if isinstance(bad['identity'][key], dict):
                    bad['identity'][key]['version'] = 'wrong'
                else:
                    bad['identity'][key] = 'wrong'
                result = _backend_status(request, [bad], settings)
                self.assertEqual(result['status'], 'pending', key)
                self.assertEqual(result['rejected_records'][0]['mismatched_fields'], [key])
            latency_only = copy.deepcopy(record)
            del latency_only['metrics']['peak_memory_bytes']
            pending = _backend_status(request, [latency_only], settings)
            self.assertEqual(pending['missing_metrics'], ['peak_memory_bytes'])
            self.assertFalse(pending['verified'])
            conflict = copy.deepcopy(record)
            conflict['metrics']['latency_ms'] += 1
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                _backend_status(request, [record, conflict], settings)
            # Export time alone is volatile; weights and semantic metadata remain bound.
            before = model_fingerprint(source)
            entry = source.metadata_props.add()
            entry.key = 'xslim_export_time'
            entry.value = 'new'
            self.assertEqual(model_fingerprint(source), before)
            entry.key = 'source_version'
            self.assertNotEqual(model_fingerprint(source), before)

    def test_json_boolean_and_number_measurement_identities_are_distinct(self):
        settings = SearchParameterSetting()
        settings.objective = 'min_latency'
        settings.backend = {'hardware': {'id': 'board', 'version': 'v1'},
                            'runtime': {'name': 'ort', 'version': '1', 'options': {'fast_math': 1}},
                            'protocol': {'id': 'benchmark', 'version': '1'}}
        settings.check(None)
        identity = {'model_fingerprint': 'm', 'plan_fingerprint': 'p', 'config_fingerprint': 'c',
                    **copy.deepcopy(settings.backend)}
        for requested, supplied in [(1, True), ([1, {'nested': 0}], [True, {'nested': False}])]:
            with self.subTest(requested=requested):
                identity['runtime']['options'] = {'fast_math': requested}
                record = _record({'schema_version': 1, 'identity': copy.deepcopy(identity),
                                  'metrics': {'latency_ms': 1.0}})
                record['identity']['runtime']['options']['fast_math'] = supplied
                status = _backend_status({'identity': identity}, [record], settings)
                self.assertEqual(status['status'], 'pending')
                self.assertEqual(status['rejected_records'][0]['mismatched_fields'], ['runtime'])

    def test_interrupted_restore_counts_preparation_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = fixture(root)
            with patch('xslim.candidate_runner.time.monotonic', return_value=50.), \
                 patch('xslim.candidate_runner.time.time', return_value=1100.):
                runner = CandidateRunner(source, XSlimSettingFactory.from_json(config),
                                         work_dir=root / 'cache', prior_elapsed_s=110., max_time_s=117.)
            runner.prepared = SimpleNamespace(context_fingerprint='ctx')
            runner._state_path.write_text(json.dumps({'context_fingerprint': 'ctx',
                'full_conversion_count': 2, 'elapsed_s': 10., 'saved_wall_time': 1000.,
                'converting': True, 'entries': {}}))
            with patch('xslim.candidate_runner.time.monotonic', return_value=55.), \
                 patch('xslim.candidate_runner.time.time', return_value=1105.):
                runner._restore_state()
                self.assertEqual(runner.elapsed_s, 115.)
                self.assertTrue(runner.budget_available())

    def test_no_callback_starts_after_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = SimpleNamespace(elapsed_s=10., max_time_s=10., full_conversion_count=1)
            state = _SearchState(Path(directory) / 'state.json', runner)
            called = []
            with self.assertRaisesRegex(RuntimeError, 'exhausted before'):
                state.call('measurement', lambda: called.append(True))
            self.assertFalse(called)


class LocalFP16SafetyTests(unittest.TestCase):
    def test_unsupported_local_fp16_boundaries_fail_closed_before_conversion(self):
        plan = SimpleNamespace(decisions={'act': SimpleNamespace(precision='fp16')})
        source = activation_model('Sigmoid')
        cases = []
        functions = copy.deepcopy(source)
        functions.functions.append(onnx.helper.make_function('local.test', 'Uncalled', ['x'], ['y'],
            [onnx.helper.make_node('Identity', ['x'], ['y'])], [onnx.helper.make_opsetid('', 18)]))
        cases.append((functions, 'ONNX functions'))
        subgraph = copy.deepcopy(source)
        branch = onnx.helper.make_graph([onnx.helper.make_node('Identity', ['a'], ['branch_y'])],
            'branch', [], [onnx.helper.make_tensor_value_info('branch_y', onnx.TensorProto.FLOAT, [1, 2])])
        subgraph.graph.node[-1].attribute.append(onnx.helper.make_attribute('body', branch))
        cases.append((subgraph, 'subgraphs'))
        shared = copy.deepcopy(source)
        shared.graph.node[-1].input.append('w')
        cases.append((shared, 'shared initializer'))
        cases.append((SimpleNamespace(ByteSize=lambda: 2_000_000_000), '>= 2GB'))
        for model, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                convert_local_fp16(model, plan=plan, batches=[])
        self.assertEqual(source.SerializeToString(), activation_model('Sigmoid').SerializeToString())

    def test_fp16_consumer_cannot_bypass_recorded_int8_output_qdq(self):
        with tempfile.TemporaryDirectory() as directory:
            source = activation_model('Sigmoid')
            source.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(
                np.array([[.91, .23], [.18, -.73]], dtype=np.float32), 'w'))
            setting = XSlimSettingFactory.from_json({
                'model_parameters': {'working_dir': directory, 'skip_onnxsim': True, 'opset': 18},
                'strategy_parameters': {'default_precision': 'int8',
                    'allowed_precisions': ['int8', 'fp32', 'fp16'], 'node_precisions': {'act': 'fp16'}},
                'calibration_parameters': {'calibration_device': 'cpu', 'calibration_type': 'minmax',
                    'calibration_step': 10, 'input_parameters': [{'input_name': 'x', 'input_shape': [1, 2],
                                                               'file_type': 'npy'}]}})
            batches = [{'x': np.array([[.3303, -.7107]], dtype=np.float32)} for _ in range(10)]
            model, plan, layout = _convert_precision_candidate(source, setting, None, batches,
                                                               str(Path(directory) / 'good.onnx'))
            self.assertTrue(layout['verified'])
            bad = copy.deepcopy(model)
            mm = next(n for n in bad.graph.node if n.name == 'mm')
            act = next(n for n in bad.graph.node if n.name == 'act')
            producers = {t: n for n in bad.graph.node for t in n.output}
            cast = producers[act.input[0]]
            self.assertEqual(cast.op_type, 'Cast')
            self.assertEqual(producers[cast.input[0]].op_type, 'DequantizeLinear')
            cast.input[0] = mm.output[0]
            # The bypass is ONNX-valid and executable, but not the recorded numerical plan.
            onnx.checker.check_model(bad, full_check=True)
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            feed = {'x': np.array([[.2137, -.5823]], dtype=np.float32)}
            def output(value):
                return ort.InferenceSession(value.SerializeToString(), options,
                    providers=['CPUExecutionProvider']).run(None, feed)[0]
            self.assertGreater(float(np.max(np.abs(output(model) - output(bad)))), 0.)
            with self.assertRaisesRegex(ValueError, 'bypasses recorded output QDQ'):
                validate_export_layout(plan, bad)
            self.assertTrue(validate_export_layout(plan, model)['verified'])

    def test_real_dtype_terminal_boundary_and_parameter_reroute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = fixture(root, mode='standard', allowed_precisions=['int8', 'fp32', 'fp16'],
                node_precisions={'locked': 'int8', 'sensitive': 'fp16', 'tail': 'fp32'})
            original = model.SerializeToString()
            result = quantize_onnx_model(config, model)
            self.assertEqual(model.SerializeToString(), original)
            onnx.checker.check_model(onnx.shape_inference.infer_shapes(result, strict_mode=True, check_type=True),
                                     full_check=True)
            self.assertTrue(all(v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
                                for v in [*result.graph.input, *result.graph.output]))
            weights = {t.name: t for t in result.graph.initializer}
            self.assertEqual(weights['w1'].data_type, onnx.TensorProto.FLOAT16)
            self.assertEqual(weights['w2'].data_type, onnx.TensorProto.FLOAT)
            for node in result.graph.node:
                if node.op_type in {'QuantizeLinear', 'DequantizeLinear'}:
                    self.assertEqual(weights[node.input[1]].data_type, onnx.TensorProto.FLOAT)
            from xslim.candidate_runner import plan_from_dict
            report = json.loads((root / 'result_precision_report.json').read_text())
            plan = plan_from_dict(report['plan'])
            self.assertTrue(validate_export_layout(plan, result)['verified'])
            for name, weight in [('tail', 'w2'), ('sensitive', 'w1')]:
                bad = copy.deepcopy(result)
                replacement = bad.graph.initializer.add()
                replacement.CopyFrom(weights[weight])
                replacement.name = 'unrelated_parameter'
                node = next(n for n in bad.graph.node if n.name == name)
                node.input[1] = replacement.name
                with self.subTest(node=name), self.assertRaisesRegex(ValueError, 'parameter provenance'):
                    validate_export_layout(plan, bad)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(result.SerializeToString(), options, providers=['CPUExecutionProvider'])
            self.assertTrue(np.isfinite(session.run(None, {'input': np.load(root / '11.npy')})[0]).all())

    def test_real_fused_group_survives_fp16_promotion(self):
        strategy = {'default_precision': 'int8', 'allowed_precisions': ['int8', 'fp32', 'fp16']}
        baseline, graph = quantized_activation('Relu', strategy)
        record_quantized_layout(baseline, graph)
        group = next(g for g in baseline.groups if g['members'] == ['act', 'mm'])
        original = baseline.fingerprint
        candidate = baseline.promote_group(group['id'], 'fp16')
        with tempfile.TemporaryDirectory() as directory:
            setting = XSlimSettingFactory.from_json({'model_parameters': {'working_dir': directory,
                'skip_onnxsim': True}, 'strategy_parameters': strategy,
                'calibration_parameters': {'calibration_device': 'cpu', 'calibration_type': 'minmax',
                    'calibration_step': 10, 'input_parameters': [{'input_name': 'x', 'input_shape': [1, 2],
                                                               'file_type': 'npy'}]}})
            batches = [{'x': np.array([[.33, -.71]], dtype=np.float32)} for _ in range(10)]
            output, actual, layout = _convert_precision_candidate(activation_model('Relu'), setting,
                candidate, batches, str(Path(directory) / 'fused.onnx'))
            self.assertTrue(layout['verified'])
            inherited = next(g for g in actual.groups if g['id'] == group['id'])
            self.assertEqual(inherited['members'], ['act', 'mm'])
            self.assertEqual(inherited['precision'], 'fp16')
            with self.assertRaisesRegex(ValueError, 'Incomplete precision override'):
                actual.with_precisions({'mm': 'fp32'})
            locked = copy.deepcopy(actual)
            next(g for g in locked.groups if g['id'] == group['id'])['locked'] = True
            with self.assertRaisesRegex(ValueError, 'locked precision group'):
                locked.promote_group(group['id'], 'fp32')
            self.assertEqual(baseline.fingerprint, original)
            self.assertTrue(any(t.data_type == onnx.TensorProto.FLOAT16 for t in output.graph.initializer))


class SearchRealORTTests(unittest.TestCase):
    def test_measurement_failures_write_reports_without_masking_or_validation(self):
        original_call = _SearchState.call
        for failure in ('exception', 'invalid', 'deadline'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, config = fixture(root, mode='search', search={
                    'max_metric_drop': 1., 'max_full_trials': 3, 'max_time_s': 180,
                    'backend': {'hardware': {'id': 'board', 'version': 'v1'},
                                'runtime': {'name': 'ort', 'version': '1'},
                                'protocol': {'id': 'test', 'version': '1'}}})
                evaluator = TaskEvaluator(source, root)
                measurements = []
                tuning_calls = []
                def callback(model, *, request):
                    measurements.append(request)
                    if failure == 'exception':
                        raise RuntimeError('measurement boom')
                    return {'invalid': True}
                def guarded_call(state, label, callback):
                    value = original_call(state, label, callback)
                    if label == 'evaluate:tuning':
                        tuning_calls.append(label)
                        if failure == 'deadline' and len(tuning_calls) == 2:
                            state.runner.prior_elapsed_s = state.runner.max_time_s
                    return value
                expected = {'exception': 'measurement boom', 'invalid': 'Measurement record',
                            'deadline': 'max_time_s exhausted before measurement'}[failure]
                with patch.object(_SearchState, 'call', guarded_call), self.assertRaisesRegex(Exception, expected):
                    quantize_onnx_model(config, source, evaluator=evaluator, measurement_callback=callback)
                for name in ('search_report', 'final_check'):
                    report = json.loads((root / f'result_{name}.json').read_text())
                    self.assertIn(expected, report['error'])
                    self.assertFalse(report['verified'])
                    self.assertEqual(report['full_conversion_count'], 1)
                    self.assertIn('task_metric', report['baseline'])
                    if failure == 'deadline' and name == 'search_report':
                        self.assertEqual(report['stop_reason'], 'max_time_s')
                    self.assertFalse(report['validation']['performed'])
                self.assertNotIn('validation', evaluator.calls)
                self.assertEqual(len(measurements), 0 if failure == 'deadline' else 1)

    def test_prepare_is_journaled_before_external_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = fixture(root, mode='search')
            evaluator = TaskEvaluator(source, root)
            def interrupted(runner):
                state = json.loads((root / 'result_search_state.json').read_text())
                self.assertEqual(state['in_progress'], 'prepare')
                raise RuntimeError('interrupted preparation')
            with patch.object(CandidateRunner, 'prepare', interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interrupted preparation'):
                    quantize_onnx_model(config, source, evaluator=evaluator)
            self.assertEqual(evaluator.calls, [])
            report = json.loads((root / 'result_search_report.json').read_text())
            self.assertEqual(report['full_conversion_count'], 0)
            self.assertIn('interrupted preparation', report['error'])

    def test_validation_failure_never_reselects_and_score_identity_keeps_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = fixture(root, mode='search', search={'max_metric_drop': 1.,
                'max_full_trials': 8, 'max_time_s': 180, 'stop_on_feasible': True})
            evaluator = TaskEvaluator(source, root)
            first = quantize_onnx_model(config, source, evaluator=evaluator)
            report_path = root / 'result_search_report.json'
            first_report = json.loads(report_path.read_text())
            self.assertTrue(first_report['verified'])
            self.assertEqual(first_report['full_conversion_count'], 2)
            self.assertEqual(evaluator.calls[-2:], ['validation', 'validation'])
            self.assertTrue(all(x == 'tuning' for x in evaluator.calls[:-2]))
            evaluator.calls.clear()
            evaluator.preprocessing_version = 'identity-v2'
            evaluator.fail_validation = True
            second = quantize_onnx_model(config, source, evaluator=evaluator)
            second_report = json.loads(report_path.read_text())
            self.assertFalse(second_report['verified'])
            self.assertEqual(second_report['validation']['status'], 'failed')
            self.assertIn('validation-only failure', second_report['validation']['error'])
            self.assertEqual(second_report['full_conversion_count'], 3)
            self.assertEqual(second_report['task_cache_misses'], 4)
            # Independent exports may reorder protobuf initializers/value_info.
            np.testing.assert_allclose(evaluator.session(first).run(None, {'input': evaluator.samples[0]})[0],
                                       evaluator.session(second).run(None, {'input': evaluator.samples[0]})[0],
                                       rtol=1e-5, atol=1e-6)
            self.assertEqual(evaluator.calls[-2:], ['validation', 'validation'])
            self.assertEqual(effective_plan_key_from_report(first_report), effective_plan_key_from_report(second_report))

    def test_runtime_implementation_and_dependency_changes_reject_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = fixture(root)
            setting = XSlimSettingFactory.from_json(config)
            runner = CandidateRunner(source, setting, work_dir=root / 'cache')
            runner.prepare()
            self.assertEqual(runner.run().status, 'ok')
            read_bytes = Path.read_bytes
            def changed(path):
                content = read_bytes(path)
                return content + b'\n# changed implementation' if path.name == 'local_fp16.py' else content
            with patch('xslim.candidate_runner.Path.read_bytes', changed):
                with self.assertRaisesRegex(ValueError, 'cache context changed'):
                    CandidateRunner(source, setting, work_dir=root / 'cache').prepare()
            with patch('xslim.candidate_runner._optional_distribution_version', return_value='changed-version'):
                with self.assertRaisesRegex(ValueError, 'cache context changed'):
                    CandidateRunner(source, setting, work_dir=root / 'cache').prepare()


def effective_plan_key_from_report(report):
    from xslim.candidate_runner import plan_from_dict
    return effective_plan_key(plan_from_dict(report['plan']))


if __name__ == '__main__':
    unittest.main()
