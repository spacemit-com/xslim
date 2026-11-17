"""Reusable static precision decisions and checks against the actual exported graph."""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase

import numpy as np
import onnx

from .logger import logger
from .ppq_decorator import QuantableOperation, QuantizationStates, TargetPlatform


@dataclass
class PrecisionDecision:
    node_name: str
    source_names: list
    precision: str
    requested_precision: object
    priority: int
    locked: bool
    reasons: list
    inputs: list
    outputs: list
    quantizable: bool
    op_type: str
    calibration_type: object = None
    planned_precision: object = None
    parameter_inputs: list = field(default_factory=list)
    quantized_parameter_inputs: list = field(default_factory=list)
    quantized_activation_sockets: list = field(default_factory=list)


@dataclass
class PrecisionPlan:
    decisions: dict
    source_to_optimized: dict
    groups: list = field(default_factory=list)
    unmapped_source_nodes: list = field(default_factory=list)
    allowed_precisions: list = field(default_factory=lambda: ["int8", "fp32"])
    quantization_domains: list = field(default_factory=list)
    layout_recorded: bool = False

    def to_dict(self):
        return asdict(self)

    @property
    def request_fingerprint(self):
        """Stable candidate key, independent of calibration/export evidence."""
        content = {"mapping": self.source_to_optimized, "allowed": self.allowed_precisions,
                   "decisions": {name: {key: value for key, value in asdict(decision).items()
                                        if key not in {"quantized_parameter_inputs", "quantized_activation_sockets",
                                                       "precision", "reasons"}}
                                 for name, decision in self.decisions.items()}}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    @property
    def fingerprint(self):
        content = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()

    def with_precisions(self, overrides):
        """Derive a candidate without mutating the baseline or its provenance."""
        result = copy.deepcopy(self)
        result.layout_recorded = False
        for name, precision in overrides.items():
            if name not in result.decisions:
                raise ValueError(f"Unknown precision-plan node {name!r}.")
            decision = result.decisions[name]
            _check_precision(precision, self.allowed_precisions)
            if decision.locked and precision != decision.precision:
                raise ValueError(f"Cannot change locked precision for {name!r}.")
            if precision == "int8" and not decision.quantizable:
                raise ValueError(f"Operation {name!r} does not support static INT8.")
            if precision != decision.precision:
                decision.precision = precision
                decision.planned_precision = precision
                decision.requested_precision = precision
                decision.reasons.append("candidate override")
        for group in result.groups:
            precisions = {result.decisions[name].precision for name in group["members"]}
            if len(precisions) != 1:
                raise ValueError(f"Incomplete precision override for shared group {group['id']}.")
            if group.get("locked") and next(iter(precisions)) != group["precision"]:
                raise ValueError(f"Cannot change locked precision group {group['id']!r}.")
            group["precision"] = next(iter(precisions))
            group["promotable"] = (not group["locked"] and group["precision"] == "int8"
                                   and "fp32" in result.allowed_precisions)
        return result

    def promote_group(self, group_id, precision="fp32"):
        group = next((item for item in self.groups if item["id"] == group_id), None)
        if group is None:
            raise ValueError(f"Unknown precision group {group_id!r}.")
        return self.with_precisions(dict.fromkeys(group["members"], precision))


def _check_precision(precision, allowed):
    if precision not in allowed:
        raise ValueError(f"Precision {precision!r} is not in allowed_precisions {allowed!r}.")
    if precision not in {"int8", "fp32", "fp16"}:
        raise ValueError(f"Unsupported precision {precision!r}.")


def _source_mapping(source_model, graph):
    nodes = list(source_model.graph.node)
    names = [node.name for node in nodes]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Precision strategies require unique, non-empty original node names; mapping is ambiguous.")
    mapping = {}
    for node in nodes:
        matches = [op.name for op in graph.operations.values()
                   if op.type == node.op_type and [v.name for v in op.outputs] == list(node.output)]
        if node.name in matches:
            mapping[node.name] = node.name
        elif len(matches) == 1:
            mapping[node.name] = matches[0]
    reverse = {}
    for source, name in mapping.items():
        reverse.setdefault(name, []).append(source)
    if any(len(sources) > 1 for sources in reverse.values()):
        raise ValueError("Ambiguous merged source node mapping; refusing to guess precision targets.")
    return {node.name: node for node in nodes}, mapping


def _subgraph_members(nodes, starts, ends):
    missing = (set(starts) | set(ends)) - set(nodes)
    if missing:
        raise ValueError(f"Subgraph boundary nodes matched no operations: {sorted(missing)}.")
    producers = {output: node.name for node in nodes.values() for output in node.output}
    forward = {name: set() for name in nodes}
    backward = {name: set() for name in nodes}
    for node in nodes.values():
        for tensor in node.input:
            parent = producers.get(tensor)
            if parent is not None:
                forward[parent].add(node.name)
                backward[node.name].add(parent)

    def reachable(seed, edges):
        result, pending = set(seed), list(seed)
        while pending:
            for name in edges[pending.pop()]:
                if name not in result:
                    result.add(name)
                    pending.append(name)
        return result

    members = reachable(starts, forward) & reachable(ends, backward)
    if not (set(starts) | set(ends)) <= members:
        raise ValueError("Subgraph boundaries must be connected by directed paths.")
    return members


def build_precision_plan(source_model, graph, setting, quantizable_types):
    """Resolve rules on original names against the graph AFTER prequantization."""
    strategy = setting.strategy_parameters
    strategy.check(setting)
    if setting.quantization_parameters.precision_level.value not in {0, 1, 2}:
        raise ValueError("Precision plans support only static INT8 precision levels 0, 1 and 2.")
    nodes, mapping = _source_mapping(source_model, graph)
    quant = setting.quantization_parameters
    assignments = {name: {} for name in graph.operations}
    methods = {}

    def assign(name, precision, priority, reason):
        _check_precision(precision, strategy.allowed_precisions)
        previous = assignments[name].get(priority)
        if previous is not None and previous[0] != precision:
            raise ValueError(f"Conflicting same-priority precision rules for {name!r}: {previous[1]} / {reason}.")
        assignments[name][priority] = (precision, ([] if previous is None else previous[1]) + [reason])

    def resolve(source):
        if source not in nodes:
            raise ValueError(f"Original node {source!r} matched no operations.")
        if source not in mapping:
            raise ValueError(f"Original node {source!r} disappeared or changed during optimization; no exact mapping.")
        return mapping[source]

    # Resolve legacy last-match ordering BEFORE combining node-priority locks.
    legacy = {}
    for index, rule in enumerate(quant.operator_quantization):
        matches = [name for name in nodes if fnmatchcase(name, rule.pattern)]
        optimized = [name for name in graph.operations if fnmatchcase(name, rule.pattern)]
        if not matches and not optimized:
            raise ValueError(f"Operator pattern {rule.pattern!r} matched no operations after graph optimization.")
        for name in {resolve(source) for source in matches} | set(optimized):
            legacy[name] = (rule, index)
    for name, (rule, index) in legacy.items():
        assign(name, rule.precision, 3, f"legacy operator_quantization[{index}]")
        methods[name] = rule.calibration_type

    rules = list(strategy.rules)
    rules.extend({"selector": "node", "names": [name], "precision": precision}
                 for name, precision in strategy.node_precisions.items())
    rules.extend({"selector": "type", "op_types": [name], "precision": precision}
                 for name, precision in strategy.op_type_precisions.items())
    for index, rule in enumerate(rules):
        selector = rule["selector"]
        if selector == "node":
            selected = set(rule["names"])
            priority = 3
        elif selector == "subgraph":
            selected = _subgraph_members(nodes, rule["start_nodes"], rule["end_nodes"])
            priority = 2
        else:
            selected = {name for name, node in nodes.items() if node.op_type in rule["op_types"]}
            priority = 1
            for op_type in rule["op_types"]:
                if not any(node.op_type == op_type for node in nodes.values()):
                    logger.warning(f"Precision rule operator type {op_type!r} matched no original nodes.")
        for source in selected:
            assign(resolve(source), rule["precision"], priority, f"{selector} rule[{index}]: {source}")

    decisions = {}
    for name, operation in graph.operations.items():
        quantizable = operation.type in quantizable_types and operation.platform != TargetPlatform.SOI
        if strategy.default_precision is not None:
            precision = strategy.default_precision
            _check_precision(precision, strategy.allowed_precisions)
            # A global INT8 default applies only to capable data operations.
            if precision == "int8" and not quantizable:
                precision = "fp32"
            assign(name, precision, 0, "explicit default_precision")
        if assignments[name]:
            priority = max(assignments[name])
            precision, reasons = assignments[name][priority]
            requested = precision
        else:
            priority, requested = -1, None
            if operation.type in quant.ignore_op_types or name in quant.ignore_op_names:
                precision, reasons = "fp32", ["legacy ignore list"]
            else:
                precision = ("int8" if quantizable and operation.platform not in {TargetPlatform.FP32, TargetPlatform.SOI}
                             else "fp32")
                reasons = ["legacy dispatch"]
        if precision == "int8" and not quantizable:
            raise ValueError(f"Operation {name!r} ({operation.type}) does not support static INT8.")
        _check_precision(precision, strategy.allowed_precisions)
        decisions[name] = PrecisionDecision(
            node_name=name, source_names=[source for source, target in mapping.items() if target == name],
            precision=precision, requested_precision=requested, priority=priority, locked=priority >= 1,
            reasons=reasons, inputs=[v.name for v in operation.inputs], outputs=[v.name for v in operation.outputs],
            quantizable=quantizable, op_type=operation.type, calibration_type=methods.get(name),
            planned_precision=precision,
            parameter_inputs=[index for index, var in enumerate(operation.inputs) if var.is_parameter])
    return PrecisionPlan(decisions, mapping, unmapped_source_nodes=sorted(set(nodes) - set(mapping)),
                         allowed_precisions=list(strategy.allowed_precisions))


def apply_precision_plan(plan, graph, target_platform):
    """Apply to a fresh prequantized graph, never to an already exported QDQ graph."""
    if set(plan.decisions) != set(graph.operations):
        raise ValueError("Precision plan no longer matches optimized graph node names.")
    for name, decision in plan.decisions.items():
        operation = graph.operations[name]
        if (operation.type != decision.op_type or [v.name for v in operation.inputs] != decision.inputs
                or [v.name for v in operation.outputs] != decision.outputs):
            raise ValueError(f"Precision plan graph layout changed for {name!r}; rebuild from floating source.")
        _check_precision(decision.precision, plan.allowed_precisions)
        if decision.precision == "int8":
            if not decision.quantizable or operation.platform == TargetPlatform.SOI:
                raise ValueError(f"Operation {name!r} does not support static INT8.")
            operation.platform = target_platform
        elif operation.platform != TargetPlatform.SOI:
            operation.platform = TargetPlatform.FP32


def record_quantized_layout(plan, graph):
    """Capture actual tensor quantization domains BEFORE the mutating exporter."""
    plan.layout_recorded = False
    roots = {}
    for name, decision in plan.decisions.items():
        operation = graph.operations.get(name)
        if operation is None:
            raise ValueError(f"Planned node {name!r} disappeared during quantization.")
        configs = operation.config_with_variable if isinstance(operation, QuantableOperation) else []
        active = []
        decision.quantized_parameter_inputs = []
        decision.quantized_activation_sockets = []
        for index, (config, variable) in enumerate(configs):
            root_config = config.dominated_by
            if root_config.state in {QuantizationStates.FP32, QuantizationStates.SOI}:
                continue
            if root_config.scale is None:
                continue
            socket = [name, "input" if index < len(operation.inputs) else "output",
                      index if index < len(operation.inputs) else index - len(operation.inputs)]
            roots.setdefault(id(root_config), []).append(socket)
            active.append(socket)
            if variable.is_parameter and index < len(operation.inputs) and config.can_export():
                decision.quantized_parameter_inputs.append(index)
            elif not variable.is_parameter and config.can_export():
                decision.quantized_activation_sockets.append({
                    "side": socket[1], "index": socket[2], "tensor": variable.name,
                    "scale": config.scale.detach().cpu().numpy().tolist(),
                    "zero_point": config.offset.detach().cpu().numpy().tolist()})
        actual = "int8" if active else "fp32"
        staged_fp16 = decision.precision == "fp16" and actual == "fp32"
        if not staged_fp16:
            _check_precision(actual, plan.allowed_precisions)
        if actual != decision.precision and not staged_fp16:
            if decision.requested_precision is not None or decision.locked:
                raise ValueError(f"Requested precision for {name!r} is {decision.precision}, actual PPQ state is {actual}.")
            decision.precision = actual
            decision.reasons.append("actual PPQ tensor states")
    # Only shared OUTPUT roots are fusion constraints. A normal producer-output /
    # consumer-input shared scale is a boundary, not a reason to group a whole branch.
    parent = {name: name for name in plan.decisions}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    # Export staging may remove physical INT8 domains; inherited fusion remains
    # an atomic precision constraint even when the whole group becomes FP16/FP32.
    inherited_groups = copy.deepcopy(plan.groups)
    for group in inherited_groups:
        members = group["members"]
        if not members or any(name not in parent for name in members):
            raise ValueError("Inherited precision group has missing members.")
        for name in members[1:]:
            parent[find(name)] = find(members[0])

    domains = []
    for sockets in roots.values():
        sockets = sorted(sockets)
        members = sorted({item[0] for item in sockets})
        domain_id = "qd_" + hashlib.sha256(json.dumps(sockets).encode()).hexdigest()[:16]
        domains.append({"id": domain_id, "members": members, "sockets": sockets})
        fused = sorted({name for name, side, _ in sockets if side == "output"})
        for name in fused[1:]:
            parent[find(name)] = find(fused[0])
    plan.quantization_domains = sorted(domains, key=lambda domain: domain["id"])
    memberships = {}
    for name in plan.decisions:
        memberships.setdefault(find(name), []).append(name)
    groups = []
    for members in memberships.values():
        members = sorted(members)
        precisions = {plan.decisions[name].precision for name in members}
        if len(precisions) != 1:
            raise ValueError(f"Conflicting user precisions in fused quantization group {members}.")
        group_id = "pg_" + hashlib.sha256(json.dumps(members).encode()).hexdigest()[:16]
        inherited = [group for group in inherited_groups if set(group["members"]) & set(members)]
        locked = (any(plan.decisions[name].locked for name in members)
                  or any(group.get("locked", False) for group in inherited))
        inherited_reasons = {reason for group in inherited for reason in group.get("reasons", [])}
        precision = next(iter(precisions))
        groups.append({"id": group_id, "members": members, "precision": precision, "locked": locked,
                       "promotable": not locked and precision == "int8" and "fp32" in plan.allowed_precisions,
                       "reasons": sorted(inherited_reasons | {
                           reason for name in members for reason in plan.decisions[name].reasons})})
    plan.groups = sorted(groups, key=lambda group: group["id"])
    plan.layout_recorded = True


def validate_export_layout(plan, exported_model):
    """Verify actual QDQ wiring; return a JSON-safe coverage report or fail closed."""
    if not plan.layout_recorded:
        raise ValueError("QDQ evidence not recorded; call record_quantized_layout before exporting.")
    for decision in plan.decisions.values():
        _check_precision(decision.precision, plan.allowed_precisions)
    onnx.checker.check_model(exported_model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(exported_model, strict_mode=True, check_type=True)
    shapes = {value.name: [dim.dim_value if dim.HasField("dim_value") else None
                           for dim in value.type.tensor_type.shape.dim]
              for value in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    types = {v.name: v.type.tensor_type.elem_type for v in
             [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    types.update({t.name: t.data_type for t in inferred.graph.initializer})
    has_fp16 = any(d.precision == "fp16" for d in plan.decisions.values())
    nodes = list(exported_model.graph.node)
    by_name = {}
    for node in nodes:
        if node.name in by_name and node.name in plan.decisions:
            raise ValueError(f"Duplicate exported precision target {node.name!r}.")
        by_name[node.name] = node
    producers = {tensor: node for node in nodes for tensor in node.output}
    consumers = {}
    for node in nodes:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)
    initializers = {tensor.name: tensor for tensor in exported_model.graph.initializer}
    evidence = {}

    def valid_qdq(node):
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"} or len(node.input) < 3:
            return False
        scale = initializers.get(node.input[1])
        zero = initializers.get(node.input[2])
        if (scale is None or zero is None or scale.data_type != onnx.TensorProto.FLOAT
                or zero.data_type not in {onnx.TensorProto.INT8, onnx.TensorProto.UINT8, onnx.TensorProto.INT32}):
            return False
        scales = onnx.numpy_helper.to_array(scale)
        zeros = onnx.numpy_helper.to_array(zero)
        if (scales.size == 0 or not np.all(np.isfinite(scales)) or np.any(scales <= 0)
                or scales.shape != zeros.shape):
            return False
        if scales.ndim > 1:
            return False
        if scales.size > 1:
            shape = shapes.get(node.input[0])
            if shape is None and node.input[0] in initializers:
                shape = list(initializers[node.input[0]].dims)
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 1)
            if shape is None or not -len(shape) <= axis < len(shape) or shape[axis] != scales.size:
                return False
        if node.op_type == "DequantizeLinear":
            weight = initializers.get(node.input[0])
            upstream = producers.get(node.input[0])
            return ((weight is not None and weight.data_type == zero.data_type)
                    or (upstream is not None and upstream.op_type == "QuantizeLinear"
                        and valid_qdq(upstream)
                        and np.array_equal(onnx.numpy_helper.to_array(initializers[upstream.input[1]]), scales)
                        and np.array_equal(onnx.numpy_helper.to_array(initializers[upstream.input[2]]), zeros)
                        and [(a.name, a.i) for a in upstream.attribute if a.name == "axis"]
                        == [(a.name, a.i) for a in node.attribute if a.name == "axis"]))
        return True

    for candidate in nodes:
        if candidate.op_type in {"QuantizeLinear", "DequantizeLinear"} and not valid_qdq(candidate):
            raise ValueError(f"Invalid QDQ scale, zero point or axis/channel layout at {candidate.name!r}.")

    expected_producers = {tensor: (name, index) for name, decision in plan.decisions.items()
                          for index, tensor in enumerate(decision.outputs) if tensor}

    def activation_origin(tensor, *, require_output_qdq=False):
        """Trace only validated QDQ pairs to an exact producer socket or external tensor.

        Internal fused edges have no QDQ of their own, but must retain the same
        producer and output index. Other wrappers require explicit validation.
        """
        visited = set()
        traversed_pairs = []
        while tensor not in visited:
            visited.add(tensor)
            producer = producers.get(tensor)
            if producer is not None and producer.op_type == "Cast" and has_fp16:
                targets = [a.i for a in producer.attribute if a.name == "to"]
                if (producer.domain or len(producer.input) != 1 or len(producer.output) != 1
                        or len(producer.attribute) != 1 or len(targets) != 1
                        or targets[0] not in {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
                        or types.get(producer.input[0]) not in {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
                        or types.get(tensor) != targets[0]):
                    raise ValueError("Invalid FP16 Cast boundary in activation provenance.")
                tensor = producer.input[0]
                continue
            if producer is None:
                return (None, tensor)
            if producer.name in plan.decisions:
                index = list(producer.output).index(tensor)
                decision = plan.decisions[producer.name]
                if require_output_qdq and decision.precision == "int8":
                    for socket in decision.quantized_activation_sockets:
                        if socket["side"] != "output" or socket["index"] != index:
                            continue
                        # A dangling QDQ branch proves nothing about this consumer's
                        # numerical path. Require its exact producer-output QDQ here.
                        matching = [q for q, _ in traversed_pairs if q.input[0] == tensor
                            and np.asarray(socket["scale"]).shape == onnx.numpy_helper.to_array(
                                initializers[q.input[1]]).shape
                            and np.allclose(onnx.numpy_helper.to_array(initializers[q.input[1]]),
                                            socket["scale"], rtol=1e-5, atol=1e-7)
                            and np.array_equal(onnx.numpy_helper.to_array(initializers[q.input[2]]),
                                               np.rint(socket["zero_point"]))]
                        if not matching:
                            raise ValueError(f"Activation path bypasses recorded output QDQ at "
                                             f"{producer.name!r} socket {index}.")
                return (producer.name, index)
            if producer.op_type != "DequantizeLinear" or not valid_qdq(producer):
                return (producer.name, list(producer.output).index(tensor))
            quantize = producers.get(producer.input[0])
            if quantize is None or quantize.op_type != "QuantizeLinear" or not valid_qdq(quantize):
                return (producer.name, list(producer.output).index(tensor))
            traversed_pairs.append((quantize, producer))
            tensor = quantize.input[0]
        raise ValueError("Cycle in exported activation provenance.")

    for name, decision in plan.decisions.items():
        node = by_name.get(name)
        if node is None or node.op_type != decision.op_type:
            raise ValueError(f"Planned node {name!r} missing or changed in actual exported layout.")
        if len(node.input) != len(decision.inputs) or len(node.output) != len(decision.outputs):
            raise ValueError(f"Planned node {name!r} changed input/output socket counts.")
        for index, original in enumerate(decision.inputs):
            if index in decision.parameter_inputs:
                continue
            expected = expected_producers.get(original, (None, original))
            if activation_origin(node.input[index], require_output_qdq=True) != expected:
                raise ValueError(f"Node {name!r} input socket {index} changed activation provenance.")
        nearby = [producers[tensor] for tensor in node.input if tensor in producers]
        nearby.extend(child for tensor in node.output for child in consumers.get(tensor, []))
        qdq = [candidate.name for candidate in nearby if valid_qdq(candidate)]
        if decision.precision in {"fp32", "fp16"}:
            if not has_fp16:
                if list(node.input) != decision.inputs or list(node.output) != decision.outputs:
                    raise ValueError(f"FP32 node {name!r} acquired quantized sockets or changed export layout.")
            else:
                expected_type = onnx.TensorProto.FLOAT16 if decision.precision == "fp16" else onnx.TensorProto.FLOAT
                floating = [t for t in [*node.input, *node.output]
                            if types.get(t) in {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}]
                if not floating or any(types[t] != expected_type for t in floating):
                    raise ValueError(f"Node {name!r} lacks actual {decision.precision} dtype evidence.")
                for index, original in enumerate(decision.outputs):
                    if activation_origin(original) != (name, index):
                        raise ValueError(f"Node {name!r} changed output provenance.")
                for index in decision.parameter_inputs:
                    if node.input[index] != decision.inputs[index] or node.input[index] not in initializers:
                        raise ValueError(f"Node {name!r} changed parameter provenance.")
        else:
            for socket in decision.quantized_activation_sockets:
                side, index = socket["side"], socket["index"]
                candidates = ([producers.get(node.input[index])] if side == "input" else
                              consumers.get(node.output[index], []))
                pairs = []
                for candidate in candidates:
                    if candidate is None or not valid_qdq(candidate):
                        continue
                    if side == "input" and candidate.op_type == "DequantizeLinear":
                        dq, q = candidate, producers.get(candidate.input[0])
                    elif side == "output" and candidate.op_type == "QuantizeLinear":
                        q = candidate
                        dq = next((child for child in consumers.get(q.output[0], [])
                                   if child.op_type == "DequantizeLinear" and valid_qdq(child)), None)
                    else:
                        continue
                    if q is None or dq is None or not valid_qdq(q):
                        continue
                    scale = onnx.numpy_helper.to_array(initializers[q.input[1]])
                    zero = onnx.numpy_helper.to_array(initializers[q.input[2]])
                    if (scale.shape != np.asarray(socket["scale"]).shape
                            or not np.allclose(scale, socket["scale"], rtol=1e-5, atol=1e-7)
                            or not np.array_equal(zero, np.rint(socket["zero_point"]))):
                        continue
                    if side == "input":
                        origin = q.input[0]
                        original = socket["tensor"]
                        if activation_origin(origin) != expected_producers.get(original, (None, original)):
                            continue
                    elif (activation_origin(dq.output[0]) != expected_producers.get(socket["tensor"], (None, socket["tensor"]))):
                        continue
                    pairs.append([q.name, dq.name])
                if not pairs:
                    raise ValueError(f"INT8 node {name!r} {side} socket {index} lacks matching QDQ evidence.")
            for index in decision.quantized_parameter_inputs:
                dq = producers.get(node.input[index])
                weight = initializers.get(dq.input[0]) if dq is not None and dq.input else None
                if (dq is None or dq.op_type != "DequantizeLinear" or not valid_qdq(dq)
                        or weight is None or weight.data_type not in {
                            onnx.TensorProto.INT8, onnx.TensorProto.UINT8, onnx.TensorProto.INT32}):
                    raise ValueError(f"INT8 node {name!r} parameter {index} lacks integer-weight DQ evidence.")
        evidence[name] = {"qdq_nodes": sorted(set(qdq)), "precision": decision.precision}
    for output in exported_model.graph.output:
        if output.name in expected_producers:
            if activation_origin(output.name, require_output_qdq=True) != expected_producers[output.name]:
                raise ValueError(f"Public output {output.name!r} changed activation provenance.")
    for name, decision in plan.decisions.items():
        if decision.precision != "int8" or evidence[name]["qdq_nodes"]:
            continue
        groups = [group for group in plan.groups if name in group["members"]]
        anchors = sorted({other for group in groups for other in group["members"]
                          if evidence[other]["qdq_nodes"]})
        if not anchors:
            raise ValueError(f"INT8 node {name!r} has no verifiable exported QDQ evidence.")
        evidence[name]["quantized_via_group"] = anchors
    total = len(plan.decisions)
    capable = sum(decision.quantizable for decision in plan.decisions.values())
    count = sum(decision.precision == "int8" for decision in plan.decisions.values())
    return {"schema_version": 1, "verified": True, "plan_fingerprint": plan.fingerprint,
            "coverage": {"optimized_nodes": total, "quantizable_nodes": capable,
                         "int8_nodes": count,
                         "fp16_nodes": sum(d.precision == "fp16" for d in plan.decisions.values()),
                         "fp32_nodes": sum(d.precision == "fp32" for d in plan.decisions.values()),
                         "int8_fraction_of_quantizable": count / capable if capable else 0.0,
                         "mapped_source_nodes": len(plan.source_to_optimized),
                         "unmapped_source_nodes": len(plan.unmapped_source_nodes)},
            "layout": evidence, "plan": plan.to_dict()}
