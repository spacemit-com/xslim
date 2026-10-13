# Copyright (c) 2025 SpacemiT. All rights reserved.
"""Support for models that carry Microsoft custom operators.

Some exported models (e.g. HuggingFace EmbeddingGemma-2) declare Microsoft private
operators such as ``SimplifiedLayerNormalization``. The onnx toolchain does not know
them, which breaks ``--fp16`` conversion in two ways:

* ``onnx.version_converter`` raises ``IndexError: unordered_map::at`` for an operator
  that has no schema, and
* ``onnx.shape_inference`` cannot propagate types through them, so
  ``onnxconverter_common`` leaves nodes with mixed fp32/fp16 inputs and ONNX Runtime
  rejects the result.

This module handles both cases without touching the model's operators:

* an operator that is already part of the ai.onnx standard at a higher opset
  (``RotaryEmbedding``, ai.onnx >= 23; ``Gelu``, ai.onnx >= 20) is converted to the ai.onnx
  domain;
* an operator that has no ai.onnx equivalent (``SimplifiedLayerNormalization``,
  ``SkipSimplifiedLayerNormalization``) is described by an *extension schema* that is
  registered with ONNX, so type/shape inference behaves normally and the operator keeps
  its own name, domain and dtype contract.

After conversion the graph is verified with onnx type inference (strict),
and any residual fp32/fp16 mismatch is repaired.
"""

from typing import Dict, List, Optional

import onnx
import onnx.helper as onnx_helper
from onnx import TensorProto
from onnx.onnx_cpp2py_export.defs import OpSchema

_FLOAT_TYPES = {TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.BFLOAT16}
_FLOAT_TYPE_STRINGS = ["tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"]
_TYPE_NAMES = {
    TensorProto.FLOAT16: "fp16",
    TensorProto.FLOAT: "fp32",
    TensorProto.DOUBLE: "fp64",
    TensorProto.BFLOAT16: "bf16",
    TensorProto.BOOL: "bool",
    TensorProto.INT64: "int64",
    TensorProto.INT32: "int32",
}

# Microsoft private operators with no ai.onnx equivalent: described by an extension
# schema so onnx can run type/shape inference. Signatures follow the operator definitions.
_EXTENSION_SCHEMAS = {
    "SimplifiedLayerNormalization": {
        "inputs": [("X", "T", False), ("scale", "T", False)],
        "outputs": [("Y", "T", False), ("Mean", "U", True), ("InvStdDev", "U", True)],
    },
    "SkipSimplifiedLayerNormalization": {
        "inputs": [("X", "T", False), ("skip", "T", False), ("gamma", "T", False)],
        "outputs": [
            ("Y", "T", False),
            ("Mean", "U", True),
            ("InvStdDev", "U", True),
            ("input_skip_bias_sum", "T", True),
        ],
    },
}

# Microsoft private operators that are ai.onnx standard from a given opset on: they are
# moved to the ai.onnx domain. ``inputs`` selects the inputs in ai.onnx order, where the
# sentinel :data:`_POSITION_IDS` stands for the synthesized position tensor; ``attributes``
# replaces the attribute set, for the operators whose signature is fixed and therefore has
# no equivalent ai.onnx attribute.
_POSITION_IDS = "@position_ids"

_MS_OPERATORS_IN_AI_ONNX = {
    # Microsoft order (X, position_ids, cos_cache, sin_cache) -> ai.onnx (X, cos, sin, pos)
    "RotaryEmbedding": {"inputs": [0, 2, 3, _POSITION_IDS], "opset": 23},
    # ai.onnx Gelu is the erf-based activation, like the com.microsoft one; pin
    # approximate="none" so the mapping stays explicit (ai.onnx also offers "tanh").
    "Gelu": {"inputs": [0], "attributes": {"approximate": "none"}},
}

_EXTENSION_SCHEMAS_REGISTERED = False


def register_extension_op_schemas() -> None:
    """Register extension schemas for the known schema-less Microsoft operators.

    Registration is global to the process and idempotent. An extension schema gives the
    operator a dtype contract but keeps its name and domain, so ``version_converter``
    and ``shape_inference`` can process a graph that contains it.
    """
    global _EXTENSION_SCHEMAS_REGISTERED
    if _EXTENSION_SCHEMAS_REGISTERED:
        return
    for op_type, spec in _EXTENSION_SCHEMAS.items():
        try:
            onnx.defs.get_schema(op_type, 1, "")
            continue
        except Exception:
            pass
        onnx.defs.register_schema(
            OpSchema(
                op_type,
                "",
                21,
                "Microsoft extension operator promoted to ai.onnx by xslim",
                inputs=[_formal_parameter(n, t, o) for n, t, o in spec["inputs"]],
                outputs=[_formal_parameter(n, t, o) for n, t, o in spec["outputs"]],
                type_constraints=[
                    ("T", _FLOAT_TYPE_STRINGS, "float tensor"),
                    ("U", _FLOAT_TYPE_STRINGS, "float tensor"),
                ],
            )
        )
    _EXTENSION_SCHEMAS_REGISTERED = True


def _formal_parameter(name: str, type_str: str, optional: bool):
    option = OpSchema.FormalParameterOption.Optional if optional else OpSchema.FormalParameterOption.Single
    return OpSchema.FormalParameter(name, type_str, "", param_option=option, is_homogeneous=True, min_arity=1)


def convert_microsoft_ops_to_ai_onnx(
    onnx_model: onnx.ModelProto, shapes: Optional[Dict[str, List[Optional[int]]]] = None
) -> int:
    """Convert Microsoft private operators that ai.onnx standardised to the default domain.

    ``RotaryEmbedding`` (ai.onnx >= 23) and ``Gelu`` (ai.onnx >= 20). The ai.onnx
    ``RotaryEmbedding`` takes ``(X, cos_cache, sin_cache, position_ids)`` while the
    Microsoft one takes ``(X, position_ids, cos_cache, sin_cache)`` and a 1D
    ``position_ids``; the extra nodes needed by the ai.onnx form (a 2D ``arange`` position
    tensor and ``num_heads``) are synthesized from the shapes. ``Gelu`` takes only the
    input tensor in both domains; the Microsoft schema has no attribute, so the ai.onnx
    ``approximate`` is pinned to ``"none"`` (the erf-based form both domains implement).

    ``shapes`` is the tensor-shape map of the pre-conversion model (see
    :func:`tensor_shapes`): the simplify passes strip the ``value_info`` used to size
    ``num_heads``.
    """
    nodes = [n for n in onnx_model.graph.node if n.domain == "com.microsoft" and n.op_type in _MS_OPERATORS_IN_AI_ONNX]
    if not nodes:
        return 0

    shapes = shapes if shapes is not None else _shape_map(onnx_model)
    rotary_nodes = [n for n in nodes if n.op_type == "RotaryEmbedding"]
    if rotary_nodes:
        position_ids = _make_position_ids(rotary_nodes[0].input[0], onnx_model)
        replacements = {str(_POSITION_IDS): position_ids}
    else:
        replacements = {}

    for node in nodes:
        spec = _MS_OPERATORS_IN_AI_ONNX[node.op_type]
        for key, value in spec.get("attributes", {}).items():
            attribute = next((a for a in node.attribute if a.name == key), None)
            if attribute is None:
                node.attribute.append(onnx_helper.make_attribute(key, value))
            elif attribute.type == onnx.AttributeProto.STRING:
                attribute.s = value.encode()
            else:
                attribute.CopyFrom(onnx_helper.make_attribute(key, value))
        inputs = [node.input[i] if isinstance(i, int) else replacements[i] for i in spec["inputs"]]
        node.input[:] = inputs
        node.domain = ""
        if node.op_type == "RotaryEmbedding" and "num_heads" not in {a.name for a in node.attribute}:
            x, cos_cache = node.input[0], node.input[1]
            x_shape, cos_shape = shapes.get(x), shapes.get(cos_cache)
            if x_shape and cos_shape and x_shape[-1] and cos_shape[-1]:
                node.attribute.append(onnx_helper.make_attribute("num_heads", x_shape[-1] // (2 * cos_shape[-1])))
    return len(nodes)


def _make_position_ids(reference: str, onnx_model: onnx.ModelProto) -> str:
    """Synthesize ``position_ids`` as ``arange(sequence_length)`` shaped ``[1, seq]``.

    The ai.onnx ``RotaryEmbedding`` needs 2D position ids; the Microsoft form indexes the
    full cos/sin cache and therefore only carries a 1D position. Building the range from
    the reference tensor's second dimension reproduces the Microsoft semantics.
    """
    names = ["/xslim/position_ids/shape", "/xslim/position_ids/seq", "/xslim/position_ids/range", "/xslim/position_ids"]
    inits = [
        "/xslim/position_ids/idx",
        "/xslim/position_ids/start",
        "/xslim/position_ids/step",
        "/xslim/position_ids/axes",
    ]
    position_tensor = names[-1]
    if any(t.name == position_tensor for t in onnx_model.graph.initializer):
        return position_tensor  # already synthesized
    existing = {n.name for n in onnx_model.graph.node}
    if names[0] in existing:
        return position_tensor
    new_nodes = [
        onnx_helper.make_node("Shape", [reference], [names[0]], name=names[0]),
        onnx_helper.make_node("Gather", [names[0], inits[0]], [names[1]], name=names[1], axis=0),
        onnx_helper.make_node("Range", [inits[1], names[1], inits[2]], [names[2]], name=names[2]),
        onnx_helper.make_node("Unsqueeze", [names[2], inits[3]], [names[3]], name=names[3]),
    ]
    new_initializers = [
        onnx_helper.make_tensor(inits[0], TensorProto.INT64, [1], [1]),
        onnx_helper.make_tensor(inits[1], TensorProto.INT64, [], [0]),
        onnx_helper.make_tensor(inits[2], TensorProto.INT64, [], [1]),
        onnx_helper.make_tensor(inits[3], TensorProto.INT64, [1], [0]),
    ]
    existing = {n.name for n in onnx_model.graph.node}
    if names[0] not in existing:
        first = next(
            n for n in onnx_model.graph.node if n.domain == "com.microsoft" and n.op_type in _MS_OPERATORS_IN_AI_ONNX
        )
        index = list(onnx_model.graph.node).index(first)
        for offset, node in enumerate(new_nodes):
            onnx_model.graph.node.insert(index + offset, node)
        onnx_model.graph.initializer.extend(new_initializers)
    return position_tensor


def tensor_shapes(onnx_model: onnx.ModelProto) -> Dict[str, List[Optional[int]]]:
    """Public accessor for :func:`_shape_map` (used before the simplify passes run)."""
    return _shape_map(onnx_model)


def _shape_map(onnx_model: onnx.ModelProto) -> Dict[str, List[Optional[int]]]:
    """Tensor shapes, from value_info/initializers; falls back to shape inference.

    The simplify passes clear ``value_info``, so a lookup may find nothing; running
    inference recovers the shapes needed to size the synthesized ``num_heads``.
    """

    def collect(source):
        shapes: Dict[str, List[Optional[int]]] = {}
        for collection in (source.graph.input, source.graph.output, source.graph.value_info):
            for value in collection:
                if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape"):
                    shapes[value.name] = [
                        dim.dim_value if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim
                    ]
        for tensor in source.graph.initializer:
            shapes[tensor.name] = list(tensor.dims)
        return shapes

    shapes = collect(onnx_model)
    if not shapes:
        return shapes
    try:
        inferred = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=False, data_prop=False)
        shapes.update(collect(inferred))
    except Exception:
        pass
    return shapes


def _check_types(onnx_model: onnx.ModelProto) -> None:
    """Strict type check on a copy with stale value_info removed.

    Raising means the graph has a genuine fp32/fp16 mismatch; the copies keep the check
    from being confused by value_info left over by the converters.
    """
    probe = onnx.ModelProto()
    probe.CopyFrom(onnx_model)
    del probe.graph.value_info[:]
    onnx.shape_inference.infer_shapes(probe, strict_mode=True, check_type=True)


def _infer_tensor_types(onnx_model: onnx.ModelProto) -> Dict[str, int]:
    probe = onnx.ModelProto()
    probe.CopyFrom(onnx_model)
    del probe.graph.value_info[:]
    try:
        inferred = onnx.shape_inference.infer_shapes(probe, strict_mode=False, data_prop=False)
    except Exception:
        return {}
    types: Dict[str, int] = {}
    for collection in (inferred.graph.input, inferred.graph.output, inferred.graph.value_info):
        for value in collection:
            if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type:
                types[value.name] = value.type.tensor_type.elem_type
    for tensor in inferred.graph.initializer:
        types[tensor.name] = tensor.data_type
    return types


def repair_fp16_dtype_mismatches(onnx_model: onnx.ModelProto, max_iterations: int = 100) -> int:
    """Repair fp32/fp16 mismatches using onnx type inference as the oracle.

    For each node whose float inputs disagree, cast the minority inputs to the node's
    intrinsic type (the type of a data input, or of a graph output, or fp16 when the node
    is fully constant). Returns the number of Cast nodes inserted.
    """
    initializers = {t.name for t in onnx_model.graph.initializer}
    graph_outputs = {o.name for o in onnx_model.graph.output}
    produced = {o for n in onnx_model.graph.node for o in n.output if o}
    cast_cache: Dict[tuple, str] = {}
    applied = 0

    for _ in range(max_iterations):
        try:
            _check_types(onnx_model)
            break
        except Exception:
            pass
        types = _infer_tensor_types(onnx_model)
        target = None
        conflicting: List[tuple] = []
        for node in onnx_model.graph.node:
            float_inputs = [(i, types.get(i)) for i in node.input if types.get(i) in _FLOAT_TYPES]
            if len({t for _, t in float_inputs}) > 1:
                target, conflicting = node, float_inputs
                break
        if target is None:
            break

        data_types = [t for i, t in conflicting if i not in initializers and i not in graph_outputs]
        if data_types:
            wanted = data_types[0]
        else:
            output_types = [t for i, t in conflicting if i in graph_outputs]
            wanted = output_types[0] if output_types else TensorProto.FLOAT16

        added = 0
        for name, current in conflicting:
            if current == wanted:
                continue
            key = (name, wanted)
            if key in cast_cache:
                replacement = cast_cache[key]
            else:
                replacement = "{}_cast_to_{}".format(name, _TYPE_NAMES.get(wanted, wanted))
                while replacement in produced:
                    replacement += "_1"
                index = list(target.input).index(name)
                onnx_model.graph.node.insert(
                    list(onnx_model.graph.node).index(target),
                    onnx_helper.make_node(
                        "Cast", [name], [replacement], name="xslim_fp16_cast_{}_{}".format(applied, added), to=wanted
                    ),
                )
                produced.add(replacement)
                cast_cache[key] = replacement
            target.input[list(target.input).index(name)] = replacement
            added += 1
        if not added:
            break
        applied += added
    return applied
