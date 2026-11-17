'Controlled local FP16 after QDQ export; PPQ calibration remains FP32.'

import copy
import types

import numpy as np
import onnx
import onnxruntime as ort


def convert_local_fp16(model, *, plan, batches):
    selected = {name for name, d in plan.decisions.items() if d.precision == 'fp16'}
    if not selected:
        return model
    if model.ByteSize() >= 2_000_000_000:
        raise ValueError('Local FP16 does not support models >= 2GB.')
    if model.functions:
        raise ValueError('Local FP16 does not support ONNX functions.')
    if any(a.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
           for n in model.graph.node for a in n.attribute):
        raise ValueError('Local FP16 does not support subgraphs.')
    users = {}
    for n in model.graph.node:
        for name in n.input:
            users.setdefault(name, set()).add(n.name)
    for tensor in model.graph.initializer:
        if users.get(tensor.name, set()) & selected and len(users[tensor.name]) > 1:
            raise ValueError(f'Local FP16 shared initializer unsupported: {tensor.name}.')
    from onnxconverter_common.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

    before = copy.deepcopy(model)
    io = {v.name: copy.deepcopy(v) for v in [*model.graph.input, *model.graph.output]}
    blocked = [n.name for n in model.graph.node if n.name not in selected]
    if any(n.op_type in DEFAULT_OP_BLOCK_LIST for n in model.graph.node if n.name in selected):
        raise ValueError('Selected operation is blocked by the FP16 converter capability list.')
    # Isolate the converter globals: its optional Cast cleanup crashes on a
    # blocked FP32 terminal. Keep all Casts for strict inference/provenance gates.
    converter_globals = dict(convert_float_to_float16.__globals__)
    converter_globals["remove_unnecessary_cast_node"] = lambda graph: None
    converter = types.FunctionType(convert_float_to_float16.__code__, converter_globals,
                                   argdefs=convert_float_to_float16.__defaults__)
    converted = converter(copy.deepcopy(model), keep_io_types=True,
        op_block_list=[*DEFAULT_OP_BLOCK_LIST, 'QuantizeLinear', 'DequantizeLinear'],
        node_block_list=blocked, disable_shape_infer=False)
    # Collapse converter-created round trips between two blocked FP32 nodes.
    # Keep a terminal FLOAT Cast as an explicit, typed public-output boundary.
    for second in list(converted.graph.node):
        if second.op_type != 'Cast' or not second.attribute or second.attribute[0].i != onnx.TensorProto.FLOAT:
            continue
        first = next((n for n in converted.graph.node if second.input[0] in n.output), None)
        if (first is None or first.op_type != 'Cast' or not first.attribute
                or first.attribute[0].i != onnx.TensorProto.FLOAT16):
            continue
        original = first.input[0]
        if second.output[0] in {v.name for v in converted.graph.output}:
            second.input[0] = original
        else:
            for n in converted.graph.node:
                for i, name in enumerate(n.input):
                    if name == second.output[0]:
                        n.input[i] = original
            converted.graph.node.remove(second)
        if not any(first.output[0] in n.input for n in converted.graph.node):
            converted.graph.node.remove(first)
    del converted.graph.value_info[:]
    # Converter 1.16 can append an FP16 Cast to a blocked terminal FP32 node.
    # Repair only that exact single-use pattern and preserve the public IO.
    for output in converted.graph.output:
        expected = io[output.name]
        producer = next((n for n in converted.graph.node if output.name in n.output), None)
        if output.type.tensor_type.elem_type != expected.type.tensor_type.elem_type:
            if (producer is None or producer.op_type != 'Cast' or producer.domain
                    or len(producer.input) != 1 or len(producer.output) != 1):
                raise ValueError(f'Unsupported local FP16 output boundary: {output.name}.')
            upstream = next((n for n in converted.graph.node if producer.input[0] in n.output), None)
            uses = [n for n in converted.graph.node if producer.input[0] in n.input]
            if (upstream is None or upstream.name not in blocked or len(uses) != 1
                    or expected.type.tensor_type.elem_type != onnx.TensorProto.FLOAT):
                raise ValueError(f'Unsafe local FP16 output boundary repair: {output.name}.')
            old = producer.input[0]
            upstream.output[list(upstream.output).index(old)] = output.name
            converted.graph.node.remove(producer)
            for vi in list(converted.graph.value_info):
                if vi.name in {old, output.name}:
                    converted.graph.value_info.remove(vi)
            output.CopyFrom(expected)
    for value in [*converted.graph.input, *converted.graph.output]:
        if value.SerializeToString() != io[value.name].SerializeToString():
            raise ValueError(f'Local FP16 changed public IO declaration: {value.name}.')
    converted = onnx.shape_inference.infer_shapes(converted, strict_mode=True, check_type=True)
    onnx.checker.check_model(converted, full_check=True)
    from .precision_strategy import validate_export_layout
    validate_export_layout(plan, converted)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    left = ort.InferenceSession(before.SerializeToString(), options, providers=['CPUExecutionProvider'])
    right = ort.InferenceSession(converted.SerializeToString(), options, providers=['CPUExecutionProvider'])
    if not batches:
        raise ValueError('Local FP16 requires actual input batches for its ORT gate.')
    for batch in batches:
        a, b = left.run(None, copy.deepcopy(batch)), right.run(None, copy.deepcopy(batch))
        if (len(a) != len(b) or any(x.shape != y.shape or not np.isfinite(y).all()
                or not np.allclose(x, y, rtol=1e-3, atol=1e-3) for x, y in zip(a, b))):
            raise ValueError('Local FP16 failed real ORT comparison (rtol=atol=1e-3).')
    return converted
