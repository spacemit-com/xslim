import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


def _get_constant_int_list(tensor):
    if not isinstance(tensor, osg.Constant) or tensor.values is None:
        return None

    values = np.asarray(tensor.values).reshape(-1)
    return [int(value) for value in values.tolist()]


def _normalize_axis(axis, rank):
    normalized_axis = int(axis)
    if normalized_axis < 0:
        normalized_axis += rank

    if normalized_axis < 0 or normalized_axis >= rank:
        return None

    return normalized_axis


def _get_static_axis_dim(variable, axis):
    shape = getattr(variable, "shape", None)
    if shape is None or axis >= len(shape):
        return None

    axis_dim = shape[axis]
    if isinstance(axis_dim, np.generic):
        axis_dim = axis_dim.item()

    if not isinstance(axis_dim, int):
        return None

    return axis_dim if axis_dim > 0 else None


def _get_split_sizes(split_node, axis_dim):
    if len(split_node.inputs) > 1:
        split_sizes = _get_constant_int_list(split_node.inputs[1])
        if split_sizes is None:
            return None
    else:
        split_sizes = split_node.attrs.get("split")
        if split_sizes is None:
            if len(split_node.outputs) == 0 or axis_dim % len(split_node.outputs) != 0:
                return None
            split_sizes = [axis_dim // len(split_node.outputs)] * len(split_node.outputs)
        else:
            split_sizes = [int(size) for size in split_sizes]

    if len(split_sizes) != len(split_node.outputs):
        return None
    if any(size <= 0 for size in split_sizes):
        return None
    if sum(split_sizes) != axis_dim:
        return None

    return split_sizes


def _collect_single_consumer_split(split_node):
    if split_node.op != "Split" or len(split_node.outputs) < 2:
        return None
    if len(split_node.inputs) == 0:
        return None

    data_input = split_node.inputs[0]
    input_shape = getattr(data_input, "shape", None)
    if input_shape is None:
        return None

    axis = _normalize_axis(split_node.attrs.get("axis", 0), len(input_shape))
    if axis is None:
        return None

    axis_dim = _get_static_axis_dim(data_input, axis)
    if axis_dim is None:
        return None

    split_sizes = _get_split_sizes(split_node, axis_dim)
    if split_sizes is None:
        return None

    consumed_outputs = []
    for index, output in enumerate(split_node.outputs):
        consumers = list(getattr(output, "outputs", []) or [])
        if len(consumers) > 0:
            consumed_outputs.append((index, output))

    if len(consumed_outputs) != 1:
        return None

    output_index, output = consumed_outputs[0]
    start = sum(split_sizes[:output_index])
    end = start + split_sizes[output_index]

    return {
        "input": data_input,
        "axis": axis,
        "start": start,
        "end": end,
        "output": output,
        "name": f"{split_node.name}_slice",
    }


def _rewrite_single_consumer_split_to_slice(split_node):
    split_info = _collect_single_consumer_split(split_node)
    if split_info is None:
        return {}

    data_input = split_info["input"]
    if split_node in getattr(data_input, "outputs", []):
        data_input.outputs.remove(split_node)

    split_node.inputs.clear()
    split_node.outputs.clear()

    slice_name = split_info["name"]
    starts = osg.Constant(
        name=f"{slice_name}_starts",
        values=np.asarray([split_info["start"]], dtype=np.int64),
    )
    ends = osg.Constant(
        name=f"{slice_name}_ends",
        values=np.asarray([split_info["end"]], dtype=np.int64),
    )
    axes = osg.Constant(
        name=f"{slice_name}_axes",
        values=np.asarray([split_info["axis"]], dtype=np.int64),
    )
    steps = osg.Constant(
        name=f"{slice_name}_steps",
        values=np.asarray([1], dtype=np.int64),
    )

    return {
        slice_name: {
            "op": "Slice",
            "inputs": [data_input, starts, ends, axes, steps],
            "outputs": [split_info["output"]],
            "name": slice_name,
            "attrs": {},
            "domain": None,
        }
    }


class SingleConsumerSplitToSlicePatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input  input   0 1 split_0
            Split  split_0 1+ 1 input output
            output output  1 0 split_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionSingleConsumerSplitToSlice"

    def rewrite(self, opset=13):
        return _rewrite_single_consumer_split_to_slice(self.split_0)


register_fusion_pattern(SingleConsumerSplitToSlicePatternMatcher(1))
