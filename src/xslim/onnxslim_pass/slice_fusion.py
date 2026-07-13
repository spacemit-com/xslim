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
    normalized_axis = axis
    if normalized_axis < 0:
        normalized_axis += rank

    if normalized_axis < 0 or normalized_axis >= rank:
        return None

    return normalized_axis


def _normalize_bound(bound, axis_dim):
    normalized_bound = int(bound)
    if normalized_bound < 0:
        normalized_bound += axis_dim

    return max(0, min(normalized_bound, axis_dim))


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


def _extract_slice_info(slice_node, data_input):
    if slice_node.op != "Slice" or len(slice_node.outputs) != 1:
        return None

    if len(slice_node.inputs) not in (4, 5):
        return None

    starts = _get_constant_int_list(slice_node.inputs[1])
    ends = _get_constant_int_list(slice_node.inputs[2])
    axes = _get_constant_int_list(slice_node.inputs[3])
    if starts is None or ends is None or axes is None:
        return None

    if len(starts) != 1 or len(ends) != 1 or len(axes) != 1:
        return None

    steps = [1]
    if len(slice_node.inputs) == 5:
        steps = _get_constant_int_list(slice_node.inputs[4])
        if steps is None or len(steps) != 1:
            return None

    if int(steps[0]) != 1:
        return None

    input_shape = getattr(data_input, "shape", None)
    if input_shape is None:
        return None

    axis = _normalize_axis(int(axes[0]), len(input_shape))
    if axis is None:
        return None

    axis_dim = _get_static_axis_dim(data_input, axis)
    if axis_dim is None:
        return None

    start = _normalize_bound(starts[0], axis_dim)
    end = _normalize_bound(ends[0], axis_dim)
    if start >= end:
        return None

    return {
        "axis": axis,
        "axis_dim": axis_dim,
        "start": start,
        "end": end,
        "size": end - start,
        "node": slice_node,
        "output": slice_node.outputs[0],
    }


def _collect_slice_group(anchor_slice):
    if len(anchor_slice.inputs) < 4:
        return None

    data_input = anchor_slice.inputs[0]
    anchor_info = _extract_slice_info(anchor_slice, data_input)
    if anchor_info is None:
        return None

    slice_infos = []
    for sibling_node in list(data_input.outputs):
        if sibling_node.op != "Slice":
            continue

        sibling_info = _extract_slice_info(sibling_node, data_input)
        if sibling_info is None:
            continue

        if sibling_info["axis"] != anchor_info["axis"]:
            continue

        slice_infos.append(sibling_info)

    if len(slice_infos) < 2:
        return None

    slice_infos.sort(
        key=lambda item: (item["start"], item["end"], item["node"].name)
    )

    current_start = 0
    for slice_info in slice_infos:
        if slice_info["start"] != current_start:
            return None
        current_start = slice_info["end"]

    if current_start != anchor_info["axis_dim"]:
        return None

    return {
        "input": data_input,
        "axis": anchor_info["axis"],
        "nodes": [slice_info["node"] for slice_info in slice_infos],
        "outputs": [slice_info["output"] for slice_info in slice_infos],
        "sizes": [slice_info["size"] for slice_info in slice_infos],
        "name": f"{slice_infos[0]['node'].name}_split",
    }


def _rewrite_slice_to_split(anchor_slice):

    slice_group = _collect_slice_group(anchor_slice)
    if slice_group is None:
        return {}

    data_input = slice_group["input"]
    split_name = slice_group["name"]
    split_sizes = osg.Constant(
        name=f"{split_name}_sizes",
        values=np.asarray(slice_group["sizes"], dtype=np.int64),
    )

    for slice_node in slice_group["nodes"]:
        # Keep the old Slice nodes connected to their input until the fusion
        # pass has finished finding matches. Disconnecting them here can make
        # an upstream multi-output Split look like it has only one consumer,
        # causing FusionSingleConsumerSplitToSlice to remove that producer in
        # the same pass. Graph cleanup will remove these output-less nodes.
        slice_node.outputs.clear()

    return {
        split_name: {
            "op": "Split",
            "inputs": [data_input, split_sizes],
            "outputs": slice_group["outputs"],
            "name": split_name,
            "attrs": {"axis": slice_group["axis"]},
            "domain": None,
        }
    }


class SliceFusionPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input   input   0 1+ slice_0
            Slice   slice_0 4+ 1 input ? ? ? output
            output  output  1 0 slice_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionSliceToSplit"

    def rewrite(self, opset=13):
        return _rewrite_slice_to_split(self.slice_0)

register_fusion_pattern(SliceFusionPatternMatcher(1))
