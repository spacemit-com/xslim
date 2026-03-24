import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


def _get_constant_pad_value(pad_node):
    if len(pad_node.inputs) < 3:
        return 0.0

    value_input = pad_node.inputs[2]
    if isinstance(value_input, osg.Variable) and value_input.name == "":
        return 0.0

    if not isinstance(value_input, osg.Constant):
        return None

    values = np.asarray(value_input.values)
    if values.size == 0:
        return 0.0

    if not np.all(values == values.reshape(-1)[0]):
        return None

    return values.reshape(-1)[0].item()


def _has_supported_axes_input(pad_node):
    if len(pad_node.inputs) < 4:
        return True

    axes_input = pad_node.inputs[3]
    return isinstance(axes_input, osg.Variable) and axes_input.name == ""


def _extract_pool_pads(pad_node):
    if not isinstance(pad_node.inputs[1], osg.Constant):
        return None

    pad_values = np.asarray(pad_node.inputs[1].values).reshape(-1).tolist()
    if len(pad_values) < 6 or len(pad_values) % 2 != 0:
        return None

    rank = len(pad_values) // 2
    if any(pad != 0 for pad in (pad_values[:2] + pad_values[rank: rank + 2])):
        return None

    return [int(pad) for pad in (pad_values[2:rank] + pad_values[rank + 2:])]


def _get_existing_pool_pads(pool_node, spatial_rank):
    auto_pad = pool_node.attrs.get("auto_pad", "NOTSET")
    if auto_pad not in {"NOTSET", "VALID"}:
        return None

    existing_pads = list(pool_node.attrs.get("pads", [0] * (2 * spatial_rank)))
    if len(existing_pads) != 2 * spatial_rank:
        return None

    if any(pad != 0 for pad in existing_pads):
        return None

    return existing_pads


def _rewrite_pool_with_input(pool_node, pad_node, merged_pads):
    input_variable = pad_node.inputs[0]
    pad_variable = pad_node.outputs[0]
    pool_node.inputs[pool_node.inputs.index(pad_variable)] = input_variable

    attrs = dict(pool_node.attrs)
    attrs["pads"] = merged_pads
    attrs.pop("auto_pad", None)

    inputs = list(pool_node.inputs)
    outputs = list(pool_node.outputs)

    pool_node.inputs.clear()
    pool_node.outputs.clear()

    if len(pad_node.users) == 0:
        if pad_node in input_variable.outputs:
            input_variable.outputs.remove(pad_node)
        pad_node.inputs.clear()
        pad_node.outputs.clear()

    return {
        pool_node.name: {
            "op": pool_node.op,
            "inputs": inputs,
            "outputs": outputs,
            "name": pool_node.name,
            "attrs": attrs,
            "domain": None,
        }
    }


class PadAveragePoolMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input       input   0 1 pad_0
            Pad         pad_0   1+ 1 input pool_0
            AveragePool pool_0  1+ 1 pad_0 output
            output      output  1 0 pool_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionPadAveragePool"

    def parameter_check(self) -> bool:
        pad_node = getattr(self, "pad_0")
        is_constant_pad = isinstance(pad_node.inputs[1], osg.Constant)
        return is_constant_pad and _has_supported_axes_input(pad_node)

    def rewrite(self, opset=11):
        pad_node = getattr(self, "pad_0")
        pool_node = getattr(self, "pool_0")

        if pad_node.attrs.get("mode", "constant") != "constant":
            return {}

        pad_fill_value = _get_constant_pad_value(pad_node)
        if pad_fill_value is None or not np.isclose(pad_fill_value, 0.0):
            return {}

        pool_pads = _extract_pool_pads(pad_node)
        if pool_pads is None:
            return {}

        existing_pads = _get_existing_pool_pads(pool_node, len(pool_pads) // 2)
        if existing_pads is None:
            return {}

        merged_pads = [a + b for a, b in zip(pool_pads, existing_pads)]
        match_case = _rewrite_pool_with_input(pool_node, pad_node, merged_pads)
        match_case[pool_node.name]["attrs"]["count_include_pad"] = 1
        return match_case


class PadMaxPoolMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input   input   0 1 pad_0
            Pad     pad_0   1+ 1 input pool_0
            MaxPool pool_0  1+ 1 pad_0 output
            output  output  1 0 pool_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionPadMaxPool"

    def parameter_check(self) -> bool:
        pad_node = getattr(self, "pad_0")
        is_constant_pad = isinstance(pad_node.inputs[1], osg.Constant)
        return is_constant_pad and _has_supported_axes_input(pad_node)

    def rewrite(self, opset=11):
        pad_node = getattr(self, "pad_0")
        pool_node = getattr(self, "pool_0")

        if pad_node.attrs.get("mode", "constant") != "constant":
            return {}

        pad_fill_value = _get_constant_pad_value(pad_node)
        if pad_fill_value is None or not np.isneginf(pad_fill_value):
            return {}

        pool_pads = _extract_pool_pads(pad_node)
        if pool_pads is None:
            return {}

        existing_pads = _get_existing_pool_pads(pool_node, len(pool_pads) // 2)
        if existing_pads is None:
            return {}

        merged_pads = [a + b for a, b in zip(pool_pads, existing_pads)]
        return _rewrite_pool_with_input(pool_node, pad_node, merged_pads)


register_fusion_pattern(PadAveragePoolMatcher(1))
register_fusion_pattern(PadMaxPoolMatcher(1))
