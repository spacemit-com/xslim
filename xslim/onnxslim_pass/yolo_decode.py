import onnx
import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnx import TensorProto, helper
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern

from xslim.defs import MIN_ONNX_OPSET_VERSION, get_default_onnx_opset_version


YOLO_DECODE_DOMAIN = "spacemit_functions"
YOLO_DECODE_OPSET_VERSION = 1
YOLO_DECODE_FUNCTION_NAME = "YoloDecode"


def _attr_as_list(node, attr_name):
    value = node.attrs.get(attr_name)
    if value is None:
        return None
    return np.asarray(value).reshape(-1).tolist()


def _attr_as_int(node, attr_name):
    value = node.attrs.get(attr_name)
    if value is None:
        return None
    return int(np.asarray(value).reshape(-1)[0])


def _find_constant_input(node, exclude_input):
    for input_value in node.inputs:
        if input_value is exclude_input:
            continue
        if isinstance(input_value, osg.Constant) and input_value.values is not None:
            return input_value
    return None


def _remove_output(variable, node):
    if node in variable.outputs:
        variable.outputs.remove(node)


def _constant_values(input_value):
    if isinstance(input_value, osg.Constant) and input_value.values is not None:
        return np.asarray(input_value.values)
    return None


def _static_dim(dim_value):
    if isinstance(dim_value, (int, np.integer)):
        return int(dim_value)
    return None


def _derive_yolo_decode_attrs(input_variable, flat_weight_size, split_0):
    reg_max = int(flat_weight_size)
    num_class = -1

    if split_0 is not None and len(split_0.inputs) > 1:
        split_sizes = _constant_values(split_0.inputs[1])
        if split_sizes is not None:
            split_sizes = split_sizes.reshape(-1)
            if split_sizes.size >= 2:
                bbox_length = int(split_sizes[0])
                num_class = int(split_sizes[1])
                if bbox_length % 4 != 0:
                    return None, None
                reg_max = bbox_length // 4
                if reg_max != int(flat_weight_size):
                    return None, None

    input_shape = getattr(input_variable, "shape", None)
    if num_class < 0 and input_shape is not None and len(input_shape) > 1:
        channel_dim = _static_dim(input_shape[1])
        if channel_dim is not None and channel_dim >= reg_max * 4:
            num_class = channel_dim - reg_max * 4

    return num_class, reg_max


def _make_tensor_constant_node(name, output_name, data_type, dims, values):
    """Create a Constant node with tensor value."""
    return helper.make_node(
        "Constant",
        [],
        [output_name],
        name=name,
        value=helper.make_tensor(name, data_type, dims, values),
    )


def _make_ref_scalar_constant_node(name, output_name, ref_attr_name):
    """Create a Constant node that references a function attribute."""
    node = helper.make_node("Constant", [], [output_name], name=name)
    attr = onnx.AttributeProto()
    attr.name = "value_int"
    attr.type = onnx.AttributeProto.INT
    attr.ref_attr_name = ref_attr_name
    node.attribute.extend([attr])
    return node


def build_yolo_decode_function(opset_version=MIN_ONNX_OPSET_VERSION):
    """
    Build YoloDecode function proto with decode logic matching the fused pattern.

    This function implements the YoloDecode operation which includes:
    - Splitting input into bbox and classification branches
    - Reshaping bbox logits into [batch, 4, reg_max, spatial]
    - Applying transpose + softmax + 1x1 conv for DFL decoding
    - Restoring the original bbox post-processing branches
    - Concatenating bbox and sigmoid-activated class predictions

    Inputs:
        input: Model output tensor [batch, (bbox_length * 4 + num_class), spatial]
        flat_weight: Flattened DFL weight from 1x16x1x1 conv
        sub_const: Constant for bbox coordinate adjustment
        add_const: Constant for additional adjustment
        mul_const: Constant for scaling

    Attributes:
        num_class: Number of object classes
        reg_max: Regression max value (typically 16)

    Output:
        Decoded bboxes and class confidences [batch, (4 + num_class), spatial]
    """

    # Build the function body so it mirrors the original YoloDecode pattern.
    nodes = [
        _make_ref_scalar_constant_node(
            "reg_max_const", "reg_max_scalar", "reg_max"
        ),
        _make_ref_scalar_constant_node(
            "num_class_const", "num_class_scalar", "num_class"
        ),
        _make_tensor_constant_node(
            "four_const", "four_scalar", TensorProto.INT64, [], [4]
        ),
        _make_tensor_constant_node(
            "one_const", "one_scalar", TensorProto.INT64, [], [1]
        ),
        _make_tensor_constant_node(
            "axes0_const", "axes0", TensorProto.INT64, [1], [0]
        ),
        _make_tensor_constant_node(
            "axes1_const", "axes1", TensorProto.INT64, [1], [1]
        ),
        _make_tensor_constant_node(
            "shape_idx_0_const", "shape_idx_0", TensorProto.INT64, [], [0]
        ),
        _make_tensor_constant_node(
            "shape_idx_2_const", "shape_idx_2", TensorProto.INT64, [], [2]
        ),
        _make_tensor_constant_node(
            "slice_0_starts_const", "slice_0_starts", TensorProto.INT64, [1], [0]
        ),
        _make_tensor_constant_node(
            "slice_0_ends_const", "slice_0_ends", TensorProto.INT64, [1], [2]
        ),
        _make_tensor_constant_node(
            "slice_1_starts_const", "slice_1_starts", TensorProto.INT64, [1], [2]
        ),
        _make_tensor_constant_node(
            "slice_1_ends_const", "slice_1_ends", TensorProto.INT64, [1], [4]
        ),
        _make_tensor_constant_node(
            "slice_steps_const", "slice_steps", TensorProto.INT64, [1], [1]
        ),
        _make_tensor_constant_node(
            "divisor_const", "divisor", TensorProto.FLOAT, [], [2.0]
        ),

        helper.make_node(
            "Mul",
            ["reg_max_scalar", "four_scalar"],
            ["bbox_length"],
            name="bbox_length_mul",
        ),
        helper.make_node(
            "Unsqueeze",
            ["bbox_length", "axes0"],
            ["bbox_length_vec"],
            name="bbox_length_unsqueeze",
        ),
        helper.make_node(
            "Unsqueeze",
            ["num_class_scalar", "axes0"],
            ["num_class_vec"],
            name="num_class_unsqueeze",
        ),
        helper.make_node(
            "Concat",
            ["bbox_length_vec", "num_class_vec"],
            ["split_sizes"],
            name="split_sizes_concat",
            axis=0,
        ),
        helper.make_node(
            "Split",
            ["input", "split_sizes"],
            ["bbox_input", "cls_input"],
            name="split_input",
            axis=1,
        ),
        helper.make_node(
            "Shape",
            ["bbox_input"],
            ["bbox_shape"],
            name="bbox_shape",
        ),
        helper.make_node(
            "Gather",
            ["bbox_shape", "shape_idx_0"],
            ["batch_dim"],
            name="gather_batch_dim",
            axis=0,
        ),
        helper.make_node(
            "Gather",
            ["bbox_shape", "shape_idx_2"],
            ["spatial_dim"],
            name="gather_spatial_dim",
            axis=0,
        ),
        helper.make_node(
            "Unsqueeze",
            ["batch_dim", "axes0"],
            ["batch_dim_vec"],
            name="batch_dim_unsqueeze",
        ),
        helper.make_node(
            "Unsqueeze",
            ["four_scalar", "axes0"],
            ["four_vec"],
            name="four_unsqueeze",
        ),
        helper.make_node(
            "Unsqueeze",
            ["reg_max_scalar", "axes0"],
            ["reg_max_vec"],
            name="reg_max_unsqueeze",
        ),
        helper.make_node(
            "Unsqueeze",
            ["spatial_dim", "axes0"],
            ["spatial_dim_vec"],
            name="spatial_dim_unsqueeze",
        ),
        helper.make_node(
            "Unsqueeze",
            ["one_scalar", "axes0"],
            ["one_vec"],
            name="one_unsqueeze",
        ),
        helper.make_node(
            "Concat",
            ["batch_dim_vec", "four_vec", "reg_max_vec", "spatial_dim_vec"],
            ["bbox_reshape_shape"],
            name="bbox_reshape_shape_concat",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            ["bbox_input", "bbox_reshape_shape"],
            ["bbox_reshaped"],
            name="bbox_reshape",
        ),
        helper.make_node(
            "Transpose",
            ["bbox_reshaped"],
            ["bbox_transposed"],
            name="bbox_transpose",
            perm=[0, 2, 1, 3],
        ),
        helper.make_node(
            "Softmax",
            ["bbox_transposed"],
            ["bbox_softmax"],
            name="bbox_softmax",
            axis=1,
        ),
        helper.make_node(
            "Concat",
            ["one_vec", "reg_max_vec", "one_vec", "one_vec"],
            ["conv_weight_shape"],
            name="conv_weight_shape_concat",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            ["flat_weight", "conv_weight_shape"],
            ["conv_weight"],
            name="conv_weight_reshape",
        ),
        helper.make_node(
            "Conv",
            ["bbox_softmax", "conv_weight"],
            ["bbox_dfl"],
            name="bbox_dfl",
            pads=[0, 0, 0, 0],
        ),
        helper.make_node(
            "Concat",
            ["batch_dim_vec", "four_vec", "spatial_dim_vec"],
            ["bbox_output_shape"],
            name="bbox_output_shape_concat",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            ["bbox_dfl", "bbox_output_shape"],
            ["bbox_output"],
            name="bbox_output_reshape",
        ),
        helper.make_node(
            "Slice",
            [
                "bbox_output",
                "slice_0_starts",
                "slice_0_ends",
                "axes1",
                "slice_steps",
            ],
            ["bbox_slice_0"],
            name="bbox_slice_0",
        ),
        helper.make_node(
            "Slice",
            [
                "bbox_output",
                "slice_1_starts",
                "slice_1_ends",
                "axes1",
                "slice_steps",
            ],
            ["bbox_slice_1"],
            name="bbox_slice_1",
        ),
        helper.make_node(
            "Sub",
            ["sub_const", "bbox_slice_0"],
            ["bbox_sub"],
            name="bbox_sub",
        ),
        helper.make_node(
            "Add",
            ["add_const", "bbox_slice_1"],
            ["bbox_add"],
            name="bbox_add",
        ),
        helper.make_node(
            "Sub",
            ["bbox_add", "bbox_sub"],
            ["bbox_adjusted"],
            name="bbox_adjusted_sub",
        ),
        helper.make_node(
            "Add",
            ["bbox_sub", "bbox_add"],
            ["bbox_sum"],
            name="bbox_sum",
        ),
        helper.make_node(
            "Div",
            ["bbox_sum", "divisor"],
            ["bbox_div"],
            name="bbox_div",
        ),
        helper.make_node(
            "Concat",
            ["bbox_div", "bbox_adjusted"],
            ["bbox_concat"],
            name="bbox_concat",
            axis=1,
        ),
        helper.make_node(
            "Mul",
            ["bbox_concat", "mul_const"],
            ["bbox_scaled"],
            name="bbox_scaled",
        ),
        helper.make_node(
            "Sigmoid",
            ["cls_input"],
            ["cls_sigmoid"],
            name="cls_sigmoid",
        ),
        helper.make_node(
            "Concat",
            ["bbox_scaled", "cls_sigmoid"],
            ["output"],
            name="output_concat",
            axis=1,
        ),
    ]

    opset_import = [helper.make_opsetid("", opset_version)]

    function_proto = helper.make_function(
        YOLO_DECODE_DOMAIN,
        YOLO_DECODE_FUNCTION_NAME,
        ["input", "flat_weight", "sub_const", "add_const", "mul_const"],  # inputs
        ["output"],  # outputs
        nodes,
        opset_imports=opset_import,
        attributes=["num_class", "reg_max"]
    )

    return function_proto



def ensure_yolo_decode_function(onnx_model):
    has_yolo_decode = any(
        node.op_type == YOLO_DECODE_FUNCTION_NAME
        and node.domain == YOLO_DECODE_DOMAIN
        for node in onnx_model.graph.node
    )
    if not has_yolo_decode:
        return onnx_model

    if not any(opset.domain == YOLO_DECODE_DOMAIN for opset in onnx_model.opset_import):
        onnx_model.opset_import.append(
            helper.make_opsetid(YOLO_DECODE_DOMAIN, YOLO_DECODE_OPSET_VERSION)
        )

    existing_functions = {
        (function.domain, function.name) for function in onnx_model.functions
    }
    if (YOLO_DECODE_DOMAIN, YOLO_DECODE_FUNCTION_NAME) in existing_functions:
        return onnx_model

    opset_version = get_default_onnx_opset_version(
        onnx_model.opset_import, MIN_ONNX_OPSET_VERSION
    )
    onnx_model.functions.append(build_yolo_decode_function(opset_version))
    return onnx_model


class YoloDecodePatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input       input           0 2 sigmoid_0 reshape_0
            Sigmoid     sigmoid_0       1 1 input concat_1
            Reshape     reshape_0       1* 1 input transpose_0
            Transpose   transpose_0     1 1 reshape_0 softmax_0
            Softmax     softmax_0       1 1 transpose_0 conv_0
            Conv        conv_0          1* 1 softmax_0 reshape_1
            Reshape     reshape_1       1* 2 conv_0 slice_0 slice_1
            Slice       slice_0         1* 1 reshape_1 sub_0
            Slice       slice_1         1* 1 reshape_1 add_0
            Sub         sub_0           1* 2 slice_0 sub_1 add_1
            Add         add_0           1* 2 slice_1 sub_1 add_1
            Sub         sub_1           2 1 add_0 sub_0 concat_0
            Add         add_1           2 1 sub_0 add_0 div_0
            Div         div_0           1* 1 add_1 concat_0
            Concat      concat_0        2 1 div_0 sub_1 mul_0
            Mul         mul_0           1* 1 concat_0 concat_1
            Concat      concat_1        2 1 mul_0 sigmoid_0 output
            output      output          1 0 concat_1
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionYoloDecode"

    def rewrite(self, opset=24):
        return _rewrite_yolo_decode(self, opset)


class YoloDecodeSplitPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input       reshape_input   0 1 reshape_0
            input       sigmoid_input   0 1 sigmoid_0
            Sigmoid     sigmoid_0       1 1 sigmoid_input concat_1
            Reshape     reshape_0       1* 1 reshape_input transpose_0
            Transpose   transpose_0     1 1 reshape_0 softmax_0
            Softmax     softmax_0       1 1 transpose_0 conv_0
            Conv        conv_0          1* 1 softmax_0 reshape_1
            Reshape     reshape_1       1* 2 conv_0 slice_0 slice_1
            Slice       slice_0         1* 1 reshape_1 sub_0
            Slice       slice_1         1* 1 reshape_1 add_0
            Sub         sub_0           1* 2 slice_0 sub_1 add_1
            Add         add_0           1* 2 slice_1 sub_1 add_1
            Sub         sub_1           2 1 add_0 sub_0 concat_0
            Add         add_1           2 1 sub_0 add_0 div_0
            Div         div_0           1* 1 add_1 concat_0
            Concat      concat_0        2 1 div_0 sub_1 mul_0
            Mul         mul_0           1* 1 concat_0 concat_1
            Concat      concat_1        2 1 mul_0 sigmoid_0 output
            output      output          1 0 concat_1
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionYoloDecodeWithSplit"

    def rewrite(self, opset=24):
        return _rewrite_yolo_decode(self, opset)


def _rewrite_yolo_decode(matcher, opset):
    del opset

    sigmoid_0 = getattr(matcher, "sigmoid_0")
    reshape_0 = getattr(matcher, "reshape_0")
    transpose_0 = getattr(matcher, "transpose_0")
    softmax_0 = getattr(matcher, "softmax_0")
    conv_0 = getattr(matcher, "conv_0")
    slice_0 = getattr(matcher, "slice_0")
    slice_1 = getattr(matcher, "slice_1")
    sub_0 = getattr(matcher, "sub_0")
    add_0 = getattr(matcher, "add_0")
    div_0 = getattr(matcher, "div_0")
    concat_0 = getattr(matcher, "concat_0")
    concat_1 = getattr(matcher, "concat_1")
    mul_0 = getattr(matcher, "mul_0")

    input_variable = None
    split_0 = None
    if hasattr(matcher, "input"):
        input_variable = getattr(matcher, "input")
    else:
        reshape_input = getattr(matcher, "reshape_input")
        sigmoid_input = getattr(matcher, "sigmoid_input")

        if len(reshape_input.inputs) != 1 or len(sigmoid_input.inputs) != 1:
            return {}

        reshape_split = reshape_input.inputs[0]
        sigmoid_split = sigmoid_input.inputs[0]
        if reshape_split is not sigmoid_split:
            return {}
        if reshape_split.op != "Split":
            return {}
        if len(reshape_split.inputs) < 1:
            return {}

        input_variable = reshape_split.inputs[0]
        split_0 = reshape_split

    transpose_perm = _attr_as_list(transpose_0, "perm")
    softmax_axis = _attr_as_int(softmax_0, "axis")
    conv_pads = _attr_as_list(conv_0, "pads") or [0, 0, 0, 0]
    concat_0_axis = _attr_as_int(concat_0, "axis")
    concat_1_axis = _attr_as_int(concat_1, "axis")

    if transpose_perm != [0, 2, 1, 3]:
        return {}
    if softmax_axis != 1:
        return {}
    if conv_pads != [0, 0, 0, 0]:
        return {}
    if concat_0_axis != 1 or concat_1_axis != 1:
        return {}

    if len(conv_0.inputs) < 2:
        return {}

    conv_weight = conv_0.inputs[1]
    if not isinstance(conv_weight, osg.Constant) or conv_weight.values is None:
        return {}

    conv_weight_values = np.asarray(conv_weight.values)
    if conv_weight_values.shape != (1, 16, 1, 1):
        return {}

    slice_0_output = slice_0.outputs[0]
    slice_1_output = slice_1.outputs[0]

    sub_const = _find_constant_input(sub_0, slice_0_output)
    add_const = _find_constant_input(add_0, slice_1_output)
    mul_const = _find_constant_input(mul_0, concat_0.outputs[0])

    if sub_const is None or add_const is None or mul_const is None:
        return {}

    if sub_0.inputs[0] is not sub_const or sub_0.inputs[1] is not slice_0_output:
        return {}

    if add_0.inputs[0] is not add_const or add_0.inputs[1] is not slice_1_output:
        return {}

    if len(div_0.inputs) < 2:
        return {}

    div_constant = div_0.inputs[1]
    if not isinstance(div_constant, osg.Constant) or div_constant.values is None:
        return {}

    div_constant_values = np.asarray(div_constant.values)
    if div_constant_values.size != 1 or not np.allclose(div_constant_values.reshape(-1)[0], 2.0):
        return {}

    output_variable = concat_1.outputs[0]
    output_variable.inputs.clear()

    if split_0 is not None:
        _remove_output(input_variable, split_0)
    else:
        _remove_output(input_variable, sigmoid_0)
        _remove_output(input_variable, reshape_0)
    _remove_output(sub_const, sub_0)
    _remove_output(add_const, add_0)
    _remove_output(mul_const, mul_0)

    flat_weight = osg.Constant(
        name=f"{conv_weight.name}_flat",
        values=conv_weight_values.reshape(-1).copy(),
    )

    num_class, reg_max = _derive_yolo_decode_attrs(
        input_variable, flat_weight.values.size, split_0
    )
    if num_class is None or reg_max is None:
        return {}
    num_class = int(num_class)
    reg_max = int(reg_max)

    yolo_decode_name = f"yolo_decode_{concat_1.name or 'concat_1'}"
    return {
        yolo_decode_name: {
            "op": YOLO_DECODE_FUNCTION_NAME,
            "inputs": [
                input_variable,
                flat_weight,
                sub_const,
                add_const,
                mul_const,
            ],
            "outputs": [output_variable],
            "domain": YOLO_DECODE_DOMAIN,
            "name": yolo_decode_name,
            "attrs": {
                "num_class": num_class,
                "reg_max": reg_max,
            },
        }
    }


register_fusion_pattern(YoloDecodePatternMatcher(1))
register_fusion_pattern(YoloDecodeSplitPatternMatcher(1))
