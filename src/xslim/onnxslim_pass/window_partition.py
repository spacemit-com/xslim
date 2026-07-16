import numpy as np
import onnx
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnx import helper

from xslim.logger import logger


WINDOW_PARTITION_DOMAIN = "spacemit_functions"
WINDOW_PARTITION_OPSET_VERSION = 1
WINDOW_PARTITION_OP_TYPE = "WindowPartition"


def _const_int_list(tensor):
    if not isinstance(tensor, osg.Constant) or tensor.values is None:
        return None
    return [int(v) for v in np.asarray(tensor.values).reshape(-1).tolist()]


def _attr_as_list(node, attr_name):
    value = node.attrs.get(attr_name)
    if value is None:
        return None
    return np.asarray(value).reshape(-1).tolist()


def build_window_partition_function(opset_version):
    nodes = [
        helper.make_node(
            "Reshape",
            ["x", "shape_before_transpose"],
            ["reshape0_out"],
            name="window_reshape_before_transpose",
        ),
        helper.make_node(
            "Transpose",
            ["reshape0_out"],
            ["transpose_out"],
            name="window_transpose",
            perm=[0, 1, 3, 2, 4, 5],
        ),
        helper.make_node(
            "Reshape",
            ["transpose_out", "shape_after_transpose"],
            ["output"],
            name="window_reshape_after_transpose",
        ),
    ]

    return helper.make_function(
        WINDOW_PARTITION_DOMAIN,
        WINDOW_PARTITION_OP_TYPE,
        ["x", "shape_before_transpose", "shape_after_transpose"],
        ["output"],
        nodes,
        opset_imports=[helper.make_opsetid("", opset_version)],
        attributes=["window_size", "perm"],
    )


def ensure_window_partition_function(onnx_model):
    has_window_partition = any(
        node.op_type == WINDOW_PARTITION_OP_TYPE
        and node.domain == WINDOW_PARTITION_DOMAIN
        for node in onnx_model.graph.node
    )
    if not has_window_partition:
        return onnx_model

    if not any(
        opset.domain == WINDOW_PARTITION_DOMAIN for opset in onnx_model.opset_import
    ):
        onnx_model.opset_import.append(
            helper.make_opsetid(
                WINDOW_PARTITION_DOMAIN, WINDOW_PARTITION_OPSET_VERSION
            )
        )

    existing_functions = {
        (function.domain, function.name) for function in onnx_model.functions
    }
    if (WINDOW_PARTITION_DOMAIN, WINDOW_PARTITION_OP_TYPE) in existing_functions:
        return onnx_model

    default_opset = next(
        (opset.version for opset in onnx_model.opset_import if opset.domain == ""),
        13,
    )
    onnx_model.functions.append(build_window_partition_function(default_opset))
    return onnx_model


def fuse_window_partition(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    graph = osg.import_onnx(onnx_model)
    updated = False

    for reshape0 in list(graph.nodes):
        if reshape0.op != "Reshape" or len(reshape0.inputs) != 2 or len(reshape0.outputs) != 1:
            continue

        reshape0_out = reshape0.outputs[0]
        if len(reshape0_out.outputs) != 1:
            continue
        transpose = reshape0_out.outputs[0]
        if transpose.op != "Transpose" or len(transpose.inputs) != 1 or len(transpose.outputs) != 1:
            continue

        transpose_perm = _attr_as_list(transpose, "perm")
        if transpose_perm != [0, 1, 3, 2, 4, 5]:
            continue

        transpose_out = transpose.outputs[0]
        if len(transpose_out.outputs) != 1:
            continue
        reshape1 = transpose_out.outputs[0]
        if reshape1.op != "Reshape" or len(reshape1.inputs) != 2 or len(reshape1.outputs) != 1:
            continue

        first_shape = _const_int_list(reshape0.inputs[1])
        second_shape = _const_int_list(reshape1.inputs[1])
        input_shape = getattr(reshape0.inputs[0], "shape", None)
        reshape0_shape = getattr(reshape0_out, "shape", None)
        reshape1_shape = getattr(reshape1.outputs[0], "shape", None)

        if first_shape is None or second_shape is None:
            continue
        if len(first_shape) != 6 or len(second_shape) != 2:
            continue
        if input_shape is None or len(input_shape) != 4:
            continue
        if reshape0_shape is None or len(reshape0_shape) != 6:
            continue
        if reshape1_shape is None or len(reshape1_shape) != 2:
            continue

        batch_dim, height_dim, width_dim, channel_dim = input_shape
        reshape0_batch, h_tiles_shape, window_h, w_tiles_shape, window_w, reshape0_channel = first_shape
        reshape0_actual = list(reshape0_shape)
        if reshape0_batch != batch_dim or reshape0_channel != channel_dim:
            continue
        if second_shape[0] != -1 or second_shape[1] != channel_dim:
            continue

        h_tiles, w_tiles = reshape0_actual[1], reshape0_actual[3]
        if h_tiles_shape not in (-1, h_tiles):
            continue
        if w_tiles_shape not in (-1, w_tiles):
            continue
        if window_h != window_w:
            continue
        if not all(isinstance(v, (int, np.integer)) for v in [height_dim, width_dim, channel_dim, h_tiles, window_h, w_tiles, window_w]):
            continue
        if int(height_dim) != int(h_tiles) * int(window_h):
            continue
        if int(width_dim) != int(w_tiles) * int(window_w):
            continue

        expected_window_tokens = (
            int(batch_dim) * int(h_tiles) * int(w_tiles) * int(window_h) * int(window_w)
        )
        if list(reshape1_shape) != [expected_window_tokens, int(channel_dim)]:
            continue

        reshape0_data_input = reshape0.inputs[0]
        reshape0_shape_input = reshape0.inputs[1]
        reshape1_shape_input = reshape1.inputs[1]
        fused_output = reshape1.outputs[0]
        fused_output.inputs.clear()
        input_var = reshape0_data_input
        if reshape0 in input_var.outputs:
            input_var.outputs.remove(reshape0)

        fused_name = (
            f"{WINDOW_PARTITION_OP_TYPE}_{reshape0.name}"
            if reshape0.name
            else WINDOW_PARTITION_OP_TYPE
        )

        fused_node = osg.Node(
            op=WINDOW_PARTITION_OP_TYPE,
            name= fused_name, 
            domain=WINDOW_PARTITION_DOMAIN,
            attrs={
                "window_size": int(window_h),
                "perm": transpose_perm,
            },
            inputs=[reshape0_data_input, reshape0_shape_input, reshape1_shape_input],
            outputs=[fused_output],
        )
        fused_output.inputs = [fused_node]
        graph.nodes.append(fused_node)

        reshape0.inputs.clear()
        reshape0.outputs.clear()
        transpose.inputs.clear()
        transpose.outputs.clear()
        reshape1.inputs.clear()
        reshape1.outputs.clear()
        updated = True

    if not updated:
        return onnx_model

    graph.cleanup().toposort()
    fused_model = osg.export_onnx(graph)
    return ensure_window_partition_function(fused_model)
