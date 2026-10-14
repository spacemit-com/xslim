import numpy as np
import onnx
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnx import helper

from xslim.logger import logger


PATCH_EMBEDDING_DOMAIN = "spacemit_functions"
PATCH_EMBEDDING_OPSET_VERSION = 1
PATCH_EMBEDDING_OP_TYPE = "PatchEmbedding"


def _attr_as_list(node, attr_name):
    value = node.attrs.get(attr_name)
    if value is None:
        return None
    return np.asarray(value).reshape(-1).tolist()


def _const_int_list(tensor):
    if not isinstance(tensor, osg.Constant) or tensor.values is None:
        return None
    return [int(v) for v in np.asarray(tensor.values).reshape(-1).tolist()]


def _int_initializer_map(onnx_model):
    init_map = {}
    for initializer in onnx_model.graph.initializer:
        try:
            init_map[initializer.name] = [
                int(v) for v in np.asarray(onnx.numpy_helper.to_array(initializer)).reshape(-1).tolist()
            ]
        except Exception:
            continue
    return init_map


def build_patch_embedding_function(opset_version):
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "weight", "bias"],
            ["conv_out"],
            name="patch_conv",
            kernel_shape=[16, 16],
            strides=[16, 16],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        ),
        helper.make_node(
            "Reshape",
            ["conv_out", "reshape_shape"],
            ["reshape_out"],
            name="patch_reshape",
        ),
        helper.make_node(
            "Transpose",
            ["reshape_out"],
            ["output"],
            name="patch_transpose",
            perm=[0, 2, 1],
        ),
    ]

    return helper.make_function(
        PATCH_EMBEDDING_DOMAIN,
        PATCH_EMBEDDING_OP_TYPE,
        ["input", "weight", "bias", "reshape_shape"],
        ["output"],
        nodes,
        opset_imports=[helper.make_opsetid("", opset_version)],
        attributes=[
            "kernel_shape",
            "strides",
            "pads",
            "dilations",
            "group",
            "perm",
            "embed_dim",
            "in_chans",
            "has_bias",
        ],
    )


def ensure_patch_embedding_function(onnx_model):
    has_patch_embedding = any(
        node.op_type == PATCH_EMBEDDING_OP_TYPE
        and node.domain == PATCH_EMBEDDING_DOMAIN
        for node in onnx_model.graph.node
    )
    if not has_patch_embedding:
        return onnx_model

    if not any(opset.domain == PATCH_EMBEDDING_DOMAIN for opset in onnx_model.opset_import):
        onnx_model.opset_import.append(
            helper.make_opsetid(PATCH_EMBEDDING_DOMAIN, PATCH_EMBEDDING_OPSET_VERSION)
        )

    existing_functions = {
        (function.domain, function.name) for function in onnx_model.functions
    }
    if (PATCH_EMBEDDING_DOMAIN, PATCH_EMBEDDING_OP_TYPE) in existing_functions:
        return onnx_model

    default_opset = next(
        (opset.version for opset in onnx_model.opset_import if opset.domain == ""),
        13,
    )
    onnx_model.functions.append(build_patch_embedding_function(default_opset))
    return onnx_model


def fuse_patch_embedding(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    initializer_map = _int_initializer_map(onnx_model)
    graph = osg.import_onnx(onnx_model)
    updated = False

    for conv_node in list(graph.nodes):
        if conv_node.op != "Conv" or len(conv_node.outputs) != 1:
            continue

        conv_out = conv_node.outputs[0]
        if len(conv_out.outputs) != 1:
            continue
        reshape_node = conv_out.outputs[0]
        if reshape_node.op != "Reshape" or len(reshape_node.inputs) != 2 or len(reshape_node.outputs) != 1:
            continue

        reshape_out = reshape_node.outputs[0]
        if len(reshape_out.outputs) != 1:
            continue
        transpose_node = reshape_out.outputs[0]
        if transpose_node.op != "Transpose" or len(transpose_node.outputs) != 1:
            continue

        perm = _attr_as_list(transpose_node, "perm")
        if perm != [0, 2, 1]:
            continue

        if len(conv_node.inputs) < 2:
            continue
        weight = conv_node.inputs[1]
        if not isinstance(weight, osg.Constant) or weight.values is None:
            continue
        weight_shape = list(weight.values.shape)
        if len(weight_shape) != 4:
            continue

        kernel_shape = _attr_as_list(conv_node, "kernel_shape")
        strides = _attr_as_list(conv_node, "strides")
        dilations = _attr_as_list(conv_node, "dilations")
        pads = _attr_as_list(conv_node, "pads")
        group = int(np.asarray(conv_node.attrs.get("group", 1)).reshape(-1)[0])
        reshape_shape = _const_int_list(reshape_node.inputs[1])
        if reshape_shape is None:
            reshape_shape = initializer_map.get(reshape_node.inputs[1].name)

        expected_kernel = [16, 16]
        expected_embed_dim = int(weight_shape[0])
        expected_in_chans = int(weight_shape[1])
        if weight_shape[2:] != expected_kernel:
            continue
        if expected_in_chans != 3:
            continue
        if kernel_shape != expected_kernel:
            continue
        if strides != expected_kernel:
            continue
        if dilations not in (None, [1, 1]):
            continue
        if pads not in (None, [0, 0, 0, 0]):
            continue
        if group != 1:
            continue

        conv_out_shape = getattr(conv_out, "shape", None)
        reshape_out_shape = getattr(reshape_out, "shape", None)
        transpose_out_shape = getattr(transpose_node.outputs[0], "shape", None)
        if conv_out_shape is None or len(conv_out_shape) != 4:
            continue
        if reshape_out_shape is None or len(reshape_out_shape) != 3:
            continue
        if transpose_out_shape is None or len(transpose_out_shape) != 3:
            continue

        batch_dim, channel_dim, h_dim, w_dim = list(conv_out_shape)
        reshape_batch_dim, reshape_channel_dim, reshape_tokens_dim = list(reshape_out_shape)
        transpose_batch_dim, transpose_tokens_dim, transpose_channel_dim = list(
            transpose_out_shape
        )

        if channel_dim != expected_embed_dim:
            continue
        if reshape_channel_dim != expected_embed_dim:
            continue
        if transpose_channel_dim != expected_embed_dim:
            continue
        if transpose_batch_dim != reshape_batch_dim:
            continue
        if transpose_tokens_dim != reshape_tokens_dim:
            continue

        expected_num_patches = None
        if isinstance(h_dim, (int, np.integer)) and isinstance(w_dim, (int, np.integer)):
            expected_num_patches = int(h_dim) * int(w_dim)
            if isinstance(reshape_tokens_dim, (int, np.integer)) and int(reshape_tokens_dim) != expected_num_patches:
                continue

        if reshape_shape is None or len(reshape_shape) != 3:
            continue
        if reshape_shape[1] != expected_embed_dim:
            continue
        if reshape_shape[0] not in (0, 1) and reshape_shape[0] != batch_dim:
            continue
        if reshape_shape[2] != -1:
            if expected_num_patches is None:
                continue
            if reshape_shape[2] != expected_num_patches:
                continue

        patch_embedding_inputs = [conv_node.inputs[0], conv_node.inputs[1]]
        if len(conv_node.inputs) > 2:
            patch_embedding_inputs.append(conv_node.inputs[2])
        else:
            patch_embedding_inputs.append(
                osg.Constant(
                    name=f"{conv_node.name or PATCH_EMBEDDING_OP_TYPE}_bias",
                    values=np.zeros((expected_embed_dim,), dtype=weight.values.dtype),
                )
            )
        patch_embedding_inputs.append(reshape_node.inputs[1])

        patch_embedding_node = osg.Node(
            op=PATCH_EMBEDDING_OP_TYPE,
            name=conv_node.name or transpose_node.name or PATCH_EMBEDDING_OP_TYPE,
            domain=PATCH_EMBEDDING_DOMAIN,
            attrs={
                "kernel_shape": expected_kernel,
                "strides": strides,
                "pads": pads or [0, 0, 0, 0],
                "dilations": dilations or [1, 1],
                "group": group,
                "perm": perm,
                "embed_dim": expected_embed_dim,
                "in_chans": expected_in_chans,
                "has_bias": len(conv_node.inputs) > 2,
            },
            inputs=patch_embedding_inputs,
            outputs=list(transpose_node.outputs),
        )

        input_var = conv_node.inputs[0]
        if conv_node in input_var.outputs:
            input_var.outputs.remove(conv_node)
        transpose_out = transpose_node.outputs[0]
        transpose_out.inputs = [patch_embedding_node]
        graph.nodes.append(patch_embedding_node)
        conv_node.outputs.clear()
        reshape_node.inputs.clear()
        reshape_node.outputs.clear()
        transpose_node.inputs.clear()
        transpose_node.outputs.clear()
        updated = True
        logger.info(
            f"Fused patch embedding pattern "
            f"{conv_node.name or 'Conv'} -> "
            f"{reshape_node.name or 'Reshape'} -> "
            f"{transpose_node.name or 'Transpose'} into "
            f"{PATCH_EMBEDDING_OP_TYPE}({patch_embedding_node.name})"
        )

    if not updated:
        return onnx_model

    graph.cleanup().toposort()
    fused_model = osg.export_onnx(graph)
    return ensure_patch_embedding_function(fused_model)
