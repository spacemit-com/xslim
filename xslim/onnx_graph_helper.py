#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import copy
import os
import pathlib
from collections import OrderedDict, deque
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence, Union

import onnx
import onnx_graphsurgeon as osg
from onnxruntime.tools.onnx_model_utils import get_optimization_level, optimize_model

from xslim.defs import MIN_ONNX_OPSET_VERSION
from xslim.logger import logger

from .onnxslim_pass import optimize_onnx_model


def get_onnx_opset(onnx_model: onnx.ModelProto) -> Dict[str, int]:
    opset_dict = {}
    for opset in onnx_model.opset_import:
        _domain = opset.domain
        _domain = "ai.onnx" if _domain == "" else _domain
        opset_dict[_domain] = opset.version

    return opset_dict


def format_onnx_model(
    onnx_model: onnx.ModelProto, sim_en: bool = True, min_onnx_version: int = MIN_ONNX_OPSET_VERSION
) -> onnx.ModelProto:
    """
    Regularize an onnx model, including removing shape fields, value_info fields, etc., to avoid entering bugs.

    Args:
        onnx_model (onnx.ModelProto): input onnx Model
        min_onnx_version (int, optional): min onnx opset version. Defaults to 13.

    Returns:
        onnx.ModelProto: output ONNX Model
    """
    onnx_model.graph.ClearField("value_info")
    for o_var in onnx_model.graph.output:
        try:
            for dim in o_var.type.tensor_type.shape.dim:
                dim.dim_value = 0
                dim.dim_param = "?"
        except:
            pass

    empty_name_count = 0
    for node in onnx_model.graph.node:
        if node.name == "":
            node.name = f"{node.op_type}_{empty_name_count}"
            empty_name_count += 1

    opset_dict = get_onnx_opset(onnx_model)
    ai_onnx_version = opset_dict.get("ai.onnx", min_onnx_version)
    if ai_onnx_version < min_onnx_version:
        logger.warning("convert ai.onnx version {} to {}...".format(ai_onnx_version, min_onnx_version))
        onnx_model = onnx.version_converter.convert_version(onnx_model, min_onnx_version)

    if sim_en:
        logger.info("simplify onnx model...")
    try:
        onnx_model = optimize_onnx_model(onnx_model)
    except Exception as e:
        logger.warning("simplify onnx model error and skip. {}".format(e))

    # try:
    #     logger.info("symplify onnx model with onnxruntime...")
    #     with TemporaryDirectory() as tempdir:
    #         src_path = os.path.join(tempdir, "onnx_model_src.onnx")
    #         dst_path = os.path.join(tempdir, "onnx_model_dst.onnx")
    #         onnx.save(onnx_model, src_path)
    #         optimize_model(pathlib.Path(src_path), pathlib.Path(dst_path), get_optimization_level("basic"))
    #         onnx_model = onnx.load(dst_path)
    # except Exception as e:
    #     logger.warning("simplify onnx model with onnxruntime error and skip. {}".format(e))

    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)
    except Exception as e:
        logger.warning("shape_inference error with {}, skiped".format(e))

    return onnx_model


def merge_onnx_model(
    onnx_model: onnx.ModelProto,
    truncate_left_graph: Optional[osg.Graph] = None,
    truncate_vars: Optional[Sequence[osg.Variable]] = None,
):
    if isinstance(truncate_left_graph, osg.Graph) and isinstance(truncate_vars, Sequence):
        osg_graph = osg.import_onnx(onnx_model)
        for idx, o_var in enumerate(osg_graph.outputs):
            o_idx = o_var.inputs[0].outputs.index(o_var)
            o_var.inputs[0].outputs[o_idx] = truncate_vars[idx]

        new_osg_graph = osg.Graph(
            nodes=osg_graph.nodes + truncate_left_graph.nodes,
            inputs=osg_graph.inputs,
            outputs=truncate_left_graph.outputs,
            name=copy.copy(osg_graph.name),
            doc_string=copy.copy(osg_graph.doc_string),
            opset=copy.copy(osg_graph.opset),
            import_domains=osg_graph.import_domains,
        )
        onnx_model = osg.export_onnx(new_osg_graph)

    return onnx_model


def truncate_onnx_model(onnx_model: onnx.ModelProto, truncate_var_names: Optional[Sequence[str]] = None):
    if isinstance(truncate_var_names, Sequence) and len(truncate_var_names) > 0:
        if len(set(truncate_var_names)) != len(truncate_var_names):
            raise RuntimeError("The incoming truncate_var_names contains duplicate tensor names")
        truncate_vars = []
        osg_graph = osg.import_onnx(onnx_model)
        tensors = osg_graph.tensors()
        for k, v in tensors.items():
            if k in set(truncate_var_names):
                truncate_vars.append(v)

        graph_valid_truncate_var_names = set([t.name for t in truncate_vars])

        invalid_var_names = set(truncate_var_names) ^ set(graph_valid_truncate_var_names)
        if len(invalid_var_names) > 0:
            raise RuntimeError(
                "The incoming truncate_var_names contains non-existent tensor names {}".format(
                    ", ".join(invalid_var_names)
                )
            )

        valid_node_names = set()
        invalid_node_names = set()
        graph_node_names = set()
        dst_node_dict = {}
        src_node_dict = {}
        for i, node in enumerate(osg_graph.nodes):
            if node.name == "":
                node.name = "{}_{}_{}".format(node.op, i, id(node))
            if node.name in graph_node_names:
                node.name = "{}_{}_{}".format(node.name, i, id(node))
            graph_node_names.add(node.name)
            dst_node_dict[node.name] = set()
            src_node_dict[node.name] = set()
            for var in node.outputs:
                for dst_node in var.outputs:
                    dst_node_dict[node.name].add(dst_node.name)
            for var in node.inputs:
                for src_node in var.inputs:
                    src_node_dict[node.name].add(src_node.name)

        def _truncate_graph_upstream(out_vars: Sequence[osg.Tensor]):
            visit_ops = deque()

            def _upstream_impl(vars: Sequence[osg.Tensor]):
                for var in vars:
                    for source_op in var.inputs:
                        if source_op.name in valid_node_names or source_op.name not in graph_node_names:
                            continue
                        valid_node_names.add(source_op.name)
                        visit_ops.append(source_op)

            _upstream_impl(out_vars)
            while len(visit_ops) > 0:
                dq_size = len(visit_ops)
                for _ in range(dq_size):
                    up_op = visit_ops.popleft()
                    _upstream_impl(up_op.inputs)

        def _truncate_graph_downstream(out_vars: Sequence[osg.Tensor]):
            visit_ops = deque()

            def _upstream_impl(vars: Sequence[osg.Tensor]):
                for var in vars:
                    for source_op in var.inputs:
                        if (
                            source_op.name in invalid_node_names
                            or source_op.name in valid_node_names
                            or source_op.name not in graph_node_names
                        ):
                            continue
                        invalid_node_names.add(source_op.name)
                        visit_ops.append(source_op)

            def _downstream_impl(vars: Sequence[osg.Tensor]):
                for var in vars:
                    for dest_op in var.outputs:
                        if dest_op.name in invalid_node_names or dest_op.name not in graph_node_names:
                            continue
                        invalid_node_names.add(dest_op.name)
                        visit_ops.append(dest_op)

            _downstream_impl(out_vars)
            while len(visit_ops) > 0:
                dq_size = len(visit_ops)
                for _ in range(dq_size):
                    up_op = visit_ops.popleft()
                    _downstream_impl(up_op.outputs)
                    _upstream_impl(up_op.inputs)

        _truncate_graph_upstream(truncate_vars)
        _truncate_graph_downstream(truncate_vars)

        if len(invalid_node_names) + len(valid_node_names) != len(graph_node_names):
            raise RuntimeError("truncate graph failed.")

        valid_nodes = []
        invalid_nodes = []

        for node in osg_graph.nodes:
            if node.name in valid_node_names:
                valid_nodes.append(node)
            elif node.name in invalid_node_names:
                invalid_nodes.append(node)
            else:
                raise RuntimeError("unexpected error for node {}".format(node.name))

        truncate_graph = osg.Graph(
            nodes=valid_nodes,
            inputs=osg_graph.inputs,
            outputs=truncate_vars,
            name=copy.copy(osg_graph.name),
            doc_string=copy.copy(osg_graph.doc_string),
            opset=copy.copy(osg_graph.opset),
            import_domains=osg_graph.import_domains,
        )

        truncate_left_graph = osg.Graph(
            nodes=invalid_nodes,
            inputs=[],
            outputs=osg_graph.outputs,
            name=copy.copy(osg_graph.name),
            doc_string=copy.copy(osg_graph.doc_string),
            opset=copy.copy(osg_graph.opset),
            import_domains=osg_graph.import_domains,
        )

        truncate_onnx_model = osg.export_onnx(truncate_graph)

        for var in truncate_vars:
            var.inputs.clear()

        return truncate_onnx_model, truncate_left_graph, truncate_vars
    else:
        return onnx_model, None, None
