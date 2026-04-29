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
from onnxruntime.tools.onnx_model_utils import (get_optimization_level,
                                                optimize_model)

from xslim.defs import MIN_ONNX_OPSET_VERSION
from xslim.logger import logger

from .onnxslim_pass import infer_onnx_model, optimize_onnx_model


def get_onnx_opset(onnx_model: onnx.ModelProto) -> Dict[str, int]:
    opset_dict = {}
    for opset in onnx_model.opset_import:
        _domain = opset.domain
        _domain = "ai.onnx" if _domain == "" else _domain
        opset_dict[_domain] = opset.version

    return opset_dict


def ensure_default_onnx_opset(onnx_model: onnx.ModelProto, min_onnx_version: int = MIN_ONNX_OPSET_VERSION) -> int:
    ai_onnx_version = None
    for opset in onnx_model.opset_import:
        if opset.domain in {"", "ai.onnx"}:
            ai_onnx_version = opset.version
            break

    if ai_onnx_version is None:
        logger.warning(
            f"Missing default ONNX opset import, defaulting to {min_onnx_version}.")
        opset = onnx_model.opset_import.add()
        opset.domain = ""
        opset.version = min_onnx_version
        ai_onnx_version = min_onnx_version
    return ai_onnx_version


def _normalize_kernel_shape_attrs(
    onnx_model: onnx.ModelProto,
) -> onnx.ModelProto:
    try:
        osg_graph = osg.import_onnx(onnx_model)
        updated = False

        def _get_tensor_shape(var: osg.Tensor):
            shape = None
            if isinstance(var, osg.Constant):
                values = getattr(var, "values", None)
                if values is not None and hasattr(values, "shape"):
                    shape = list(values.shape)

            if shape is None:
                inferred_shape = getattr(var, "shape", None)
                if inferred_shape is not None:
                    shape = list(inferred_shape)

            return shape

        def _to_int_list(values):
            try:
                return [int(value) for value in values]
            except (TypeError, ValueError):
                return None

        def _normalize_attr_list(
            node: osg.Node,
            attr_name: str,
            expected_len: int,
            default_values: List[int],
            min_value: int,
        ) -> bool:
            current_values = node.attrs.get(attr_name)
            if current_values is None:
                return False

            normalized_values = _to_int_list(current_values)
            if (
                normalized_values is None
                or len(normalized_values) != expected_len
                or any(value < min_value for value in normalized_values)
            ):
                node.attrs[attr_name] = default_values
                logger.warning(
                    (
                        f"Normalizing {node.op} node {node.name or node.op} "
                        f"attr {attr_name} from {current_values} "
                        f"to {default_values}."
                    )
                )
                return True

            if (
                not isinstance(current_values, list)
                or current_values != normalized_values
            ):
                node.attrs[attr_name] = normalized_values
                return True

            return False

        for node in osg_graph.nodes:
            if node.op not in {"Conv", "ConvTranspose"}:
                continue
            if len(node.inputs) < 2:
                continue

            weight_var = node.inputs[1]
            weight_shape = _get_tensor_shape(weight_var)

            if weight_shape is None or len(weight_shape) < 3:
                logger.warning(
                    (
                        "skip filling kernel_shape for %s because "
                        "weight shape is unavailable"
                    ),
                    node.name or node.op,
                )
                continue

            kernel_shape = weight_shape[2:]
            if any(dim is None or isinstance(dim, str) for dim in kernel_shape):
                logger.warning(
                    (
                        "skip filling kernel_shape for %s because "
                        "weight shape %s is not static"
                    ),
                    node.name or node.op,
                    weight_shape,
                )
                continue

            node.attrs["kernel_shape"] = [int(dim) for dim in kernel_shape]
            updated = True

            spatial_rank = len(kernel_shape)
            updated = _normalize_attr_list(
                node,
                "strides",
                spatial_rank,
                [1] * spatial_rank,
                1,
            ) or updated
            updated = _normalize_attr_list(
                node,
                "dilations",
                spatial_rank,
                [1] * spatial_rank,
                1,
            ) or updated
            updated = _normalize_attr_list(
                node,
                "pads",
                spatial_rank * 2,
                [0] * (spatial_rank * 2),
                0,
            ) or updated

            group = node.attrs.get("group")
            if group is not None:
                try:
                    normalized_group = int(str(group))
                except (TypeError, ValueError):
                    normalized_group = 1

                if normalized_group < 1:
                    normalized_group = 1

                if group != normalized_group:
                    node.attrs["group"] = normalized_group
                    logger.warning(
                        (
                            f"Normalizing {node.op} node "
                            f"{node.name or node.op} attr group "
                            f"from {group} to {normalized_group}."
                        )
                    )
                    updated = True

            if node.op == "ConvTranspose":
                updated = _normalize_attr_list(
                    node,
                    "output_padding",
                    spatial_rank,
                    [0] * spatial_rank,
                    0,
                ) or updated

                output_shape = node.attrs.get("output_shape")
                if output_shape is not None:
                    normalized_output_shape = _to_int_list(output_shape)
                    if (
                        normalized_output_shape is None
                        or len(normalized_output_shape) != spatial_rank
                        or any(value < 1 for value in normalized_output_shape)
                    ):
                        del node.attrs["output_shape"]
                        logger.warning(
                            (
                                "Dropping invalid ConvTranspose "
                                f"output_shape {output_shape} from "
                                f"node {node.name or node.op}."
                            )
                        )
                        updated = True
                    elif (
                        not isinstance(output_shape, list)
                        or output_shape != normalized_output_shape
                    ):
                        node.attrs["output_shape"] = normalized_output_shape
                        updated = True

        if not updated:
            return onnx_model

        return osg.export_onnx(osg_graph)
    except Exception as exc:
        logger.warning(
            "Failed to normalize kernel_shape attributes via GraphSurgeon; "
            "returning original ONNX model. Error: %s",
            exc,
        )
        return onnx_model


def _deduplicate_node_names(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    used_names = set()
    next_suffix = {}

    for index, node in enumerate(onnx_model.graph.node):
        base_name = node.name or f"{node.op_type}_{index}"
        new_name = base_name

        if new_name in used_names:
            suffix = next_suffix.get(base_name, 1)
            while f"{base_name}_{suffix}" in used_names:
                suffix += 1
            new_name = f"{base_name}_{suffix}"
            next_suffix[base_name] = suffix + 1
            logger.warning(
                f"Renaming duplicated op name {base_name} to {new_name}."
            )
        else:
            next_suffix.setdefault(base_name, 1)

        if node.name != new_name:
            node.name = new_name
        used_names.add(new_name)

    return onnx_model


def format_onnx_model(
    onnx_model: onnx.ModelProto, sim_en: bool = True, min_onnx_version: Optional[int] = None
) -> onnx.ModelProto:
    """
    Regularize an onnx model, including removing shape fields, value_info fields, etc., to avoid entering bugs.

    Args:
        onnx_model (onnx.ModelProto): input onnx Model
        min_onnx_version (int, optional): target default ai.onnx opset version.

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

    onnx_model = _deduplicate_node_names(onnx_model)
    onnx_model = _normalize_kernel_shape_attrs(onnx_model)

    target_onnx_version = min_onnx_version
    required_onnx_version = MIN_ONNX_OPSET_VERSION if target_onnx_version is None else target_onnx_version

    ai_onnx_version = ensure_default_onnx_opset(onnx_model, required_onnx_version)
    if target_onnx_version is None and ai_onnx_version < required_onnx_version:
        logger.warning("convert ai.onnx version {} to {}...".format(
            ai_onnx_version, required_onnx_version))
        onnx_model = onnx.version_converter.convert_version(
            onnx_model, required_onnx_version)
    elif target_onnx_version is not None and ai_onnx_version != target_onnx_version:
        logger.warning("convert ai.onnx version {} to {}...".format(
            ai_onnx_version, target_onnx_version))
        onnx_model = onnx.version_converter.convert_version(
            onnx_model, target_onnx_version)

    if sim_en:
        logger.info("simplify onnx model...")
    try:
        onnx_model = optimize_onnx_model(onnx_model)
    except Exception as e:
        logger.warning("simplify onnx model error and skip. {}".format(e))

    try:
        onnx_model = infer_onnx_model(onnx_model)
    except Exception as e:
        logger.warning("shape_inference error with {}, skiped".format(e))

    onnx_model = _deduplicate_node_names(onnx_model)
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
            inputs=truncate_left_graph.inputs,
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
            raise RuntimeError(
                "The incoming truncate_var_names contains duplicate tensor names")
        truncate_vars = []
        osg_graph = osg.import_onnx(onnx_model)
        tensors = osg_graph.tensors()
        for k, v in tensors.items():
            if k in set(truncate_var_names):
                truncate_vars.append(v)

        graph_valid_truncate_var_names = set([t.name for t in truncate_vars])

        invalid_var_names = set(truncate_var_names) ^ set(
            graph_valid_truncate_var_names)
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
                raise RuntimeError(
                    "unexpected error for node {}".format(node.name))

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
            inputs=osg_graph.inputs,
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
