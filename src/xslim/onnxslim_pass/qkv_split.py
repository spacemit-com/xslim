import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


def _get_constant_scalar_value(tensor):
    if not isinstance(tensor, osg.Constant) or tensor.values is None:
        return None

    values = np.asarray(tensor.values)
    if values.size != 1:
        return None

    return int(values.reshape(-1)[0])


def _is_supported_qkv_weight(weight_values):
    if weight_values is None:
        return False

    weight_array = np.asarray(weight_values)
    return weight_array.ndim == 2 and weight_array.shape[1] % 3 == 0


def _split_qkv_weight(weight_values):
    weight_array = np.asarray(weight_values)
    qkv_num = weight_array.shape[1] // 3
    reshaped_weight = weight_array.reshape(weight_array.shape[0], qkv_num, 3)
    return [
        np.ascontiguousarray(reshaped_weight[:, :, index])
        for index in range(3)
    ]


def _split_qkv_bias(bias_values):
    bias_array = np.asarray(bias_values)
    qkv_num = bias_array.shape[-1] // 3
    reshaped_bias = bias_array.reshape(qkv_num, 3)
    return [
        np.ascontiguousarray(reshaped_bias[:, index])
        for index in range(3)
    ]


def _get_add_constant_and_tensor(add_node):
    if len(add_node.inputs) != 2:
        return None, None

    lhs_input, rhs_input = add_node.inputs
    lhs_is_const = isinstance(lhs_input, osg.Constant) and lhs_input.values is not None
    rhs_is_const = isinstance(rhs_input, osg.Constant) and rhs_input.values is not None

    if lhs_is_const and rhs_is_const:
        return lhs_input, rhs_input
    if lhs_is_const:
        return lhs_input, rhs_input
    if rhs_is_const:
        return rhs_input, lhs_input
    return None, None


class MatMulQKVSplitPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input   input      0 1 matmul_0
            MatMul  matmul_0   2 1 input ? add_0
            Add     add_0      2 1 ? matmul_0 reshape_0
            Reshape reshape_0  2 1+ add_0 ? gather_0
            Gather  gather_0   2 1 reshape_0 ? output_0
            output  output_0   1 0 gather_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "SplitMatMulQKV"

    def rewrite(self, opset=23):
        matmul_node = self.matmul_0
        add_node = self.add_0
        reshape_node = self.reshape_0

        if not isinstance(matmul_node.inputs[1], osg.Constant):
            return {}

        bias_constant, add_tensor_input = _get_add_constant_and_tensor(add_node)
        if bias_constant is None:
            return {}

        if add_tensor_input is not matmul_node.outputs[0]:
            return {}

        weight_values = matmul_node.inputs[1].values
        bias_values = bias_constant.values
        if not _is_supported_qkv_weight(weight_values):
            return {}

        bias_array = np.asarray(bias_values)
        if bias_array.ndim != 1 or bias_array.shape[0] != np.asarray(weight_values).shape[1]:
            return {}

        reshape_shape_input = reshape_node.inputs[1]
        if not isinstance(reshape_shape_input, osg.Constant):
            return {}

        reshape_shape = np.asarray(reshape_shape_input.values).reshape(-1)
        if reshape_shape.size == 0:
            return {}

        qkv_dim = int(reshape_shape[-1])
        if qkv_dim != 3:
            return {}

        if len(reshape_node.outputs) != 1:
            return {}

        reshape_output = reshape_node.outputs[0]
        reshape_consumers = reshape_output.outputs
        if len(reshape_consumers) != 3:
            return {}

        sibling_gathers = [
            node for node in reshape_output.outputs if node.op == "Gather" and len(node.inputs) == 2
        ]
        if len(sibling_gathers) != 3:
            return {}

        gather_nodes = {}
        for gather_node in sibling_gathers:
            gather_axis = int(gather_node.attrs.get("axis", 0))
            if gather_axis < 0:
                gather_axis += reshape_shape.size
            if gather_axis != reshape_shape.size - 1:
                return {}

            gather_index = _get_constant_scalar_value(gather_node.inputs[1])
            if gather_index not in (0, 1, 2):
                return {}
            if gather_index in gather_nodes:
                return {}
            gather_nodes[gather_index] = gather_node

        if set(gather_nodes.keys()) != {0, 1, 2}:
            return {}

        gather_nodes = [gather_nodes[0], gather_nodes[1], gather_nodes[2]]

        q_weight, k_weight, v_weight = _split_qkv_weight(weight_values)
        q_bias, k_bias, v_bias = _split_qkv_bias(bias_values)

        input_variable = matmul_node.inputs[0]
        gather_outputs = [gather_node.outputs[0] for gather_node in gather_nodes]

        if matmul_node in input_variable.outputs:
            input_variable.outputs.remove(matmul_node)

        new_ops = {}
        for name_suffix, weight_chunk, bias_chunk, gather_output in zip(
            ["q", "k", "v"],
            [q_weight, k_weight, v_weight],
            [q_bias, k_bias, v_bias],
            gather_outputs,
        ):
            weight_const = osg.Constant(
                name=f"{matmul_node.name}_{name_suffix}_weight",
                values=weight_chunk,
            )
            bias_const = osg.Constant(
                name=f"{add_node.name}_{name_suffix}_bias",
                values=bias_chunk,
            )
            matmul_output = osg.Variable(
                name=f"{matmul_node.name}_{name_suffix}_out",
                dtype=gather_output.dtype,
                shape=gather_output.shape,
            )

            new_ops[f"{matmul_node.name}_{name_suffix}"] = {
                "op": "MatMul",
                "inputs": [input_variable, weight_const],
                "outputs": [matmul_output],
                "name": f"{matmul_node.name}_{name_suffix}",
                "attrs": {},
                "domain": None,
            }

            gather_output.inputs.clear()
            add_inputs = [matmul_output, bias_const]
            if add_node.inputs[0] is bias_constant:
                add_inputs = [bias_const, matmul_output]

            new_ops[f"{add_node.name}_{name_suffix}"] = {
                "op": "Add",
                "inputs": add_inputs,
                "outputs": [gather_output],
                "name": f"{add_node.name}_{name_suffix}",
                "attrs": {},
                "domain": None,
            }

        return new_ops


register_fusion_pattern(MatMulQKVSplitPatternMatcher(1))
