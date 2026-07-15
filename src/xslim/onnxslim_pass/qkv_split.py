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


def _split_contiguous_qkv_weight(weight_values):
    weight_array = np.asarray(weight_values)
    qkv_num = weight_array.shape[1] // 3
    return [
        np.ascontiguousarray(weight_array[:, qkv_num * index:qkv_num * (index + 1)])
        for index in range(3)
    ]


def _split_contiguous_qkv_bias(bias_values):
    bias_array = np.asarray(bias_values)
    qkv_num = bias_array.shape[-1] // 3
    return [
        np.ascontiguousarray(bias_array[qkv_num * index:qkv_num * (index + 1)])
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


def _get_constant_int_values(tensor):
    if not isinstance(tensor, osg.Constant) or tensor.values is None:
        return None

    values = np.asarray(tensor.values).reshape(-1)
    return [int(value) for value in values.tolist()]


def _get_split_sizes(split_node, axis_dim):
    if len(split_node.inputs) > 1:
        split_sizes = _get_constant_int_values(split_node.inputs[1])
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


def _normalize_axis(axis, rank):
    normalized_axis = int(axis)
    if normalized_axis < 0:
        normalized_axis += rank

    if normalized_axis < 0 or normalized_axis >= rank:
        return None

    return normalized_axis


def _get_squeeze_axes(squeeze_node, input_rank):
    axes = squeeze_node.attrs.get("axes")
    if axes is None:
        if len(squeeze_node.inputs) < 2:
            return None
        axes = _get_constant_int_values(squeeze_node.inputs[1])

    if axes is None:
        return None

    normalized_axes = []
    for axis in axes:
        normalized_axis = _normalize_axis(axis, input_rank)
        if normalized_axis is None:
            return None
        normalized_axes.append(normalized_axis)

    return normalized_axes


def _variable_consumers(variable):
    return list(getattr(variable, "outputs", []) or [])


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


class MatMulReshapeTransposeSplitQKVSqueezePatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input     input       0 1 matmul_0
            MatMul    matmul_0    2 1 input ? add_0
            Add       add_0       2 1 ? matmul_0 reshape_0
            Reshape   reshape_0   2 1 add_0 ? transpose_0
            Transpose transpose_0 1 1 reshape_0 split_0
            Split     split_0     1+ 1 transpose_0 output_0
            output    output_0    1 0 split_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "SplitMatMulReshapeTransposeQKVSqueeze"

    def rewrite(self, opset=23):
        matmul_node = self.matmul_0
        add_node = self.add_0
        reshape_node = self.reshape_0
        transpose_node = self.transpose_0
        split_node = self.split_0

        if len(matmul_node.inputs) != 2 or not isinstance(matmul_node.inputs[1], osg.Constant):
            return {}

        bias_constant, add_tensor_input = _get_add_constant_and_tensor(add_node)
        if bias_constant is None or add_tensor_input is not matmul_node.outputs[0]:
            return {}

        weight_values = matmul_node.inputs[1].values
        bias_values = bias_constant.values
        if not _is_supported_qkv_weight(weight_values):
            return {}

        weight_array = np.asarray(weight_values)
        bias_array = np.asarray(bias_values)
        if bias_array.ndim != 1 or bias_array.shape[0] != weight_array.shape[1]:
            return {}

        if len(reshape_node.inputs) != 2 or not isinstance(reshape_node.inputs[1], osg.Constant):
            return {}

        reshape_shape = np.asarray(reshape_node.inputs[1].values).reshape(-1)
        if reshape_shape.size != 5:
            return {}

        reshape_shape = [int(dim) for dim in reshape_shape.tolist()]
        if any(dim <= 0 for dim in reshape_shape):
            return {}
        if reshape_shape[2] != 3:
            return {}

        qkv_num = weight_array.shape[1] // 3
        if reshape_shape[3] * reshape_shape[4] != qkv_num:
            return {}

        transpose_perm = [int(dim) for dim in transpose_node.attrs.get("perm", [])]
        if transpose_perm != [2, 0, 3, 1, 4]:
            return {}

        split_axis = _normalize_axis(int(split_node.attrs.get("axis", 0)), len(transpose_perm))
        if split_axis != 0:
            return {}

        split_sizes = _get_split_sizes(split_node, reshape_shape[2])
        if split_sizes != [1, 1, 1]:
            return {}

        if len(reshape_node.outputs) != 1 or len(transpose_node.outputs) != 1:
            return {}

        original_reshape_output = reshape_node.outputs[0]
        original_transpose_output = transpose_node.outputs[0]
        if _variable_consumers(original_reshape_output) != [transpose_node]:
            return {}
        if _variable_consumers(original_transpose_output) != [split_node]:
            return {}

        split_outputs = list(split_node.outputs)
        if len(split_outputs) != 3:
            return {}

        split_output_names = {split_output.name for split_output in split_outputs}
        squeeze_by_split_output = {}
        for split_output in split_outputs:
            consumers = _variable_consumers(split_output)
            if len(consumers) != 1 or consumers[0].op != "Squeeze":
                return {}

            squeeze_node = consumers[0]
            if len(squeeze_node.inputs) < 1 or len(squeeze_node.outputs) != 1:
                return {}

            squeeze_input = squeeze_node.inputs[0]
            if squeeze_input.name not in split_output_names:
                return {}

            squeeze_axes = _get_squeeze_axes(squeeze_node, len(transpose_perm))
            if squeeze_axes != [0]:
                return {}

            if squeeze_input.name in squeeze_by_split_output:
                return {}
            squeeze_by_split_output[squeeze_input.name] = squeeze_node

        if set(squeeze_by_split_output.keys()) != split_output_names:
            return {}

        q_weight, k_weight, v_weight = _split_contiguous_qkv_weight(weight_values)
        q_bias, k_bias, v_bias = _split_contiguous_qkv_bias(bias_values)

        input_variable = matmul_node.inputs[0]
        if matmul_node in input_variable.outputs:
            input_variable.outputs.remove(matmul_node)

        new_ops = {}
        branch_shape = [reshape_shape[0], reshape_shape[1], reshape_shape[3], reshape_shape[4]]
        for index, name_suffix, weight_chunk, bias_chunk in zip(
            range(3),
            ["q", "k", "v"],
            [q_weight, k_weight, v_weight],
            [q_bias, k_bias, v_bias],
        ):
            squeeze_node = squeeze_by_split_output[split_outputs[index].name]
            branch_output = squeeze_node.outputs[0]

            weight_const = osg.Constant(
                name=f"{matmul_node.name}_{name_suffix}_weight",
                values=weight_chunk,
            )
            bias_const = osg.Constant(
                name=f"{add_node.name}_{name_suffix}_bias",
                values=bias_chunk,
            )
            branch_shape_const = osg.Constant(
                name=f"{reshape_node.name}_{name_suffix}_shape",
                values=np.asarray(branch_shape, dtype=np.int64),
            )

            matmul_output = osg.Variable(
                name=f"{matmul_node.name}_{name_suffix}_out",
                dtype=matmul_node.outputs[0].dtype,
                shape=None,
            )
            add_output = osg.Variable(
                name=f"{add_node.name}_{name_suffix}_out",
                dtype=add_node.outputs[0].dtype,
                shape=None,
            )
            branch_reshape_output = osg.Variable(
                name=f"{reshape_node.name}_{name_suffix}_out",
                dtype=branch_output.dtype,
                shape=branch_shape,
            )

            new_ops[f"{matmul_node.name}_{name_suffix}"] = {
                "op": "MatMul",
                "inputs": [input_variable, weight_const],
                "outputs": [matmul_output],
                "name": f"{matmul_node.name}_{name_suffix}",
                "attrs": {},
                "domain": None,
            }

            add_inputs = [matmul_output, bias_const]
            if add_node.inputs[0] is bias_constant:
                add_inputs = [bias_const, matmul_output]

            new_ops[f"{add_node.name}_{name_suffix}"] = {
                "op": "Add",
                "inputs": add_inputs,
                "outputs": [add_output],
                "name": f"{add_node.name}_{name_suffix}",
                "attrs": {},
                "domain": None,
            }

            new_ops[f"{reshape_node.name}_{name_suffix}"] = {
                "op": "Reshape",
                "inputs": [add_output, branch_shape_const],
                "outputs": [branch_reshape_output],
                "name": f"{reshape_node.name}_{name_suffix}",
                "attrs": dict(reshape_node.attrs),
                "domain": None,
            }

            branch_output.inputs.clear()
            squeeze_node.inputs.clear()
            squeeze_node.outputs.clear()
            new_ops[f"{transpose_node.name}_{name_suffix}"] = {
                "op": "Transpose",
                "inputs": [branch_reshape_output],
                "outputs": [branch_output],
                "name": f"{transpose_node.name}_{name_suffix}",
                "attrs": {"perm": [0, 2, 1, 3]},
                "domain": None,
            }

        return new_ops


register_fusion_pattern(MatMulQKVSplitPatternMatcher(1))
register_fusion_pattern(MatMulReshapeTransposeSplitQKVSqueezePatternMatcher(1))
