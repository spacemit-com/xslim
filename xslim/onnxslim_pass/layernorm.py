import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class LayernormPatternCase0Matcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input            0 2 reduce_mean_0 sub_0
            ReduceMean     reduce_mean_0    1 1 input sub_0
            Sub            sub_0            2 1 input reduce_mean_0 pow_0
            Pow            pow_0            2 1 sub_0 ? reduce_mean_1
            ReduceMean     reduce_mean_1    1 1 pow_0 add_0
            Add            add_0            2 1 reduce_mean_1 ? sqrt_0
            Sqrt           sqrt_0           1 1 add_0 div_0
            Div            div_0            2 1 sub_0 sqrt_0 mul_0
            Mul            mul_0            2 1 div_0 ? add_1
            Add            add_1            2 1 mul_0 ? output
            output         output           1 0 add_1
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern, 'FusionLayerNorm'."""
        return "FusionLayerNormCase0"

    def rewrite(self, opset=11):
        reduce_mean_0_node = self.reduce_mean_0
        reduce_mean_1_node = self.reduce_mean_1
        sub_0_node = self.sub_0
        pow_0_node = self.pow_0
        add_0_node = self.add_0
        div_node = self.div_0
        mul_0_node = self.mul_0
        add_1_node = self.add_1

        axes_0 = reduce_mean_0_node.attrs.get("axes", [])
        axes_1 = reduce_mean_1_node.attrs.get("axes", [])

        input_variable = reduce_mean_0_node.inputs[0]

        if (
            isinstance(pow_0_node.inputs[1], osg.Constant)
            and pow_0_node.inputs[1].values is not None
            and np.allclose(2.0, pow_0_node.inputs[1].values)
            and isinstance(add_0_node.inputs[1], osg.Constant)
            and isinstance(mul_0_node.inputs[1], osg.Constant)
            and isinstance(add_1_node.inputs[1], osg.Constant)
        ):

            norm_scale_constant = mul_0_node.inputs[1]
            norm_bias_constant = add_1_node.inputs[1]

            axis = max(axes_0)
            epsilon = float(add_0_node.inputs[1].values)

            if not len(norm_scale_constant.shape) == 1 or not len(norm_bias_constant.shape) == 1:
                return {}

            input_variable.outputs.remove(reduce_mean_0_node)
            input_variable.outputs.remove(sub_0_node)

            output_variable = add_1_node.outputs[0]
            output_variable.inputs.clear()

            return {
                reduce_mean_0_node.name: {
                    "op": "LayerNormalization",
                    "inputs": [input_variable, mul_0_node.inputs[1], add_1_node.inputs[1]],
                    "outputs": [output_variable],
                    "domain": None,
                    "attrs": {"axis": axis, "epsilon": epsilon},
                    "name": reduce_mean_0_node.name,
                }
            }
        else:
            return {}


class LayernormPatternCase1Matcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input            0 2 reduce_mean_0 sub_0
            ReduceMean     reduce_mean_0    1 1 input sub_0
            Sub            sub_0            2 1 input reduce_mean_0 pow_0
            Pow            pow_0            2 1 sub_0 ? reduce_mean_1
            ReduceMean     reduce_mean_1    1 1 pow_0 add_0
            Add            add_0            2 1 reduce_mean_1 ? sqrt_0
            Sqrt           sqrt_0           1 1 add_0 div_0
            Div            div_0            2 1 sub_0 sqrt_0 mul_0
            Mul            mul_0            2 1 ? div_0 add_1
            Add            add_1            2 1 mul_0 ? output
            output         output           1 0 add_1
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern, 'FusionLayerNorm'."""
        return "FusionLayerNormCase1"

    def rewrite(self, opset=11):
        reduce_mean_0_node = self.reduce_mean_0
        reduce_mean_1_node = self.reduce_mean_1
        sub_0_node = self.sub_0
        pow_0_node = self.pow_0
        add_0_node = self.add_0
        div_node = self.div_0
        mul_0_node = self.mul_0
        add_1_node = self.add_1

        axes_0 = reduce_mean_0_node.attrs.get("axes", [])
        axes_1 = reduce_mean_1_node.attrs.get("axes", [])

        input_variable = reduce_mean_0_node.inputs[0]

        if (
            isinstance(pow_0_node.inputs[1], osg.Constant)
            and pow_0_node.inputs[1].values is not None
            and np.allclose(2.0, pow_0_node.inputs[1].values)
            and isinstance(add_0_node.inputs[1], osg.Constant)
            and isinstance(mul_0_node.inputs[0], osg.Constant)
            and isinstance(add_1_node.inputs[1], osg.Constant)
        ):

            norm_scale_constant = mul_0_node.inputs[0]
            norm_bias_constant = add_1_node.inputs[1]

            axis = max(axes_0)
            epsilon = float(add_0_node.inputs[1].values)

            append_ops = {}

            input_variable.outputs.remove(reduce_mean_0_node)
            input_variable.outputs.remove(sub_0_node)

            output_variable = add_1_node.outputs[0]
            output_variable.inputs.clear()

            ln_input = input_variable
            ln_output = output_variable
            if not len(norm_scale_constant.shape) == 1 or not len(norm_bias_constant.shape) == 1:
                ndim = len(norm_scale_constant.shape) + axis
                permute_in = [i for i in range(axis)] + [i for i in range(axis + 1, ndim)] + [axis]
                permute_out = [i for i in range(axis)] + [ndim - 1] + [i - 1 for i in range(axis + 1, ndim)]

                ln_input = osg.Variable("{}_ln_input".format(reduce_mean_0_node.name), np.float32)
                ln_output = osg.Variable("{}_ln_output".format(reduce_mean_0_node.name), np.float32)

                append_ops["{}_permute_in".format(reduce_mean_0_node.name)] = {
                    "op": "Transpose",
                    "inputs": [input_variable],
                    "outputs": [ln_input],
                    "domain": None,
                    "attrs": {"perm": permute_in},
                    "name": "{}_permute_in".format(reduce_mean_0_node.name),
                }

                append_ops["{}_permute_out".format(reduce_mean_0_node.name)] = {
                    "op": "Transpose",
                    "inputs": [ln_output],
                    "outputs": [output_variable],
                    "domain": None,
                    "attrs": {"perm": permute_out},
                    "name": "{}_permute_out".format(reduce_mean_0_node.name),
                }

                axis = -1
                mul_0_node.inputs[0].values = norm_scale_constant.values.reshape(-1)
                add_1_node.inputs[1].values = norm_bias_constant.values.reshape(-1)

            append_ops[reduce_mean_0_node.name] = {
                "op": "LayerNormalization",
                "inputs": [ln_input, mul_0_node.inputs[0], add_1_node.inputs[1]],
                "outputs": [ln_output],
                "domain": None,
                "attrs": {"axis": axis, "epsilon": epsilon},
                "name": reduce_mean_0_node.name,
            }

            return append_ops
        else:
            return {}


register_fusion_pattern(LayernormPatternCase0Matcher(1))
register_fusion_pattern(LayernormPatternCase1Matcher(1))
