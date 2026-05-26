import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern

from xslim.defs import MIN_ONNX_OPSET_VERSION, resolve_operator_domain


def _get_reduce_mean_axes(node):
    axes = node.attrs.get("axes")
    if axes is not None:
        return np.asarray(axes).reshape(-1).tolist()

    if len(node.inputs) < 2 or not isinstance(node.inputs[1], osg.Constant):
        return []

    if node.inputs[1].values is None:
        return []

    return np.asarray(node.inputs[1].values).reshape(-1).tolist()


class RMSNormPatternCase0Matcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input            0 2 pow_0 div_0
            Pow            pow_0            2 1 input ? reduce_mean_0
            ReduceMean     reduce_mean_0    1+ 1 pow_0 add_0
            Add            add_0            2 1 reduce_mean_0 ? sqrt_0
            Sqrt           sqrt_0           1 1 add_0 div_0
            Div            div_0            2 1 input sqrt_0 mul_0
            Mul            mul_0            2 1 div_0 ? output
            output         output           1 0 mul_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionRMSNormCase0"

    def rewrite(self, opset=MIN_ONNX_OPSET_VERSION):
        reduce_mean_node = self.reduce_mean_0
        pow_0_node = self.pow_0
        add_0_node = self.add_0
        div_node = self.div_0
        mul_0_node = self.mul_0

        axes = _get_reduce_mean_axes(reduce_mean_node)
        input_variable = div_node.inputs[0]

        if (
            pow_0_node.inputs[0] is input_variable
            and isinstance(pow_0_node.inputs[1], osg.Constant)
            and pow_0_node.inputs[1].values is not None
            and np.allclose(2.0, pow_0_node.inputs[1].values)
            and isinstance(add_0_node.inputs[1], osg.Constant)
            and isinstance(mul_0_node.inputs[1], osg.Constant)
            and div_node.inputs[0] is input_variable
            and axes
        ):
            input_variable.outputs.remove(pow_0_node)
            input_variable.outputs.remove(div_node)

            output_variable = mul_0_node.outputs[0]
            output_variable.inputs.clear()
            axis = max(axes)
            epsilon = float(add_0_node.inputs[1].values)

            return {
                reduce_mean_node.name: {
                    "op": "RMSNormalization",
                    "inputs": [input_variable, mul_0_node.inputs[1]],
                    "outputs": [output_variable],
                    "domain": resolve_operator_domain("RMSNormalization", opset),
                    "attrs": {"axis": axis, "epsilon": epsilon},
                    "name": reduce_mean_node.name,
                }
            }

        return {}


class RMSNormPatternCase1Matcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input            0 2 pow_0 div_0
            Pow            pow_0            2 1 input ? reduce_mean_0
            ReduceMean     reduce_mean_0    1+ 1 pow_0 add_0
            Add            add_0            2 1 reduce_mean_0 ? sqrt_0
            Sqrt           sqrt_0           1 1 add_0 div_0
            Div            div_0            2 1 input sqrt_0 mul_0
            Mul            mul_0            2 1 ? div_0 output
            output         output           1 0 mul_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionRMSNormCase1"

    def rewrite(self, opset=MIN_ONNX_OPSET_VERSION):
        reduce_mean_node = self.reduce_mean_0
        pow_0_node = self.pow_0
        add_0_node = self.add_0
        div_node = self.div_0
        mul_0_node = self.mul_0

        axes = _get_reduce_mean_axes(reduce_mean_node)
        input_variable = div_node.inputs[0]

        if (
            pow_0_node.inputs[0] is input_variable
            and isinstance(pow_0_node.inputs[1], osg.Constant)
            and pow_0_node.inputs[1].values is not None
            and np.allclose(2.0, pow_0_node.inputs[1].values)
            and isinstance(add_0_node.inputs[1], osg.Constant)
            and isinstance(mul_0_node.inputs[0], osg.Constant)
            and div_node.inputs[0] is input_variable
            and axes
        ):
            input_variable.outputs.remove(pow_0_node)
            input_variable.outputs.remove(div_node)

            output_variable = mul_0_node.outputs[0]
            output_variable.inputs.clear()
            axis = max(axes)
            epsilon = float(add_0_node.inputs[1].values)

            return {
                reduce_mean_node.name: {
                    "op": "RMSNormalization",
                    "inputs": [input_variable, mul_0_node.inputs[0]],
                    "outputs": [output_variable],
                    "domain": resolve_operator_domain("RMSNormalization", opset),
                    "attrs": {"axis": axis, "epsilon": epsilon},
                    "name": reduce_mean_node.name,
                }
            }

        return {}


register_fusion_pattern(RMSNormPatternCase0Matcher(2))
register_fusion_pattern(RMSNormPatternCase1Matcher(2))
