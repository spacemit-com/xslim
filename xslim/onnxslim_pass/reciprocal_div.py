import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


def _is_one_constant(tensor):
    return (
        isinstance(tensor, osg.Constant)
        and tensor.values is not None
        and np.asarray(tensor.values).size == 1
        and np.allclose(np.asarray(tensor.values), 1.0)
    )


class ReciprocalDivMulPatternCase0Matcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input_x          0 1 div_0
            input          input_y          0 1 mul_0
            Div            div_0            2 1 ? input_x mul_0
            Mul            mul_0            2 1 div_0 input_y output
            output         output           1 0 mul_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionReciprocalDivMulCase0"

    def rewrite(self, opset=11):
        div_node = self.div_0
        mul_node = self.mul_0

        if not _is_one_constant(div_node.inputs[0]):
            return {}

        input_x = div_node.inputs[1]
        input_y = mul_node.inputs[1]

        input_x.outputs.remove(div_node)
        input_y.outputs.remove(mul_node)

        output_variable = mul_node.outputs[0]
        output_variable.inputs.clear()

        return {
            div_node.name: {
                "op": "Div",
                "inputs": [input_y, input_x],
                "outputs": [output_variable],
                "domain": None,
                "attrs": {},
                "name": div_node.name,
            }
        }


class ReciprocalDivMulPatternCase1Matcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input_x          0 1 div_0
            input          input_y          0 1 mul_0
            Div            div_0            2 1 ? input_x mul_0
            Mul            mul_0            2 1 input_y div_0 output
            output         output           1 0 mul_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionReciprocalDivMulCase1"

    def rewrite(self, opset=11):
        div_node = self.div_0
        mul_node = self.mul_0

        if not _is_one_constant(div_node.inputs[0]):
            return {}

        input_x = div_node.inputs[1]
        input_y = mul_node.inputs[0]

        input_x.outputs.remove(div_node)
        input_y.outputs.remove(mul_node)

        output_variable = mul_node.outputs[0]
        output_variable.inputs.clear()

        return {
            div_node.name: {
                "op": "Div",
                "inputs": [input_y, input_x],
                "outputs": [output_variable],
                "domain": None,
                "attrs": {},
                "name": div_node.name,
            }
        }


register_fusion_pattern(ReciprocalDivMulPatternCase0Matcher(2))
register_fusion_pattern(ReciprocalDivMulPatternCase1Matcher(2))
