import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class LayernormPatternMatcher(PatternMatcher):
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
        return "FusionLayerNorm"

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

        if isinstance(pow_0_node.inputs[1], osg.Constant) and \
            pow_0_node.inputs[1].values is not None and \
            np.allclose(2.0, pow_0_node.inputs[1].values) and \
            isinstance(add_0_node.inputs[1], osg.Constant) and \
            isinstance(mul_0_node.inputs[1], osg.Constant) and \
            isinstance(add_1_node.inputs[1], osg.Constant):

            axis = max(axes_0)
            epsilon = float(add_0_node.inputs[1].values)

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
                    "name": reduce_mean_0_node.name
                }
            }


register_fusion_pattern(LayernormPatternMatcher(1))
