import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class SwishPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        """Initializes a `SwishPatternMatcher` to identify and fuse Swish patterns in a computational graph."""
        pattern = Pattern(
            """
            input       input  0 2 sig_0 mul_0
            Sigmoid     sig_0  1 1 input mul_0
            Mul         mul_0  2 1 input sig_0 output
            output      output 1 0 mul_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern, 'FusionSwish'."""
        return "FusionSwish"

    def rewrite(self, opset=24):
        """Rewrite the computation graph pattern to fuse Swish operations."""
        input_variable = self.sig_0.inputs[0]
        output_variable = self.mul_0.outputs[0]

        input_variable.outputs.remove(self.sig_0)
        input_variable.outputs.remove(self.mul_0)
        output_variable.inputs.clear()

        alpha = 1.0

        return {
            self.mul_0.name: {
                "op": "Swish",
                "inputs": [input_variable],
                "outputs": [output_variable],
                "domain": None,
                "name": self.mul_0.name,
                "attrs": {"alpha": alpha},
            }
        }


register_fusion_pattern(SwishPatternMatcher(1))
