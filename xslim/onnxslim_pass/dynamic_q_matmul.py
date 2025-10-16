import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class DynamicQuantizeMatMulIntegerPatternMatcher(PatternMatcher):
    def __init__(self, priority):

        pattern = Pattern(
            """
            input                   input               0 1 lhs_dyn_quant_0
            DynamicQuantizeLinear   lhs_dyn_quant_0     1 3 input matmul_0 mul_0 matmul_0
            MatMulInteger           matmul_0            4 1 lhs_dyn_quant_0 ? lhs_dyn_quant_0 ? cast_0
            Cast                    cast_0              1 1 matmul_0 mul_1
            Mul                     mul_0               2 1 lhs_dyn_quant_0 ? mul_1
            Mul                     mul_1               2 1 cast_0 mul_0 output
            output                  output              1 0 mul_1
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern, 'FusionDynamicQuantizeMatMulInteger'."""
        return "FusionDynamicQuantizeMatMulInteger"

    def rewrite(self, opset=11):
        lhs_dyn_quant_0 = self.lhs_dyn_quant_0
        matmul_0 = self.matmul_0
        mul_0 = self.mul_0

        if len(lhs_dyn_quant_0.outputs[0].outputs) == 1:
            input_variable = lhs_dyn_quant_0.inputs[0]
            input_variable.outputs.remove(lhs_dyn_quant_0)

            weight_variable = matmul_0.inputs[1]
            weight_scale_variable = mul_0.inputs[1]
            weight_zp_variable = matmul_0.inputs[3]

            weight_variable.outputs.remove(matmul_0)
            weight_scale_variable.outputs.remove(mul_0)
            weight_zp_variable.outputs.remove(matmul_0)

            output_variable = self.mul_1.outputs[0]
            output_variable.inputs.clear()
            return {
                matmul_0.name: {
                    "op": "DynamicQuantizeMatMul",
                    "inputs": [input_variable, weight_variable, weight_scale_variable, weight_zp_variable],
                    "outputs": [output_variable],
                    "domain": "com.microsoft",
                    "name": matmul_0.name
                }
            }
        else:
            return {}

class DynamicQuantizeMatMulPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input                   input               0 1 lhs_dyn_quant_0
            DynamicQuantizeLinear   lhs_dyn_quant_0     1 3 input lhs_dq_0 lhs_dq_0 lhs_dq_0
            DequantizeLinear        lhs_dq_0            3 1 lhs_dyn_quant_0 lhs_dyn_quant_0 lhs_dyn_quant_0 matmul_0
            DequantizeLinear        rhs_dq_0            3 1 ? ? ? matmul_0
            MatMul                  matmul_0            2 1 lhs_dq_0 rhs_dq_0 output
            output                  output              1 0 matmul_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern, 'FusionDynamicQuantizeMatMul'."""
        return "FusionDynamicQuantizeMatMul"

    def rewrite(self, opset=11):
        lhs_dyn_quant_0 = self.lhs_dyn_quant_0
        rhs_dq_0 = self.rhs_dq_0
        matmul_0 = self.matmul_0

        if len(lhs_dyn_quant_0.outputs[0].outputs) == 1:
            input_variable = lhs_dyn_quant_0.inputs[0]
            input_variable.outputs.remove(lhs_dyn_quant_0)

            weight_variable = rhs_dq_0.inputs[0]
            weight_scale_variable = rhs_dq_0.inputs[1]
            weight_zp_variable = rhs_dq_0.inputs[2]

            weight_variable.outputs.remove(rhs_dq_0)
            weight_scale_variable.outputs.remove(rhs_dq_0)
            weight_zp_variable.outputs.remove(rhs_dq_0)

            output_variable = matmul_0.outputs[0]
            output_variable.inputs.clear()
            return {
                matmul_0.name: {
                    "op": "DynamicQuantizeMatMul",
                    "inputs": [input_variable, weight_variable, weight_scale_variable, weight_zp_variable],
                    "outputs": [output_variable],
                    "domain": "com.microsoft",
                    "name": matmul_0.name
                }
            }
        else:
            return {}

class DynamicQuantizeMatMulBiasRHSPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input                   input               0 1 dyn_matmul_0
            DynamicQuantizeMatMul   dyn_matmul_0        4 1 input ? ? ? bias_add_0
            Add                     bias_add_0          2 1 dyn_matmul_0 ? output
            output                  output              1 0 bias_add_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern, 'FusionDynamicQuantizeMatMulBiasRHS'."""
        return "FusionDynamicQuantizeMatMulBiasRHS"

    def rewrite(self, opset=11):
        dyn_matmul_0 = self.dyn_matmul_0
        bias_add_0 = self.bias_add_0

        if isinstance(bias_add_0.inputs[1], osg.Constant) and \
            bias_add_0.inputs[1].values is not None and \
            len(bias_add_0.inputs[1].values.shape) <= 1:
            input_variable = dyn_matmul_0.inputs[0]
            weight_variable = dyn_matmul_0.inputs[1]
            weight_scale_variable = dyn_matmul_0.inputs[2]
            weight_zp_variable = dyn_matmul_0.inputs[3]

            input_variable.outputs.remove(dyn_matmul_0)
            weight_variable.outputs.remove(dyn_matmul_0)
            weight_scale_variable.outputs.remove(dyn_matmul_0)
            weight_zp_variable.outputs.remove(dyn_matmul_0)

            bias_variable = bias_add_0.inputs[1]
            bias_variable.outputs.remove(bias_add_0)

            output_variable = bias_add_0.outputs[0]
            output_variable.inputs.clear()

            return {
                dyn_matmul_0.name: {
                    "op": "DynamicQuantizeMatMul",
                    "inputs": [input_variable, weight_variable, weight_scale_variable, weight_zp_variable, bias_variable],
                    "outputs": [output_variable],
                    "domain": "com.microsoft",
                    "name": dyn_matmul_0.name
                }
            }
        else:
            return {}

class DynamicQuantizeMatMulBiasLHSPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input                   input               0 1 dyn_matmul_0
            DynamicQuantizeMatMul   dyn_matmul_0        4 1 input ? ? ? bias_add_0
            Add                     bias_add_0          2 1 ? dyn_matmul_0 output
            output                  output              1 0 bias_add_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern, 'FusionDynamicQuantizeMatMulBiasLHS'."""
        return "FusionDynamicQuantizeMatMulBiasLHS"

    def rewrite(self, opset=11):
        dyn_matmul_0 = self.dyn_matmul_0
        bias_add_0 = self.bias_add_0

        if isinstance(bias_add_0.inputs[0], osg.Constant) and \
            bias_add_0.inputs[0].values is not None and \
            len(bias_add_0.inputs[0].values.shape) <= 1:
            input_variable = dyn_matmul_0.inputs[0]
            weight_variable = dyn_matmul_0.inputs[1]
            weight_scale_variable = dyn_matmul_0.inputs[2]
            weight_zp_variable = dyn_matmul_0.inputs[3]

            input_variable.outputs.remove(dyn_matmul_0)
            weight_variable.outputs.remove(dyn_matmul_0)
            weight_scale_variable.outputs.remove(dyn_matmul_0)
            weight_zp_variable.outputs.remove(dyn_matmul_0)

            bias_variable = bias_add_0.inputs[0]
            bias_variable.outputs.remove(bias_add_0)

            output_variable = bias_add_0.outputs[0]
            output_variable.inputs.clear()

            return {
                dyn_matmul_0.name: {
                    "op": "DynamicQuantizeMatMul",
                    "inputs": [input_variable, weight_variable, weight_scale_variable, weight_zp_variable, bias_variable],
                    "outputs": [output_variable],
                    "domain": "com.microsoft",
                    "name": dyn_matmul_0.name
                }
            }
        else:
            return {}

register_fusion_pattern(DynamicQuantizeMatMulIntegerPatternMatcher(1))
#register_fusion_pattern(DynamicQuantizeMatMulPatternMatcher(1))
register_fusion_pattern(DynamicQuantizeMatMulBiasRHSPatternMatcher(1))
register_fusion_pattern(DynamicQuantizeMatMulBiasLHSPatternMatcher(1))
