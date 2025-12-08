import numpy as np
import onnxslim.third_party.onnx_graphsurgeon as osg
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern
from xslim.logger import logger


class BatchNormDecompositionMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input      X        0 1 bn
            BatchNormalization bn 5 1 X ? ? ? ? output
            output     output   1 0 bn
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "DecomposeBatchNormalization"

    def rewrite(self, opset=11):
        bn_node = self.bn
        X = bn_node.inputs[0]
        scale_const = bn_node.inputs[1]
        B_const = bn_node.inputs[2]
        mean_const = bn_node.inputs[3]
        var_const = bn_node.inputs[4]
        output_var = bn_node.outputs[0]

        if not all(isinstance(t, osg.Constant) for t in [scale_const, B_const, mean_const, var_const]):
            return {}

        scale = scale_const.values
        B = B_const.values
        mean = mean_const.values
        var = var_const.values
        epsilon = bn_node.attrs.get("epsilon", 0.000009999999747378752)

        try:
            denom = np.sqrt(var + epsilon)
            a_1d = scale / denom
            b_1d = B - a_1d * mean
            x_shape = X.shape
            if not x_shape:
                logger.warning("[Warning] No shape info, skip BN optimize.")
                return {}
            else:
                axis = 1
                broadcast_shape = [1] * len(x_shape)
                broadcast_shape[axis] = -1
                a = a_1d.reshape(broadcast_shape)
                b = b_1d.reshape(broadcast_shape)
        except Exception as e:
            logger.warning(f"[Warning] Failed to compute a/b for BN {bn_node.name}: {e}")
            return {}

        a_const = osg.Constant(name=f"{bn_node.name}_a", values=a)
        b_const = osg.Constant(name=f"{bn_node.name}_b", values=b)

        mul_output = osg.Variable(name=f"{bn_node.name}_mul_out", dtype=X.dtype)

        new_ops = {}
        X.outputs.remove(bn_node)
        output_var.inputs.clear()

        new_ops[f"{bn_node.name}_mul"] = {
            "op": "Mul",
            "inputs": [X, a_const],
            "outputs": [mul_output],
            "domain": None,
            "attrs": {},
            "name": f"{bn_node.name}_mul",
        }

        new_ops[f"{bn_node.name}_add"] = {
            "op": "Add",
            "inputs": [mul_output, b_const],
            "outputs": [output_var],
            "domain": None,
            "attrs": {},
            "name": f"{bn_node.name}_add",
        }

        return new_ops

register_fusion_pattern(BatchNormDecompositionMatcher(1))