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


def _build_ln_ops(reduce_mean_0_node, add_0_node, scale_input, bias_input,
                  axes_0, input_variable, output_variable, opset):
    """Build the LayerNormalization op dict, with Transpose wrap if needed.

    When scale/bias are not 1-D (channel-wise LayerNorm, e.g. scale shaped
    ``(1, C, 1, 1)``), insert Transpose in/out to move the normalisation axis to
    the last position so a standard ``LayerNormalization(axis=-1)`` applies.
    """
    axis = max(axes_0)
    epsilon = float(add_0_node.inputs[1].values)

    append_ops = {}

    ln_input = input_variable
    ln_output = output_variable

    if len(scale_input.shape) != 1 or len(bias_input.shape) != 1:
        # The permutation rank must equal the input rank.  value_info may have
        # been cleared upstream (format_onnx_model) so input_variable.shape can
        # be None; fall back to the (constant) scale rank, which equals the
        # input rank for channel-wise LayerNorm.
        if input_variable.shape is not None:
            ndim = len(input_variable.shape)
        elif axis >= 0:
            ndim = len(scale_input.shape)
        else:
            # Cannot resolve a non-negative axis without the input rank.
            return {}
        perm_in = (
            [i for i in range(axis)]
            + [i for i in range(axis + 1, ndim)]
            + [axis]
        )
        perm_out = list(np.argsort(perm_in).tolist())

        ln_dtype = input_variable.dtype
        ln_input = osg.Variable(
            f"{reduce_mean_0_node.name}_ln_input", ln_dtype
        )
        ln_output = osg.Variable(
            f"{reduce_mean_0_node.name}_ln_output", ln_dtype
        )

        append_ops[f"{reduce_mean_0_node.name}_permute_in"] = {
            "op": "Transpose",
            "inputs": [input_variable],
            "outputs": [ln_input],
            "domain": None,
            "attrs": {"perm": perm_in},
            "name": f"{reduce_mean_0_node.name}_permute_in",
        }
        append_ops[f"{reduce_mean_0_node.name}_permute_out"] = {
            "op": "Transpose",
            "inputs": [ln_output],
            "outputs": [output_variable],
            "domain": None,
            "attrs": {"perm": perm_out},
            "name": f"{reduce_mean_0_node.name}_permute_out",
        }

        axis = -1
        scale_input.values = scale_input.values.reshape(-1)
        bias_input.values = bias_input.values.reshape(-1)

    append_ops[reduce_mean_0_node.name] = {
        "op": "LayerNormalization",
        "inputs": [ln_input, scale_input, bias_input],
        "outputs": [ln_output],
        "domain": resolve_operator_domain("LayerNormalization", opset),
        "attrs": {"axis": axis, "epsilon": epsilon},
        "name": reduce_mean_0_node.name,
    }
    return append_ops


class _LayernormMatcherBase(PatternMatcher):
    """Shared rewrite for the LayerNorm subgraph.

    Subclasses parameterise two dimensions:

    * ``_scale_index`` -- which input of ``mul_0`` holds the affine scale
      constant (0 or 1); the other input is the normalised value ``div_0``.
    * ``_check_variance(var_node)`` -- validate the node computing variance of
      ``sub_0``.  ``Pow(sub_0, 2)`` and ``Mul(sub_0, sub_0)`` are both accepted
      by different subclasses.

    Pattern strings must be defined per subclass because the pattern engine
    matches a fixed op type at the variance node (Pow vs Mul) and the two forms
    have different input structures.  The variance node is always aliased
    ``var_0`` so this base can reference ``self.var_0`` uniformly.
    """

    _scale_index = 1

    def _check_variance(self, var_node):
        raise NotImplementedError

    def rewrite(self, opset=MIN_ONNX_OPSET_VERSION):
        reduce_mean_0_node = self.reduce_mean_0
        reduce_mean_1_node = self.reduce_mean_1
        sub_0_node = self.sub_0
        var_0_node = self.var_0
        add_0_node = self.add_0
        mul_0_node = self.mul_0
        add_1_node = self.add_1

        axes_0 = _get_reduce_mean_axes(reduce_mean_0_node)
        axes_1 = _get_reduce_mean_axes(reduce_mean_1_node)

        input_variable = reduce_mean_0_node.inputs[0]

        scale_input = mul_0_node.inputs[self._scale_index]
        bias_input = add_1_node.inputs[1]

        if not (
            self._check_variance(var_0_node)
            and isinstance(add_0_node.inputs[1], osg.Constant)
            and isinstance(scale_input, osg.Constant)
            and isinstance(bias_input, osg.Constant)
            and axes_0
            and axes_0 == axes_1
        ):
            return {}

        output_variable = add_1_node.outputs[0]

        # Build the replacement ops first.  _build_ln_ops can bail out with an
        # empty dict (e.g. non-1-D scale whose rank cannot be resolved), and it
        # only mutates scale/bias values on its success path -- so compute it
        # before touching the graph topology to avoid leaving dangling nodes
        # when no fusion is applied.
        append_ops = _build_ln_ops(
            reduce_mean_0_node, add_0_node, scale_input, bias_input,
            axes_0, input_variable, output_variable, opset,
        )
        if not append_ops:
            return {}

        input_variable.outputs.remove(reduce_mean_0_node)
        input_variable.outputs.remove(sub_0_node)
        output_variable.inputs.clear()

        return append_ops


def _check_pow_square(var_node):
    """Pow(sub_0, exponent) with exponent == 2."""
    exponent = var_node.inputs[1]
    return (
        isinstance(exponent, osg.Constant)
        and exponent.values is not None
        and np.allclose(2.0, exponent.values)
    )


def _check_mul_self_square(var_node):
    """Mul(sub_0, sub_0): both inputs are the same variable (self-multiply)."""
    return var_node.inputs[0] is var_node.inputs[1]


class LayernormPatternCase0Matcher(_LayernormMatcherBase):
    """Pow variance; scale at mul_0.inputs[1]."""

    _scale_index = 1
    _check_variance = staticmethod(_check_pow_square)

    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input            0 2 reduce_mean_0 sub_0
            ReduceMean     reduce_mean_0    1+ 1 input sub_0
            Sub            sub_0            2 1 input reduce_mean_0 var_0
            Pow            var_0            2 1 sub_0 ? reduce_mean_1
            ReduceMean     reduce_mean_1    1+ 1 var_0 add_0
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
        return "FusionLayerNormCase0"


class LayernormPatternCase1Matcher(_LayernormMatcherBase):
    """Pow variance; scale at mul_0.inputs[0]."""

    _scale_index = 0
    _check_variance = staticmethod(_check_pow_square)

    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input            0 2 reduce_mean_0 sub_0
            ReduceMean     reduce_mean_0    1+ 1 input sub_0
            Sub            sub_0            2 1 input reduce_mean_0 var_0
            Pow            var_0            2 1 sub_0 ? reduce_mean_1
            ReduceMean     reduce_mean_1    1+ 1 var_0 add_0
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
        return "FusionLayerNormCase1"


class LayernormMulSquareCase0Matcher(_LayernormMatcherBase):
    """Mul(sub, sub) variance variant; scale at mul_0.inputs[1].

    Some exporters emit variance as ``Mul(x - mean, x - mean)`` instead of
    ``Pow(x - mean, 2)``; the Pow-based matchers miss those subgraphs.
    """

    _scale_index = 1
    _check_variance = staticmethod(_check_mul_self_square)

    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input            0 2 reduce_mean_0 sub_0
            ReduceMean     reduce_mean_0    1+ 1 input sub_0
            Sub            sub_0            2 1 input reduce_mean_0 var_0
            Mul            var_0            2 1 sub_0 sub_0 reduce_mean_1
            ReduceMean     reduce_mean_1    1+ 1 var_0 add_0
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
        return "FusionLayerNormMulSquareCase0"


class LayernormMulSquareCase1Matcher(_LayernormMatcherBase):
    """Mul(sub, sub) variance variant; scale at mul_0.inputs[0]."""

    _scale_index = 0
    _check_variance = staticmethod(_check_mul_self_square)

    def __init__(self, priority):
        pattern = Pattern(
            """
            input          input            0 2 reduce_mean_0 sub_0
            ReduceMean     reduce_mean_0    1+ 1 input sub_0
            Sub            sub_0            2 1 input reduce_mean_0 var_0
            Mul            var_0            2 1 sub_0 sub_0 reduce_mean_1
            ReduceMean     reduce_mean_1    1+ 1 var_0 add_0
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
        return "FusionLayerNormMulSquareCase1"


register_fusion_pattern(LayernormPatternCase0Matcher(1))
register_fusion_pattern(LayernormPatternCase1Matcher(1))
register_fusion_pattern(LayernormMulSquareCase0Matcher(1))
register_fusion_pattern(LayernormMulSquareCase1Matcher(1))
