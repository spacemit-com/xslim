from typing import Iterable, List, Set, Union, Dict, Callable, Tuple
import torch
from ppq.core import (
    OBSERVER_MSE_HIST_BINS,
    PASSIVE_OPERATIONS,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    RoundingPolicy,
    TargetPlatform,
    empty_ppq_cache,
    ppq_warning,
)
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.quantization.optim import (
    QuantizationOptimizationPipeline,
    QuantizationOptimizationPass,
    RuntimeCalibrationPass,
)
from ppq.IR.search import SearchableGraph
from ppq.executor import BaseGraphExecutor
from ppq.quantization.qfunction import PPQuantFunction


class BiasParameterBakingPass(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name="BiasParameterBaking Pass")
        self._quantize_function = PPQuantFunction

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation):
                continue
            if operation.type in {"Conv", "ConvTranspose", "Gemm"}:
                if operation.num_of_input == 3:
                    i_cfg, w_cfg, b_cfg = operation.config.input_quantization_config
                    o_cfg = operation.config.output_quantization_config[0]
                    if b_cfg.state not in {QuantizationStates.FP32}:
                        continue
                    bias = operation.inputs[-1].value
                    if bias is None:
                        raise ValueError(
                            f"Bias Varaible {operation.inputs[-1].name} must be a constant. " "Please check it again."
                        )
                    assert bias.numel() == bias.shape[-1], (
                        f"For op {operation.name}, expect Bias shape to be {[bias.numel()]}, "
                        f"however {bias.shape} was given"
                    )
                    operation.inputs[-1].value = bias.squeeze()

                    if operation.inputs[-1].value.ndim == 0 and operation.inputs[-1].value.numel() == 1:
                        operation.inputs[-1].value = operation.inputs[-1].value.unsqueeze(0)
                    if w_cfg.scale is None or i_cfg.scale is None:
                        continue
                    _b_scale = w_cfg.scale * i_cfg.scale
                    _i_bias = bias.to(torch.float64) / _b_scale.to(torch.float64)
                    if torch.all(torch.abs(_i_bias) < 2 ** (b_cfg.num_of_bits - 1)):
                        b_cfg.scale = _b_scale
                    else:
                        # in frac + w frac无法表示就使用 out frac
                        b_cfg.scale = o_cfg.scale
                    b_cfg.state = QuantizationStates.PASSIVE
                    b_cfg.offset = torch.zeros_like(b_cfg.scale)


class AsymmetricaUnsignlAlignSign(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name="AsymmetricalAlignS8 Pass")

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation):
                continue
            for config, var in [_ for _ in operation.config_with_variable]:
                if config.policy.has_property(QuantizationProperty.ASYMMETRICAL) and config.quant_min == 0:
                    config.quant_min = -(2 ** (config.num_of_bits - 1))
                    config.quant_max = 2 ** (config.num_of_bits - 1) - 1
                    if config.dominated_by == config and config.offset is not None:
                        store_state = config.state
                        config.state = QuantizationStates.INITIAL
                        config.offset = config.offset + config.quant_min
                        config.state = store_state


class QuantizeFusionPass(QuantizationOptimizationPass):
    """
    ## PPQ Quantize Fusion Pass(通用量化图融合过程)

    Operation fusion (or kernel/layer fusion) is key optimization in many state-of-the-art execution frameworks.

    Graph fusion can combine operations into a single op to obtain higher accuracy and performance,
        Pattern like: Conv + Relu can be reduced to ConvRelu. This fusion will reduce memory accesses,
        and the quantization point after conv can also be removed.

    Technically we can fuse those layers before quantization, while fused layers are not supported by onnx standard.
        So to say ConvRelu is not a valid onnx operation, no execution framework can parse it.

    Therefore, PPQ will simulate the graph fusion by adjusting quantization config: if PPQ finds their is a
        pattern like Conv + Relu, the output quantization of Conv will be disabled, pretending that the Conv + Relu
        fusion has happened.

    This Pass is designed for 2 types graph fusion:
        1. activation fusion
        2. passive operation fusion

    For activation fusion, PPQ will identify the pattern: Computing op + Activation Op from your network. The output
        quantization of computing op will be disabled with their state being set to QuantizationState.OVERLAPPED.

    Activation fusion here supports only simple activation patterns,
        for complex activation functions like mish, swish,
        will be represented as mish = tanh + mul + softplus, swish = sigmoid + mul in onnx,
        cause onnx does not have a op defination for them.
        Identifying those complex patterns requires pattern matching, which is implemented in ppq.IR.search.py

    Complex quantization fusions must be invoked manually, PPQ implemented softplus & swish fusion functions in
        ppq.quantization.optim.refine.MishFusionPass
        ppq.quantization.optim.refine.SwishFusionPass

    For passive operation fusion, PPQ will keep the input and the output variable share a same scale for passive operations.
        An operation is identified as passive op only if its attribute "is_active_quant_op" = False, this
        attribute is initialized by quantizer.

    If there is a passive operation having multiple input and output, the fusion procedure will make its
    FIRST input variable and ALL output variables share the same scale(the same scale as its first input).
    The quantization states of all output variables will be set to QuantizationState.OVERLAPPED.

    ### Parameters:

    * activation_type(Set[str]):

            A collection contains all activation types.

            The pattern will be recognized as [Computing Op -> Activation Op],

            By graph fusion, the output quantization of the Computing Op and
                the input quantization of the activation op will be disabled.

    * fuse_activation(bool)

            Whether to fuse activation op with computing op.

    # fuse_passive_op(bool)

            Whether to fuse passive op so that the input variable and output variable will share a same scale.

    * fuse_matmul_add(bool)

            Fuse MatMul + Bias Add

    ### Usage
    This pass is included in PPQ Quantization Setting, you can calling this optimization by:

        setting = QuantizationSettingFactory.default_setting()

        setting.fusion = True

        # calling ppq.api.quantize_onnx_model function with this setting.
        ir = quantize_torch_model(
        model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
        platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE,
        collate_fn=collate_fn)
    """

    def __init__(
        self,
        activation_type: Set[str],
        fuse_activation: bool = True,
        fuse_passive_op: bool = True,
        fuse_relu_clip: bool = True,
    ) -> None:
        self.fuse_activation = fuse_activation
        self.fuse_passive_op = fuse_passive_op
        self.fuse_relu_clip = fuse_relu_clip
        self.activation_types = activation_type
        super().__init__(name="PPQ Quantization Fusion Pass")

    def is_same_platform(self, operations: List[Operation]):
        platforms = [operation.platform for operation in operations]
        return all([platform == platforms[0] for platform in platforms])

    @empty_ppq_cache
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        processor = SearchableGraph(graph)

        # fuse computing operations and its following activation.
        if self.fuse_activation:
            patterns = processor.pattern_matching(
                patterns=[lambda x: x.is_computing_op, lambda x: x.type in self.activation_types],
                edges=[[0, 1]],
                exclusive=True,
            )

            for computing_op, act_op in patterns:
                if not isinstance(act_op, QuantableOperation):
                    continue
                if not isinstance(computing_op, QuantableOperation):
                    continue

                if (
                    computing_op.platform != act_op.platform
                    and computing_op.config.output_quantization_config[0].state != QuantizationStates.FP32
                ):
                    ppq_warning(
                        f"Unexpected dispatching was found: "
                        f"Op {computing_op.name} and {act_op.name} should be send to a same platform."
                    )
                    continue

                if (
                    len(graph.get_downstream_operations(computing_op)) == 1
                    and len(graph.get_upstream_operations(act_op)) == 1
                ):
                    computing_op.config.output_quantization_config[
                        0
                    ].dominated_by = act_op.config.output_quantization_config[0]
                    act_op.config.input_quantization_config[0].dominated_by = act_op.config.output_quantization_config[
                        0
                    ]

            if "Swish" in self.activation_types:
                search_engine = SearchableGraph(graph)
                patterns = search_engine.pattern_matching(
                    patterns=[lambda x: x.is_computing_op, "Sigmoid", "Mul"],
                    edges=[[0, 1], [1, 2], [0, 2]],
                    exclusive=True,
                )

                for pattern in patterns:
                    if any([not isinstance(op, QuantableOperation) for op in pattern]):
                        ppq_warning(
                            f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                            "however part of your swish activation is not quantable, "
                            "so that graph fusion can not merge their quantization configuration."
                        )
                        continue
                    if any([op.platform != pattern[0].platform for op in pattern]):
                        ppq_warning(
                            f"There is a pattern of swish activation in your network start from {pattern[0]}, "
                            "however part of your swish activation is not quantable, "
                            "so that graph fusion can not merge their quantization configuration."
                        )
                        continue
                    computing, sigmoid, mul = pattern

                    assert isinstance(computing, QuantableOperation)
                    assert isinstance(sigmoid, QuantableOperation)
                    assert isinstance(mul, QuantableOperation)

                    master_config = mul.config.output_quantization_config[0]
                    computing.config.output_quantization_config[0].dominated_by = master_config
                    sigmoid.config.input_quantization_config[0].dominated_by = master_config
                    sigmoid.config.output_quantization_config[0].dominated_by = master_config
                    mul.config.input_quantization_config[0].dominated_by = master_config
                    mul.config.input_quantization_config[1].dominated_by = master_config

            if "Mish" in self.activation_types:
                search_engine = SearchableGraph(graph)
                patterns = search_engine.pattern_matching(
                    patterns=[lambda x: x.is_computing_op, "Softplus", "Tanh", "Mul"],
                    edges=[[0, 1], [1, 2], [2, 3], [0, 3]],
                    exclusive=True,
                )

                for pattern in patterns:
                    if any([not isinstance(op, QuantableOperation) for op in pattern]):
                        ppq_warning(
                            f"There is a pattern of mish activation in your network start from {pattern[0]}, "
                            "however part of your mish activation is not quantable, "
                            "so that graph fusion can not merge their quantization configuration."
                        )
                        continue
                    if any([op.platform != pattern[0].platform for op in pattern]):
                        ppq_warning(
                            f"There is a pattern of mish activation in your network start from {pattern[0]}, "
                            "however part of your mish activation is not quantable, "
                            "so that graph fusion can not merge their quantization configuration."
                        )
                        continue
                    computing, softplus, tanh, mul = pattern

                    assert isinstance(computing, QuantableOperation)
                    assert isinstance(softplus, QuantableOperation)
                    assert isinstance(tanh, QuantableOperation)
                    assert isinstance(mul, QuantableOperation)

                    master_config = mul.config.output_quantization_config[0]
                    computing.config.output_quantization_config[0].dominated_by = master_config
                    tanh.config.input_quantization_config[0].dominated_by = master_config
                    tanh.config.output_quantization_config[0].dominated_by = master_config
                    softplus.config.input_quantization_config[0].dominated_by = master_config
                    softplus.config.output_quantization_config[0].dominated_by = master_config
                    mul.config.input_quantization_config[0].dominated_by = master_config
                    mul.config.input_quantization_config[1].dominated_by = master_config

        if self.fuse_passive_op:
            # all passive operations should never changes quantization configuration of its input
            # so to say their input and output share a same scale.
            for op in graph.operations.values():
                if op.type not in PASSIVE_OPERATIONS:
                    continue
                source_op = op.inputs[0].source_op
                if source_op is None:
                    continue  # beginning op, can not merge.
                if isinstance(op, QuantableOperation) and self.is_same_platform([op, source_op]):
                    TQC = op.config.input_quantization_config[0]
                    for output_cfg in op.config.output_quantization_config:
                        output_cfg.dominated_by = TQC

        if self.fuse_relu_clip:
            patterns = processor.pattern_matching(
                patterns=[lambda x: True, lambda x: x.type in {"Relu", "Clip"}], edges=[[0, 1]], exclusive=True
            )
            for computing_op, act_op in patterns:
                if not isinstance(act_op, QuantableOperation):
                    continue
                if not isinstance(computing_op, QuantableOperation):
                    continue

                if (
                    len(graph.get_downstream_operations(computing_op)) == 1
                    and len(graph.get_upstream_operations(act_op)) == 1
                ):
                    computing_op.config.output_quantization_config[
                        0
                    ].dominated_by = act_op.config.output_quantization_config[0]
                    act_op.config.input_quantization_config[0].dominated_by = act_op.config.output_quantization_config[
                        0
                    ]
