from typing import Callable, Dict, Iterator, List

import torch
from ppq.core import PASSIVE_OPERATIONS, ppq_warning
from ppq.executor import RuntimeHook, TorchExecutor
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.quantization.measure import torch_snr_error, torch_cosine_similarity, torch_mean_square_error
from ppq.utils.fetch import batch_random_fetch, tensor_random_fetch, generate_torch_indexer
from tqdm import tqdm

from ppq.quantization.analyse.util import MeasurePrinter, MeasureRecorder


class OutputRecorder(RuntimeHook):
    def __init__(self, operation: Operation, fetchs: int = 4096) -> None:
        self.fetched = None
        self.fetchs = fetchs
        super().__init__(operation)

    def pre_forward_hook(self, inputs: list, **kwargs) -> list:
        return super().pre_forward_hook(inputs, **kwargs)

    def post_forward_hook(self, outputs: list, **kwargs) -> list:
        output_tensor = outputs[0]
        assert isinstance(output_tensor, torch.Tensor), "Output of monitoring operation is not a torch.Tensor"
        self.fetched = batch_random_fetch(output_tensor, seed=10086, fetches_per_batch=self.fetchs).to("cpu")
        return super().post_forward_hook(outputs, **kwargs)

    def pop(self) -> torch.Tensor:
        fetched = self.fetched
        self.fetched = None
        return fetched


class DetailedRecorder(RuntimeHook):
    def __init__(self, operation: Operation, fetchs: int = 1024) -> None:
        self.fetchs = fetchs
        self.i_storage = [[] for _ in range(operation.num_of_input)]
        self.o_storage = [[] for _ in range(operation.num_of_output)]
        self.i_indexer = [None for _ in range(operation.num_of_input)]
        self.o_indexer = [None for _ in range(operation.num_of_output)]
        super().__init__(operation)

    def pre_forward_hook(self, inputs: List[torch.Tensor], **kwargs) -> list:
        for idx, input in enumerate(inputs):
            if input.numel() > 0:
                if self.i_indexer[idx] is None:
                    self.i_indexer[idx] = generate_torch_indexer(self.fetchs, input.numel())
                self.i_storage[idx].append(input.flatten()[self.i_indexer[idx]].to("cpu"))
            else:
                self.i_storage[idx].append(torch.ones([1]))
        return super().pre_forward_hook(inputs, **kwargs)

    def post_forward_hook(self, outputs: List[torch.Tensor], **kwargs) -> list:
        for idx, output in enumerate(outputs):
            if output.numel() > 0:
                if self.o_indexer[idx] is None:
                    self.o_indexer[idx] = generate_torch_indexer(self.fetchs, output.numel())
                self.o_storage[idx].append(output.flatten()[self.o_indexer[idx]].to("cpu"))
            else:
                self.o_storage[idx].append(torch.ones([1]))
        return super().post_forward_hook(outputs, **kwargs)

    def clear(self):
        self.i_storage = [[] for _ in range(self._hook_to.num_of_input)]
        self.o_storage = [[] for _ in range(self._hook_to.num_of_output)]


def graphwise_error_analyse(
    graph: BaseGraph,
    running_device: str,
    dataloader: Iterator,
    collate_fn: Callable = None,
    method: str = "snr",
    steps: int = 8,
    verbose: bool = True,
    fetchs: int = 4096,
) -> Dict[str, float]:
    """Measure the difference from a quantized graph to its dequantized graph.

    A dictionary contains output differences for all operation will be returned as a result.

        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}

    if verbose is set as True, this function will display error report at last.

    The key of the dictionary is an operation name while the value of corresponding key
        is the difference between quantized output and float output of this operation.

    Result {'operation name 1': 0.933} means that quantized graph and fp32 graph have a difference
        (or similarity, based on your measurement) 0.933 at the output tensor of 'operation name 1'.

    ATTENTION: Output difference is measured at graph-level, it includes the difference accmulated from the
        very beginning operation along to the target operation.

    Args:
        graph (BaseGraph):
            A fully quantized graph instance.

        running_device (str):
            A device string used to initialize a graph executor for the graph execution.
                if a executor was given, this parameter will be skipped.

        dataloader (Iterator):
            Test dataloader, this function will measure the output difference based on given data.

        collate_fn (Callable, optional):
            An data preprocessing function provided by user to convert data from dataloader towards
                executable format. If set as None, then no action will be taken during preprocessing.

        method (str, optional):
            A string indicates a measurement to calculate the difference of quantized output and fp32 one.
                'cosine', 'snr', and 'mse' is supported in PPQ for now.

        steps (Int, optional)
            computation steps.

    Returns:
        A dictionary contains output differences for all operation will be returned from this function.

        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}
    """
    executor = TorchExecutor(graph=graph, device=running_device)

    # find all quantable operations.
    interested_op = [
        operation
        for operation in graph.operations.values()
        if (isinstance(operation, QuantableOperation) and operation.is_computing_op)
    ]
    if len(interested_op) == 0:
        print("Oops. you got nothing to analyse.")
        return

    # set up all hooks.
    recorders, hooks, caches = {}, {}, {}
    for operation in interested_op:
        if isinstance(operation, QuantableOperation):
            if operation.num_of_output > 1:
                ppq_warning(
                    f"Operation {operation.name} has more than 1 output, "
                    "analyser will process the first output of it."
                )

            recorders[operation.name] = MeasureRecorder(measurement=method)
            hooks[operation.name] = OutputRecorder(operation=operation, fetchs=fetchs)
            caches[operation.name] = []

    # dequantize all
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.dequantize()

    # run for each quantable operations:
    for idx, batch in tqdm(
        enumerate(dataloader),
        desc="Analysing Graphwise Quantization Error(Phrase 1):",
        total=(min(len(dataloader), steps)),
    ):
        if collate_fn is not None:
            batch = collate_fn(batch)
        executor.forward(inputs=batch, hooks=hooks)

        for operation in interested_op:
            hook = hooks[operation.name]
            caches[operation.name].append(hook.pop())

        if idx >= steps:
            break

    # restore all
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.restore_quantize_state()

    # run for each quantable operations:
    for idx, batch in tqdm(
        enumerate(dataloader),
        desc="Analysing Graphwise Quantization Error(Phrase 2):",
        total=(min(len(dataloader), steps)),
    ):
        if collate_fn is not None:
            batch = collate_fn(batch)
        executor.forward(inputs=batch, hooks=hooks)

        for operation in interested_op:
            recorder = recorders[operation.name]
            hook = hooks[operation.name]
            cache = caches[operation.name]
            recorder.update(y_real=cache[idx], y_pred=hook.pop())

        if idx >= steps:
            break

    results = {}
    for operation in interested_op:
        assert isinstance(operation, QuantableOperation)
        results[operation.name] = recorders[operation.name].measure

    if verbose:
        method_str = "MEASUREMENT"
        if method == "snr":
            method_str = "NOISE:SIGNAL POWER RATIO"
        if method == "cosine":
            method_str = "COSINE SIMILARITY"
        if method == "mse":
            method_str = "MSE LOSS(UNSCALED)"
        MeasurePrinter(
            results, order="large_to_small", measure=method_str, percentage=method in {"snr", "cosine"}
        ).print()
    return results


def statistical_analyse(
    graph: BaseGraph,
    running_device: str,
    dataloader: Iterator,
    collate_fn: Callable = None,
    steps: int = 8,
) -> List[dict]:
    """It is time to do some statistical work.

    statistical_analyse is a powerful analying function
        that provides a in-depth study of your network.

    use report = statistical_analyse() to invoke this function

    The return value of this function is a collection of statistics parameters
    You are recommended to processing them with pandas

    from pandas import DataFrame
    report_df = DataFrame(report)

    Args:
        graph (BaseGraph): _description_
        running_device (str): _description_
        dataloader (Iterator): _description_
        collate_fn (Callable, optional): _description_. Defaults to None.
        steps (int, optional): _description_. Defaults to 8.

    Returns:
        Dict[str, float]: _description_
    """

    class StatisticalErrorAnalyser:
        def __init__(self, x_fp: List[torch.Tensor], x_qt: List[torch.Tensor], op: Operation, var: Variable) -> None:
            self.x_qt = torch.cat(x_qt, dim=0)
            self.x_fp = torch.cat(x_fp, dim=0)
            self.x_er = self.x_qt - self.x_fp
            self.op = op
            self.var = var

            self.num_of_samples = self.x_fp.shape[0]

        def stat(self) -> dict:
            x_er, x_fp, x_qt = self.x_er, self.x_fp, self.x_qt
            er_mean = x_er.mean().item()
            er_std = x_er.std().item()

            qt_mean = x_qt.mean().item()
            qt_std = x_qt.std().item()
            qt_min = x_qt.min().item()
            qt_max = x_qt.max().item()

            fp_mean = x_fp.mean().item()
            fp_std = x_fp.std().item()
            fp_min = x_fp.min().item()
            fp_max = x_fp.max().item()
            fp_hist = torch.histc(x_fp, bins=32, min=x_fp.min(), max=x_fp.max()).cpu().tolist()

            snr = torch_snr_error(x_qt, x_fp).item()
            cosine = torch_cosine_similarity(x_qt, x_fp).item()
            mse = torch_mean_square_error(x_qt, x_fp).item()
            return {
                "Op": "{}[{}]".format(self.op.name, self.op.type),
                "Var": self.var.name,
                "SNR": snr,
                "MSE": mse,
                "Cosine": cosine,
                "Q.MinMax": "{:.3f}, {:.3f}".format(qt_min, qt_max),
                "F.MinMax": "{:.3f}, {:.3f}".format(fp_min, fp_max),
                "F.Hist": fp_hist,
            }

    executor = TorchExecutor(graph=graph, device=running_device)
    # find all quantable operations.
    interested_op = []
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation) and operation.type not in PASSIVE_OPERATIONS:
            interested_op.append(operation)
    if len(interested_op) == 0:
        print("Oops. you got nothing to analyse.")
        return

    # set up all hooks.
    hooks, caches = {}, {}
    for operation in interested_op:
        if isinstance(operation, QuantableOperation):
            hooks[operation.name] = DetailedRecorder(operation=operation, fetchs=2048)
            caches[operation.name] = {
                "Quantized Input": [],
                "Quantized Output": [],
                "Dequantized Input": [],
                "Dequantized Output": [],
            }

    # dequantize all
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.dequantize()

    # run for each quantable operations:
    analyse_data_list = []
    for idx, batch in enumerate(dataloader):
        if collate_fn is not None:
            batch = collate_fn(batch)
        analyse_data_list.append(batch)
        if idx >= steps:
            break

    for batch in tqdm(analyse_data_list, desc="Analysing Phrase 1"):
        executor.forward(inputs=batch, hooks=hooks)

    for operation in interested_op:
        hook = hooks[operation.name]
        assert isinstance(hook, DetailedRecorder)
        caches[operation.name]["Dequantized Input"] = hook.i_storage.copy()
        caches[operation.name]["Dequantized Output"] = hook.o_storage.copy()
        hook.clear()

    # restore all
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.restore_quantize_state()

    # run for each quantable operations:
    for batch in tqdm(analyse_data_list, desc="Analysing Phrase 2"):
        executor.forward(inputs=batch, hooks=hooks)

    for operation in interested_op:
        hook = hooks[operation.name]
        assert isinstance(hook, DetailedRecorder)
        caches[operation.name]["Quantized Input"] = hook.i_storage.copy()
        caches[operation.name]["Quantized Output"] = hook.o_storage.copy()
        hook.clear()

    # analysing cache
    records = []
    visited_var = set()
    for name, record in caches.items():
        operation = graph.operations[name]
        assert isinstance(operation, Operation)
        for idx, input_var in enumerate(operation.inputs):
            if input_var in visited_var or input_var.is_parameter:
                continue
            visited_var.add(input_var)
            x_qt = record["Quantized Input"][idx]
            x_fp = record["Dequantized Input"][idx]
            if x_fp[0].dtype not in {torch.float32, torch.float64, torch.float16}:
                continue
            records.append(StatisticalErrorAnalyser(x_fp=x_fp, x_qt=x_qt, op=operation, var=input_var).stat())

        for idx, output_var in enumerate(operation.outputs):
            if output_var in visited_var:
                continue
            visited_var.add(output_var)
            x_qt = record["Quantized Output"][idx]
            x_fp = record["Dequantized Output"][idx]
            if x_fp[0].dtype not in {torch.float32, torch.float64, torch.float16}:
                continue
            records.append(StatisticalErrorAnalyser(x_fp=x_fp, x_qt=x_qt, op=operation, var=output_var).stat())

    return records
