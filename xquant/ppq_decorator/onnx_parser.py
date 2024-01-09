from typing import Any, Dict, Iterable, List, Union

import onnx
from onnx import helper, mapping, numpy_helper
from ppq.core import DEFAULT_OPSET_DOMAIN, DEFAULT_OPSET_VERSION, GRAPH_OPSET_ATTRIB, NetworkFramework, is_file_exist
from ppq.IR import BaseGraph, GraphBuilder, Operation, Opset, Variable
from ppq.parser.onnx_parser import OnnxParser


class OnnxParserDecorator(OnnxParser):
    def build(self, file_path_or_proto: Union[str, onnx.ModelProto]) -> BaseGraph:
        _rand_seed = 0  # used for name generation.
        if isinstance(file_path_or_proto, str):
            if not is_file_exist(file_path_or_proto):
                raise FileNotFoundError(f"file {file_path_or_proto} does not exist, or it is a directory.")
            model_pb = onnx.load(file_path_or_proto)
        elif isinstance(file_path_or_proto, onnx.ModelProto):
            model_pb = file_path_or_proto
        else:
            raise TypeError("type for file_path_or_proto {} error".format(type(file_path_or_proto)))

        opsets = model_pb.opset_import

        assert isinstance(
            model_pb, onnx.ModelProto
        ), f"onnx load failed, only ProtoBuffer object is expected here, while {type(model_pb)} is loaded."
        graph_pb = model_pb.graph
        graph = BaseGraph(name=graph_pb.name, built_from=NetworkFramework.ONNX)
        graph._detail[GRAPH_OPSET_ATTRIB] = self.convert_opsets_to_str(opsets)
        graph._detail["ir_version"] = model_pb.ir_version

        onnx_import_opset = DEFAULT_OPSET_VERSION
        for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
            if opset["domain"] == DEFAULT_OPSET_DOMAIN or opset["domain"] == "":
                onnx_import_opset = opset["version"]
                break

        # a temporary storage for operation's inputs and outputs
        op_inputs_dict, op_outputs_dict = {}, {}
        for node in graph_pb.node:
            op_name = node.name
            if len(op_name) == 0:  # some operation do not have a name, we just generate one.
                op_name = "generated_name_" + str(_rand_seed)
                _rand_seed += 1

            if op_name in graph.operations:
                raise KeyError(f"Duplicated operation {op_name} was found.")

            graph.operations[op_name] = Operation(
                name=op_name,
                op_type=node.op_type,
                attributes={item.name: helper.get_attribute_value(item) for item in node.attribute},
                opset=Opset(domain=DEFAULT_OPSET_DOMAIN, version=onnx_import_opset),
            )
            op_inputs_dict[op_name] = [var_name for var_name in node.input]
            op_outputs_dict[op_name] = [var_name for var_name in node.output]

        initializer = {}
        for item in graph_pb.initializer:
            init_name = item.name
            value = numpy_helper.to_array(item)
            initializer[init_name] = value

        inputs = [item.name for item in graph_pb.input]
        outputs = [item.name for item in graph_pb.output]
        graph._detail["pb_inputs"] = inputs
        graph._detail["pb_outputs"] = outputs
        graph = self.build_variables(
            graph, graph_inputs=inputs, graph_outputs=outputs, op_inputs=op_inputs_dict, op_outputs=op_outputs_dict
        )
        graph = self.initialize_params(graph, initializer)
        self.de_inplace(graph)
        return self.refine_graph(graph)
