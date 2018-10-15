#!/usr/bin/python

import tflite.Model
import tflite.SubGraph
import tflite.Operator
import tflite.OperatorCode
import tflite.BuiltinOperator
from operator_wrapping import Operator, EnumStrMaps
from tensor_wrapping import Tensor, SetTensorTypeStr
from operation import Operation


class OperatorParser(object):
    def __init__(self, tf_model, tf_subgraph, perf_predictor=None):
        self.tf_model = tf_model
        self.tf_subgraph = tf_subgraph
        self.perf_predictor = perf_predictor
        self.operators_in_list = list()
        self.operators_per_type = dict()
        # Tensor type string table
        SetTensorTypeStr()

    def Parse(self):
        for operator_idx in range(self.tf_subgraph.OperatorsLength()):
            tf_operator = self.tf_subgraph.Operators(operator_idx)
            opcode_str = self.GetOpcodeStr(tf_operator)
            input_tensors = self.GetInputTensors(tf_operator)
            output_tensors = self.GetOutputTensors(tf_operator)

            op = Operator(operator_idx, tf_operator, input_tensors, output_tensors,
                          opcode_str)
            self.AppendOperator(op)

    def GetOpcodeStr(self, tf_operator):
        opcode_list_idx = tf_operator.OpcodeIndex()
        opcode_id = self.tf_model.OperatorCodes(opcode_list_idx).BuiltinCode()
        opcode_str = EnumStrMaps.BuiltinOpcode[opcode_id]
        if opcode_id == 32:
            # Custom operator
            custom_operator = self.tf_model.OperatorCodes(tf_operator.OpcodeIndex())
            custom_op_name = custom_operator.CustomCode().decode('utf-8')
            opcode_str = opcode_str + "(" + custom_op_name + ")"
        return opcode_str

    def GetInputTensors(self, tf_operator):
        operator_inputs = tf_operator.InputsAsNumpy()
        return self.GetTensors(operator_inputs)

    def GetOutputTensors(self, tf_operator):
        operator_outputs = tf_operator.OutputsAsNumpy()
        return self.GetTensors(operator_outputs)

    def GetTensors(self, tf_tensors_index):
        return_list = list()
        for tensor_idx in tf_tensors_index:
            if (tensor_idx < 0):
                return_list.append(Tensor(tensor_idx, 0, 0))
                continue
            tf_tensor = self.tf_subgraph.Tensors(tensor_idx)
            buffer_idx = tf_tensor.Buffer()
            tf_buffer = self.tf_model.Buffers(buffer_idx)
            return_list.append(Tensor(tensor_idx, tf_tensor, tf_buffer))
        return return_list

    def AppendOperator(self, operator):
        self.operators_in_list.append(operator)

        opcode_str = operator.opcode_str
        if opcode_str not in self.operators_per_type:
            self.operators_per_type[opcode_str] = list()
        self.operators_per_type[opcode_str].append(operator)

    def PrintAll(self):
        print('')
        self.PrintAllOperatorsInList()
        print('')
        self.PrintAllTypesInfo()
        print('')

    def PrintAllOperatorsInList(self):
        for operator in self.operators_in_list:
            operator.PrintInfo(self.perf_predictor)
            print('')

    def PrintAllTypesInfo(self):
        print("Number of all operator types: {0}".format(len(self.operators_per_type)))

        # number of instructions of all operator types
        total_instrs = 0

        # (a string of the operator type, a list of operators which are the same operator type)
        for type_str, oper_list in self.operators_per_type.items():
            # this operator type can be computed?
            can_compute = oper_list[0].operation.can_compute

            # number of occurrence of this operator type
            occur = len(oper_list)

            # total number of instructions of the same operator types
            if can_compute:
                instrs = sum(operator.operation.TotalInstrNum() for operator in oper_list)
                total_instrs = total_instrs + instrs
                instrs = "{:,}".format(instrs)
            else:
                instrs = "???"

            print("\t{type_str:38}: {occur:4} \t (instrs: {instrs})".format(
                type_str=type_str, occur=occur, instrs=instrs))

        total_instrs = "{:,}".format(total_instrs)
        print("{0:46}: {1:4} \t (total instrs: {2})".format("Number of all operators",
                                                            len(self.operators_in_list),
                                                            total_instrs))
