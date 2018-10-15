#!/usr/bin/python

import tflite.Operator
import tflite.OperatorCode
import tflite.BuiltinOperator
import tflite.ActivationFunctionType
from tensor_wrapping import Tensor
from operation import Operation
from perf_predictor import PerfPredictor


# Match enum value integer to name string
# Assumption 1: enum value is defined by old style (can be used on python 2)
# Assumption 2: when class define enum value, only constant value is defined and methods are not defined
# Assumption 3: only integer value is set by constant definition
def BuildEnumClassStrMap(obj):
    ret = {}
    for fieldName in dir(obj):
        if (not fieldName.startswith('_')):
            fieldValue = getattr(obj, fieldName)
            if (isinstance(fieldValue, (int))):
                ret[fieldValue] = fieldName
    return ret


class EnumStrMaps():
    BuiltinOpcode = BuildEnumClassStrMap(tflite.BuiltinOperator.BuiltinOperator())
    ActivationFunctionType = BuildEnumClassStrMap(
        tflite.ActivationFunctionType.ActivationFunctionType())
    BuiltinOptions = BuildEnumClassStrMap(tflite.BuiltinOptions.BuiltinOptions())


def GetStrTensorIndex(tensors):
    return_string = "["
    for idx in range(len(tensors)):
        if idx != 0:
            return_string += ", "
        return_string += str(tensors[idx].tensor_idx)
    return_string += "]"
    return return_string


def GetAttribute(o, *args):
    import functools
    return functools.reduce(getattr, args, o)


def BuildBuiltinOptionGen():
    bo_gen = {}
    for val_enum in EnumStrMaps.BuiltinOptions:
        val_str = EnumStrMaps.BuiltinOptions[val_enum]
        try:
            # Dynamically import Builtin Option classes
            # 0 (NONE) is the only exception that does not have no corresponding flatbuffer-generated class
            module = __import__("tflite." + val_str)
            bo_gen[val_enum] = GetAttribute(module, val_str, val_str)
        except ImportError as e:
            assert val_enum == 0 and val_str == "NONE"
    return bo_gen


class OptionLoader:
    builtinOptionGen = BuildBuiltinOptionGen()

    @staticmethod
    def GetBuiltinOptions(options_type, options_table):
        options = OptionLoader.builtinOptionGen[options_type]()
        options.Init(options_table.Bytes, options_table.Pos)
        return options


class Operator(object):
    def __init__(self, operator_idx, tf_operator, input_tensors, output_tensors,
                 opcode_str):
        self.operator_idx = operator_idx
        self.tf_operator = tf_operator
        self.inputs = input_tensors
        self.outputs = output_tensors
        self.opcode_str = opcode_str
        self.operation = Operation(self.tf_operator, self.opcode_str, self.inputs,
                                   self.outputs)

    def PrintInfo(self, perf_predictor=None):
        # total instruction num
        instrs = "{:,}".format(
            self.operation.TotalInstrNum()) if self.operation.can_compute else "???"

        # total operation cycles
        cycles = "{:,}".format(
            (perf_predictor.PredictCycles(self.operation)
             )) if self.operation.can_compute and perf_predictor != None else "???"

        print("Operator {0}: {1} (instrs: {2}, cycls: {3})".format(
            self.operator_idx, self.opcode_str, instrs, cycles))

        self.PrintOptionInfo()

        print("\tInput Tensors" + GetStrTensorIndex(self.inputs))
        for tensor in self.inputs:
            tensor.PrintInfo("\t\t")
        print("\tOutput Tensors" + GetStrTensorIndex(self.outputs))
        for tensor in self.outputs:
            tensor.PrintInfo("\t\t")

    def PrintOptionInfo(self):
        # FIXME: workaround for ops such as custom
        try:
            options = OptionLoader.GetBuiltinOptions(
                self.tf_operator.BuiltinOptionsType(), self.tf_operator.BuiltinOptions())
        except KeyError:
            return

        # fused activation function
        try:
            activation_code = options.FusedActivationFunction()
            fused_activation = EnumStrMaps.ActivationFunctionType[activation_code]
            print("\tFused Activation: " + fused_activation)
        except AttributeError:
            # This operator does not support FusedActivationFunction
            pass
