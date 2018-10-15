#!/usr/bin/python

import tflite.Tensor
import tflite.TensorType

TensorTypeList = {}


def SetTensorTypeStr():
    tensorTypeObj = tflite.TensorType.TensorType()

    for fieldName in dir(tensorTypeObj):
        if (not fieldName.startswith('_')):
            fieldValue = getattr(tensorTypeObj, fieldName)
            if (isinstance(fieldValue, (int))):
                TensorTypeList[fieldValue] = fieldName


class Tensor(object):
    def __init__(self, tensor_idx, tf_tensor, tf_buffer):
        self.tensor_idx = tensor_idx
        self.tf_tensor = tf_tensor
        self.tf_buffer = tf_buffer

    def PrintInfo(self, depth_str=""):
        print_str = ""
        if self.tensor_idx < 0:
            print_str = "Tensor {0:4}".format(self.tensor_idx)
        else:
            buffer_idx = self.tf_tensor.Buffer()
            isEmpty = "Filled"
            if (self.tf_buffer.DataLength() == 0):
                isEmpty = " Empty"
            shape_str = self.GetShapeString()
            type_name = TensorTypeList[self.tf_tensor.Type()]

            shape_name = ""
            if self.tf_tensor.Name() != 0:
                shape_name = self.tf_tensor.Name()

            print_str = "Tensor {0:4} : buffer {1:4} | {2} | {3:7} | Shape {4} ({5})".format(
                self.tensor_idx, buffer_idx, isEmpty, type_name, shape_str, shape_name)
        print(depth_str + print_str)

    def GetShapeString(self):
        if self.tf_tensor.ShapeLength() == 0:
            return "Scalar"
        return_string = "["
        for shape_idx in range(self.tf_tensor.ShapeLength()):
            if (shape_idx != 0):
                return_string += ", "
            return_string += str(self.tf_tensor.Shape(shape_idx))
        return_string += "]"
        return return_string
