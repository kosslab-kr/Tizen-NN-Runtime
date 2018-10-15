#!/usr/bin/python
import os
import sys
import numpy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tflite'))
flatbuffersPath = '../../externals/flatbuffers'
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), flatbuffersPath + '/python'))

import flatbuffers
import tflite.Model
import tflite.SubGraph
import argparse
from operator_parser import OperatorParser
from perf_predictor import PerfPredictor


class TFLiteModelFileParser(object):
    def __init__(self, args):
        # Read flatbuffer file descriptor using argument
        self.tflite_file = args.input_file

        # Set print level (0 ~ 2)
        # TODO: print information based on level
        self.print_level = args.verbose
        if (args.verbose > 2):
            self.print_level = 2
        if (args.verbose < 0):
            self.print_level = 0

        # Set tensor index list to print information
        # TODO:
        #   Print tensors in list only
        #   Print all tensors if argument used and not specified index number
        if (args.tensor != None):
            if (len(args.tensor) == 0):
                self.print_all_tensor = True
            else:
                self.print_all_tensor = False
                self.print_tensor_index = []

                for tensor_index in args.tensor:
                    self.print_tensor_index.append(int(tensor_index))

        # Set operator index list to print information
        # TODO:
        #   Print operators in list only
        #   Print all operators if argument used and not specified index number
        if (args.operator != None):
            if (len(args.operator) == 0):
                self.print_all_oeprator = True
            else:
                self.print_all_oeprator = False
                self.print_operator_index = []

                for operator_index in args.operator:
                    self.print_operator_index.append(int(operator_index))

    def main(self):
        # Generate Model: top structure of tflite model file
        buf = self.tflite_file.read()
        buf = bytearray(buf)
        tf_model = tflite.Model.Model.GetRootAsModel(buf, 0)

        # Model file can have many models
        # 1st subgraph is main model
        model_name = "Main model"
        for subgraph_index in range(tf_model.SubgraphsLength()):
            tf_subgraph = tf_model.Subgraphs(subgraph_index)
            if (subgraph_index != 0):
                model_name = "Model #" + str(subgraph_index)

            print("[" + model_name + "]\n")

            # Model inputs & outputs
            model_inputs = tf_subgraph.InputsAsNumpy()
            model_outputs = tf_subgraph.OutputsAsNumpy()

            print(model_name + " input tensors: " + str(model_inputs))
            print(model_name + " output tensors: " + str(model_outputs))

            # Parse Operators and print all of operators
            op_parser = OperatorParser(tf_model, tf_subgraph, PerfPredictor())
            op_parser.Parse()
            op_parser.PrintAll()


if __name__ == '__main__':
    # Define argument and read
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "input_file", type=argparse.FileType('rb'), help="tflite file to read")
    arg_parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help="set print level (0~2, default: 0)")
    arg_parser.add_argument(
        '-t', '--tensor', nargs='*', help="tensor ID to print information (default: all)")
    arg_parser.add_argument(
        '-o',
        '--operator',
        nargs='*',
        help="operator ID to print information (default: all)")
    args = arg_parser.parse_args()

    # Call main function
    TFLiteModelFileParser(args).main()
