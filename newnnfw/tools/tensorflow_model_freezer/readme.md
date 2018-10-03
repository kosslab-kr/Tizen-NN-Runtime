## What this tool is about

This tool generaes the following files:
1. __Tensorflow model__ files in *.pb, *.pbtxt
1. Tensorflow model after __freezing__ in *.pb, *.pbtxt
1. __Tensorboard__ log file to visually see the above 1 and 2.
1. __TFLITE__ file after running TOCO

By define `Test Cases`, you can easily and quickly generate files for various ranks of operands.

## How to use

- Copy `MUL_gen.py` or `TOPK_gen.py` and modify for your taste.
  - Note that `TOPK_gen.py` fails while generating TFLITE file since TOCO does not support `TOPK` oeration.

- Run `~/nnfw$ PYTHONPATH=$PYTHONPATH:./tools/tensorflow_model_freezer/ python tools/tensorflow_model_freezer/sample/MUL_gen.py  ~/temp`
  - Files will be generated under `~/temp`

## Note
- This tool is tested with Python 2.7 and 3
