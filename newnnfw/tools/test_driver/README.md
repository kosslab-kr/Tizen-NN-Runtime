# How the test driver works

## Unittest
- There are two kinds of unittest:
    - Kernel ACL
    - Runtime
- Related file : `run_unittest.sh`
- Usage :
```
$ ./tools/test_driver/test_driver.sh \
    --artifactpath=. \
    --unittest
```
- The `run_unittest.sh` usage :

```
$ LD_LIBRARY_PATH=Product/out/lib \
    ./tools/test_driver/run_unittest.sh \
    --reportdir=report \
    --unittestdir=Product/out/unittest
```

### Kernel ACL Unittest
- Test whether the various operations are performed successfully and whether the output and the expected value are the same.
- TC location : `libs/kernel/acl/src/`

### Runtime Unittest
- Test whether the expected value and the actual output value are the same when the model is configured, compiled and executed.
- TC location : `runtimes/tests/neural_networks_test/`

## Framework test
- Execute the **tflite model** using the given **driver**.
- There is a TC directory for each model, and a `config.sh` file exists in each TC directory.
- When `run_test.sh`, refer to the **tflite model** information in `config.sh`, download the file, and run the **tflite model** with the given **driver**.
- Related files : `run_test.sh` and `run_frameworktest.sh`
- TC location :
    - `tests/framework/tests/` : Config directory for TC
    - `tests/framework/cache/` : TC (Downloaded tflite model files)

### Run tflite_run with various tflite models
- Driver : `tflite_run`
- Driver source location : `tools/tflite_run/`
- Usage :
```
$ ./tools/test_driver/test_driver.sh \
    --artifactpath=. \
    --frameworktest
```
- Related pages : [tflite_run](https://github.sec.samsung.net/STAR/nnfw/tree/master/tools/tflite_run)

### Run nnapi_test with various tflite models
- `nnapi_test` runs tflite in two ways and compares the result:
    1. tflite interpreter
    2. `libneuralnetworks.so`, which could be PureACL or NNAPI depending on `--ldlibrarypath`(`LD_LIBRARY_PATH`)
- Driver : `nnapi_test`
- Driver source location : `tools/nnapi_test/`
- Usage :
```
$ ./tools/test_driver/test_driver.sh \
    --artifactpath=. \
    --verification .
```

