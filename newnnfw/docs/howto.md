## Build Requires

If you are building this project, then the following modules must be installed on your system:

- CMake
- Boost C++ libraries

```
$ sudo apt-get install cmake libboost-all-dev
```

## How to use (simple) NNAPI Binding

This repo provides a T/F Lite Model loader(named ``tflite_run``), and simple NNAPI binding.

Let's type the following commands, and see what happens!
```
$ make install
$ USE_NNAPI=1 LD_LIBRARY_PATH="$(pwd)/Product/obj/runtimes/logging:$(pwd)/Product/out/lib" Product/out/bin/tflite_run [T/F Lite Flatbuffer Model Path]
```

## How to get pre-built T/F Lite Flatbuffer models?
Google provides several pre-built T/F Lite models. Please check [this article](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md)


## Build How-to
- [Cross building for ARM](howto/CrossBuildForArm.md)
- [Cross building for AARCH64](howto/CrossBuildForAarch64.md)
- [Build using prebuilt docker image](howto/HowToUseDockerImage.md)


## Other how-to documents
- [Building TensorFlow and TOCO from source](howto/BuildTFfromSource.md)
- [How to setup XU3 with Ubuntu 16.04](howto/device/xu3_ubuntu.md)
- [How to setup XU4 with Ubuntu 16.04](howto/device/xu4_ubuntu.md)
- [How to add unittest using gtest](howto/HowToAddUnittest.md)
