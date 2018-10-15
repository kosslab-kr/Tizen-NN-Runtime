Build
```
CROSS_BUILD=1 TARGET_ARCH=armv7l make
CROSS_BUILD=1 TARGET_ARCH=armv7l make install
```

Test
```
USE_NNAPI=1 \
LD_LIBRARY_PATH="$(pwd)/Product/out/lib:$(pwd)/Product/obj/contrib/bindacl" \
Product/out/bin/tflite_run \
[T/F Lite Flatbuffer Model Path]
```
