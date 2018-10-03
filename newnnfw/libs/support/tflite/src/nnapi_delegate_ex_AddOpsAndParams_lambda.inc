// This file is included from AddOpsAndParams defined in nnapi_delegate.cc
// and contains lambda for extened implementation to original Tensorflow Lite.
    auto add_resize_bilinear_params = [&add_scalar_int32, &interpreter, &augmented_inputs](void* data) {
      auto builtin = reinterpret_cast<TfLiteResizeBilinearParams*>(data);
      if  (builtin->align_corners) {
        FATAL("Resize bilinear does not support align corners in NNAPI");
      }

      TfLiteTensor* tensor = interpreter->tensor(augmented_inputs.back());
      assert(tensor->type == kTfLiteInt32);
      assert(tensor->bytes == sizeof(int)*2);
      augmented_inputs.pop_back();

      int height = ((int*)(tensor->data.raw))[1];
      int width = ((int*)(tensor->data.raw))[0];
      add_scalar_int32(height);
      add_scalar_int32(width);
    };

    auto add_strided_slice_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteStridedSliceParams*>(data);
      add_scalar_int32(builtin->begin_mask);
      add_scalar_int32(builtin->end_mask);
      // ellipsis_mask and new_axis_mask are not supported on nn runtime
      // cf) tflite interpreter supports both operations
      if (builtin->ellipsis_mask) {
        FATAL("STRIDE_SLICE does not support ellipsis_mask in NNAPI");
      }
      if (builtin->new_axis_mask) {
        FATAL("STRIDE_SLICE does not support new_axis_mask in NNAPI");
      }
      add_scalar_int32(builtin->shrink_axis_mask);
    };

    auto add_gather_ex_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteGatherParams*>(data);
      add_scalar_int32(builtin->axis);
      if (builtin->axis != 0) {
        FATAL("GATHER does not support axis>0 in NNAPI");
      }
    };
