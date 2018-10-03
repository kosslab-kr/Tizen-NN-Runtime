function(_TensorFlowSource_import)
  if(NOT DOWNLOAD_TENSORFLOW)
    set(TensorFlowSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_TENSORFLOW)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  envoption(TENSORFLOW_URL https://github.com/tensorflow/tensorflow/archive/v1.9.0.zip)

  ExternalSource_Download("tensorflow" ${TENSORFLOW_URL})

  set(TensorFlowSource_DIR ${tensorflow_SOURCE_DIR} PARENT_SCOPE)
  set(TensorFlowSource_FOUND TRUE PARENT_SCOPE)
endfunction(_TensorFlowSource_import)

_TensorFlowSource_import()
