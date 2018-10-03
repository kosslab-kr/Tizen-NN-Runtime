function(_FlatBuffersSource_import)
  if(NOT DOWNLOAD_FLATBUFFERS)
    set(FlatBuffersSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FLATBUFFERS)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  # NOTE TensorFlow 1.9 downloads FlatBuffers from the following URL
  envoption(FLATBUFFERS_URL https://github.com/google/flatbuffers/archive/971a68110e4fc1bace10fcb6deeb189e7e1a34ce.zip)

  ExternalSource_Download("flatbuffers" ${FLATBUFFERS_URL})

  set(FlatBuffersSource_DIR ${flatbuffers_SOURCE_DIR} PARENT_SCOPE)
  set(FlatBuffersSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FlatBuffersSource_import)

_FlatBuffersSource_import()
