function(_EigenSource_import)
  if(NOT DOWNLOAD_EIGEN)
    set(EigenSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_EIGEN)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  # NOTE The following URL comes from TensorFlow 1.9
  envoption(EIGEN_URL https://bitbucket.org/eigen/eigen/get/fd6845384b86.zip)

  ExternalSource_Download("eigen" ${EIGEN_URL})

  set(EigenSource_DIR ${eigen_SOURCE_DIR} PARENT_SCOPE)
  set(EigenSource_FOUND TRUE PARENT_SCOPE)
endfunction(_EigenSource_import)

_EigenSource_import()
