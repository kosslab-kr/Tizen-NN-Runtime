function(_FarmhashSource_import)
  if(NOT DOWNLOAD_FARMHASH)
    set(FarmhashSource_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT DOWNLOAD_FARMHASH)

  nnfw_include(ExternalSourceTools)
  nnfw_include(OptionTools)

  # NOTE TensorFlow 1.9 downloads farmhash from the following URL
  envoption(FARMHASH_URL https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.zip)

  ExternalSource_Download("farmhash" ${FARMHASH_URL})

  set(FarmhashSource_DIR ${farmhash_SOURCE_DIR} PARENT_SCOPE)
  set(FarmhashSource_FOUND TRUE PARENT_SCOPE)
endfunction(_FarmhashSource_import)

_FarmhashSource_import()
