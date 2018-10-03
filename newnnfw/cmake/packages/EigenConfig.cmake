function(_Eigen_import)
  nnfw_find_package(EigenSource QUIET)

  if(NOT EigenSource_FOUND)
    set(Eigen_FOUND FALSE PARENT_SCOPE)
    return()
  endif(NOT EigenSource_FOUND)

  if(NOT TARGET eigen)
    add_library(eigen INTERFACE)
    target_include_directories(eigen INTERFACE "${EigenSource_DIR}")
  endif(NOT TARGET eigen)

  set(Eigen_FOUND TRUE PARENT_SCOPE)
endfunction(_Eigen_import)

_Eigen_import()
