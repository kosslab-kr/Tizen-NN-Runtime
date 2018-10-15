macro(add_extdirectory DIR TAG)
  add_subdirectory(${DIR} "${CMAKE_BINARY_DIR}/externals/${TAG}")
endmacro(add_extdirectory)

set(ExternalProjectTools_FOUND TRUE)
