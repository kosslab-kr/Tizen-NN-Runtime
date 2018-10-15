function(ExternalSource_Download PREFIX URL)
  get_filename_component(FILENAME ${URL} NAME)

  set(CACHE_DIR "${CMAKE_SOURCE_DIR}/externals")
  set(OUT_DIR "${CACHE_DIR}/${PREFIX}")
  set(TMP_DIR "${CACHE_DIR}/${PREFIX}-tmp")

  set(DOWNLOAD_PATH "${CACHE_DIR}/${PREFIX}-${FILENAME}")
  set(STAMP_PATH "${CACHE_DIR}/${PREFIX}.stamp")

  if(NOT EXISTS "${CACHE_DIR}")
    file(MAKE_DIRECTORY "${CACHE_DIR}")
  endif(NOT EXISTS "${CACHE_DIR}")

  if(NOT EXISTS "${STAMP_PATH}")
    file(REMOVE_RECURSE "${OUT_DIR}")
    file(REMOVE_RECURSE "${TMP_DIR}")

    file(MAKE_DIRECTORY "${TMP_DIR}")

    message("-- Download ${PREFIX} from ${URL}")
    file(DOWNLOAD ${URL} "${DOWNLOAD_PATH}")
    message("-- Download ${PREFIX} from ${URL} - done")

    message("-- Extract ${PREFIX}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xfz "${DOWNLOAD_PATH}"
                    WORKING_DIRECTORY "${TMP_DIR}")
    file(REMOVE "${DOWNLOAD_PATH}")
    message("-- Extract ${PREFIX} - done")

    message("-- Cleanup ${PREFIX}")
    file(GLOB contents "${TMP_DIR}/*")
    list(LENGTH contents n)
    if(NOT n EQUAL 1 OR NOT IS_DIRECTORY "${contents}")
      set(contents "${TMP_DIR}")
    endif()

    get_filename_component(contents ${contents} ABSOLUTE)

    file(RENAME ${contents} "${OUT_DIR}")
    file(REMOVE_RECURSE "${TMP_DIR}")
    file(WRITE "${STAMP_PATH}" "${URL}")
    message("-- Cleanup ${PREFIX} - done")
  endif()

  set(${PREFIX}_SOURCE_DIR "${OUT_DIR}" PARENT_SCOPE)
endfunction(ExternalSource_Download)

set(ExternalSourceTools_FOUND TRUE)
