cmake_minimum_required(VERSION 3.1)

PROJECT(cam)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv)
include_directories($OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_CFLAGS_OTHER})
set(LIBS ${LIBS} ${OPENCV_LIBRARIES})

SET(SRCS
	src/cam.cpp
)
ADD_EXECUTABLE(cam ${SRCS})


target_link_libraries(cam ${LIBS})
