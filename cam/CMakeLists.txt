cmake_minimum_required(VERSION 3.1)

PROJECT(nn_cam)

SET(INSTALL_EXEC_PREFIX "${INSTALL_PREFIX}/bin")

INCLUDE(FindPkgConfig)
pkg_check_modules(APP_PKGS REQUIRED opencv)

FOREACH (flag ${APP_PKGS_CFLAGS})
    SET(EXTRA_CFLAGS "${EXTRA_CFLAGS} ${flag}")
ENDFOREACH(flag)

SET(SRCS
        src/bitmap_helpers.cc
        src/label_image.cc
)

ADD_EXECUTABLE(cam ${SRCS})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${APP_PKGS_LDFLAGS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} tensorflow-lite pthread dl)

INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${INSTALL_EXEC_PREFIX})
