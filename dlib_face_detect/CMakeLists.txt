CMAKE_MINIMUM_REQUIRED( VERSION 3.10.1 )

PROJECT(face_test)

ADD_DEFINITIONS( -std=c++17 )

#find required opencv
FIND_PACKAGE( OpenCV REQUIRED )

#directory of opencv headers
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include" )

AUX_SOURCE_DIRECTORY( ${PROJECT_SOURCE_DIR}/ src_file )

#directory of opencv link libs
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})

ADD_EXECUTABLE( face_test ${src_file} )

TARGET_LINK_LIBRARIES( face_test ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( face_test libdlib.a )
