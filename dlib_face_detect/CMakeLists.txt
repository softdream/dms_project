CMAKE_MINIMUM_REQUIRED( VERSION 3.10.1 )

PROJECT(face_test)

# Set a default build type if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	MESSAGE("Setting build type to 'RelWithDebInfo' as none was specified.")
	SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  	# Set the possible values of build type for cmake-gui
	SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

ADD_DEFINITIONS( -std=c++17 )

#find required opencv
FIND_PACKAGE( OpenCV REQUIRED )

# find required dlib
FIND_PACKAGE( dlib REQUIRED )

#directory of opencv headers
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# directory of dlib headers
INCLUDE_DIRECTORIES( ${dlib_INCLUDE_DIRS} )

# directory of this project
INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include" )


AUX_SOURCE_DIRECTORY( ${PROJECT_SOURCE_DIR}/ src_file )

#directory of opencv link libs
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})

ADD_EXECUTABLE( face_test ${src_file} )

TARGET_LINK_LIBRARIES( face_test ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( face_test ${dlib_LIBRARIES} )
