# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/build

# Include any dependencies generated for this target.
include CMakeFiles/body_detect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/body_detect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/body_detect.dir/flags.make

CMakeFiles/body_detect.dir/src/body_detect.cpp.o: CMakeFiles/body_detect.dir/flags.make
CMakeFiles/body_detect.dir/src/body_detect.cpp.o: ../src/body_detect.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/body_detect.dir/src/body_detect.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/body_detect.dir/src/body_detect.cpp.o -c /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/src/body_detect.cpp

CMakeFiles/body_detect.dir/src/body_detect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/body_detect.dir/src/body_detect.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/src/body_detect.cpp > CMakeFiles/body_detect.dir/src/body_detect.cpp.i

CMakeFiles/body_detect.dir/src/body_detect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/body_detect.dir/src/body_detect.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/src/body_detect.cpp -o CMakeFiles/body_detect.dir/src/body_detect.cpp.s

CMakeFiles/body_detect.dir/src/image_utils.cpp.o: CMakeFiles/body_detect.dir/flags.make
CMakeFiles/body_detect.dir/src/image_utils.cpp.o: ../src/image_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/body_detect.dir/src/image_utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/body_detect.dir/src/image_utils.cpp.o -c /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/src/image_utils.cpp

CMakeFiles/body_detect.dir/src/image_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/body_detect.dir/src/image_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/src/image_utils.cpp > CMakeFiles/body_detect.dir/src/image_utils.cpp.i

CMakeFiles/body_detect.dir/src/image_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/body_detect.dir/src/image_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/src/image_utils.cpp -o CMakeFiles/body_detect.dir/src/image_utils.cpp.s

CMakeFiles/body_detect.dir/src/main.cpp.o: CMakeFiles/body_detect.dir/flags.make
CMakeFiles/body_detect.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/body_detect.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/body_detect.dir/src/main.cpp.o -c /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/src/main.cpp

CMakeFiles/body_detect.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/body_detect.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/src/main.cpp > CMakeFiles/body_detect.dir/src/main.cpp.i

CMakeFiles/body_detect.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/body_detect.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/src/main.cpp -o CMakeFiles/body_detect.dir/src/main.cpp.s

# Object files for target body_detect
body_detect_OBJECTS = \
"CMakeFiles/body_detect.dir/src/body_detect.cpp.o" \
"CMakeFiles/body_detect.dir/src/image_utils.cpp.o" \
"CMakeFiles/body_detect.dir/src/main.cpp.o"

# External object files for target body_detect
body_detect_EXTERNAL_OBJECTS =

body_detect: CMakeFiles/body_detect.dir/src/body_detect.cpp.o
body_detect: CMakeFiles/body_detect.dir/src/image_utils.cpp.o
body_detect: CMakeFiles/body_detect.dir/src/main.cpp.o
body_detect: CMakeFiles/body_detect.dir/build.make
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_viz.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.2.0
body_detect: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.2.0
body_detect: CMakeFiles/body_detect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable body_detect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/body_detect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/body_detect.dir/build: body_detect

.PHONY : CMakeFiles/body_detect.dir/build

CMakeFiles/body_detect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/body_detect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/body_detect.dir/clean

CMakeFiles/body_detect.dir/depend:
	cd /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/build /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/build /home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/build/CMakeFiles/body_detect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/body_detect.dir/depend
