project(detect)
 
cmake_minimum_required(VERSION 2.8.10)
set(OpenCV_DIR "/Volumes/HD/Applications/opencv-4.2.0/release")
find_package(OpenCV)#REQUIRED core highgui imgproc imgcodecs
find_package(OpenMP)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
 
SET(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb -D DEBUG")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
 
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
link_directories(
	/usr/local/lib
)
 
#add_executable(detect detect.cpp)
add_executable(detect detect_img.cpp read_cfg.cpp)
target_link_libraries(detect ${OpenCV_LIBS} png )

