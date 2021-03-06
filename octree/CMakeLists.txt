cmake_minimum_required(VERSION 3.16)
project(OpenCV_octree)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
if(OpenCV_FOUND)
    message(STATUS "The OpenCV lib is found!")
    message(${OpenCV_LIBS})
endif()

set(SOURCE_FILES ./src/main.cpp ./src/octree.h ./src/octree.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )