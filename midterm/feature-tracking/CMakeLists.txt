cmake_minimum_required(VERSION 3.5)
project(feature-tracking)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(feature-tracking.o ./src/main.cpp)

target_link_libraries(feature-tracking.o PUBLIC ${OpenCV_LIBS})