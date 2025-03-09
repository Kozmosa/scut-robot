cmake_minimum_required(VERSION 3.5)
project(aruco-analysis)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(aruco-analysis.o ./src/main.cpp)

target_link_libraries(aruco-analysis.o PUBLIC ${OpenCV_LIBS})