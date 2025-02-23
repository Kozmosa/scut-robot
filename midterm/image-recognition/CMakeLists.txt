cmake_minimum_required(VERSION 3.5)
project(image-recognition)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(image-recognition.o ./src/main.cpp)

target_link_libraries(image-recognition.o PUBLIC ${OpenCV_LIBS})