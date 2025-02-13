cmake_minimum_required(VERSION 3.9)

project(scut-robot-cxx)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(video_test.o ./src/armor_cpp.cpp)

add_executable(pic_test.o ./src/armor_pic.cpp)

target_link_libraries(video_test.o PUBLIC ${OpenCV_LIBS})
target_link_libraries(pic_test.o PUBLIC ${OpenCV_LIBS})