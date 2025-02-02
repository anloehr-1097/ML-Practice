cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example)


list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/MachineLearning/Cpp/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/home/libtorch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(example example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 17)
