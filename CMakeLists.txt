cmake_minimum_required(VERSION 3.10)
include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")
set(EXECUTABLE_OUTPUT_PATH "bin")
add_executable(a src/main.cpp)
target_link_libraries(a cudart)
target_link_libraries(a cusparse)