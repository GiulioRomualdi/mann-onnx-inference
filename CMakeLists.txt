cmake_minimum_required(VERSION 3.16.0)

## MAIN project
project(mann-onnx-inference
  VERSION 0.1.0)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(onnxruntime REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} onnxruntime::onnxruntime)
