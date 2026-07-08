cmake_minimum_required(VERSION 3.20)
project(QR_Code)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)
add_subdirectory(benchmark)

add_executable(qrcode_detection src/main.cpp)

target_link_libraries(qrcode_detection PRIVATE detection)

