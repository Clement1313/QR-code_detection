cmake_minimum_required(VERSION 3.20)
project(QR_Code)

set(CMAKE_CXX_STANDARD 20)

add_executable(qrcode_detection main.cpp detection/detection.cpp detection/preprocess.cpp image/historigram.cpp image/image_io.cpp image/image.cpp)

