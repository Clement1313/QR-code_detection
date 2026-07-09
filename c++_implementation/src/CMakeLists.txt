add_library(detection SHARED detection/detection.cpp detection/preprocess.cpp util/historigram.cpp util/image.cpp util/image_io.cpp)

target_include_directories(detection PUBLIC ${CMAKE_SOURCE_DIR}/include)
