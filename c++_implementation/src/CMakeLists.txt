add_library(detection SHARED detection/detection.cpp detection/preprocess.cpp detection/pipeline.cpp util/histogram.cpp util/image.cpp util/image_io.cpp evaluation/evaluation.cpp debug/debug_visualization.cpp)

target_include_directories(detection PUBLIC ${CMAKE_SOURCE_DIR}/include)
