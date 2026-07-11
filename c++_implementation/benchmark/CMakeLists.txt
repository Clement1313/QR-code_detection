include(FetchContent)

get_filename_component(REPO_ROOT ${CMAKE_SOURCE_DIR} DIRECTORY)

FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(detection_bench benchmark.cpp)
target_link_libraries(detection_bench PRIVATE detection benchmark::benchmark)
target_include_directories(detection_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(detection_bench PRIVATE
        TEST_DATA_DIR="${REPO_ROOT}/test_suite/test_integration")