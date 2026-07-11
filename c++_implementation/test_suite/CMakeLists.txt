include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(detection_tests
        test_suite.cpp)

target_link_libraries(detection_tests PRIVATE detection GTest::gtest_main)
target_include_directories(detection_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(detection_tests)