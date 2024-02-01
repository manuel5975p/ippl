add_executable (TestVariant TestVariant.cpp)
gtest_discover_tests(TestVariant)
include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries (
    TestVariant
    ippl
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)