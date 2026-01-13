# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

if(NOT BUILD_TESTS)
    return()
endif()

message(STATUS "Configuring tests...")

# Trova GoogleTest (opzionale)
option(USE_GOOGLETEST "Use GoogleTest framework" OFF)

if(USE_GOOGLETEST)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.13.0
    )
    FetchContent_MakeAvailable(googletest)
#else()
#    # Simple test executable
#    add_executable(test_regression test_regression.cpp)
#    target_link_libraries(test_regression PRIVATE regression::core)
#    
#    add_test(NAME regression_tests COMMAND test_regression)
endif()

# Aggiungi altri test qui
# Test principale
add_executable(test_regression test_regression.cpp)
target_link_libraries(test_regression PRIVATE regression::core)
add_test(NAME regression_tests COMMAND test_regression)

# Test specifici per linear regression
add_executable(test_linear test_linear.cpp)
target_link_libraries(test_linear PRIVATE regression::core)
add_test(NAME linear_regression_tests COMMAND test_linear)

# Test specifici per logistic regression
add_executable(test_logistic test_logistic.cpp)
target_link_libraries(test_logistic PRIVATE regression::core)
add_test(NAME logistic_regression_tests COMMAND test_logistic)

# Aggiungi directory di output per i test
set_target_properties(test_regression test_linear test_logistic
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)