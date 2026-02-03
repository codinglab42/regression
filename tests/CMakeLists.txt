# ============================================================================
# TEST C++
# ============================================================================
message(STATUS "🧪 Configuring C++ tests...")

# Test per Linear Regression
add_executable(test_linear test_linear.cpp)
target_link_libraries(test_linear ml_library)
add_test(NAME test_linear COMMAND test_linear)

# Test per Logistic Regression
add_executable(test_logistic test_logistic.cpp)
target_link_libraries(test_logistic ml_library)
add_test(NAME test_logistic COMMAND test_logistic)

# Test per Neural Network
add_executable(test_neural test_neural.cpp)
target_link_libraries(test_neural ml_library)
add_test(NAME test_neural COMMAND test_neural)

# Test completo
add_executable(test_all test_regression.cpp)
target_link_libraries(test_all ml_library)
add_test(NAME test_all COMMAND test_all)

message(STATUS "✅ C++ tests configured")
message(STATUS "  Available tests:")
message(STATUS "    - test_linear: Linear Regression")
message(STATUS "    - test_logistic: Logistic Regression")
message(STATUS "    - test_neural: Neural Network")
message(STATUS "    - test_all: Complete test")
message(STATUS "")
message(STATUS "  Run tests with: ctest --output-on-failure")