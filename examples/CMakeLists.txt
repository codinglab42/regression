# examples/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

if(NOT BUILD_EXAMPLES)
    return()
endif()

message(STATUS "Configuring examples...")

# Esempio 1: Uso base
add_executable(example_linear example_linear.cpp)
target_link_libraries(example_linear PRIVATE regression::core)

# Esempio 2: Uso avanzato
add_executable(example_logistic example_logistic.cpp)
target_link_libraries(example_logistic PRIVATE regression::core)

# Esempio 3: Python binding demo
#if(BUILD_PYTHON_BINDINGS)
#    add_executable(example_python example_python.cpp)
#    target_link_libraries(example_python PRIVATE regression::core)
#endif()