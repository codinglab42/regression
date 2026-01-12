
# === PYBIND11 INSTALLATO VIA PIP / PYENV ===
# Dì a CMake quale Python usare
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Recupera il path ai file CMake di pybind11 (installato via pip)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")


# Trova pybind11
find_package(pybind11 REQUIRED)

# Crea modulo Python
pybind11_add_module(pyregression
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bindRegression.cpp
)

target_include_directories(pyregression 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Link alla libreria core e pybind11
target_link_libraries(pyregression 
    PRIVATE 
    regression_core
)



