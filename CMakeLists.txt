cmake_minimum_required(VERSION 3.15)
project(RegressionProject LANGUAGES CXX)

# Export dei comandi per l'editor (code)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Imposta standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Directory di output per TUTTE le librerie
set(OUTPUT_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)
file(MAKE_DIRECTORY ${OUTPUT_LIB_DIR})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_LIB_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_LIB_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_LIB_DIR})

# ricerca dipendenze
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


# Trova pybind11 Eigen3
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)



# Sub directory di progetto
add_subdirectory(src)
add_subdirectory(pybinding)


