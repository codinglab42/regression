cmake_minimum_required(VERSION 3.15)
project(RegressionProject VERSION 1.0.0 LANGUAGES CXX)

# --- Opzioni ---
option(BUILD_PYTHON_BINDINGS "Costruisci i binding Python" ON)
option(BUILD_TESTS "Costruisci i test C++" ON)

# --- Impostazioni Globali ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Obbligatorio per librerie usate in Python

# --- Output Directories ---
# Mettiamo tutto in build/lib e build/bin per ordine
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Ricerca Dipendenze ---

# 1. Trova Python (dialoga con pyenv automaticamente)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python trovato: ${Python3_EXECUTABLE}")

# 2. Trova Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 3. Trova o Scarica pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.1)
    FetchContent_MakeAvailable(pybind11)
endif()

# --- Sottodirectory ---
add_subdirectory(src)

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(pybinding)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()