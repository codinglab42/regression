cmake_minimum_required(VERSION 3.15)

# Politiche CMake per gestire meglio i warning
if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)  # Per Python3
endif()

project(RegressionProject 
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "C++ Regression Library with Python Bindings"
)

# ==================== INIZIO MODIFICHE ====================
# Forza CMake a usare il Python dell'environment attivo
if(DEFINED ENV{VIRTUAL_ENV})
    # Virtualenv (pyenv virtualenv, venv, etc.)
    set(PYTHON_FROM_ENV "$ENV{VIRTUAL_ENV}/bin/python3")
    message(STATUS "Detected virtualenv: $ENV{VIRTUAL_ENV}")
elseif(DEFINED ENV{CONDA_PREFIX})
    # Conda environment
    set(PYTHON_FROM_ENV "$ENV{CONDA_PREFIX}/bin/python3")
    message(STATUS "Detected conda environment: $ENV{CONDA_PREFIX}")
elseif(DEFINED ENV{PYENV_ROOT})
    # Pyenv shim - più robusto
    execute_process(
        COMMAND bash -c "which python3 || command -v python3"
        OUTPUT_VARIABLE PYTHON_FROM_ENV
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYTHON_FROM_ENV STREQUAL "")
        execute_process(
            COMMAND bash -c "pyenv which python3"
            OUTPUT_VARIABLE PYTHON_FROM_ENV
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    message(STATUS "Detected pyenv: $ENV{PYENV_ROOT}")
endif()

# Se abbiamo trovato un Python dall'environment, forzalo
if(PYTHON_FROM_ENV AND EXISTS ${PYTHON_FROM_ENV})
    message(STATUS "Using Python from environment: ${PYTHON_FROM_ENV}")
    set(Python3_EXECUTABLE ${PYTHON_FROM_ENV} CACHE FILEPATH "Python executable from environment" FORCE)
    
    # Test più robusto per verificare Python
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "
import sys
import sysconfig
print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'Executable: {sys.executable}')
print(f'Includes: {sysconfig.get_path(\"include\")}')
"
        OUTPUT_VARIABLE PYTHON_TEST_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYTHON_TEST_RESULT
    )
    
    if(NOT PYTHON_TEST_RESULT EQUAL 0)
        message(WARNING "Python test failed, pybind11 may not work correctly")
    else()
        message(STATUS "Python test successful:\n${PYTHON_TEST_OUTPUT}")
    endif()
endif()
# ==================== FINE MODIFICHE ====================

# Export dei comandi per l'editor
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Standard C++17 con estensioni GNU disabilitate
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Imposta build type di default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    message(STATUS "Build type not specified, defaulting to ${CMAKE_BUILD_TYPE}")
endif()

# Posizione indipendente per le librerie (PIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Directory di output con gestione configurazione
set(OUTPUT_BASE_DIR ${CMAKE_SOURCE_DIR}/output)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_BASE_DIR}/debug/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_BASE_DIR}/release/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${OUTPUT_BASE_DIR}/relwithdebinfo/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${OUTPUT_BASE_DIR}/minsizerel/bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_BASE_DIR}/debug/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_BASE_DIR}/release/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${OUTPUT_BASE_DIR}/relwithdebinfo/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${OUTPUT_BASE_DIR}/minsizerel/lib)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_BASE_DIR}/debug/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_BASE_DIR}/release/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${OUTPUT_BASE_DIR}/relwithdebinfo/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${OUTPUT_BASE_DIR}/minsizerel/lib)

# Crea le directory per tutte le configurazioni
foreach(CONFIG_TYPE debug release relwithdebinfo minsizerel)
    file(MAKE_DIRECTORY ${OUTPUT_BASE_DIR}/${CONFIG_TYPE}/bin)
    file(MAKE_DIRECTORY ${OUTPUT_BASE_DIR}/${CONFIG_TYPE}/lib)
endforeach()

# Trova Python3 (necessario per pybind11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")

# Prova a trovare pybind11 installato via pip
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    RESULT_VARIABLE PYBIND11_RESULT
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_RESULT EQUAL 0)
    message(STATUS "Found pybind11 via pip: ${PYBIND11_CMAKE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
else()
    message(STATUS "pybind11 not found via pip, trying system...")
endif()

# Trova pybind11 (prima cerca via find_package)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    find_package(pybind11 MODULE QUIET)
endif()

if(NOT pybind11_FOUND)
    # Fallback: scarica pybind11 automaticamente
    message(STATUS "Downloading pybind11...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.0
    )
    FetchContent_MakeAvailable(pybind11)
else()
    # Verifica versione pybind11
    message(STATUS "pybind11 version: ${pybind11_VERSION}")
    if(pybind11_VERSION VERSION_LESS "2.10.0")
        message(WARNING "pybind11 version ${pybind11_VERSION} may not be fully compatible")
    endif()
endif()

message(STATUS "pybind11 found: ${pybind11_FOUND}")

# Trova Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

# Opzioni di configurazione
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# Subdirectory
add_subdirectory(src)

if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    add_subdirectory(pybinding)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "Eigen3: ${Eigen3_VERSION}")
message(STATUS "pybind11: ${pybind11_VERSION}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "Output directory: ${OUTPUT_BASE_DIR}")
message(STATUS "=============================")