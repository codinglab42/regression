cmake_minimum_required(VERSION 3.15)
project(ML_Library VERSION 2.0.0 LANGUAGES CXX)

# ============================================================================
# OPZIONI DI CONFIGURAZIONE
# ============================================================================
option(BUILD_PYTHON_BINDINGS "Costruisci i binding Python" ON)
option(BUILD_TESTS "Costruisci i test C++" ON)
option(BUILD_EXAMPLES "Costruisci esempi" OFF)
option(USE_OPENMP "Usa OpenMP per parallelizzazione" OFF)
option(VERBOSE_OUTPUT "Output verboso di configurazione" ON)
option(BUILD_SHARED_LIBS "Build shared libraries instead of static" OFF)

# ============================================================================
# IMPOSTAZIONI GLOBALI
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Directory di output
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================================================
# RICERCA DIPENDENZE
# ============================================================================
if(VERBOSE_OUTPUT)
    message(STATUS "========================================")
    message(STATUS "🔍 Ricerca dipendenze...")
endif()

# 1. Python3 (compatibile con pyenv)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(VERBOSE_OUTPUT)
    message(STATUS "✅ Python trovato: ${Python3_EXECUTABLE}")
    message(STATUS "   Version: ${Python3_VERSION}")
    
    # Info pyenv se presente
    if(DEFINED ENV{PYENV_ROOT})
        message(STATUS "   Pyenv: $ENV{PYENV_ROOT}")
    endif()
endif()

# 2. Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(VERBOSE_OUTPUT)
    message(STATUS "✅ Eigen3 trovato: ${EIGEN3_VERSION}")
endif()

# 3. pybind11 (con fallback a FetchContent)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    if(VERBOSE_OUTPUT)
        message(STATUS "📦 pybind11 non trovato, scaricando...")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()
if(VERBOSE_OUTPUT)
    message(STATUS "✅ pybind11: ${pybind11_VERSION}")
endif()

# 4. OpenMP opzionale
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        if(VERBOSE_OUTPUT)
            message(STATUS "✅ OpenMP abilitato")
        endif()
    else()
        message(WARNING "⚠️  OpenMP non trovato")
    endif()
endif()

# ============================================================================
# SOTTODIRECTORY
# ============================================================================
if(VERBOSE_OUTPUT)
    message(STATUS "========================================")
    message(STATUS "🏗️  Configurazione sottodirectory...")
endif()

# Libreria principale
add_subdirectory(src)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    if(VERBOSE_OUTPUT)
        message(STATUS "🐍 Building Python bindings...")
    endif()
    add_subdirectory(pybinding)
endif()

# Test
if(BUILD_TESTS)
    if(VERBOSE_OUTPUT)
        message(STATUS "🧪 Building tests...")
    endif()
    enable_testing()
    add_subdirectory(tests)
endif()

# Esempi
if(BUILD_EXAMPLES)
    if(VERBOSE_OUTPUT)
        message(STATUS "📚 Building examples...")
    endif()
    add_subdirectory(examples)
endif()

# ============================================================================
# INSTALLAZIONE
# ============================================================================
install(DIRECTORY include/ DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================
if(VERBOSE_OUTPUT)
    message(STATUS "")
    message(STATUS "========================================")
    message(STATUS "✅ Configuration Complete")
    message(STATUS "========================================")
    message(STATUS "Project:      ML Library v${PROJECT_VERSION}")
    message(STATUS "Build type:   ${CMAKE_BUILD_TYPE}")
    message(STATUS "")
    message(STATUS "Features:")
    message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
    message(STATUS "  Tests:           ${BUILD_TESTS}")
    message(STATUS "  Examples:        ${BUILD_EXAMPLES}")
    message(STATUS "  OpenMP:          ${USE_OPENMP}")
    message(STATUS "  Shared libs:     ${BUILD_SHARED_LIBS}")
    message(STATUS "")
    message(STATUS "Dependencies:")
    message(STATUS "  Python:          ${Python3_VERSION}")
    message(STATUS "  Eigen3:          ${EIGEN3_VERSION}")
    message(STATUS "  pybind11:        ${pybind11_VERSION}")
    message(STATUS "")
    message(STATUS "Next commands:")
    message(STATUS "  make -j\$(nproc)")
    if(BUILD_PYTHON_BINDINGS)
        message(STATUS "  make test_python_module")
    endif()
    if(BUILD_TESTS)
        message(STATUS "  ctest --output-on-failure")
    endif()
    message(STATUS "  make install")
    message(STATUS "========================================")
endif()