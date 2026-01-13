# CMakeLists.txt per i binding Python
cmake_minimum_required(VERSION 3.15)

# Assicurati che pybind11 sia stato trovato
if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found. Cannot build Python bindings.")
endif()

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 not found. Cannot build Python bindings.")
endif()

message(STATUS "Configuring Python bindings...")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python version: ${Python3_VERSION}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python3_LIBRARY}")

# Ottieni il suffisso per l'estensione Python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Ottieni il percorso di installazione
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import site; import sys; print(site.getusersitepackages() if '--user' in sys.argv else site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python extension suffix: ${PYTHON_EXTENSION_SUFFIX}")
message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")

# Opzioni per i binding
option(PYTHON_BINDINGS_STANDALONE "Build Python bindings as standalone module" OFF)
option(PYTHON_BINDINGS_INSTALL "Install Python bindings to site-packages" ON)
option(PYTHON_BINDINGS_TEST "Build tests for Python bindings" ON)

# Trova tutti i file sorgente per i binding
file(GLOB PYTHON_SOURCES "*.cpp")
message(STATUS "Found Python binding sources: ${PYTHON_SOURCES}")

# Crea il modulo Python
pybind11_add_module(
    regression_pybind  # Nome del modulo Python
    ${PYTHON_SOURCES}
)

# Rinomina l'output per avere il nome corretto
set_target_properties(regression_pybind PROPERTIES
    OUTPUT_NAME "regression"
    SUFFIX ${PYTHON_EXTENSION_SUFFIX}
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Includi directory
target_include_directories(regression_pybind
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Link alle librerie
target_link_libraries(regression_pybind
    PRIVATE
        regression::core  # La tua libreria C++
        Eigen3::Eigen
        pybind11::module
)

# Flags del compilatore specifici per pybind11
target_compile_options(regression_pybind PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -fvisibility=hidden
        # Flags per ridurre la dimensione
        -ffunction-sections
        -fdata-sections
    >
    $<$<CXX_COMPILER_ID:Clang>:
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -fvisibility=hidden
        -ffunction-sections
        -fdata-sections
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4
        /wd4100  # unused parameter
        /wd4505  # unreferenced local function removed
    >
)

# Flags del linker per ridurre la dimensione (solo GCC/Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(regression_pybind PRIVATE
        -Wl,--gc-sections
        -Wl,--as-needed
    )
endif()

# Definizioni per pybind11
target_compile_definitions(regression_pybind PRIVATE
    PYBIND11_STRICT_TYPE_SAFETY
    PYBIND11_DETAILED_ERROR_MESSAGES
    # Disabilita alcuni warning di Eigen in pybind11
    EIGEN_MPL2_ONLY
    EIGEN_NO_DEBUG
)

# Configura installazione
if(PYTHON_BINDINGS_INSTALL)
    install(TARGETS regression_pybind
        DESTINATION ${PYTHON_SITE_PACKAGES}
    )
    
    # Installa anche i file Python puri se presenti
    file(GLOB PYTHON_PURE_FILES "*.py")
    if(PYTHON_PURE_FILES)
        install(FILES ${PYTHON_PURE_FILES}
            DESTINATION ${PYTHON_SITE_PACKAGES}
        )
    endif()
    
    message(STATUS "Python bindings will be installed to: ${PYTHON_SITE_PACKAGES}")
endif()

# Creazione di un target per testare il modulo Python
if(PYTHON_BINDINGS_TEST AND Python3_EXECUTABLE)
    # Crea un test per verificare che il modulo sia importabile
    add_test(
        NAME test_python_import
        COMMAND ${Python3_EXECUTABLE} -c "import sys; sys.path.insert(0, '${CMAKE_LIBRARY_OUTPUT_DIRECTORY}'); import regression; print('Python module imported successfully')"
    )
    
    # Crea uno script di test Python
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/test_python_bindings.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/test_python_bindings.py
        @ONLY
    )
    
    # Aggiungi il test Python
    add_test(
        NAME test_python_functionality
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_python_bindings.py
    )
    
    # Aggiungi un target per eseguire test Python
    add_custom_target(test_python
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_python_bindings.py
        DEPENDS regression_pybind
        COMMENT "Running Python tests..."
    )
endif()

# Target per generare la documentazione dei binding (opzionale)
find_program(SPHINX_BUILD sphinx-build)
if(SPHINX_BUILD AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs")
    add_custom_target(docs
        COMMAND ${SPHINX_BUILD} -b html ${CMAKE_CURRENT_SOURCE_DIR}/docs ${CMAKE_CURRENT_BINARY_DIR}/docs/html
        COMMENT "Building documentation..."
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# Target per pulire i file Python generati
add_custom_target(clean_python
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/*.py
        ${CMAKE_CURRENT_BINARY_DIR}/*.pyc
        ${CMAKE_CURRENT_BINARY_DIR}/__pycache__
    COMMENT "Cleaning Python generated files"
)

# Aggiungi dipendenze
add_dependencies(clean_python regression_pybind)