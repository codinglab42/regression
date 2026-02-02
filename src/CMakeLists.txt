# ============================================================================
# CONFIGURAZIONE SORGENTI ML LIBRARY
# ============================================================================
message(STATUS "📦 Configuring ML Library sources...")

# ============================================================================
# 1. ELENCO COMPLETO DEI SORGENTI
# ============================================================================
set(ML_LIBRARY_SOURCES
    # Regressione
    linear_regression.cpp
    logistic_regression.cpp
    
    # Neural Network
    neural_network/neural_network.cpp
    
    # Layers
    layers/dense.cpp
    layers/convolutional.cpp
    layers/pooling.cpp
    layers/recurrent.cpp
    
    # Activation
    activation/activation.cpp
    
    # Optimizers
    optimizers/sgd.cpp
    optimizers/adam.cpp
    
    ## Serialization
    #serialization/model_serializer.cpp
)

# Informazioni
list(LENGTH ML_LIBRARY_SOURCES NUM_SOURCES)
message(STATUS "  Total source files: ${NUM_SOURCES}")

# ============================================================================
# 2. INCLUDES
# ============================================================================
set(PUBLIC_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
)

set(PRIVATE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/include/exceptions
    ${CMAKE_SOURCE_DIR}/include/serialization
)

# ============================================================================
# 3. LIBRERIA STATICA
# ============================================================================
add_library(ml_library_static STATIC ${ML_LIBRARY_SOURCES})

target_include_directories(ml_library_static
    PUBLIC ${PUBLIC_INCLUDE_DIRS}
    PRIVATE ${PRIVATE_INCLUDE_DIRS}
)

target_link_libraries(ml_library_static PUBLIC Eigen3::Eigen)

# ============================================================================
# 4. LIBRERIA DINAMICA
# ============================================================================
add_library(ml_library_shared SHARED ${ML_LIBRARY_SOURCES})

target_include_directories(ml_library_shared
    PUBLIC ${PUBLIC_INCLUDE_DIRS}
    PRIVATE ${PRIVATE_INCLUDE_DIRS}
)

target_link_libraries(ml_library_shared PUBLIC Eigen3::Eigen)

# Export symbols
target_compile_definitions(ml_library_shared PRIVATE ML_LIBRARY_EXPORT)

# ============================================================================
# 5. PROPRIETÀ COMUNI
# ============================================================================
# Nomi uniformi
set_target_properties(ml_library_static PROPERTIES
    OUTPUT_NAME "ml_library"
    ARCHIVE_OUTPUT_NAME "ml_library"
)

set_target_properties(ml_library_shared PROPERTIES
    OUTPUT_NAME "ml_library"
    ARCHIVE_OUTPUT_NAME "ml_library"
    LIBRARY_OUTPUT_NAME "ml_library"
    POSITION_INDEPENDENT_CODE ON
)

# Flag di compilazione
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Flag comuni
    set(COMMON_FLAGS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wno-unused-parameter
    )
    
    target_compile_options(ml_library_static PRIVATE ${COMMON_FLAGS})
    target_compile_options(ml_library_shared PRIVATE ${COMMON_FLAGS})
    
    # Specifico per shared
    target_compile_options(ml_library_shared PRIVATE
        -fvisibility=hidden
        -fvisibility-inlines-hidden
    )
    
    # Debug vs Release
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(ml_library_static PRIVATE -O0 -g -DDEBUG)
        target_compile_options(ml_library_shared PRIVATE -O0 -g -DDEBUG)
    else()
        target_compile_options(ml_library_static PRIVATE -O3 -DNDEBUG)
        target_compile_options(ml_library_shared PRIVATE -O3 -DNDEBUG)
    endif()
endif()

# ============================================================================
# 6. ALIAS (VERSIONE SICURA)
# ============================================================================
# MODIFICA: Usa nomi senza namespace se CMake < 3.11
# Oppure controlla la versione di CMake

# Controlla versione CMake per namespace support
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.11)
    # Namespace supportato
    add_library(ML::Static ALIAS ml_library_static)
    add_library(ML::Shared ALIAS ml_library_shared)
    
    # Target unificato (usa BUILD_SHARED_LIBS)
    if(BUILD_SHARED_LIBS)
        add_library(ml_library ALIAS ml_library_shared)
        add_library(ML::Library ALIAS ml_library_shared)
        message(STATUS "  Default: Shared library (.so/.dll)")
    else()
        add_library(ml_library ALIAS ml_library_static)
        add_library(ML::Library ALIAS ml_library_static)
        message(STATUS "  Default: Static library (.a/.lib)")
    endif()
else()
    # Fallback senza namespace
    message(STATUS "  ⚠️  CMake < 3.11, using simple aliases")
    add_library(ml_library_static_alias ALIAS ml_library_static)
    add_library(ml_library_shared_alias ALIAS ml_library_shared)
    
    # Target unificato
    if(BUILD_SHARED_LIBS)
        add_library(ml_library ALIAS ml_library_shared)
        message(STATUS "  Default: Shared library (.so/.dll)")
    else()
        add_library(ml_library ALIAS ml_library_static)
        message(STATUS "  Default: Static library (.a/.lib)")
    endif()
endif()

# ============================================================================
# 7. INSTALLAZIONE
# ============================================================================
install(TARGETS ml_library_static ml_library_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# ============================================================================
# 8. CONFIGURAZIONE COMPLETATA
# ============================================================================
message(STATUS "✅ ML Library sources configured")
message(STATUS "  Available targets:")
message(STATUS "    - ml_library_static (static)")
message(STATUS "    - ml_library_shared (shared)")
message(STATUS "    - ml_library (alias, uses BUILD_SHARED_LIBS)")
message(STATUS "    - ML::Library (recommended alias)")