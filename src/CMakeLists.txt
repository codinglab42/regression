# Directory include
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Lista MANUALE dei file sorgente (MEGLIO di file GLOB)
set(CORE_SOURCES
    linear_regression.cpp
    logistic_regression.cpp
    # Aggiungi altri file qui manualmente
)

# Controlla che i file esistano
foreach(src_file ${CORE_SOURCES})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${src_file})
        message(WARNING "Source file not found: ${src_file}")
    endif()
endforeach()

# Aggiungi libreria
add_library(regression_core STATIC ${CORE_SOURCES})
add_library(regression::core ALIAS regression_core)

# Metadati della libreria
set_target_properties(regression_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "regression_core"
    DEBUG_POSTFIX "_d"
    RELEASE_POSTFIX ""
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories
target_include_directories(regression_core
    PUBLIC
        $<BUILD_INTERFACE:${INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link librerie
target_link_libraries(regression_core PUBLIC Eigen3::Eigen)

# Compiler flags per diverse configurazioni
target_compile_options(regression_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wdouble-promotion
        -Wformat=2
        $<$<CONFIG:DEBUG>:-O0 -g3 -fno-inline -DDEBUG>
        $<$<CONFIG:RELEASE>:-O3 -DNDEBUG>
        $<$<CONFIG:RELWITHDEBINFO>:-O2 -g -DNDEBUG>
        $<$<CONFIG:MINSIZEREL>:-Os -DNDEBUG>
    >
    $<$<CXX_COMPILER_ID:Clang>:
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        $<$<CONFIG:DEBUG>:-O0 -g3 -fno-inline -DDEBUG>
        $<$<CONFIG:RELEASE>:-O3 -DNDEBUG>
        $<$<CONFIG:RELWITHDEBINFO>:-O2 -g -DNDEBUG>
        $<$<CONFIG:MINSIZEREL>:-Os -DNDEBUG>
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4
        /permissive-
        $<$<CONFIG:DEBUG>:/Od /Zi /MDd /DDEBUG>
        $<$<CONFIG:RELEASE>:/O2 /MD /DNDEBUG>
        $<$<CONFIG:RELWITHDEBINFO>:/O2 /Zi /MD /DNDEBUG>
        $<$<CONFIG:MINSIZEREL>:/O1 /MD /DNDEBUG>
    >
)

# Preprocessor definitions
target_compile_definitions(regression_core
    PUBLIC
        REGRESSION_EXPORTS
        EIGEN_NO_DEBUG  # Disabilita debug in Eigen per performance
    PRIVATE
        $<$<CONFIG:DEBUG>:REGRESSION_DEBUG=1>
        $<$<NOT:$<CONFIG:DEBUG>>:REGRESSION_DEBUG=0>
)

# Installazione (opzionale)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    install(TARGETS regression_core
        EXPORT RegressionTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    install(DIRECTORY ${INCLUDE_DIR}/regression
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    )
endif()