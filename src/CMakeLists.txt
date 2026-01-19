# Definiamo i sorgenti
set(LIB_SOURCES
    linear_regression.cpp
    logistic_regression.cpp
)

# 1. Libreria Statica
add_library(regression_static STATIC ${LIB_SOURCES})
target_include_directories(regression_static PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(regression_static PUBLIC Eigen3::Eigen)

# 2. Libreria Dinamica (Shared)
add_library(regression_shared SHARED ${LIB_SOURCES})
target_include_directories(regression_shared PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(regression_shared PUBLIC Eigen3::Eigen)

# Flag di compilazione per pulizia codice (solo per Linux/macOS)
if(NOT MSVC)
    target_compile_options(regression_static PRIVATE -Wall -Wextra -O3)
    target_compile_options(regression_shared PRIVATE -Wall -Wextra -O3)
endif()

# Setta il nome del file finale uguale per entrambi (es. libregression.a e libregression.so)
set_target_properties(regression_static PROPERTIES OUTPUT_NAME "regression")
set_target_properties(regression_shared PROPERTIES OUTPUT_NAME "regression")