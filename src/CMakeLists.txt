
# File c++ per la libreria
file(GLOB CORE_SOURCES "*.cpp")

# Libreria statica
add_library(regression_core STATIC ${CORE_SOURCES})

# Eigen3
target_link_libraries(regression_core PUBLIC Eigen3::Eigen)

# headers
target_include_directories(regression_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)