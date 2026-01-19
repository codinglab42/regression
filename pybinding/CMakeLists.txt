# Crea il modulo Python (.so)
pybind11_add_module(regression_module regression_module.cpp)

# Collega la libreria dinamica e gli header di Eigen
target_link_libraries(regression_module PRIVATE regression_shared Eigen3::Eigen)

# Specifichiamo dove mettere il modulo compilato per trovarlo subito con Python
set_target_properties(regression_module PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python_module
    PREFIX ""
    SUFFIX ".so"
)