# ============================================================================
# PYTHON BINDINGS
# ============================================================================
message(STATUS "🐍 Configuring Python bindings...")

# Crea modulo Python
pybind11_add_module(regression_module
    regression_module.cpp
)

# Link con la libreria principale
target_link_libraries(regression_module PRIVATE ml_library)

# Include directories
target_include_directories(regression_module PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Eigen3_INCLUDE_DIRS}
)

# Proprietà del modulo
set_target_properties(regression_module PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""
    OUTPUT_NAME "regression_module"
)

# ============================================================================
# INSTALLAZIONE INTELLIGENTE PER PYENV
# ============================================================================
if(DEFINED ENV{PYENV_ROOT})
    # Determina site-packages di pyenv
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "
import site
if hasattr(site, 'getsitepackages'):
    print(site.getsitepackages()[0])
else:
    import sys
    print(sys.prefix + '/lib/python' + str(sys.version_info[0]) + '.' + str(sys.version_info[1]) + '/site-packages')
"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(PYTHON_SITE_PACKAGES)
        message(STATUS "📁 Installing to pyenv site-packages: ${PYTHON_SITE_PACKAGES}")
        install(TARGETS regression_module DESTINATION ${PYTHON_SITE_PACKAGES})
        
        # Symlink per testing
        add_custom_command(TARGET regression_module POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:regression_module>
            ${CMAKE_BINARY_DIR}/regression_module$<TARGET_FILE_NAME:regression_module>
            COMMENT "Creating symlink for testing"
        )
    else()
        message(WARNING "⚠️  Could not determine pyenv site-packages")
        install(TARGETS regression_module DESTINATION ${CMAKE_INSTALL_PREFIX}/python)
    endif()
else()
    # Installazione normale
    install(TARGETS regression_module DESTINATION ${CMAKE_INSTALL_PREFIX}/python)
endif()

# ============================================================================
# TARGET PER TEST PYTHON
# ============================================================================
add_custom_target(test_python_module
    COMMAND ${Python3_EXECUTABLE} -c "
import sys
import os

# Aggiungi directory build al path
build_dir = os.path.join('${CMAKE_BINARY_DIR}', 'pybinding')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
else:
    sys.path.insert(0, '${CMAKE_BINARY_DIR}')

try:
    import regression_module as ml
    print('✅ SUCCESS: Python module imported')
    print(f'   Module: {ml.__name__}')
    print(f'   Version: {ml.__version__}')
    
    # Test base
    import numpy as np
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([1, 2], dtype=np.float64)
    
    lr = ml.LinearRegression()
    lr.fit(X, y)
    pred = lr.predict(X)
    print(f'   LinearRegression test: ✓')
    
    print('✅ All Python tests passed')
    
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
    DEPENDS regression_module
    COMMENT "Testing Python module import"
    VERBATIM
)

message(STATUS "✅ Python bindings configured")
message(STATUS "  Module: regression_module")
message(STATUS "  Test: make test_python_module")