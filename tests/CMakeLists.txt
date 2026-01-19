# =========================================================
# TESTS/CMAKELISTS.TXT - Test indipendenti
# =========================================================

# 1. Trova o scarica GoogleTest
find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/heads/main.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# 2. Lista dei file di test (ognuno con il suo 'main')
set(TEST_FILES
    test_linear
    test_logistic
    test_regression
)

# 3. Crea un eseguibile per ogni file di test
foreach(test_name ${TEST_FILES})
    add_executable(${test_name} ${test_name}.cpp)
    
    # Link alla libreria statica e a GTest
    target_link_libraries(${test_name} 
        PRIVATE 
            regression_static 
            GTest::gtest_main
            Eigen3::Eigen
    )
    
    # Registra il test per 'ctest'
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()