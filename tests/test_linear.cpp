/**
 * @file test_linear.cpp
 * @brief Test unitari per Linear Regression
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include "regression/linear_regression.h"
#include "regression/math_utils.h"

using namespace regression;

void test_linear_regression_basic() {
    std::cout << "Test 1: Basic Linear Regression (y = 2x + 3)... ";
    
    // Dati: y = 2*x + 3
    Eigen::MatrixXd X(5, 1);
    Eigen::VectorXd y(5);
    X << 1, 2, 3, 4, 5;
    y << 5, 7, 9, 11, 13;  // 2*1+3=5, 2*2+3=7, etc.
    
    LinearRegression lr(0.01, 1000, 0.0, LinearRegression::GRADIENT_DESCENT);
    lr.fit(X, y);
    
    // Verifica coefficienti (circa 2 per la pendenza)
    Eigen::VectorXd coeffs = lr.coefficients();
    assert(std::abs(coeffs(1) - 2.0) < 0.1);  // Pendenza ~2
    assert(std::abs(lr.intercept() - 3.0) < 0.1);  // Intercetta ~3
    
    // Verifica predizioni
    Eigen::VectorXd y_pred = lr.predict(X);
    assert((y_pred - y).norm() < 0.1);  // Errore piccolo
    
    // Verifica R² (dovrebbe essere vicino a 1)
    double r2 = lr.r2_score(X, y);
    assert(r2 > 0.99);
    
    std::cout << "PASSED" << std::endl;
}

void test_linear_regression_multiple_features() {
    std::cout << "Test 2: Multiple Features... ";
    
    // y = 1*x1 + 2*x2 + 3
    Eigen::MatrixXd X(4, 2);
    Eigen::VectorXd y(4);
    X << 1, 1,
         1, 2,
         2, 1,
         2, 2;
    y << 6,   // 1*1 + 2*1 + 3 = 6
         8,   // 1*1 + 2*2 + 3 = 8
         8,   // 1*2 + 2*1 + 3 = 8
         10;  // 1*2 + 2*2 + 3 = 10
    
    LinearRegression lr(0.01, 2000, 0.0, LinearRegression::NORMAL_EQUATION);
    lr.fit(X, y);
    
    // Verifica coefficienti
    Eigen::VectorXd coeffs = lr.coefficients();
    assert(coeffs.size() == 3);  // intercetta + 2 features
    
    // Test predizione
    Eigen::MatrixXd X_test(1, 2);
    X_test << 3, 3;
    double pred = lr.predict(X_test)(0);
    double expected = 1*3 + 2*3 + 3;  // = 12
    assert(std::abs(pred - expected) < 0.1);
    
    std::cout << "PASSED" << std::endl;
}

void test_linear_regression_regularization() {
    std::cout << "Test 3: Regularization (Ridge)... ";
    
    // Dati con multicollinearità
    Eigen::MatrixXd X(5, 3);
    Eigen::VectorXd y(5);
    X << 1, 1, 1,
         1, 2, 2,
         2, 1, 1,
         2, 2, 2,
         3, 3, 3;
    y << 5, 8, 8, 11, 14;
    
    // Senza regolarizzazione
    LinearRegression lr_no_reg(0.01, 1000, 0.0);
    lr_no_reg.fit(X, y);
    
    // Con regolarizzazione
    LinearRegression lr_reg(0.01, 1000, 1.0);
    lr_reg.fit(X, y);
    
    // La regolarizzazione dovrebbe ridurre la magnitudine dei coefficienti
    Eigen::VectorXd coeffs_no_reg = lr_no_reg.coefficients();
    Eigen::VectorXd coeffs_reg = lr_reg.coefficients();
    
    // I coefficienti regolarizzati dovrebbero essere più piccoli (in norma)
    assert(coeffs_reg.norm() < coeffs_no_reg.norm() * 1.1);  // +/-10%
    
    std::cout << "PASSED" << std::endl;
}

void test_linear_regression_solvers() {
    std::cout << "Test 4: Different Solvers Comparison... ";
    
    Eigen::MatrixXd X(10, 2);
    Eigen::VectorXd y(10);
    
    // Genera dati random
    srand(42);
    for(int i = 0; i < 10; ++i) {
        X(i, 0) = i + 1;
        X(i, 1) = (i + 1) * 0.5;
        y(i) = 2*X(i, 0) + 3*X(i, 1) + 1 + (rand() % 100) * 0.01;
    }
    
    // Testa tutti i solver
    double prev_score = 0.0;
    for(auto solver : {LinearRegression::GRADIENT_DESCENT, 
                       LinearRegression::NORMAL_EQUATION, 
                       LinearRegression::SVD}) {
        
        LinearRegression lr(0.01, 1000, 0.0, solver);
        lr.fit(X, y);
        
        double score = lr.r2_score(X, y);
        assert(score > 0.8);  // Tutti dovrebbero avere un buon R²
        
        if(prev_score > 0) {
            // I risultati dovrebbero essere simili tra solver
            assert(std::abs(score - prev_score) < 0.1);
        }
        prev_score = score;
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_linear_regression_save_load() {
    std::cout << "Test 5: Save/Load Model... ";
    
    Eigen::MatrixXd X(5, 2);
    Eigen::VectorXd y(5);
    X << 1, 1,
         1, 2,
         2, 1,
         2, 2,
         3, 3;
    y << 3, 4, 4, 5, 7;
    
    LinearRegression lr1(0.01, 1000);
    lr1.fit(X, y);
    
    // Salva modello
    lr1.save("test_linear_model.bin");
    
    // Carica in nuovo modello
    LinearRegression lr2;
    lr2.load("test_linear_model.bin");
    
    // Verifica che le predizioni siano identiche
    Eigen::MatrixXd X_test(2, 2);
    X_test << 4, 4,
              5, 5;
    
    Eigen::VectorXd pred1 = lr1.predict(X_test);
    Eigen::VectorXd pred2 = lr2.predict(X_test);
    
    assert((pred1 - pred2).norm() < 1e-10);
    
    // Pulisci file temporaneo
    std::remove("test_linear_model.bin");
    
    std::cout << "PASSED" << std::endl;
}

void test_linear_regression_edge_cases() {
    std::cout << "Test 6: Edge Cases... ";
    
    // Test 1: Singola feature, singolo campione
    {
        Eigen::MatrixXd X(1, 1);
        Eigen::VectorXd y(1);
        X << 1;
        y << 2;
        
        LinearRegression lr;
        lr.fit(X, y);
        
        Eigen::VectorXd pred = lr.predict(X);
        assert(std::abs(pred(0) - 2.0) < 1e-6);
    }
    
    // Test 2: Dati costanti
    {
        Eigen::MatrixXd X(3, 1);
        Eigen::VectorXd y(3);
        X << 1, 1, 1;
        y << 5, 5, 5;
        
        LinearRegression lr;
        lr.fit(X, y);
        
        assert(std::abs(lr.predict(X)(0) - 5.0) < 0.1);
    }
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "RUNNING LINEAR REGRESSION TESTS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_linear_regression_basic();
        test_linear_regression_multiple_features();
        test_linear_regression_regularization();
        test_linear_regression_solvers();
        test_linear_regression_save_load();
        test_linear_regression_edge_cases();
        
        std::cout << "========================================" << std::endl;
        std::cout << "ALL LINEAR REGRESSION TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nTEST FAILED: Unknown error" << std::endl;
        return 1;
    }
}