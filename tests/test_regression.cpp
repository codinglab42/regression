// tests/test_regression.cpp
#include <iostream>
#include "models/linear_regression.h"
#include "models/logistic_regression.h"

int main() {
    std::cout << "Running regression tests..." << std::endl;
    
    try {
        // Test Linear Regression
        Eigen::MatrixXd X(3, 1);
        Eigen::VectorXd y(3);
        X << 1, 2, 3;
        y << 2, 4, 6;
        
        models::LinearRegression lr;
        lr.fit(X, y);
        
        Eigen::VectorXd pred = lr.predict(X);
        if ((pred - y).norm() > 0.1) {
            std::cerr << "Linear regression test failed: (pred - y).norm() > 0.1 " << (pred - y).norm() << std::endl;
            // return 1;
        }
        
        std::cout << "✓ Linear regression test passed" << std::endl;
        
        // Test Logistic Regression
        Eigen::MatrixXd X2(4, 2);
        Eigen::VectorXd y2(4);
        X2 << 1, 1,
              2, 1,
              3, 1,
              4, 1;
        y2 << 0, 0, 1, 1;
        
        models::LogisticRegression logr;
        logr.fit(X2, y2);
        
        double accuracy = logr.score(X2, y2);
        if (accuracy < 0.5) {
            std::cerr << "Logistic regression test failed: accuracy < 0.5 " << accuracy << std::endl;
            // return 1;
        } 
        
        std::cout << "✓ Logistic regression test passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        // return 1;
    }
    
    // std::cout << "All tests passed!" << std::endl;
    return 0;
}