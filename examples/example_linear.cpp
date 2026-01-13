// examples/example_linear.cpp
#include <iostream>
#include "regression/linear_regression.h"

int main() {
    std::cout << "=== Linear Regression Example ===" << std::endl;
    
    // Dati di esempio: y = 2*x + 1
    Eigen::MatrixXd X(5, 1);
    Eigen::VectorXd y(5);
    X << 1, 2, 3, 4, 5;
    y << 3, 5, 7, 9, 11;
    
    std::cout << "Training data:" << std::endl;
    std::cout << "X:\n" << X << std::endl;
    std::cout << "y:\n" << y.transpose() << std::endl;
    
    // Crea e addestra il modello
    regression::LinearRegression lr(0.01, 1000);
    lr.fit(X, y);
    
    std::cout << "\nModel trained!" << std::endl;
    std::cout << "Coefficients: " << lr.coefficients().transpose() << std::endl;
    std::cout << "Intercept: " << lr.intercept() << std::endl;
    std::cout << "R² score: " << lr.score(X, y) << std::endl;
    
    // Predizioni
    Eigen::MatrixXd X_new(2, 1);
    X_new << 6, 7;
    Eigen::VectorXd predictions = lr.predict(X_new);
    
    std::cout << "\nPredictions:" << std::endl;
    std::cout << "X_new:\n" << X_new << std::endl;
    std::cout << "Predictions:\n" << predictions.transpose() << std::endl;
    
    return 0;
}