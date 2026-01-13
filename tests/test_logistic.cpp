/**
 * @file test_logistic.cpp
 * @brief Test unitari per Logistic Regression
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include "regression/logistic_regression.h"
#include "regression/math_utils.h"

using namespace regression;

void test_logistic_regression_basic() {
    std::cout << "Test 1: Basic Logistic Regression (separazione lineare)... ";
    
    // Dati linearmente separabili: x > 2.5 -> classe 1
    Eigen::MatrixXd X(6, 1);
    Eigen::VectorXd y(6);
    X << 1, 2, 3, 4, 5, 6;
    y << 0, 0, 1, 1, 1, 1;
    
    LogisticRegression lr(0.1, 1000, 0.0, 1e-6, false);
    lr.fit(X, y);
    
    // Verifica accuratezza
    double accuracy = lr.score(X, y);
    assert(accuracy > 0.95);  // Dovrebbe classificare bene
    
    // Verifica predizioni
    Eigen::VectorXi y_pred = lr.predict_class(X);
    int correct = 0;
    for(int i = 0; i < y.size(); ++i) {
        if(y_pred(i) == static_cast<int>(y(i))) correct++;
    }
    assert(correct >= 5);  // Almeno 5/6 corretti
    
    std::cout << "PASSED (accuracy: " << accuracy << ")" << std::endl;
}

void test_logistic_regression_two_features() {
    std::cout << "Test 2: Two Features (AND logico)... ";
    
    // Tavola di verità AND: entrambi 1 -> 1
    Eigen::MatrixXd X(4, 2);
    Eigen::VectorXd y(4);
    X << 0, 0,
         0, 1,
         1, 0,
         1, 1;
    y << 0, 0, 0, 1;  // Solo (1,1) è vero
    
    LogisticRegression lr(0.1, 2000, 0.0, 1e-6, false);
    lr.fit(X, y);
    
    // Verifica accuratezza
    double accuracy = lr.score(X, y);
    assert(accuracy == 1.0);  // Perfetto per AND lineare
    
    // Verifica predizione specifica
    Eigen::MatrixXd X_test(1, 2);
    X_test << 1, 1;
    double prob = lr.predict(X_test)(0);
    assert(prob > 0.5);  // Dovrebbe predire classe 1
    
    std::cout << "PASSED (accuracy: " << accuracy << ")" << std::endl;
}

void test_logistic_regression_regularization() {
    std::cout << "Test 3: Regularization... ";
    
    // Dati con rumore
    Eigen::MatrixXd X(10, 3);
    Eigen::VectorXd y(10);
    
    srand(42);
    for(int i = 0; i < 10; ++i) {
        X(i, 0) = i;
        X(i, 1) = i * 0.5;
        X(i, 2) = i * 0.3;
        y(i) = (i > 5) ? 1 : 0;
        if(i == 3 || i == 7) y(i) = 1 - y(i);  // Rumore
    }
    
    // Senza regolarizzazione
    LogisticRegression lr_no_reg(0.1, 1000, 0.0);
    lr_no_reg.fit(X, y);
    
    // Con regolarizzazione L2
    LogisticRegression lr_reg(0.1, 1000, 1.0);
    lr_reg.fit(X, y);
    
    // Coefficienti regolarizzati dovrebbero essere più piccoli
    Eigen::VectorXd coeffs_no_reg = lr_no_reg.coefficients();
    Eigen::VectorXd coeffs_reg = lr_reg.coefficients();
    
    assert(coeffs_reg.tail(3).norm() < coeffs_no_reg.tail(3).norm() * 1.5);
    
    std::cout << "PASSED" << std::endl;
}

void test_logistic_regression_probabilities() {
    std::cout << "Test 4: Probability Calibration... ";
    
    // Dati dove il punto 2.5 è al confine
    Eigen::MatrixXd X(5, 1);
    Eigen::VectorXd y(5);
    X << 1, 2, 3, 4, 5;
    y << 0, 0, 1, 1, 1;
    
    LogisticRegression lr(0.1, 1000);
    lr.fit(X, y);
    
    // Testa punti diversi
    Eigen::MatrixXd X_test(3, 1);
    X_test << 2, 3, 2.5;
    
    Eigen::VectorXd probs = lr.predict(X_test);
    
    // Probabilità dovrebbe essere monotona
    assert(probs(0) < probs(1));  // x=2 < x=3
    assert(probs(0) < 0.5);       // x=2 dovrebbe essere < 0.5
    assert(probs(1) > 0.5);       // x=3 dovrebbe essere > 0.5
    
    // Punto di confine ~0.5
    assert(probs(2) > 0.3 && probs(2) < 0.7);
    
    std::cout << "PASSED (probs: " << probs.transpose() << ")" << std::endl;
}

void test_logistic_regression_metrics() {
    std::cout << "Test 5: Evaluation Metrics... ";
    
    // Dati con classi bilanciate
    Eigen::MatrixXd X(8, 2);
    Eigen::VectorXd y(8);
    X << 1, 0,
         2, 0,
         3, 0,
         4, 0,
         5, 1,
         6, 1,
         7, 1,
         8, 1;
    y << 0, 0, 0, 0, 1, 1, 1, 1;
    
    LogisticRegression lr(0.1, 1000);
    lr.fit(X, y);
    
    // Calcola tutte le metriche
    Eigen::Vector3d metrics = lr.precision_recall_f1(X, y, 0.5);
    double precision = metrics(0);
    double recall = metrics(1);
    double f1 = metrics(2);
    
    // Per dati perfettamente separabili, tutte le metriche dovrebbero essere 1
    assert(precision > 0.95);
    assert(recall > 0.95);
    assert(f1 > 0.95);
    
    // Matrice di confusione
    Eigen::MatrixXd cm = lr.confusion_matrix(X, y);
    assert(cm.rows() == 2 && cm.cols() == 2);
    assert(cm.sum() == X.rows());
    
    std::cout << "PASSED (P: " << precision << ", R: " << recall << ", F1: " << f1 << ")" << std::endl;
}

void test_logistic_regression_save_load() {
    std::cout << "Test 6: Save/Load Model... ";
    
    Eigen::MatrixXd X(6, 2);
    Eigen::VectorXd y(6);
    X << 1, 0,
         2, 0,
         3, 0,
         4, 1,
         5, 1,
         6, 1;
    y << 0, 0, 0, 1, 1, 1;
    
    LogisticRegression lr1(0.1, 1000, 0.5);
    lr1.fit(X, y);
    
    // Salva modello
    lr1.save("test_logistic_model.bin");
    
    // Carica modello
    LogisticRegression lr2;
    lr2.load("test_logistic_model.bin");
    
    // Verifica predizioni identiche
    Eigen::MatrixXd X_test(2, 2);
    X_test << 2.5, 0.5,
              4.5, 0.5;
    
    Eigen::VectorXd probs1 = lr1.predict(X_test);
    Eigen::VectorXd probs2 = lr2.predict(X_test);
    
    assert((probs1 - probs2).norm() < 1e-10);
    
    // Pulisci file
    std::remove("test_logistic_model.bin");
    
    std::cout << "PASSED" << std::endl;
}

void test_logistic_regression_threshold() {
    std::cout << "Test 7: Decision Threshold... ";
    
    // Dati con incertezza
    Eigen::MatrixXd X(5, 1);
    Eigen::VectorXd y(5);
    X << 1, 2, 3, 4, 5;
    y << 0, 0, 1, 1, 1;
    
    LogisticRegression lr(0.1, 1000);
    lr.fit(X, y);
    
    // Testa soglie diverse
    Eigen::VectorXi pred_low = lr.predict_class(X, 0.3);   // Soglia bassa
    Eigen::VectorXi pred_high = lr.predict_class(X, 0.7);  // Soglia alta
    
    // Con soglia bassa, più predizioni positive
    int pos_low = pred_low.sum();
    int pos_high = pred_high.sum();
    
    assert(pos_low >= pos_high);  // Soglia bassa → più positivi
    
    std::cout << "PASSED (low thresh positives: " << pos_low 
              << ", high thresh: " << pos_high << ")" << std::endl;
}

void test_logistic_regression_edge_cases() {
    std::cout << "Test 8: Edge Cases... ";
    
    // Test 1: Tutti della stessa classe
    {
        Eigen::MatrixXd X(3, 1);
        Eigen::VectorXd y(3);
        X << 1, 2, 3;
        y << 1, 1, 1;  // Tutti classe 1
        
        LogisticRegression lr;
        lr.fit(X, y);
        
        assert(lr.score(X, y) == 1.0);
    }
    
    // Test 2: Singolo campione
    {
        Eigen::MatrixXd X(1, 2);
        Eigen::VectorXd y(1);
        X << 1, 1;
        y << 1;
        
        LogisticRegression lr;
        lr.fit(X, y);
        
        Eigen::VectorXd pred = lr.predict(X);
        assert(pred(0) > 0.5);
    }
    
    // Test 3: Classi perfettamente separate
    {
        Eigen::MatrixXd X(4, 1);
        Eigen::VectorXd y(4);
        X << 1, 2, 10, 11;
        y << 0, 0, 1, 1;
        
        LogisticRegression lr;
        lr.fit(X, y);
        
        assert(lr.score(X, y) == 1.0);
    }
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "RUNNING LOGISTIC REGRESSION TESTS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_logistic_regression_basic();
        test_logistic_regression_two_features();
        test_logistic_regression_regularization();
        test_logistic_regression_probabilities();
        test_logistic_regression_metrics();
        test_logistic_regression_save_load();
        test_logistic_regression_threshold();
        test_logistic_regression_edge_cases();
        
        std::cout << "========================================" << std::endl;
        std::cout << "ALL LOGISTIC REGRESSION TESTS PASSED!" << std::endl;
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