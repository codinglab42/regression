#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "neural_network/neural_network.h"
#include "regression/linear_regression.h"
#include "regression/logistic_regression.h"

using namespace Eigen;
using namespace neural_network;

void test_neural_network_regression() {
    std::cout << "=== Test Neural Network Regression ===" << std::endl;
    
    // Crea dati sintetici per regressione
    int n_samples = 1000;
    int n_features = 5;
    
    MatrixXd X = MatrixXd::Random(n_samples, n_features);
    VectorXd true_weights = VectorXd::LinSpaced(n_features, 1.0, 5.0);
    VectorXd y = X * true_weights + VectorXd::Random(n_samples) * 0.1;
    
    // Crea e addestra rete neurale
    NeuralNetwork nn({n_features, 64, 32, 1}, "relu", "linear");
    nn.set_epochs(50);
    nn.set_batch_size(32);
    nn.set_validation_split(0.2);
    nn.set_verbose(true);
    
    nn.fit(X, y);
    
    // Predizioni
    VectorXd y_pred = nn.predict(X);
    
    // Calcola MSE
    double mse = (y - y_pred).squaredNorm() / n_samples;
    double r2 = 1.0 - (y - y_pred).squaredNorm() / (y.array() - y.mean()).square().sum();
    
    std::cout << "Neural Network Results:" << std::endl;
    std::cout << "  MSE: " << mse << std::endl;
    std::cout << "  R²: " << r2 << std::endl;
    
    // Confronta con Linear Regression
    regression::LinearRegression lr;
    lr.fit(X, y);
    VectorXd y_lr = lr.predict(X);
    double lr_mse = (y - y_lr).squaredNorm() / n_samples;
    
    std::cout << "\nLinear Regression Results:" << std::endl;
    std::cout << "  MSE: " << lr_mse << std::endl;
    std::cout << "  Difference: " << std::abs(mse - lr_mse) << std::endl;
}

void test_neural_network_classification() {
    std::cout << "\n=== Test Neural Network Classification ===" << std::endl;
    
    // Crea dati sintetici per classificazione binaria
    int n_samples = 1000;
    int n_features = 10;
    
    MatrixXd X = MatrixXd::Random(n_samples, n_features);
    
    // Funzione non lineare per separare le classi
    VectorXd y_raw = (X.col(0).array().square() + X.col(1).array().sin() 
                     + X.col(2).array() * X.col(3).array()).matrix();
    
    // Binarizza
    VectorXd y = (y_raw.array() > y_raw.mean()).cast<double>();
    
    // Crea e addestra rete neurale
    NeuralNetwork nn({n_features, 128, 64, 32, 1}, "relu", "sigmoid");
    nn.set_epochs(100);
    nn.set_batch_size(64);
    nn.set_validation_split(0.2);
    nn.set_verbose(true);
    nn.set_loss_function("binary_crossentropy");
    
    nn.fit(X, y);
    
    // Predizioni e accuracy
    VectorXd y_pred_prob = nn.predict(X);
    VectorXi y_pred = (y_pred_prob.array() > 0.5).cast<int>();
    VectorXi y_true = y.cast<int>();
    
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (y_pred(i) == y_true(i)) {
            correct++;
        }
    }
    
    double accuracy = static_cast<double>(correct) / n_samples;
    
    std::cout << "Neural Network Classification Results:" << std::endl;
    std::cout << "  Accuracy: " << accuracy << std::endl;
    std::cout << "  Loss history size: " << nn.get_loss_history().size() << std::endl;
    
    // Confronta con Logistic Regression
    regression::LogisticRegression logreg(0.1, 1000, 0.0, 1e-4, false);
    logreg.fit(X, y);
    double logreg_acc = logreg.score(X, y);
    
    std::cout << "\nLogistic Regression Results:" << std::endl;
    std::cout << "  Accuracy: " << logreg_acc << std::endl;
    std::cout << "  Difference: " << std::abs(accuracy - logreg_acc) << std::endl;
}

void test_model_serialization() {
    std::cout << "\n=== Test Model Serialization ===" << std::endl;
    
    // Crea un modello semplice
    MatrixXd X = MatrixXd::Random(100, 3);
    VectorXd y = X.col(0) + 2 * X.col(1) - 0.5 * X.col(2);
    
    NeuralNetwork nn({3, 16, 8, 1});
    nn.set_epochs(10);
    nn.set_verbose(false);
    nn.fit(X, y);
    
    // Salva il modello
    std::string filename = "test_nn_model.bin";
    nn.save(filename);
    std::cout << "Model saved to: " << filename << std::endl;
    
    // Carica il modello
    NeuralNetwork nn_loaded;
    nn_loaded.load(filename);
    std::cout << "Model loaded from: " << filename << std::endl;
    
    // Confronta predizioni
    VectorXd y_orig = nn.predict(X);
    VectorXd y_loaded = nn_loaded.predict(X);
    
    double diff = (y_orig - y_loaded).norm() / y_orig.norm();
    std::cout << "Prediction difference (relative): " << diff << std::endl;
    
    if (diff < 1e-10) {
        std::cout << "✓ Serialization test PASSED" << std::endl;
    } else {
        std::cout << "✗ Serialization test FAILED" << std::endl;
    }
    
    // Pulisci file temporaneo
    std::remove(filename.c_str());
}

void test_exception_handling() {
    std::cout << "\n=== Test Exception Handling ===" << std::endl;
    
    try {
        NeuralNetwork nn;
        MatrixXd X = MatrixXd::Random(10, 5);
        VectorXd y = VectorXd::Random(8); // Dimensione errata
        
        std::cout << "Testing dimension mismatch..." << std::endl;
        nn.fit(X, y);
        
        std::cout << "✗ Exception test FAILED (should have thrown)" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✓ Exception caught: " << e.what() << std::endl;
    }
    
    try {
        NeuralNetwork nn;
        MatrixXd X; // Matrice vuota
        VectorXd y;
        
        std::cout << "\nTesting empty dataset..." << std::endl;
        nn.fit(X, y);
        
        std::cout << "✗ Exception test FAILED (should have thrown)" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✓ Exception caught: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Machine Learning Library v2.0 - Neural Network Tests" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    try {
        test_neural_network_regression();
        test_neural_network_classification();
        test_model_serialization();
        test_exception_handling();
        
        std::cout << "\n=== All tests completed ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}