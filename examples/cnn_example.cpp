// examples/cnn_example.cpp
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "models/neural_network.h"

using namespace Eigen;
using namespace neural_network;

void test_cnn_mnist_like() {
    std::cout << "=== CNN Example (MNIST-like) ===" << std::endl;
    
    // Simula dataset MNIST: 28x28 immagini in scala di grigi
    int batch_size = 32;
    int channels = 1; // Scala di grigi
    int height = 28;
    int width = 28;
    int n_classes = 10;
    
    // Crea dati sintetici
    MatrixXd X = MatrixXd::Random(batch_size * 100, channels * height * width);
    VectorXd y = VectorXd::Random(batch_size * 100);
    y = (y.array() > 0).cast<double>(); // Binarizza per semplicità
    
    // Crea architettura CNN
    NeuralNetwork nn;
    
    // Layer 1: Convoluzione
    nn.add_convolutional_layer(1, 32, 3, 1, 1, "relu"); // 32 filtri 3x3
    // Output: 32 x 28 x 28
    
    // Layer 2: Pooling
    nn.add_pooling_layer(2, 2, components::layers::Pooling::MAX, 32);
    // Output: 32 x 14 x 14
    
    // Layer 3: Convoluzione
    nn.add_convolutional_layer(32, 64, 3, 1, 1, "relu"); // 64 filtri 3x3
    // Output: 64 x 14 x 14
    
    // Layer 4: Pooling
    nn.add_pooling_layer(2, 2, components::layers::Pooling::MAX, 64);
    // Output: 64 x 7 x 7
    
    // Appiattisci per layer fully connected
    // 64 * 7 * 7 = 3136 features
    nn.add_layer(std::make_unique<components::layers::Dense>(64 * 7 * 7, 128, "relu"));
    nn.add_layer(std::make_unique<components::layers::Dense>(128, 64, "relu"));
    nn.add_layer(std::make_unique<components::layers::Dense>(64, 1, "sigmoid"));
    
    // Configura training
    nn.set_epochs(50);
    nn.set_batch_size(batch_size);
    nn.set_validation_split(0.2);
    nn.set_verbose(true);
    nn.set_loss_function("binary_crossentropy");
    
    std::cout << "CNN Architecture:" << std::endl;
    std::cout << nn.to_string() << std::endl;
    
    // Train
    nn.fit(X, y);
    
    // Test
    MatrixXd X_test = MatrixXd::Random(batch_size, channels * height * width);
    VectorXd y_pred = nn.predict(X_test);
    
    std::cout << "\nPredictions shape: " << y_pred.rows() << " x " << y_pred.cols() << std::endl;
    std::cout << "First 5 predictions: ";
    for (int i = 0; i < std::min(5, (int)y_pred.size()); ++i) {
        std::cout << y_pred(i) << " ";
    }
    std::cout << std::endl;
}

void test_rnn_sequence() {
    std::cout << "\n=== RNN Sequence Example ===" << std::endl;
    
    // Dati sequenziali: time series prediction
    int seq_length = 10;
    int feature_dim = 5;
    int batch_size = 16;
    
    MatrixXd X = MatrixXd::Random(batch_size * 50, seq_length * feature_dim);
    VectorXd y = VectorXd::Random(batch_size * 50);
    
    // Crea RNN per regressione sequenziale
    NeuralNetwork nn;
    
    // Layer RNN con return_sequences=true per ottenere output per ogni timestep
    nn.add_recurrent_layer(64, "tanh", true); // Output: seq_length * 64
    
    // Layer denso per ogni timestep
    nn.add_layer(std::make_unique<components::layers::Dense>(seq_length * 64, 32, "relu"));
    nn.add_layer(std::make_unique<components::layers::Dense>(32, 1, "linear"));
    
    // Configura
    nn.set_epochs(30);
    nn.set_batch_size(batch_size);
    nn.set_validation_split(0.2);
    nn.set_verbose(true);
    nn.set_loss_function("mse");
    
    std::cout << "RNN Architecture:" << std::endl;
    std::cout << nn.to_string() << std::endl;
    
    // Train
    nn.fit(X, y);
    
    // Test
    MatrixXd X_test = MatrixXd::Random(batch_size, seq_length * feature_dim);
    VectorXd y_pred = nn.predict(X_test);
    
    std::cout << "\nSequence prediction complete!" << std::endl;
    std::cout << "Test predictions MSE: " << nn.score(X_test, y_test) << std::endl;
}

int main() {
    try {
        test_cnn_mnist_like();
        test_rnn_sequence();
        
        std::cout << "\n=== CNN/RNN Tests Completed Successfully ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}