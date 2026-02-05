#include "models/neural_network.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <random>
#include "utils/serializable.h"
#include "utils/math_utils.h"
#include "components/layers/recurrent.h"
#include "components/layers/convolutional.h"

using namespace Eigen;
using namespace models;

// Costruttori
NeuralNetwork::NeuralNetwork()
    : batch_size_(32), epochs_(100), validation_split_(0.2), verbose_(false),
      loss_function_("binary_crossentropy") {
    optimizer_ = std::make_unique<optimizers::SGD>(0.01);
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                           const std::string& activation,
                           const std::string& output_activation)
    : batch_size_(32), epochs_(100), validation_split_(0.2), verbose_(false),
      loss_function_("binary_crossentropy") {
    
    ML_CHECK_PARAM(layer_sizes.size() >= 2, "layer_sizes", 
                  "must have at least 2 layers (input and output)", get_model_type());
    
    // Crea layers
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        std::string act = (i == layer_sizes.size() - 2) ? output_activation : activation;
        layers_.push_back(std::make_unique<layers::Dense>(
            layer_sizes[i], layer_sizes[i + 1], act));
    }
    
    optimizer_ = std::make_unique<optimizers::SGD>(0.01);
}

// Configurazione
void NeuralNetwork::add_layer(std::unique_ptr<layers::Layer> layer) {
    if (!layers_.empty()) {
        int last_output = layers_.back()->get_output_size();
        int new_input = layer->get_input_size();
        
        if (last_output != new_input) {
            throw ml_exception::DimensionMismatchException(
                "layer connection",
                last_output, 1,
                new_input, 1,
                get_model_type());
        }
    }
    layers_.push_back(std::move(layer));
}

void NeuralNetwork::set_optimizer(std::unique_ptr<optimizers::Optimizer> optimizer) {
    optimizer_ = std::move(optimizer);
}

void NeuralNetwork::set_loss_function(const std::string& loss) {
    const std::vector<std::string> valid_losses = {
        "binary_crossentropy", "categorical_crossentropy", "mse"
    };
    
    if (std::find(valid_losses.begin(), valid_losses.end(), loss) == valid_losses.end()) {
        throw ml_exception::InvalidParameterException(
            "loss_function", "must be one of: binary_crossentropy, "
            "categorical_crossentropy, mse", get_model_type());
    }
    loss_function_ = loss;
}

void NeuralNetwork::set_batch_size(int batch_size) {
    ML_CHECK_PARAM(batch_size > 0, "batch_size", "must be > 0", get_model_type());
    batch_size_ = batch_size;
}

void NeuralNetwork::set_epochs(int epochs) {
    ML_CHECK_PARAM(epochs > 0, "epochs", "must be > 0", get_model_type());
    epochs_ = epochs;
}

void NeuralNetwork::set_validation_split(double split) {
    ML_CHECK_PARAM(split >= 0 && split < 1, "validation_split", 
                  "must be >= 0 and < 1", get_model_type());
    validation_split_ = split;
}

void NeuralNetwork::set_verbose(bool verbose) {
    verbose_ = verbose;
}

// Metodi privati
void NeuralNetwork::initialize_weights(const Eigen::MatrixXd& X) {
    // Inizializzazione è già gestita nei layer Dense
    // Qui potremmo aggiungere inizializzazione specifica se necessario
}

Eigen::MatrixXd NeuralNetwork::forward_pass(const Eigen::MatrixXd& X, 
                                          bool training) const {
    ML_CHECK_FITTED(!layers_.empty(), get_model_type());
    
    Eigen::MatrixXd activation = X;
    forward_cache_.clear();
    forward_cache_.reserve(layers_.size());
    
    for (const auto& layer : layers_) {
        layers::LayerCache cache;
        cache.input = activation;
        activation = layer->forward(activation);
        cache.output = activation;
        cache.has_activation = true;
        
        if (training) {
            forward_cache_.push_back(cache);
        }
    }
    
    return activation;
}

void NeuralNetwork::backward_pass(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                double learning_rate) {
    // Forward pass (con cache)
    Eigen::MatrixXd y_pred = forward_pass(X, true);
    
    // Calcolo gradiente della loss
    Eigen::MatrixXd dA = compute_loss_gradient(y_pred, y);
    
    // Backward pass attraverso i layer
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        dA = layers_[i]->backward(dA, learning_rate);
    }
}

double NeuralNetwork::compute_loss(const Eigen::MatrixXd& y_pred, 
                                 const Eigen::VectorXd& y_true) const {
    if (loss_function_ == "binary_crossentropy") {
        // Binary cross-entropy
        Eigen::MatrixXd y_pred_clipped = y_pred.array().max(1e-15).min(1 - 1e-15);
        Eigen::MatrixXd loss = - (y_true.array() * y_pred_clipped.array().log() + 
                                (1 - y_true.array()) * (1 - y_pred_clipped.array()).log());
        return loss.mean();
    } else if (loss_function_ == "mse") {
        // Mean squared error
        return (y_pred - y_true).array().square().mean();
    }
    
    throw ml_exception::InvalidConfigurationException(
        "Unknown loss function: " + loss_function_, get_model_type());
}

Eigen::MatrixXd NeuralNetwork::compute_loss_gradient(const Eigen::MatrixXd& y_pred,
                                                   const Eigen::VectorXd& y_true) const {
    if (loss_function_ == "binary_crossentropy") {
        // Derivata della binary cross-entropy
        Eigen::MatrixXd y_pred_clipped = y_pred.array().max(1e-15).min(1 - 1e-15);
        return (y_pred_clipped - y_true) / static_cast<double>(y_true.size());
    } else if (loss_function_ == "mse") {
        // Derivata del MSE
        return 2.0 * (y_pred - y_true) / static_cast<double>(y_true.size());
    }
    
    throw ml_exception::InvalidConfigurationException(
        "Unknown loss function: " + loss_function_, get_model_type());
}

// Metodo fit principale
void NeuralNetwork::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    ML_CHECK_NOT_EMPTY(y, "y", get_model_type());
    ML_CHECK_DIMENSIONS(X.rows(), y.size(), X.cols(), 1, 
                       "X and y rows", get_model_type());
    
    // Inizializza weights se necessario
    if (layers_.empty()) {
        // Architettura di default se non specificata
        int input_size = X.cols();
        int hidden_size = std::max(32, input_size * 2);
        layers_.push_back(std::make_unique<layers::Dense>(input_size, hidden_size, "relu"));
        layers_.push_back(std::make_unique<layers::Dense>(hidden_size, 1, "sigmoid"));
    }
    
    // Valida architettura
    validate_architecture();
    
    // Split train/validation
    auto [X_train, y_train, X_val, y_val] = [&]() {
        if (validation_split_ > 0) {
            auto splits = utils::MathUtils::train_test_split(
                X, y, validation_split_, 42, get_model_type());
            return std::make_tuple(splits[0].first, splits[0].second,
                                  splits[1].first, splits[1].second);
        }
        return std::make_tuple(X, y, MatrixXd(), VectorXd());
    }();
    
    // Training loop
    loss_history_.clear();
    val_loss_history_.clear();
    accuracy_history_.clear();
    
    for (int epoch = 0; epoch < epochs_; ++epoch) {
        double epoch_loss = 0.0;
        int num_batches = 0;
        
        // Mini-batch training
        Eigen::Index n_samples = X_train.rows();
        std::vector<Eigen::Index> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{42 + epoch});
        
        for (Eigen::Index start = 0; start < n_samples; start += batch_size_) {
            Eigen::Index end = std::min(start + batch_size_, n_samples);
            Eigen::Index batch_size = end - start;
            
            // Crea batch
            Eigen::MatrixXd X_batch(batch_size, X_train.cols());
            Eigen::VectorXd y_batch(batch_size);
            
            for (Eigen::Index i = 0; i < batch_size; ++i) {
                X_batch.row(i) = X_train.row(indices[start + i]);
                y_batch(i) = y_train(indices[start + i]);
            }
            
            // Forward pass
            Eigen::MatrixXd y_pred = forward_pass(X_batch, false);
            
            // Calcola loss
            double batch_loss = compute_loss(y_pred, y_batch);
            epoch_loss += batch_loss;
            
            // Backward pass
            backward_pass(X_batch, y_batch, optimizer_->get_learning_rate());
            
            num_batches++;
        }
        
        // Calcola loss media dell'epoch
        epoch_loss /= num_batches;
        loss_history_.push_back(epoch_loss);
        
        // Validation
        if (validation_split_ > 0 && X_val.rows() > 0) {
            Eigen::MatrixXd y_val_pred = predict(X_val);
            double val_loss = compute_loss(y_val_pred, y_val);
            val_loss_history_.push_back(val_loss);
        }
        
        // Log progresso
        if (verbose_ && (epoch % 10 == 0 || epoch == epochs_ - 1)) {
            std::cout << "Epoch " << std::setw(4) << epoch 
                      << " - Loss: " << std::fixed << std::setprecision(6) << epoch_loss;
            if (!val_loss_history_.empty()) {
                std::cout << " - Val Loss: " << val_loss_history_.back();
            }
            std::cout << std::endl;
        }
    }
}

// Metodi predict
VectorXd NeuralNetwork::predict(const MatrixXd& X) const {
    ML_CHECK_FITTED(!layers_.empty(), get_model_type());
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    
    // Verifica dimensioni input
    int expected_input = layers_.front()->get_input_size();
    if (X.cols() != expected_input) {
        throw ml_exception::FeatureMismatchException(
            expected_input, static_cast<int>(X.cols()), get_model_type());
    }
    
    return forward_pass(X, false).col(0); // Assume output singola colonna
}

Eigen::MatrixXd NeuralNetwork::predict_proba(const MatrixXd& X) const {
    ML_CHECK_FITTED(!layers_.empty(), get_model_type());
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    
    return forward_pass(X, false);
}

double NeuralNetwork::score(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    
    if (loss_function_ == "binary_crossentropy") {
        // Accuracy per classificazione binaria
        VectorXi y_pred_class = (y_pred.array() > 0.5).cast<int>();
        VectorXi y_true_class = y.cast<int>();
        int correct = (y_pred_class.array() == y_true_class.array()).cast<int>().sum();
        return static_cast<double>(correct) / static_cast<double>(y.size());
    } else {
        // R² per regressione
        double ss_res = (y - y_pred).squaredNorm();
        double ss_tot = (y.array() - y.mean()).square().sum();
        return 1.0 - (ss_res / ss_tot);
    }
}

int NeuralNetwork::get_num_parameters() const {
    int total = 0;
    for (const auto& layer : layers_) {
        total += layer->get_parameter_count();
    }
    return total;
}

void NeuralNetwork::validate_architecture() const {
    if (layers_.empty()) {
        throw ml_exception::InvalidConfigurationException(
            "Network has no layers", get_model_type());
    }
    
    // Verifica connessioni tra layer
    for (size_t i = 1; i < layers_.size(); ++i) {
        int prev_output = layers_[i-1]->get_output_size();
        int curr_input = layers_[i]->get_input_size();
        
        if (prev_output != curr_input) {
            throw ml_exception::DimensionMismatchException(
                "layer connection",
                prev_output, 1,
                curr_input, 1,
                get_model_type());
        }
    }
}

// In neural_network.cpp
void NeuralNetwork::add_convolutional_layer(int input_channels, int output_channels,
                                          int kernel_size, int stride, int padding,
                                          const std::string& activation) {
    
    // Determina dimensioni input dal layer precedente
    int prev_output_size = layers_.empty() ? 0 : layers_.back()->get_output_size();
    
    // Se è il primo layer, abbiamo bisogno di specificare le dimensioni spaziali
    if (layers_.empty()) {
        throw ml_exception::InvalidConfigurationException(
            "For first convolutional layer, use constructor with spatial dimensions",
            get_model_type());
    }
    
    // Crea layer convoluzionale
    auto conv_layer = std::make_unique<layers::Convolutional>(
        input_channels, output_channels, kernel_size, stride, padding, activation);
    
    add_layer(std::move(conv_layer));
}

void NeuralNetwork::add_pooling_layer(int pool_size, int stride,
                                    layers::Pooling::PoolType type, int channels) {
    auto pool_layer = std::make_unique<layers::Pooling>(
        pool_size, stride, type, channels);
    
    add_layer(std::move(pool_layer));
}

void NeuralNetwork::add_recurrent_layer(int hidden_size,
                                      const std::string& activation,
                                      bool return_sequences) {
    
    int input_size = layers_.empty() ? 0 : layers_.back()->get_output_size();
    if (input_size == 0) {
        throw ml_exception::InvalidConfigurationException(
            "Cannot determine input size for recurrent layer", get_model_type());
    }
    
    auto rnn_layer = std::make_unique<layers::Recurrent>(
        input_size, hidden_size, activation, return_sequences);
    
    add_layer(std::move(rnn_layer));
}

void NeuralNetwork::add_lstm_layer(int hidden_size, bool return_sequences) {
    int input_size = layers_.empty() ? 0 : layers_.back()->get_output_size();
    if (input_size == 0) {
        throw ml_exception::InvalidConfigurationException(
            "Cannot determine input size for LSTM layer", get_model_type());
    }
    
    auto lstm_layer = std::make_unique<layers::Recurrent>(
        input_size, hidden_size, layers::Recurrent::LSTM, return_sequences);
    
    add_layer(std::move(lstm_layer));
}

void NeuralNetwork::add_gru_layer(int hidden_size, bool return_sequences) {
    int input_size = layers_.empty() ? 0 : layers_.back()->get_output_size();
    if (input_size == 0) {
        throw ml_exception::InvalidConfigurationException(
            "Cannot determine input size for GRU layer", get_model_type());
    }
    
    auto gru_layer = std::make_unique<layers::Recurrent>(
        input_size, hidden_size, layers::Recurrent::GRU, return_sequences);
    
    add_layer(std::move(gru_layer));
}

// Serializzazione
void NeuralNetwork::serialize_binary(std::ostream& out) const {
    using namespace utils;
    
    // Serializza parametri di configurazione
    out.write(reinterpret_cast<const char*>(&batch_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&epochs_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&validation_split_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&verbose_), sizeof(bool));
    
    // Serializza loss function
    size_t loss_len = loss_function_.size();
    out.write(reinterpret_cast<const char*>(&loss_len), sizeof(size_t));
    out.write(loss_function_.c_str(), loss_len);
    
    // Serializza numero di layer
    size_t num_layers = layers_.size();
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    
    // Serializza ogni layer
    for (const auto& layer : layers_) {
        // Serializza tipo di layer
        std::string layer_type = layer->get_type();
        size_t type_len = layer_type.size();
        out.write(reinterpret_cast<const char*>(&type_len), sizeof(size_t));
        out.write(layer_type.c_str(), type_len);
        
        // Serializza il layer
        layer->serialize(out);
    }
    
    // Serializza optimizer
    if (optimizer_) {
        std::string optimizer_type = optimizer_->get_type();
        size_t opt_len = optimizer_type.size();
        out.write(reinterpret_cast<const char*>(&opt_len), sizeof(size_t));
        out.write(optimizer_type.c_str(), opt_len);
        optimizer_->serialize(out);
    }
    
    // Serializza history
    eigen_utils::serialize_eigen_vector(
        Eigen::Map<const Eigen::VectorXd>(loss_history_.data(), loss_history_.size()), 
        out);
    eigen_utils::serialize_eigen_vector(
        Eigen::Map<const Eigen::VectorXd>(val_loss_history_.data(), val_loss_history_.size()), 
        out);
    eigen_utils::serialize_eigen_vector(
        Eigen::Map<const Eigen::VectorXd>(accuracy_history_.data(), accuracy_history_.size()), 
        out);
}

void NeuralNetwork::deserialize_binary(std::istream& in) {
    using namespace utils;
    
    // Deserializza parametri di configurazione
    in.read(reinterpret_cast<char*>(&batch_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&epochs_), sizeof(int));
    in.read(reinterpret_cast<char*>(&validation_split_), sizeof(double));
    in.read(reinterpret_cast<char*>(&verbose_), sizeof(bool));
    
    // Deserializza loss function
    size_t loss_len;
    in.read(reinterpret_cast<char*>(&loss_len), sizeof(size_t));
    loss_function_.resize(loss_len);
    in.read(&loss_function_[0], loss_len);
    
    // Deserializza numero di layer
    size_t num_layers;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    layers_.clear();
    layers_.reserve(num_layers);
    
    // Deserializza ogni layer
    for (size_t i = 0; i < num_layers; ++i) {
        // Leggi tipo di layer
        size_t type_len;
        in.read(reinterpret_cast<char*>(&type_len), sizeof(size_t));
        std::string layer_type(type_len, '\0');
        in.read(&layer_type[0], type_len);
        
        // Crea layer del tipo corretto
        if (layer_type == "Dense") {
            auto layer = std::make_unique<layers::Dense>(1, 1); // Dummy, verrà sovrascritto
            layer->deserialize(in);
            layers_.push_back(std::move(layer));
        } else {
            throw ml_exception::DeserializationException(
                "model file", "unknown layer type: " + layer_type, get_model_type());
        }
    }
    
    // Deserializza optimizer
    size_t opt_len;
    in.read(reinterpret_cast<char*>(&opt_len), sizeof(size_t));
    std::string optimizer_type(opt_len, '\0');
    in.read(&optimizer_type[0], opt_len);
    
    if (optimizer_type == "SGD") {
        optimizer_ = std::make_unique<optimizers::SGD>(0.01);
        optimizer_->deserialize(in);
    } else {
        throw ml_exception::DeserializationException(
            "model file", "unknown optimizer type: " + optimizer_type, get_model_type());
    }
    
    // Deserializza history
    Eigen::VectorXd loss_vec, val_loss_vec, acc_vec;
    eigen_utils::deserialize_eigen_vector(loss_vec, in);
    eigen_utils::deserialize_eigen_vector(val_loss_vec, in);
    eigen_utils::deserialize_eigen_vector(acc_vec, in);
    
    loss_history_.assign(loss_vec.data(), loss_vec.data() + loss_vec.size());
    val_loss_history_.assign(val_loss_vec.data(), val_loss_vec.data() + val_loss_vec.size());
    accuracy_history_.assign(acc_vec.data(), acc_vec.data() + acc_vec.size());
}

std::string NeuralNetwork::to_string() const {
    std::ostringstream oss;
    oss << "NeuralNetwork ["
        << "Layers: " << layers_.size()
        << ", Parameters: " << get_num_parameters()
        << ", Loss: " << loss_function_
        << ", Batch size: " << batch_size_
        << ", Epochs: " << epochs_;
    
    if (!layers_.empty()) {
        oss << "\nArchitecture: ";
        oss << layers_.front()->get_input_size();
        for (const auto& layer : layers_) {
            oss << " → " << layer->get_output_size();
        }
    }
    
    oss << "]";
    return oss.str();
}

void NeuralNetwork::summary() const {
    std::cout << std::string(65, '-') << std::endl;
    std::cout << std::left << std::setw(25) << "Layer (type)" 
              << std::setw(20) << "Output Shape" 
              << std::setw(15) << "Param #" << std::endl;
    std::cout << std::string(65, '=') << std::endl;

    int total_params = 0;
    for (const auto& layer : layers_) {
        std::string layer_info = layer->get_type();
        int params = layer->get_parameter_count();
        total_params += params;

        std::cout << std::left << std::setw(25) << layer_info
                  << std::setw(20) << (std::to_string(layer->get_output_size()))
                  << std::setw(15) << params << std::endl;
    }

    std::cout << std::string(65, '=') << std::endl;
    std::cout << "Total params: " << total_params << std::endl;
    std::cout << "Loss Function: " << loss_function_ << std::endl;
    std::cout << std::string(65, '-') << std::endl;
}