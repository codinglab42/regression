#include "components/layers/dense.h"
#include "components/activation/activation.h"
#include "utils/serializable.h"
#include <cmath>
#include <random>
#include <stdexcept>

using namespace Eigen;

namespace layers {

    // Costruttore
    Dense::Dense(int input_size, int output_size, 
                const std::string& activation,
                const std::string& weight_initializer,
                const std::string& bias_initializer)
        : input_size_(input_size), output_size_(output_size),
        l1_lambda_(0.0), l2_lambda_(0.0) {
        
        if (input_size <= 0 || output_size <= 0) {
            throw ml_exception::InvalidParameterException(
                "layer dimensions", "must be > 0", "Dense");
        }
        
        weights_.resize(output_size, input_size);
        biases_.resize(output_size);
        
        initialize_weights(weight_initializer);
        initialize_biases(bias_initializer);
        
        activation_ = activation::create_activation(activation);
        if (!activation_) {
            throw ml_exception::InvalidParameterException(
                "activation", "unknown activation function: " + activation, "Dense");
        }
        
        clear_cache();
    }

    // Inizializzazione pesi
    void Dense::initialize_weights(const std::string& initializer) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if (initializer == "he") {
            // He initialization per ReLU
            double stddev = std::sqrt(2.0 / input_size_);
            std::normal_distribution<> dist(0.0, stddev);
            for (int i = 0; i < output_size_; ++i) {
                for (int j = 0; j < input_size_; ++j) {
                    weights_(i, j) = dist(gen);
                }
            }
        } else if (initializer == "xavier" || initializer == "glorot") {
            // Xavier/Glorot initialization
            double limit = std::sqrt(6.0 / (input_size_ + output_size_));
            std::uniform_real_distribution<> dist(-limit, limit);
            for (int i = 0; i < output_size_; ++i) {
                for (int j = 0; j < input_size_; ++j) {
                    weights_(i, j) = dist(gen);
                }
            }
        } else if (initializer == "zeros") {
            weights_.setZero();
        } else if (initializer == "ones") {
            weights_.setOnes();
        } else if (initializer == "random") {
            std::normal_distribution<> dist(0.0, 0.01);
            for (int i = 0; i < output_size_; ++i) {
                for (int j = 0; j < input_size_; ++j) {
                    weights_(i, j) = dist(gen);
                }
            }
        } else {
            throw ml_exception::InvalidParameterException(
                "weight_initializer", 
                "unknown initializer: " + initializer, "Dense");
        }
    }

    // Inizializzazione bias
    void Dense::initialize_biases(const std::string& initializer) {
        if (initializer == "zeros") {
            biases_.setZero();
        } else if (initializer == "ones") {
            biases_.setOnes();
        } else if (initializer == "random") {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> dist(0.0, 0.01);
            for (int i = 0; i < output_size_; ++i) {
                biases_(i) = dist(gen);
            }
        } else {
            throw ml_exception::InvalidParameterException(
                "bias_initializer", 
                "unknown initializer: " + initializer, "Dense");
        }
    }

    // Forward propagation
    MatrixXd Dense::forward(const MatrixXd& input) {
        if (input.cols() != input_size_) {
            throw ml_exception::DimensionMismatchException(
                "input columns",
                input_size_, 1,
                input.cols(), 1,
                "Dense");
        }
        
        // Salva input nella cache
        cache_.input = input;
        
        // Calcola z = input * weights^T + bias
        cache_.z = input * weights_.transpose();
        cache_.z.rowwise() += biases_.transpose();
        
        // Applica funzione di attivazione
        cache_.output = activation_->forward(cache_.z);
        cache_.has_activation = true;
        
        return cache_.output;
    }

    // Backward propagation
    MatrixXd Dense::backward(const MatrixXd& gradient, double learning_rate) {
        if (!cache_.has_activation) {
            throw ml_exception::InvalidConfigurationException(
                "Cache not initialized. Call forward() first.", "Dense");
        }
        
        // Gradiente rispetto a z
        MatrixXd dZ = activation_->backward(gradient, cache_.z);
        
        // Gradiente rispetto ai pesi e bias
        MatrixXd dW = dZ.transpose() * cache_.input;
        VectorXd db = dZ.colwise().sum();
        
        // Aggiungi regolarizzazione se specificata
        if (l1_lambda_ > 0) {
            dW += l1_regularization_gradient();
        }
        if (l2_lambda_ > 0) {
            dW += l2_regularization_gradient();
        }
        
        // Normalizza per batch size
        double batch_size = cache_.input.rows();
        dW /= batch_size;
        db /= batch_size;
        
        // Aggiorna pesi e bias
        weights_ -= learning_rate * dW;
        biases_ -= learning_rate * db;
        
        // Gradiente rispetto all'input (per propagare al layer precedente)
        MatrixXd dA_prev = dZ * weights_;
        
        // Pulisci cache (opzionale, per risparmiare memoria)
        // clear_cache();
        
        return dA_prev;
    }

    // Gradienti di regolarizzazione
    MatrixXd Dense::l1_regularization_gradient() const {
        MatrixXd grad = weights_.unaryExpr([](double w) {
            return (w > 0) ? 1.0 : ((w < 0) ? -1.0 : 0.0);
        });
        return l1_lambda_ * grad;
    }

    MatrixXd Dense::l2_regularization_gradient() const {
        return l2_lambda_ * weights_;
    }

    // Setters
    void Dense::set_weights(const MatrixXd& weights) {
        if (weights.rows() != output_size_ || weights.cols() != input_size_) {
            throw ml_exception::DimensionMismatchException(
                "weights",
                output_size_, input_size_,
                weights.rows(), weights.cols(),
                "Dense");
        }
        weights_ = weights;
    }

    void Dense::set_biases(const VectorXd& biases) {
        if (biases.size() != output_size_) {
            throw ml_exception::DimensionMismatchException(
                "biases",
                output_size_, 1,
                biases.size(), 1,
                "Dense");
        }
        biases_ = biases;
    }

    void Dense::set_activation(const std::string& activation) {
        activation_ = activation::create_activation(activation);
        if (!activation_) {
            throw ml_exception::InvalidParameterException(
                "activation", "unknown activation function: " + activation, "Dense");
        }
    }

    void Dense::set_regularization(double l1, double l2) {
        if (l1 < 0 || l2 < 0) {
            throw ml_exception::InvalidParameterException(
                "regularization", "lambda values must be >= 0", "Dense");
        }
        l1_lambda_ = l1;
        l2_lambda_ = l2;
    }

    // Informazioni
    int Dense::get_parameter_count() const {
        return weights_.size() + biases_.size();
    }

    std::string Dense::get_config() const {
        std::ostringstream oss;
        oss << "Dense(input=" << input_size_ 
            << ", output=" << output_size_
            << ", activation=" << activation_->get_type()
            << ", params=" << get_parameter_count() << ")";
        return oss.str();
    }

    // Cache management
    void Dense::clear_cache() {
        cache_.input = MatrixXd();
        cache_.z = MatrixXd();
        cache_.output = MatrixXd();
        cache_.has_activation = false;
    }

    // Serializzazione
    void Dense::serialize(std::ostream& out) const {
        using namespace utils;
        
        // Serializza dimensioni
        out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&output_size_), sizeof(int));
        
        // Serializza pesi e bias
        eigen_utils::serialize_eigen(weights_, out);
        eigen_utils::serialize_eigen_vector(biases_, out);
        
        // Serializza parametri di regolarizzazione
        out.write(reinterpret_cast<const char*>(&l1_lambda_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&l2_lambda_), sizeof(double));
        
        // Serializza tipo di attivazione
        std::string act_type = activation_->get_type();
        size_t act_len = act_type.size();
        out.write(reinterpret_cast<const char*>(&act_len), sizeof(size_t));
        out.write(act_type.c_str(), act_len);
    }

    void Dense::deserialize(std::istream& in) {
        using namespace utils;
        
        // Deserializza dimensioni
        in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
        in.read(reinterpret_cast<char*>(&output_size_), sizeof(int));
        
        // Re-inizializza matrici
        weights_.resize(output_size_, input_size_);
        biases_.resize(output_size_);
        
        // Deserializza pesi e bias
        eigen_utils::deserialize_eigen(weights_, in);
        eigen_utils::deserialize_eigen_vector(biases_, in);
        
        // Deserializza parametri di regolarizzazione
        in.read(reinterpret_cast<char*>(&l1_lambda_), sizeof(double));
        in.read(reinterpret_cast<char*>(&l2_lambda_), sizeof(double));
        
        // Deserializza tipo di attivazione
        size_t act_len;
        in.read(reinterpret_cast<char*>(&act_len), sizeof(size_t));
        std::string act_type(act_len, '\0');
        in.read(&act_type[0], act_len);
        
        // Ricrea funzione di attivazione
        activation_ = activation::create_activation(act_type);
        if (!activation_) {
            throw ml_exception::DeserializationException(
                "layer file", "unknown activation type: " + act_type, "Dense");
        }
        
        // Pulisci cache
        clear_cache();
    }
}