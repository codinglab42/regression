#include "components/optimizers/sgd.h"
#include "utils/serializable.h"
#include <stdexcept>

using namespace Eigen;
using namespace optimizers;

// Costruttore
SGD::SGD(double learning_rate, double momentum)
    : learning_rate_(learning_rate), momentum_(momentum),
      velocity_initialized_(false) {
    
    if (learning_rate <= 0) {
        throw ml_exception::InvalidParameterException(
            "learning_rate", "must be > 0", "SGD");
    }
    
    if (momentum < 0 || momentum >= 1) {
        throw ml_exception::InvalidParameterException(
            "momentum", "must be in [0, 1)", "SGD");
    }
}

void SGD::initialize_velocity(const MatrixXd& weights, const VectorXd& biases) {
    if (!velocity_initialized_) {
        velocity_weights_ = MatrixXd::Zero(weights.rows(), weights.cols());
        velocity_biases_ = VectorXd::Zero(biases.size());
        velocity_initialized_ = true;
    }
}

void SGD::update(MatrixXd& weights, const MatrixXd& gradients) {
    if (weights.rows() != gradients.rows() || weights.cols() != gradients.cols()) {
        throw ml_exception::DimensionMismatchException(
            "weights and gradients",
            weights.rows(), weights.cols(),
            gradients.rows(), gradients.cols(),
            "SGD");
    }
    
    if (momentum_ > 0) {
        initialize_velocity(weights, VectorXd::Zero(1));
        
        // Aggiorna velocity con momentum
        velocity_weights_ = momentum_ * velocity_weights_ + learning_rate_ * gradients;
        weights -= velocity_weights_;
    } else {
        // SGD semplice
        weights -= learning_rate_ * gradients;
    }
}

void SGD::update(VectorXd& biases, const VectorXd& gradients) {
    if (biases.size() != gradients.size()) {
        throw ml_exception::DimensionMismatchException(
            "biases and gradients",
            biases.size(), 1,
            gradients.size(), 1,
            "SGD");
    }
    
    if (momentum_ > 0) {
        initialize_velocity(MatrixXd::Zero(1, 1), biases);
        
        // Aggiorna velocity con momentum
        velocity_biases_ = momentum_ * velocity_biases_ + learning_rate_ * gradients;
        biases -= velocity_biases_;
    } else {
        // SGD semplice
        biases -= learning_rate_ * gradients;
    }
}

void SGD::set_momentum(double momentum) {
    if (momentum < 0 || momentum >= 1) {
        throw ml_exception::InvalidParameterException(
            "momentum", "must be in [0, 1)", "SGD");
    }
    momentum_ = momentum;
    
    // Reset velocity se momentum cambia
    if (momentum_ == 0) {
        velocity_initialized_ = false;
    }
}

// Serializzazione
void SGD::serialize(std::ostream& out) const {
    using namespace utils;
    
    out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&momentum_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&velocity_initialized_), sizeof(bool));
    
    if (velocity_initialized_) {
        eigen_utils::serialize_eigen(velocity_weights_, out);
        eigen_utils::serialize_eigen_vector(velocity_biases_, out);
    }
}

void SGD::deserialize(std::istream& in) {
    using namespace utils;
    
    in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(double));
    in.read(reinterpret_cast<char*>(&momentum_), sizeof(double));
    in.read(reinterpret_cast<char*>(&velocity_initialized_), sizeof(bool));
    
    if (velocity_initialized_) {
        eigen_utils::deserialize_eigen(velocity_weights_, in);
        eigen_utils::deserialize_eigen_vector(velocity_biases_, in);
    }
}