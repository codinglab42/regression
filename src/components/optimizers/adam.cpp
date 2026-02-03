#include "components/optimizers/adam.h"
#include "utils/serializable.h"
#include <cmath>
#include <stdexcept>

using namespace Eigen;
using namespace optimizers;

// Costruttore
Adam::Adam(double learning_rate, double beta1, double beta2, double epsilon)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      t_(0), initialized_(false) {
    
    if (learning_rate <= 0) {
        throw ml_exception::InvalidParameterException(
            "learning_rate", "must be > 0", "Adam");
    }
    
    if (beta1 <= 0 || beta1 >= 1) {
        throw ml_exception::InvalidParameterException(
            "beta1", "must be in (0, 1)", "Adam");
    }
    
    if (beta2 <= 0 || beta2 >= 1) {
        throw ml_exception::InvalidParameterException(
            "beta2", "must be in (0, 1)", "Adam");
    }
    
    if (epsilon <= 0) {
        throw ml_exception::InvalidParameterException(
            "epsilon", "must be > 0", "Adam");
    }
}

void Adam::initialize_moments(const MatrixXd& weights, const VectorXd& biases) {
    if (!initialized_) {
        m_weights_ = MatrixXd::Zero(weights.rows(), weights.cols());
        v_weights_ = MatrixXd::Zero(weights.rows(), weights.cols());
        m_biases_ = VectorXd::Zero(biases.size());
        v_biases_ = VectorXd::Zero(biases.size());
        t_ = 0;
        initialized_ = true;
    }
}

void Adam::update(MatrixXd& weights, const MatrixXd& gradients) {
    if (weights.rows() != gradients.rows() || weights.cols() != gradients.cols()) {
        throw ml_exception::DimensionMismatchException(
            "weights and gradients",
            weights.rows(), weights.cols(),
            gradients.rows(), gradients.cols(),
            "Adam");
    }
    
    initialize_moments(weights, VectorXd::Zero(1));
    
    // Incrementa timestep
    t_++;
    
    // Aggiorna moment estimates
    m_weights_ = beta1_ * m_weights_ + (1.0 - beta1_) * gradients;
    v_weights_ = beta2_ * v_weights_ + (1.0 - beta2_) * gradients.array().square().matrix();
    
    // Bias correction
    double m_hat_correction = 1.0 / (1.0 - std::pow(beta1_, t_));
    double v_hat_correction = 1.0 / (1.0 - std::pow(beta2_, t_));
    
    MatrixXd m_hat = m_weights_ * m_hat_correction;
    MatrixXd v_hat = v_weights_ * v_hat_correction;
    
    // Aggiorna pesi
    weights.array() -= learning_rate_ * m_hat.array() / (v_hat.array().sqrt() + epsilon_);
}

void Adam::update(VectorXd& biases, const VectorXd& gradients) {
    if (biases.size() != gradients.size()) {
        throw ml_exception::DimensionMismatchException(
            "biases and gradients",
            biases.size(), 1,
            gradients.size(), 1,
            "Adam");
    }
    
    initialize_moments(MatrixXd::Zero(1, 1), biases);
    
    // Incrementa timestep (già fatto per weights, ma manteniamo consistenza)
    if (t_ == 0) t_ = 1;
    
    // Aggiorna moment estimates
    m_biases_ = beta1_ * m_biases_ + (1.0 - beta1_) * gradients;
    v_biases_ = beta2_ * v_biases_ + (1.0 - beta2_) * gradients.array().square().matrix();
    
    // Bias correction
    double m_hat_correction = 1.0 / (1.0 - std::pow(beta1_, t_));
    double v_hat_correction = 1.0 / (1.0 - std::pow(beta2_, t_));
    
    VectorXd m_hat = m_biases_ * m_hat_correction;
    VectorXd v_hat = v_biases_ * v_hat_correction;
    
    // Aggiorna bias
    biases.array() -= learning_rate_ * m_hat.array() / (v_hat.array().sqrt() + epsilon_);
}

void Adam::set_betas(double beta1, double beta2) {
    if (beta1 <= 0 || beta1 >= 1) {
        throw ml_exception::InvalidParameterException(
            "beta1", "must be in (0, 1)", "Adam");
    }
    
    if (beta2 <= 0 || beta2 >= 1) {
        throw ml_exception::InvalidParameterException(
            "beta2", "must be in (0, 1)", "Adam");
    }
    
    beta1_ = beta1;
    beta2_ = beta2;
    reset();
}

void Adam::set_epsilon(double epsilon) {
    if (epsilon <= 0) {
        throw ml_exception::InvalidParameterException(
            "epsilon", "must be > 0", "Adam");
    }
    epsilon_ = epsilon;
}

void Adam::reset() {
    initialized_ = false;
    t_ = 0;
}

// Serializzazione
void Adam::serialize(std::ostream& out) const {
    using namespace utils;
    
    out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&beta1_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&beta2_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&epsilon_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&t_), sizeof(long long));
    out.write(reinterpret_cast<const char*>(&initialized_), sizeof(bool));
    
    if (initialized_) {
        eigen_utils::serialize_eigen(m_weights_, out);
        eigen_utils::serialize_eigen(v_weights_, out);
        eigen_utils::serialize_eigen_vector(m_biases_, out);
        eigen_utils::serialize_eigen_vector(v_biases_, out);
    }
}

void Adam::deserialize(std::istream& in) {
    using namespace utils;
    
    in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(double));
    in.read(reinterpret_cast<char*>(&beta1_), sizeof(double));
    in.read(reinterpret_cast<char*>(&beta2_), sizeof(double));
    in.read(reinterpret_cast<char*>(&epsilon_), sizeof(double));
    in.read(reinterpret_cast<char*>(&t_), sizeof(long long));
    in.read(reinterpret_cast<char*>(&initialized_), sizeof(bool));
    
    if (initialized_) {
        eigen_utils::deserialize_eigen(m_weights_, in);
        eigen_utils::deserialize_eigen(v_weights_, in);
        eigen_utils::deserialize_eigen_vector(m_biases_, in);
        eigen_utils::deserialize_eigen_vector(v_biases_, in);
    }
}