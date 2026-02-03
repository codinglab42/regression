#include "components/activation/activation.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;
using namespace activation;

// Factory per creare funzioni di attivazione
std::unique_ptr<Activation> activation::create_activation(const std::string& type) {
    if (type == "relu") {
        return std::make_unique<ReLU>();
    } else if (type == "sigmoid") {
        return std::make_unique<Sigmoid>();
    } else if (type == "tanh") {
        return std::make_unique<Tanh>();
    } else if (type == "softmax") {
        return std::make_unique<Softmax>();
    } else if (type == "leaky_relu") {
        return std::make_unique<LeakyReLU>();
    } else if (type == "linear" || type == "identity") {
        // Implementazione lineare (identità)
        class Linear : public Activation {
        public:
            MatrixXd forward(const MatrixXd& z) override { return z; }
            MatrixXd backward(const MatrixXd& dA, const MatrixXd& z) override { return dA; }
            std::string get_type() const override { return "linear"; }
            Linear* clone() const override { return new Linear(*this); }
        };
        return std::make_unique<Linear>();
    }
    return nullptr;
}

// ReLU
MatrixXd ReLU::forward(const MatrixXd& z) {
    return z.unaryExpr([](double v) { return std::max(0.0, v); });
}

MatrixXd ReLU::backward(const MatrixXd& dA, const MatrixXd& z) {
    MatrixXd dZ = z.unaryExpr([](double v) { return (v > 0) ? 1.0 : 0.0; });
    return dA.array() * dZ.array();
}

// Sigmoid
MatrixXd Sigmoid::forward(const MatrixXd& z) {
    // Implementazione numericamente stabile
    return z.unaryExpr([](double v) {
        if (v >= 0) {
            return 1.0 / (1.0 + std::exp(-v));
        } else {
            double exp_v = std::exp(v);
            return exp_v / (1.0 + exp_v);
        }
    });
}

MatrixXd Sigmoid::backward(const MatrixXd& dA, const MatrixXd& z) {
    MatrixXd s = forward(z);
    MatrixXd dZ = s.array() * (1.0 - s.array());
    return dA.array() * dZ.array();
}

// Tanh
MatrixXd Tanh::forward(const MatrixXd& z) {
    return z.unaryExpr([](double v) { return std::tanh(v); });
}

MatrixXd Tanh::backward(const MatrixXd& dA, const MatrixXd& z) {
    MatrixXd tanh_z = forward(z);
    MatrixXd dZ = 1.0 - tanh_z.array().square();
    return dA.array() * dZ.array();
}

// Softmax
MatrixXd Softmax::forward(const MatrixXd& z) {
    // Implementazione numericamente stabile
    MatrixXd exp_z = z.rowwise() - z.colwise().maxCoeff();
    exp_z = exp_z.array().exp();
    MatrixXd sum_exp = exp_z.rowwise().sum();
    
    // Normalizza
    for (int i = 0; i < exp_z.rows(); ++i) {
        exp_z.row(i) /= sum_exp(i);
    }
    
    return exp_z;
}

MatrixXd Softmax::backward(const MatrixXd& dA, const MatrixXd& z) {
    // Per softmax combinato con cross-entropy, il gradiente è semplice
    // dZ = y_pred - y_true (se usato con cross-entropy)
    // Qui restituiamo dA come placeholder, la logica completa è nella loss
    return dA;
}

// Leaky ReLU
MatrixXd LeakyReLU::forward(const MatrixXd& z) {
    return z.unaryExpr([this](double v) { 
        return (v > 0) ? v : alpha_ * v; 
    });
}

MatrixXd LeakyReLU::backward(const MatrixXd& dA, const MatrixXd& z) {
    MatrixXd dZ = z.unaryExpr([this](double v) { 
        return (v > 0) ? 1.0 : alpha_; 
    });
    return dA.array() * dZ.array();
}