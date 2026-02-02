#include "layers/recurrent.h"
#include "activation/activation.h"
#include "serialization/serializable.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;
using namespace layers;

// ===================== COSTRUTTORI =====================

Recurrent::Recurrent(int input_size, int hidden_size, 
                     const std::string& activation,
                     bool return_sequences)
    : input_size_(input_size), hidden_size_(hidden_size),
      sequence_length_(-1), rnn_type_(SIMPLE),
      return_sequences_(return_sequences),
      dropout_rate_(0.0), recurrent_dropout_rate_(0.0),
      clip_value_(5.0) {
    
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "Recurrent");
    ML_CHECK_PARAM(hidden_size > 0, "hidden_size", "must be > 0", "Recurrent");
    
    initialize_weights_simple();
    activation_ = activation::create_activation(activation);
    gate_activation_ = activation::create_activation("sigmoid");
    
    if (!activation_ || !gate_activation_) {
        throw ml_exception::InvalidParameterException(
            "activation", "unknown activation function", "Recurrent");
    }
    
    clear_cache();
}

Recurrent::Recurrent(int input_size, int hidden_size,
                     RNNType type, bool return_sequences)
    : input_size_(input_size), hidden_size_(hidden_size),
      sequence_length_(-1), rnn_type_(type),
      return_sequences_(return_sequences),
      dropout_rate_(0.0), recurrent_dropout_rate_(0.0),
      clip_value_(5.0) {
    
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "Recurrent");
    ML_CHECK_PARAM(hidden_size > 0, "hidden_size", "must be > 0", "Recurrent");
    
    switch (type) {
        case LSTM:
            initialize_weights_lstm();
            activation_ = activation::create_activation("tanh");
            gate_activation_ = activation::create_activation("sigmoid");
            break;
        case GRU:
            initialize_weights_gru();
            activation_ = activation::create_activation("tanh");
            gate_activation_ = activation::create_activation("sigmoid");
            break;
        case SIMPLE:
            initialize_weights_simple();
            activation_ = activation::create_activation("tanh");
            gate_activation_ = activation::create_activation("sigmoid");
            break;
    }
    
    if (!activation_ || !gate_activation_) {
        throw ml_exception::InvalidParameterException(
            "activation", "unknown activation function", "Recurrent");
    }
    
    clear_cache();
}

// ===================== INIZIALIZZAZIONE PESI =====================

void Recurrent::initialize_weights_simple() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double limit = std::sqrt(6.0 / (input_size_ + hidden_size_));
    std::uniform_real_distribution<> dist(-limit, limit);
    
    W_x_ = MatrixXd::Zero(hidden_size_, input_size_);
    W_h_ = MatrixXd::Zero(hidden_size_, hidden_size_);
    b_ = VectorXd::Zero(hidden_size_);
    
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            W_x_(i, j) = dist(gen);
        }
        for (int j = 0; j < hidden_size_; ++j) {
            W_h_(i, j) = dist(gen);
        }
        b_(i) = dist(gen) * 0.1;
    }
}

void Recurrent::initialize_weights_lstm() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double limit = std::sqrt(6.0 / (input_size_ + hidden_size_));
    std::uniform_real_distribution<> dist(-limit, limit);
    
    // Variabili locali
    MatrixXd W_i = MatrixXd::Zero(hidden_size_, input_size_);
    MatrixXd W_f = MatrixXd::Zero(hidden_size_, input_size_);
    MatrixXd W_c = MatrixXd::Zero(hidden_size_, input_size_);
    MatrixXd W_o = MatrixXd::Zero(hidden_size_, input_size_);
    
    MatrixXd U_i = MatrixXd::Zero(hidden_size_, hidden_size_);
    MatrixXd U_f = MatrixXd::Zero(hidden_size_, hidden_size_);
    MatrixXd U_c = MatrixXd::Zero(hidden_size_, hidden_size_);
    MatrixXd U_o = MatrixXd::Zero(hidden_size_, hidden_size_);
    
    VectorXd b_i = VectorXd::Zero(hidden_size_);
    VectorXd b_f = VectorXd::Zero(hidden_size_);
    VectorXd b_c = VectorXd::Zero(hidden_size_);
    VectorXd b_o = VectorXd::Zero(hidden_size_);
    
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            W_i(i, j) = dist(gen);
            W_f(i, j) = dist(gen);
            W_c(i, j) = dist(gen);
            W_o(i, j) = dist(gen);
        }
        for (int j = 0; j < hidden_size_; ++j) {
            U_i(i, j) = dist(gen);
            U_f(i, j) = dist(gen);
            U_c(i, j) = dist(gen);
            U_o(i, j) = dist(gen);
        }
        b_i(i) = dist(gen) * 0.1;
        b_f(i) = dist(gen) * 0.1;
        b_c(i) = dist(gen) * 0.1;
        b_o(i) = dist(gen) * 0.1;
    }
    
    // Combina nelle matrici membro
    W_x_.resize(hidden_size_ * 4, input_size_);
    W_x_ << W_i, W_f, W_c, W_o;
    
    W_h_.resize(hidden_size_ * 4, hidden_size_);
    W_h_ << U_i, U_f, U_c, U_o;
    
    b_.resize(hidden_size_ * 4);
    b_ << b_i, b_f, b_c, b_o;
}

void Recurrent::initialize_weights_gru() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double limit = std::sqrt(6.0 / (input_size_ + hidden_size_));
    std::uniform_real_distribution<> dist(-limit, limit);
    
    // Variabili locali
    MatrixXd W_z = MatrixXd::Zero(hidden_size_, input_size_);
    MatrixXd W_r = MatrixXd::Zero(hidden_size_, input_size_);
    MatrixXd W_h = MatrixXd::Zero(hidden_size_, input_size_);
    
    MatrixXd U_z = MatrixXd::Zero(hidden_size_, hidden_size_);
    MatrixXd U_r = MatrixXd::Zero(hidden_size_, hidden_size_);
    MatrixXd U_h = MatrixXd::Zero(hidden_size_, hidden_size_);
    
    VectorXd b_z = VectorXd::Zero(hidden_size_);
    VectorXd b_r = VectorXd::Zero(hidden_size_);
    VectorXd b_h = VectorXd::Zero(hidden_size_);
    
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            W_z(i, j) = dist(gen);
            W_r(i, j) = dist(gen);
            W_h(i, j) = dist(gen);
        }
        for (int j = 0; j < hidden_size_; ++j) {
            U_z(i, j) = dist(gen);
            U_r(i, j) = dist(gen);
            U_h(i, j) = dist(gen);
        }
        b_z(i) = dist(gen) * 0.1;
        b_r(i) = dist(gen) * 0.1;
        b_h(i) = dist(gen) * 0.1;
    }
    
    // Combina nelle matrici membro
    W_x_.resize(hidden_size_ * 3, input_size_);
    W_x_ << W_z, W_r, W_h;
    
    W_h_.resize(hidden_size_ * 3, hidden_size_);
    W_h_ << U_z, U_r, U_h;
    
    b_.resize(hidden_size_ * 3);
    b_ << b_z, b_r, b_h;
}

// ===================== FORWARD PASS =====================

MatrixXd Recurrent::forward(const Eigen::MatrixXd& input) {
    // Determina sequence_length
    sequence_length_ = input.cols() / input_size_;
    if (sequence_length_ * input_size_ != input.cols()) {
        throw ml_exception::DimensionMismatchException(
            "input columns",
            input.cols(), 1,
            sequence_length_ * input_size_, 1,
            "Recurrent");
    }
    
    int batch_size = input.rows();
    
    // Salva input nella cache
    cache_.input = input;
    cache_.has_activation = false;
    
    // Risistema input per timestep
    inputs_.clear();
    inputs_.reserve(sequence_length_);
    for (int t = 0; t < sequence_length_; ++t) {
        MatrixXd input_t(batch_size, input_size_);
        for (int b = 0; b < batch_size; ++b) {
            input_t.row(b) = input.block(b, t * input_size_, 1, input_size_);
        }
        inputs_.push_back(input_t);
    }
    
    // Esegui forward in base al tipo
    MatrixXd output;
    switch (rnn_type_) {
        case SIMPLE:
            output = forward_simple(input);
            break;
        case LSTM:
            output = forward_lstm(input);
            break;
        case GRU:
            output = forward_gru(input);
            break;
    }
    
    cache_.output = output;
    return output;
}

MatrixXd Recurrent::forward_simple(const MatrixXd& input) {
    int batch_size = input.rows();
    hidden_states_.clear();
    hidden_states_.resize(sequence_length_ + 1);
    hidden_states_[0] = MatrixXd::Zero(batch_size, hidden_size_);
    
    for (int t = 0; t < sequence_length_; ++t) {
        MatrixXd h_prev = hidden_states_[t];
        MatrixXd z = inputs_[t] * W_x_.transpose() + 
                    h_prev * W_h_.transpose() + 
                    b_.transpose().replicate(batch_size, 1);
        
        hidden_states_[t + 1] = activation_->forward(z);
    }
    
    if (return_sequences_) {
        MatrixXd output(batch_size, sequence_length_ * hidden_size_);
        for (int t = 0; t < sequence_length_; ++t) {
            output.block(0, t * hidden_size_, batch_size, hidden_size_) = hidden_states_[t + 1];
        }
        return output;
    } else {
        return hidden_states_[sequence_length_];
    }
}

MatrixXd Recurrent::forward_lstm(const MatrixXd& input) {
    int batch_size = input.rows();
    hidden_states_.clear();
    cell_states_.clear();
    lstm_gates_raw_.clear();
    
    hidden_states_.resize(sequence_length_ + 1);
    cell_states_.resize(sequence_length_ + 1);
    lstm_gates_raw_.resize(sequence_length_);
    
    hidden_states_[0] = MatrixXd::Zero(batch_size, hidden_size_);
    cell_states_[0] = MatrixXd::Zero(batch_size, hidden_size_);
    
    for (int t = 0; t < sequence_length_; ++t) {
        MatrixXd h_prev = hidden_states_[t];
        MatrixXd c_prev = cell_states_[t];
        
        MatrixXd gates_raw = inputs_[t] * W_x_.transpose() + 
                            h_prev * W_h_.transpose() + 
                            b_.transpose().replicate(batch_size, 1);
        
        lstm_gates_raw_[t] = gates_raw;
        
        MatrixXd i_gate_raw = gates_raw.block(0, 0, batch_size, hidden_size_);
        MatrixXd f_gate_raw = gates_raw.block(0, hidden_size_, batch_size, hidden_size_);
        MatrixXd c_candidate_raw = gates_raw.block(0, 2 * hidden_size_, batch_size, hidden_size_);
        MatrixXd o_gate_raw = gates_raw.block(0, 3 * hidden_size_, batch_size, hidden_size_);
        
        MatrixXd i_gate = gate_activation_->forward(i_gate_raw);
        MatrixXd f_gate = gate_activation_->forward(f_gate_raw);
        MatrixXd c_candidate = activation_->forward(c_candidate_raw);
        MatrixXd o_gate = gate_activation_->forward(o_gate_raw);
        
        MatrixXd c_t = f_gate.array() * c_prev.array() + i_gate.array() * c_candidate.array();
        MatrixXd tanh_c_t = activation_->forward(c_t);
        MatrixXd h_t = o_gate.array() * tanh_c_t.array();
        
        cell_states_[t + 1] = c_t;
        hidden_states_[t + 1] = h_t;
    }
    
    if (return_sequences_) {
        MatrixXd output(batch_size, sequence_length_ * hidden_size_);
        for (int t = 0; t < sequence_length_; ++t) {
            output.block(0, t * hidden_size_, batch_size, hidden_size_) = hidden_states_[t + 1];
        }
        return output;
    } else {
        return hidden_states_[sequence_length_];
    }
}

MatrixXd Recurrent::forward_gru(const Eigen::MatrixXd& input) {
    int batch_size = input.rows();
    hidden_states_.clear();
    z_gates_.clear();
    r_gates_.clear();
    h_candidate_raw_.clear();
    
    hidden_states_.resize(sequence_length_ + 1);
    z_gates_.resize(sequence_length_);
    r_gates_.resize(sequence_length_);
    h_candidate_raw_.resize(sequence_length_);
    
    hidden_states_[0] = MatrixXd::Zero(batch_size, hidden_size_);
    
    MatrixXd W_z = W_x_.block(0, 0, hidden_size_, input_size_);
    MatrixXd W_r = W_x_.block(hidden_size_, 0, hidden_size_, input_size_);
    MatrixXd W_h = W_x_.block(2 * hidden_size_, 0, hidden_size_, input_size_);
    
    MatrixXd U_z = W_h_.block(0, 0, hidden_size_, hidden_size_);
    MatrixXd U_r = W_h_.block(hidden_size_, 0, hidden_size_, hidden_size_);
    MatrixXd U_h = W_h_.block(2 * hidden_size_, 0, hidden_size_, hidden_size_);
    
    VectorXd b_z = b_.segment(0, hidden_size_);
    VectorXd b_r = b_.segment(hidden_size_, hidden_size_);
    VectorXd b_h = b_.segment(2 * hidden_size_, hidden_size_);
    
    for (int t = 0; t < sequence_length_; ++t) {
        MatrixXd h_prev = hidden_states_[t];
        
        MatrixXd z_gate_raw = inputs_[t] * W_z.transpose() + 
                             h_prev * U_z.transpose() + 
                             b_z.transpose().replicate(batch_size, 1);
        z_gates_[t] = z_gate_raw;
        MatrixXd z_gate = gate_activation_->forward(z_gate_raw);
        
        MatrixXd r_gate_raw = inputs_[t] * W_r.transpose() + 
                             h_prev * U_r.transpose() + 
                             b_r.transpose().replicate(batch_size, 1);
        r_gates_[t] = r_gate_raw;
        MatrixXd r_gate = gate_activation_->forward(r_gate_raw);
        
        MatrixXd r_h_prev = r_gate.array() * h_prev.array();
        MatrixXd h_candidate_raw_val = inputs_[t] * W_h.transpose() + 
                                      r_h_prev * U_h.transpose() + 
                                      b_h.transpose().replicate(batch_size, 1);
        h_candidate_raw_[t] = h_candidate_raw_val;
        MatrixXd h_candidate = activation_->forward(h_candidate_raw_val);
        
        MatrixXd h_t = (1 - z_gate.array()) * h_prev.array() + z_gate.array() * h_candidate.array();
        hidden_states_[t + 1] = h_t;
    }
    
    if (return_sequences_) {
        MatrixXd output(batch_size, sequence_length_ * hidden_size_);
        for (int t = 0; t < sequence_length_; ++t) {
            output.block(0, t * hidden_size_, batch_size, hidden_size_) = hidden_states_[t + 1];
        }
        return output;
    } else {
        return hidden_states_[sequence_length_];
    }
}

// ===================== BACKWARD PASS =====================

MatrixXd Recurrent::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
    if (!cache_.has_activation) {
        throw ml_exception::InvalidConfigurationException(
            "Cache not initialized. Call forward() first.", "Recurrent");
    }
    
    switch (rnn_type_) {
        case SIMPLE:
            return backward_simple(gradient, learning_rate);
        case LSTM:
            return backward_lstm(gradient, learning_rate);
        case GRU:
            return backward_gru(gradient, learning_rate);
    }
    
    return MatrixXd();
}

MatrixXd Recurrent::backward_simple(const Eigen::MatrixXd& gradient, double learning_rate) {
    int batch_size = gradient.rows();
    
    MatrixXd dW_x = MatrixXd::Zero(hidden_size_, input_size_);
    MatrixXd dW_h = MatrixXd::Zero(hidden_size_, hidden_size_);
    VectorXd db = VectorXd::Zero(hidden_size_);
    
    MatrixXd dInput = MatrixXd::Zero(batch_size, sequence_length_ * input_size_);
    MatrixXd dh_next = MatrixXd::Zero(batch_size, hidden_size_);
    
    for (int t = sequence_length_ - 1; t >= 0; --t) {
        MatrixXd h_curr = hidden_states_[t + 1];
        MatrixXd h_prev = hidden_states_[t];
        
        MatrixXd dh;
        if (return_sequences_) {
            MatrixXd dh_t = gradient.block(0, t * hidden_size_, batch_size, hidden_size_);
            dh = dh_t + dh_next;
        } else if (t == sequence_length_ - 1) {
            dh = gradient + dh_next;
        } else {
            dh = dh_next;
        }
        
        MatrixXd z = inputs_[t] * W_x_.transpose() + 
                    h_prev * W_h_.transpose() + 
                    b_.transpose().replicate(batch_size, 1);
        
        MatrixXd dz = activation_->backward(dh, z);
        
        dW_x += dz.transpose() * inputs_[t];
        dW_h += dz.transpose() * h_prev;
        db += dz.colwise().sum().transpose();
        
        MatrixXd dx = dz * W_x_;
        dh_next = dz * W_h_;
        
        dInput.block(0, t * input_size_, batch_size, input_size_) = dx;
    }
    
    dW_x /= batch_size;
    dW_h /= batch_size;
    db /= batch_size;
    
    clip_gradient(dW_x, clip_value_);
    clip_gradient(dW_h, clip_value_);
    clip_gradient(db, clip_value_);
    
    W_x_ -= learning_rate * dW_x;
    W_h_ -= learning_rate * dW_h;
    b_ -= learning_rate * db;
    
    return dInput;
}

MatrixXd Recurrent::backward_lstm(const Eigen::MatrixXd& gradient, double learning_rate) {
    int batch_size = gradient.rows();
    
    MatrixXd dW_x = MatrixXd::Zero(W_x_.rows(), W_x_.cols());
    MatrixXd dW_h = MatrixXd::Zero(W_h_.rows(), W_h_.cols());
    VectorXd db = VectorXd::Zero(b_.size());
    
    MatrixXd dInput = MatrixXd::Zero(batch_size, sequence_length_ * input_size_);
    MatrixXd dh_next = MatrixXd::Zero(batch_size, hidden_size_);
    MatrixXd dc_next = MatrixXd::Zero(batch_size, hidden_size_);
    
    for (int t = sequence_length_ - 1; t >= 0; --t) {
        MatrixXd h_curr = hidden_states_[t + 1];
        MatrixXd h_prev = hidden_states_[t];
        MatrixXd c_curr = cell_states_[t + 1];
        MatrixXd c_prev = cell_states_[t];
        
        MatrixXd gates_raw = lstm_gates_raw_[t];
        
        MatrixXd i_gate_raw = gates_raw.block(0, 0, batch_size, hidden_size_);
        MatrixXd f_gate_raw = gates_raw.block(0, hidden_size_, batch_size, hidden_size_);
        MatrixXd c_candidate_raw = gates_raw.block(0, 2 * hidden_size_, batch_size, hidden_size_);
        MatrixXd o_gate_raw = gates_raw.block(0, 3 * hidden_size_, batch_size, hidden_size_);
        
        MatrixXd i_gate = gate_activation_->forward(i_gate_raw);
        MatrixXd f_gate = gate_activation_->forward(f_gate_raw);
        MatrixXd c_candidate = activation_->forward(c_candidate_raw);
        MatrixXd o_gate = gate_activation_->forward(o_gate_raw);
        MatrixXd tanh_c_curr = activation_->forward(c_curr);
        
        MatrixXd dh;
        if (return_sequences_) {
            MatrixXd dh_t = gradient.block(0, t * hidden_size_, batch_size, hidden_size_);
            dh = dh_t + dh_next;
        } else if (t == sequence_length_ - 1) {
            dh = gradient + dh_next;
        } else {
            dh = dh_next;
        }
        
        MatrixXd d_o = dh.array() * tanh_c_curr.array();
        d_o = gate_activation_->backward(d_o, o_gate_raw);
        
        MatrixXd d_tanh_c_curr = dh.array() * o_gate.array();
        MatrixXd d_cell = d_tanh_c_curr.array() * (1 - tanh_c_curr.array().square());
        d_cell += dc_next;
        
        MatrixXd d_f = d_cell.array() * c_prev.array();
        d_f = gate_activation_->backward(d_f, f_gate_raw);
        
        MatrixXd d_i = d_cell.array() * c_candidate.array();
        d_i = gate_activation_->backward(d_i, i_gate_raw);
        
        MatrixXd d_c_candidate = d_cell.array() * i_gate.array();
        d_c_candidate = activation_->backward(d_c_candidate, c_candidate_raw);
        
        MatrixXd dgates(batch_size, 4 * hidden_size_);
        dgates.block(0, 0, batch_size, hidden_size_) = d_i;
        dgates.block(0, hidden_size_, batch_size, hidden_size_) = d_f;
        dgates.block(0, 2 * hidden_size_, batch_size, hidden_size_) = d_c_candidate;
        dgates.block(0, 3 * hidden_size_, batch_size, hidden_size_) = d_o;
        
        clip_gradient(dgates, clip_value_);
        
        dW_x += dgates.transpose() * inputs_[t];
        dW_h += dgates.transpose() * h_prev;
        db += dgates.colwise().sum().transpose();
        
        MatrixXd dx = dgates * W_x_;
        dh_next = dgates * W_h_;
        dc_next = d_cell.array() * f_gate.array();
        
        dInput.block(0, t * input_size_, batch_size, input_size_) = dx;
    }
    
    dW_x /= batch_size;
    dW_h /= batch_size;
    db /= batch_size;
    
    clip_gradient(dW_x, clip_value_);
    clip_gradient(dW_h, clip_value_);
    clip_gradient(db, clip_value_);
    
    W_x_ -= learning_rate * dW_x;
    W_h_ -= learning_rate * dW_h;
    b_ -= learning_rate * db;
    
    return dInput;
}

MatrixXd Recurrent::backward_gru(const Eigen::MatrixXd& gradient, double learning_rate) {
    int batch_size = gradient.rows();
    
    MatrixXd dW_x = MatrixXd::Zero(W_x_.rows(), W_x_.cols());
    MatrixXd dW_h = MatrixXd::Zero(W_h_.rows(), W_h_.cols());
    VectorXd db = VectorXd::Zero(b_.size());
    
    MatrixXd dInput = MatrixXd::Zero(batch_size, sequence_length_ * input_size_);
    MatrixXd dh_next = MatrixXd::Zero(batch_size, hidden_size_);
    
    MatrixXd W_z = W_x_.block(0, 0, hidden_size_, input_size_);
    MatrixXd W_r = W_x_.block(hidden_size_, 0, hidden_size_, input_size_);
    MatrixXd W_h = W_x_.block(2 * hidden_size_, 0, hidden_size_, input_size_);
    
    MatrixXd U_z = W_h_.block(0, 0, hidden_size_, hidden_size_);
    MatrixXd U_r = W_h_.block(hidden_size_, 0, hidden_size_, hidden_size_);
    MatrixXd U_h = W_h_.block(2 * hidden_size_, 0, hidden_size_, hidden_size_);
    
    VectorXd b_z = b_.segment(0, hidden_size_);
    VectorXd b_r = b_.segment(hidden_size_, hidden_size_);
    VectorXd b_h = b_.segment(2 * hidden_size_, hidden_size_);
    
    for (int t = sequence_length_ - 1; t >= 0; --t) {
        MatrixXd h_curr = hidden_states_[t + 1];
        MatrixXd h_prev = hidden_states_[t];
        
        MatrixXd z_gate_raw = z_gates_[t];
        MatrixXd r_gate_raw = r_gates_[t];
        MatrixXd h_candidate_raw_val = h_candidate_raw_[t];
        
        MatrixXd z_gate = gate_activation_->forward(z_gate_raw);
        MatrixXd r_gate = gate_activation_->forward(r_gate_raw);
        MatrixXd h_candidate = activation_->forward(h_candidate_raw_val);
        
        MatrixXd dh;
        if (return_sequences_) {
            MatrixXd dh_t = gradient.block(0, t * hidden_size_, batch_size, hidden_size_);
            dh = dh_t + dh_next;
        } else if (t == sequence_length_ - 1) {
            dh = gradient + dh_next;
        } else {
            dh = dh_next;
        }
        
        MatrixXd dh_dh_candidate = dh.array() * z_gate.array();
        MatrixXd dh_dh_prev = dh.array() * (1 - z_gate.array());
        MatrixXd dh_dz = dh.array() * (h_candidate.array() - h_prev.array());
        
        MatrixXd d_h_candidate = activation_->backward(dh_dh_candidate, h_candidate_raw_val);
        MatrixXd d_z = gate_activation_->backward(dh_dz, z_gate_raw);
        
        MatrixXd temp = (h_prev * U_h.transpose()).array();
        MatrixXd d_h_candidate_wrt_r = d_h_candidate.array() * temp.array();
        MatrixXd d_r = gate_activation_->backward(d_h_candidate_wrt_r, r_gate_raw);
        
        MatrixXd d_h_prev = dh_dh_prev;
        d_h_prev += d_z * U_z;
        d_h_prev += d_r * U_r;
        
        MatrixXd r_gate_expanded = r_gate.replicate(1, hidden_size_);
        MatrixXd d_h_candidate_times_r = d_h_candidate.array() * r_gate_expanded.array();
        d_h_prev += d_h_candidate_times_r * U_h;
        
        MatrixXd dgates(batch_size, 3 * hidden_size_);
        dgates.block(0, 0, batch_size, hidden_size_) = d_z;
        dgates.block(0, hidden_size_, batch_size, hidden_size_) = d_r;
        dgates.block(0, 2 * hidden_size_, batch_size, hidden_size_) = d_h_candidate;
        
        clip_gradient(dgates, clip_value_);
        
        dW_x.block(0, 0, hidden_size_, input_size_) += d_z.transpose() * inputs_[t];
        dW_x.block(hidden_size_, 0, hidden_size_, input_size_) += d_r.transpose() * inputs_[t];
        dW_x.block(2 * hidden_size_, 0, hidden_size_, input_size_) += d_h_candidate.transpose() * inputs_[t];
        
        dW_h.block(0, 0, hidden_size_, hidden_size_) += d_z.transpose() * h_prev;
        dW_h.block(hidden_size_, 0, hidden_size_, hidden_size_) += d_r.transpose() * h_prev;
        
        MatrixXd r_h_prev = (r_gate.array() * h_prev.array()).matrix();
        dW_h.block(2 * hidden_size_, 0, hidden_size_, hidden_size_) += 
            d_h_candidate.transpose() * r_h_prev;
        
        db.segment(0, hidden_size_) += d_z.colwise().sum().transpose();
        db.segment(hidden_size_, hidden_size_) += d_r.colwise().sum().transpose();
        db.segment(2 * hidden_size_, hidden_size_) += d_h_candidate.colwise().sum().transpose();
        
        MatrixXd dx = d_z * W_z + d_r * W_r + d_h_candidate * W_h;
        dInput.block(0, t * input_size_, batch_size, input_size_) = dx;
        
        dh_next = d_h_prev;
    }
    
    dW_x /= batch_size;
    dW_h /= batch_size;
    db /= batch_size;
    
    clip_gradient(dW_x, clip_value_);
    clip_gradient(dW_h, clip_value_);
    clip_gradient(db, clip_value_);
    
    W_x_ -= learning_rate * dW_x;
    W_h_ -= learning_rate * dW_h;
    b_ -= learning_rate * db;
    
    return dInput;
}

// ===================== UTILITY METHODS =====================

void Recurrent::clip_gradient(Eigen::MatrixXd& gradient, double clip_value) {
    double norm = gradient.norm();
    if (norm > clip_value) {
        gradient *= clip_value / norm;
    }
}

void Recurrent::clip_gradient(Eigen::VectorXd& gradient, double clip_value) {
    double norm = gradient.norm();
    if (norm > clip_value) {
        gradient *= clip_value / norm;
    }
}

void Recurrent::apply_dropout(Eigen::MatrixXd& matrix, 
                            std::vector<Eigen::MatrixXd>& masks, 
                            double rate) {
    if (rate <= 0) return;
    
    int rows = matrix.rows();
    int cols = matrix.cols();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    MatrixXd mask(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mask(i, j) = (dist(gen) > rate) ? 1.0 / (1.0 - rate) : 0.0;
        }
    }
    
    matrix = matrix.array() * mask.array();
    masks.push_back(mask);
}

void Recurrent::apply_dropout_gradient(Eigen::MatrixXd& gradient, 
                                     const std::vector<Eigen::MatrixXd>& masks) {
    if (masks.empty()) return;
    // Implementazione semplificata
}

int Recurrent::calculate_output_size() const {
    if (return_sequences_) {
        return sequence_length_ > 0 ? sequence_length_ * hidden_size_ : hidden_size_;
    } else {
        return hidden_size_;
    }
}

// ===================== GETTERS & SETTERS =====================

int Recurrent::get_output_size() const {
    return calculate_output_size();
}

int Recurrent::get_parameter_count() const {
    return W_x_.size() + W_h_.size() + b_.size();
}

std::string Recurrent::get_config() const {
    std::ostringstream oss;
    oss << "Recurrent(input=" << input_size_
        << ", hidden=" << hidden_size_
        << ", type=";
    
    switch (rnn_type_) {
        case SIMPLE: oss << "simple"; break;
        case LSTM: oss << "lstm"; break;
        case GRU: oss << "gru"; break;
    }
    
    oss << ", return_sequences=" << (return_sequences_ ? "true" : "false")
        << ", params=" << get_parameter_count() << ")";
    
    return oss.str();
}

Eigen::MatrixXd Recurrent::get_weights() const {
    return W_x_;
}

Eigen::VectorXd Recurrent::get_biases() const {
    return b_;
}

void Recurrent::set_weights(const Eigen::MatrixXd& weights) {
    if (weights.rows() != W_x_.rows() || weights.cols() != W_x_.cols()) {
        throw ml_exception::DimensionMismatchException(
            "weights",
            W_x_.rows(), W_x_.cols(),
            weights.rows(), weights.cols(),
            "Recurrent");
    }
    W_x_ = weights;
}

void Recurrent::set_biases(const Eigen::VectorXd& biases) {
    if (biases.size() != b_.size()) {
        throw ml_exception::DimensionMismatchException(
            "biases",
            b_.size(), 1,
            biases.size(), 1,
            "Recurrent");
    }
    b_ = biases;
}

void Recurrent::set_dropout(double dropout_rate) {
    ML_CHECK_PARAM(dropout_rate >= 0 && dropout_rate < 1, 
                  "dropout_rate", "must be in [0, 1)", "Recurrent");
    dropout_rate_ = dropout_rate;
}

void Recurrent::set_recurrent_dropout(double recurrent_dropout_rate) {
    ML_CHECK_PARAM(recurrent_dropout_rate >= 0 && recurrent_dropout_rate < 1,
                  "recurrent_dropout_rate", "must be in [0, 1)", "Recurrent");
    recurrent_dropout_rate_ = recurrent_dropout_rate;
}

void Recurrent::set_sequence_length(int seq_length) {
    ML_CHECK_PARAM(seq_length > 0, "sequence_length", "must be > 0", "Recurrent");
    sequence_length_ = seq_length;
}

void Recurrent::reset_states() {
    hidden_states_.clear();
    cell_states_.clear();
}

void Recurrent::set_gradient_clip(double clip_value) {
    ML_CHECK_PARAM(clip_value > 0, "clip_value", "must be > 0", "Recurrent");
    clip_value_ = clip_value;
}

double Recurrent::get_gradient_clip() const { 
    return clip_value_; 
}

Eigen::MatrixXd Recurrent::get_last_hidden_state() const {
    if (hidden_states_.empty()) {
        return MatrixXd();
    }
    return hidden_states_.back();
}

Eigen::MatrixXd Recurrent::get_last_cell_state() const {
    if (cell_states_.empty()) {
        return MatrixXd();
    }
    return cell_states_.back();
}

void Recurrent::clear_cache() {
    cache_.input = MatrixXd();
    cache_.output = MatrixXd();
    cache_.has_activation = false;
    hidden_states_.clear();
    cell_states_.clear();
    inputs_.clear();
    dropout_masks_.clear();
    recurrent_dropout_masks_.clear();
    z_gates_.clear();
    r_gates_.clear();
    h_candidate_raw_.clear();
    lstm_gates_raw_.clear();
    sequence_length_ = -1;
}

// ===================== SERIALIZZAZIONE =====================

void Recurrent::serialize(std::ostream& out) const {
    using namespace serialization;
    
    out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&sequence_length_), sizeof(int));
    
    int type_int = static_cast<int>(rnn_type_);
    out.write(reinterpret_cast<const char*>(&type_int), sizeof(int));
    
    out.write(reinterpret_cast<const char*>(&return_sequences_), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&dropout_rate_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&recurrent_dropout_rate_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&clip_value_), sizeof(double));
    
    eigen_utils::serialize_eigen(W_x_, out);
    eigen_utils::serialize_eigen(W_h_, out);
    eigen_utils::serialize_eigen_vector(b_, out);
    
    std::string act_type = activation_->get_type();
    size_t act_len = act_type.size();
    out.write(reinterpret_cast<const char*>(&act_len), sizeof(size_t));
    out.write(act_type.c_str(), act_len);
    
    std::string gate_act_type = gate_activation_->get_type();
    size_t gate_act_len = gate_act_type.size();
    out.write(reinterpret_cast<const char*>(&gate_act_len), sizeof(size_t));
    out.write(gate_act_type.c_str(), gate_act_len);
}

void Recurrent::deserialize(std::istream& in) {
    using namespace serialization;
    
    in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&hidden_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&sequence_length_), sizeof(int));
    
    int type_int;
    in.read(reinterpret_cast<char*>(&type_int), sizeof(int));
    rnn_type_ = static_cast<RNNType>(type_int);
    
    in.read(reinterpret_cast<char*>(&return_sequences_), sizeof(bool));
    in.read(reinterpret_cast<char*>(&dropout_rate_), sizeof(double));
    in.read(reinterpret_cast<char*>(&recurrent_dropout_rate_), sizeof(double));
    in.read(reinterpret_cast<char*>(&clip_value_), sizeof(double));
    
    eigen_utils::deserialize_eigen(W_x_, in);
    eigen_utils::deserialize_eigen(W_h_, in);
    eigen_utils::deserialize_eigen_vector(b_, in);
    
    size_t act_len;
    in.read(reinterpret_cast<char*>(&act_len), sizeof(size_t));
    std::string act_type(act_len, '\0');
    in.read(&act_type[0], act_len);
    
    size_t gate_act_len;
    in.read(reinterpret_cast<char*>(&gate_act_len), sizeof(size_t));
    std::string gate_act_type(gate_act_len, '\0');
    in.read(&gate_act_type[0], gate_act_len);
    
    activation_ = activation::create_activation(act_type);
    gate_activation_ = activation::create_activation(gate_act_type);
    
    if (!activation_ || !gate_activation_) {
        throw ml_exception::DeserializationException(
            "layer file", "unknown activation type", "Recurrent");
    }
    
    clear_cache();
}