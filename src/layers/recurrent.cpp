#include "layers/recurrent.h"
#include "activation/activation.h"
#include "serialization/serializable.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;
using namespace layers;

// Costruttore RNN semplice
Recurrent::Recurrent(int input_size, int hidden_size,
                    const std::string& activation, bool return_sequences)
    : input_size_(input_size), hidden_size_(hidden_size),
      sequence_length_(0), rnn_type_(SIMPLE), return_sequences_(return_sequences),
      dropout_rate_(0.0), recurrent_dropout_rate_(0.0) {
    
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", get_type());
    ML_CHECK_PARAM(hidden_size > 0, "hidden_size", "must be > 0", get_type());
    
    initialize_weights_simple();
    
    activation_ = activation::create_activation(activation);
    if (!activation_) {
        throw ml_exception::InvalidParameterException(
            "activation", "unknown activation function: " + activation, get_type());
    }
    
    // Per RNN semplice, gate_activation non è usato
    gate_activation_ = nullptr;
    
    clear_cache();
}

// Costruttore LSTM/GRU
Recurrent::Recurrent(int input_size, int hidden_size,
                    RNNType type, bool return_sequences)
    : input_size_(input_size), hidden_size_(hidden_size),
      sequence_length_(0), rnn_type_(type), return_sequences_(return_sequences),
      dropout_rate_(0.0), recurrent_dropout_rate_(0.0) {
    
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", get_type());
    ML_CHECK_PARAM(hidden_size > 0, "hidden_size", "must be > 0", get_type());
    
    if (type == LSTM) {
        initialize_weights_lstm();
    } else if (type == GRU) {
        initialize_weights_gru();
    } else {
        initialize_weights_simple();
    }
    
    // Per LSTM/GRU usiamo tanh per l'output e sigmoid per i gate
    activation_ = activation::create_activation("tanh");
    gate_activation_ = activation::create_activation("sigmoid");
    
    if (!activation_ || !gate_activation_) {
        throw ml_exception::InvalidParameterException(
            "activation", "failed to create activation functions", get_type());
    }
    
    clear_cache();
}

// Inizializzazione pesi per RNN semplice
void Recurrent::initialize_weights_simple() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Inizializzazione Xavier per RNN
    double limit_x = std::sqrt(6.0 / (input_size_ + hidden_size_));
    double limit_h = std::sqrt(6.0 / (hidden_size_ + hidden_size_));
    
    std::uniform_real_distribution<> dist_x(-limit_x, limit_x);
    std::uniform_real_distribution<> dist_h(-limit_h, limit_h);
    
    // Pesi input-hidden
    W_x_ = MatrixXd::Zero(hidden_size_, input_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            W_x_(i, j) = dist_x(gen);
        }
    }
    
    // Pesi hidden-hidden (ricorrenti)
    W_h_ = MatrixXd::Zero(hidden_size_, hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            W_h_(i, j) = dist_h(gen);
        }
    }
    
    // Bias
    b_ = VectorXd::Zero(hidden_size_);
}

// Inizializzazione pesi per LSTM (semplificata)
void Recurrent::initialize_weights_lstm() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Inizializzazione per LSTM
    double limit = std::sqrt(2.0 / (input_size_ + hidden_size_));
    std::normal_distribution<> dist(0.0, limit);
    
    // Pesi per ogni gate: forget, input, cell, output
    W_f_ = MatrixXd::Zero(hidden_size_, input_size_ + hidden_size_);
    W_i_ = MatrixXd::Zero(hidden_size_, input_size_ + hidden_size_);
    W_c_ = MatrixXd::Zero(hidden_size_, input_size_ + hidden_size_);
    W_o_ = MatrixXd::Zero(hidden_size_, input_size_ + hidden_size_);
    
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < input_size_ + hidden_size_; ++j) {
            W_f_(i, j) = dist(gen);
            W_i_(i, j) = dist(gen);
            W_c_(i, j) = dist(gen);
            W_o_(i, j) = dist(gen);
        }
    }
    
    // Bias per ogni gate
    b_f_ = VectorXd::Zero(hidden_size_);
    b_i_ = VectorXd::Zero(hidden_size_);
    b_c_ = VectorXd::Zero(hidden_size_);
    b_o_ = VectorXd::Zero(hidden_size_);
    
    // Inizializzazione bias del forget gate a 1 (raccomandato)
    b_f_.setConstant(1.0);
}

// Inizializzazione pesi per GRU (semplificata)
void Recurrent::initialize_weights_gru() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Inizializzazione per GRU
    double limit = std::sqrt(2.0 / (input_size_ + hidden_size_));
    std::normal_distribution<> dist(0.0, limit);
    
    // Pesi per update, reset e hidden gates
    W_z_ = MatrixXd::Zero(hidden_size_, input_size_ + hidden_size_);
    W_r_ = MatrixXd::Zero(hidden_size_, input_size_ + hidden_size_);
    W_hh_ = MatrixXd::Zero(hidden_size_, input_size_ + hidden_size_);
    
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < input_size_ + hidden_size_; ++j) {
            W_z_(i, j) = dist(gen);
            W_r_(i, j) = dist(gen);
            W_hh_(i, j) = dist(gen);
        }
    }
    
    // Bias
    b_z_ = VectorXd::Zero(hidden_size_);
    b_r_ = VectorXd::Zero(hidden_size_);
    b_hh_ = VectorXd::Zero(hidden_size_);
}

// Calcolo dimensioni output
int Recurrent::calculate_output_size() const {
    if (return_sequences_) {
        return sequence_length_ * hidden_size_;
    } else {
        return hidden_size_;
    }
}

// Forward propagation per RNN semplice
MatrixXd Recurrent::forward_simple(const MatrixXd& input) {
    // input shape: [batch_size, sequence_length * input_size]
    // Oppure: [batch_size * sequence_length, input_size]
    
    int batch_size = input.rows();
    int total_elements = input.cols();
    
    // Determina sequence_length
    if (sequence_length_ == 0) {
        // Proviamo a dedurlo
        if (total_elements % input_size_ == 0) {
            sequence_length_ = total_elements / input_size_;
        } else {
            // Assume input già in formato sequenza appiattita
            sequence_length_ = 1;
        }
    }
    
    // Riorganizza input in sequenza
    inputs_.clear();
    inputs_.resize(sequence_length_);
    
    for (int t = 0; t < sequence_length_; ++t) {
        inputs_[t] = MatrixXd::Zero(batch_size, input_size_);
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < input_size_; ++i) {
                int input_idx = t * input_size_ + i;
                if (input_idx < total_elements) {
                    inputs_[t](b, i) = input(b, input_idx);
                }
            }
        }
    }
    
    // Inizializza stati nascosti
    hidden_states_.clear();
    hidden_states_.resize(sequence_length_ + 1); // +1 per stato iniziale
    
    // Stato iniziale nascosto (zero)
    hidden_states_[0] = MatrixXd::Zero(batch_size, hidden_size_);
    
    // Forward pass attraverso la sequenza
    for (int t = 0; t < sequence_length_; ++t) {
        // h_t = activation(W_x * x_t + W_h * h_{t-1} + b)
        MatrixXd combined = inputs_[t] * W_x_.transpose() + 
                           hidden_states_[t] * W_h_.transpose();
        combined.rowwise() += b_.transpose();
        
        hidden_states_[t + 1] = activation_->forward(combined);
        
        // Applica dropout se specificato
        if (dropout_rate_ > 0 && t < sequence_length_ - 1) {
            apply_dropout(hidden_states_[t + 1], dropout_masks_, dropout_rate_);
        }
    }
    
    // Prepara output
    int output_size = calculate_output_size();
    MatrixXd output = MatrixXd::Zero(batch_size, output_size);
    
    if (return_sequences_) {
        // Restituisce tutti gli stati nascosti
        for (int t = 0; t < sequence_length_; ++t) {
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < hidden_size_; ++h) {
                    int output_idx = t * hidden_size_ + h;
                    output(b, output_idx) = hidden_states_[t + 1](b, h);
                }
            }
        }
    } else {
        // Restituisce solo l'ultimo stato nascosto
        output = hidden_states_[sequence_length_];
    }
    
    cache_.output = output;
    cache_.has_activation = false; // L'attivazione è già applicata
    
    return output;
}

// Forward propagation principale
MatrixXd Recurrent::forward(const MatrixXd& input) {
    cache_.input = input;
    
    switch (rnn_type_) {
        case SIMPLE:
            return forward_simple(input);
        case LSTM:
            return forward_lstm(input);
        case GRU:
            return forward_gru(input);
        default:
            throw ml_exception::InvalidConfigurationException(
                "Unknown RNN type", get_type());
    }
}

// Backward propagation per RNN semplice
MatrixXd Recurrent::backward_simple(const MatrixXd& gradient, double learning_rate) {
    if (hidden_states_.empty()) {
        throw ml_exception::InvalidConfigurationException(
            "Hidden states not initialized. Call forward() first.", get_type());
    }
    
    int batch_size = gradient.rows();
    int grad_cols = gradient.cols();
    
    // Prepara gradienti per gli stati nascosti
    std::vector<MatrixXd> dhidden(sequence_length_ + 1, 
                                 MatrixXd::Zero(batch_size, hidden_size_));
    
    // Inizializza gradienti per l'ultimo timestep
    if (return_sequences_) {
        // Il gradiente viene distribuito su tutta la sequenza
        for (int t = 0; t < sequence_length_; ++t) {
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < hidden_size_; ++h) {
                    int grad_idx = t * hidden_size_ + h;
                    if (grad_idx < grad_cols) {
                        dhidden[t + 1](b, h) = gradient(b, grad_idx);
                    }
                }
            }
        }
    } else {
        // Il gradiente va solo all'ultimo stato nascosto
        dhidden[sequence_length_] = gradient;
    }
    
    // Gradienti accumulati per i pesi
    MatrixXd dW_x_acc = MatrixXd::Zero(hidden_size_, input_size_);
    MatrixXd dW_h_acc = MatrixXd::Zero(hidden_size_, hidden_size_);
    VectorXd db_acc = VectorXd::Zero(hidden_size_);
    
    // Gradienti per l'input
    MatrixXd dinput = MatrixXd::Zero(batch_size, sequence_length_ * input_size_);
    
    // Backward pass attraverso il tempo (BPTT)
    for (int t = sequence_length_ - 1; t >= 0; --t) {
        // Gradiente rispetto a z (pre-attivazione)
        MatrixXd dz = activation_->backward(dhidden[t + 1], hidden_states_[t + 1]);
        
        // Applica dropout gradient se necessario
        if (dropout_rate_ > 0 && t < sequence_length_ - 1) {
            apply_dropout_gradient(dz, dropout_masks_);
        }
        
        // Gradienti per i pesi
        dW_x_acc += dz.transpose() * inputs_[t];
        dW_h_acc += dz.transpose() * hidden_states_[t];
        db_acc += dz.colwise().sum().transpose();
        
        // Gradiente per lo stato nascosto precedente
        dhidden[t] = dz * W_h_;
        
        // Gradiente per l'input corrente
        MatrixXd dx_t = dz * W_x_;
        
        // Salva nel gradiente dell'input
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < input_size_; ++i) {
                int input_idx = t * input_size_ + i;
                dinput(b, input_idx) = dx_t(b, i);
            }
        }
    }
    
    // Normalizza gradienti per batch size
    dW_x_acc /= batch_size;
    dW_h_acc /= batch_size;
    db_acc /= batch_size;
    
    // Aggiorna pesi
    W_x_ -= learning_rate * dW_x_acc;
    W_h_ -= learning_rate * dW_h_acc;
    b_ -= learning_rate * db_acc;
    
    return dinput;
}

// Backward propagation principale
MatrixXd Recurrent::backward(const MatrixXd& gradient, double learning_rate) {
    switch (rnn_type_) {
        case SIMPLE:
            return backward_simple(gradient, learning_rate);
        case LSTM:
            return backward_lstm(gradient, learning_rate);
        case GRU:
            return backward_gru(gradient, learning_rate);
        default:
            throw ml_exception::InvalidConfigurationException(
                "Unknown RNN type", get_type());
    }
}

// Utility per dropout
void Recurrent::apply_dropout(Eigen::MatrixXd& matrix, 
                            std::vector<Eigen::MatrixXd>& masks, 
                            double rate) {
    if (rate <= 0.0) return;
    
    int rows = matrix.rows();
    int cols = matrix.cols();
    
    // Crea maschera di dropout
    Eigen::MatrixXd mask = Eigen::MatrixXd::Random(rows, cols);
    mask = (mask.array() > rate).cast<double>() / (1.0 - rate);
    
    // Applica maschera
    matrix.array() *= mask.array();
    
    // Salva maschera per backward
    masks.push_back(mask);
}

void Recurrent::apply_dropout_gradient(Eigen::MatrixXd& gradient, 
                                     const std::vector<Eigen::MatrixXd>& masks) {
    if (masks.empty()) return;
    
    // Applica la stessa maschera al gradiente
    static size_t mask_idx = 0;
    if (mask_idx < masks.size()) {
        gradient.array() *= masks[mask_idx].array();
        mask_idx++;
    }
}

// Metodi di interfaccia
int Recurrent::get_output_size() const {
    return calculate_output_size();
}

int Recurrent::get_parameter_count() const {
    if (rnn_type_ == SIMPLE) {
        return W_x_.size() + W_h_.size() + b_.size();
    } else if (rnn_type_ == LSTM) {
        return (W_f_.size() + W_i_.size() + W_c_.size() + W_o_.size() +
                b_f_.size() + b_i_.size() + b_c_.size() + b_o_.size());
    } else if (rnn_type_ == GRU) {
        return (W_z_.size() + W_r_.size() + W_hh_.size() +
                b_z_.size() + b_r_.size() + b_hh_.size());
    }
    return 0;
}

std::string Recurrent::get_config() const {
    std::string type_str;
    switch (rnn_type_) {
        case SIMPLE: type_str = "SIMPLE"; break;
        case LSTM: type_str = "LSTM"; break;
        case GRU: type_str = "GRU"; break;
    }
    
    std::ostringstream oss;
    oss << "Recurrent(input=" << input_size_
        << ", hidden=" << hidden_size_
        << ", type=" << type_str
        << ", return_seq=" << (return_sequences_ ? "true" : "false")
        << ", params=" << get_parameter_count() << ")";
    return oss.str();
}

void Recurrent::clear_cache() {
    cache_.input = MatrixXd();
    cache_.z = MatrixXd();
    cache_.output = MatrixXd();
    cache_.has_activation = false;
    
    hidden_states_.clear();
    cell_states_.clear();
    inputs_.clear();
    dropout_masks_.clear();
    recurrent_dropout_masks_.clear();
    
    sequence_length_ = 0;
}

// Setters
void Recurrent::set_dropout(double dropout_rate) {
    ML_CHECK_PARAM(dropout_rate >= 0 && dropout_rate < 1, 
                  "dropout_rate", "must be in [0, 1)", get_type());
    dropout_rate_ = dropout_rate;
}

void Recurrent::set_recurrent_dropout(double recurrent_dropout_rate) {
    ML_CHECK_PARAM(recurrent_dropout_rate >= 0 && recurrent_dropout_rate < 1,
                  "recurrent_dropout_rate", "must be in [0, 1)", get_type());
    recurrent_dropout_rate_ = recurrent_dropout_rate;
}

void Recurrent::set_sequence_length(int seq_length) {
    ML_CHECK_PARAM(seq_length > 0, "seq_length", "must be > 0", get_type());
    sequence_length_ = seq_length;
}

void Recurrent::reset_states() {
    clear_cache();
}

// Getters per stati
MatrixXd Recurrent::get_last_hidden_state() const {
    if (hidden_states_.empty()) {
        return MatrixXd::Zero(1, hidden_size_);
    }
    return hidden_states_.back();
}

MatrixXd Recurrent::get_last_cell_state() const {
    if (cell_states_.empty() || rnn_type_ != LSTM) {
        return MatrixXd::Zero(1, hidden_size_);
    }
    return cell_states_.back();
}

// Implementazioni forward per LSTM e GRU (scheletri)
MatrixXd Recurrent::forward_lstm(const MatrixXd& input) {
    // Implementazione LSTM completa sarebbe qui
    throw ml_exception::NotImplementedException(
        "LSTM forward not fully implemented", get_type());
}

MatrixXd Recurrent::forward_gru(const MatrixXd& input) {
    // Implementazione GRU completa sarebbe qui
    throw ml_exception::NotImplementedException(
        "GRU forward not fully implemented", get_type());
}

MatrixXd Recurrent::backward_lstm(const MatrixXd& gradient, double learning_rate) {
    throw ml_exception::NotImplementedException(
        "LSTM backward not fully implemented", get_type());
}

MatrixXd Recurrent::backward_gru(const MatrixXd& gradient, double learning_rate) {
    throw ml_exception::NotImplementedException(
        "GRU backward not fully implemented", get_type());
}

// Serializzazione (semplificata)
void Recurrent::serialize(std::ostream& out) const {
    using namespace serialization;
    
    out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&sequence_length_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&rnn_type_), sizeof(RNNType));
    out.write(reinterpret_cast<const char*>(&return_sequences_), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&dropout_rate_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&recurrent_dropout_rate_), sizeof(double));
    
    // Serializza pesi in base al tipo
    if (rnn_type_ == SIMPLE) {
        eigen_utils::serialize_eigen(W_x_, out);
        eigen_utils::serialize_eigen(W_h_, out);
        eigen_utils::serialize_eigen_vector(b_, out);
    }
    // Serializzazione per LSTM e GRU omessa per brevità
}

void Recurrent::deserialize(std::istream& in) {
    using namespace serialization;
    
    in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&hidden_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&sequence_length_), sizeof(int));
    in.read(reinterpret_cast<char*>(&rnn_type_), sizeof(RNNType));
    in.read(reinterpret_cast<char*>(&return_sequences_), sizeof(bool));
    in.read(reinterpret_cast<char*>(&dropout_rate_), sizeof(double));
    in.read(reinterpret_cast<char*>(&recurrent_dropout_rate_), sizeof(double));
    
    // Ricrea le attivazioni
    if (rnn_type_ == SIMPLE) {
        activation_ = activation::create_activation("tanh");
    } else {
        activation_ = activation::create_activation("tanh");
        gate_activation_ = activation::create_activation("sigmoid");
    }
    
    // Deserializza pesi in base al tipo
    if (rnn_type_ == SIMPLE) {
        eigen_utils::deserialize_eigen(W_x_, in);
        eigen_utils::deserialize_eigen(W_h_, in);
        eigen_utils::deserialize_eigen_vector(b_, in);
    }
    
    clear_cache();
}