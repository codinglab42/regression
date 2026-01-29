#include "layers/convolutional.h"
#include "activation/activation.h"
#include "serialization/serializable.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;
using namespace layers;

// Costruttori
Convolutional::Convolutional(int input_channels, int output_channels,
                           int kernel_size, int stride, int padding,
                           const std::string& activation)
    : input_channels_(input_channels), output_channels_(output_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      dilation_(1), groups_(1), input_height_(0), input_width_(0) {
    
    ML_CHECK_PARAM(input_channels > 0, "input_channels", "must be > 0", get_type());
    ML_CHECK_PARAM(output_channels > 0, "output_channels", "must be > 0", get_type());
    ML_CHECK_PARAM(kernel_size > 0, "kernel_size", "must be > 0", get_type());
    ML_CHECK_PARAM(stride > 0, "stride", "must be > 0", get_type());
    ML_CHECK_PARAM(padding >= 0, "padding", "must be >= 0", get_type());
    
    initialize_kernels();
    initialize_biases();
    
    activation_ = activation::create_activation(activation);
    if (!activation_) {
        throw ml_exception::InvalidParameterException(
            "activation", "unknown activation function: " + activation, get_type());
    }
    
    clear_cache();
}

Convolutional::Convolutional(int input_channels, int input_height, int input_width,
                           int output_channels, int kernel_size,
                           int stride, int padding, const std::string& activation)
    : input_channels_(input_channels), input_height_(input_height), input_width_(input_width),
      output_channels_(output_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding), dilation_(1), groups_(1) {
    
    ML_CHECK_PARAM(input_channels > 0, "input_channels", "must be > 0", get_type());
    ML_CHECK_PARAM(input_height > 0, "input_height", "must be > 0", get_type());
    ML_CHECK_PARAM(input_width > 0, "input_width", "must be > 0", get_type());
    ML_CHECK_PARAM(output_channels > 0, "output_channels", "must be > 0", get_type());
    ML_CHECK_PARAM(kernel_size > 0, "kernel_size", "must be > 0", get_type());
    ML_CHECK_PARAM(stride > 0, "stride", "must be > 0", get_type());
    ML_CHECK_PARAM(padding >= 0, "padding", "must be >= 0", get_type());
    
    initialize_kernels();
    initialize_biases();
    
    activation_ = activation::create_activation(activation);
    if (!activation_) {
        throw ml_exception::InvalidParameterException(
            "activation", "unknown activation function: " + activation, get_type());
    }
    
    clear_cache();
}

// Inizializzazione kernels
void Convolutional::initialize_kernels() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Inizializzazione He per CNN
    double stddev = std::sqrt(2.0 / (input_channels_ * kernel_size_ * kernel_size_));
    std::normal_distribution<> dist(0.0, stddev);
    
    kernels_.resize(output_channels_);
    for (int oc = 0; oc < output_channels_; ++oc) {
        kernels_[oc].resize(input_channels_, kernel_size_ * kernel_size_);
        for (int ic = 0; ic < input_channels_; ++ic) {
            for (int k = 0; k < kernel_size_ * kernel_size_; ++k) {
                kernels_[oc](ic, k) = dist(gen);
            }
        }
    }
}

void Convolutional::initialize_biases() {
    biases_.resize(output_channels_);
    biases_.setZero(); // Bias inizializzati a zero per CNN
}

// Calcolo dimensioni output
int Convolutional::calculate_output_height() const {
    if (input_height_ == 0) return 0;
    return (input_height_ + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
}

int Convolutional::calculate_output_width() const {
    if (input_width_ == 0) return 0;
    return (input_width_ + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
}

// im2col: trasforma l'input in una matrice colonna per convoluzione efficiente
MatrixXd Convolutional::im2col(const MatrixXd& input) const {
    int batch_size = input.rows();
    int input_size = input_channels_ * input_height_ * input_width_;
    
    // Verifica dimensioni input
    if (input.cols() != input_size) {
        throw ml_exception::DimensionMismatchException(
            "input columns",
            input_size, 1,
            input.cols(), 1,
            get_type());
    }
    
    int output_h = calculate_output_height();
    int output_w = calculate_output_width();
    int kernel_elements = kernel_size_ * kernel_size_;
    
    // Dimensione della matrice colonna: (kernel_elements * input_channels_) x (batch_size * output_h * output_w)
    int col_rows = input_channels_ * kernel_elements;
    int col_cols = batch_size * output_h * output_w;
    MatrixXd col_matrix = MatrixXd::Zero(col_rows, col_cols);
    
    // Applica padding se necessario
    MatrixXd padded_input = apply_padding(input);
    
    // Riempimento della matrice colonna
    for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < output_h; ++oh) {
            for (int ow = 0; ow < output_w; ++ow) {
                int col_index = b * output_h * output_w + oh * output_w + ow;
                
                for (int ic = 0; ic < input_channels_; ++ic) {
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int input_h = oh * stride_ + kh;
                            int input_w = ow * stride_ + kw;
                            
                            int row_index = ic * kernel_elements + kh * kernel_size_ + kw;
                            
                            // Calcola l'indice nell'input appiattito
                            int input_index = ic * (input_height_ + 2 * padding_) * (input_width_ + 2 * padding_) +
                                            input_h * (input_width_ + 2 * padding_) + input_w;
                            
                            col_matrix(row_index, col_index) = padded_input(b, input_index);
                        }
                    }
                }
            }
        }
    }
    
    return col_matrix;
}

// col2im: inverso di im2col
MatrixXd Convolutional::col2im(const MatrixXd& col_matrix) const {
    int batch_size = cache_.input.rows();
    int output_h = calculate_output_height();
    int output_w = calculate_output_width();
    int kernel_elements = kernel_size_ * kernel_size_;
    
    int padded_h = input_height_ + 2 * padding_;
    int padded_w = input_width_ + 2 * padding_;
    MatrixXd gradient_padded = MatrixXd::Zero(batch_size, input_channels_ * padded_h * padded_w);
    
    // Riempimento del gradiente con padding
    for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < output_h; ++oh) {
            for (int ow = 0; ow < output_w; ++ow) {
                int col_index = b * output_h * output_w + oh * output_w + ow;
                
                for (int ic = 0; ic < input_channels_; ++ic) {
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int input_h = oh * stride_ + kh;
                            int input_w = ow * stride_ + kw;
                            
                            int row_index = ic * kernel_elements + kh * kernel_size_ + kw;
                            
                            // Calcola l'indice nell'input con padding
                            int input_index = ic * padded_h * padded_w + input_h * padded_w + input_w;
                            
                            gradient_padded(b, input_index) += col_matrix(row_index, col_index);
                        }
                    }
                }
            }
        }
    }
    
    // Rimuovi il padding se necessario
    return remove_padding(gradient_padded);
}

// Forward propagation
MatrixXd Convolutional::forward(const MatrixXd& input) {
    // Memorizza l'input nella cache
    cache_.input = input;
    
    // Estrai dimensioni batch
    int batch_size = input.rows();
    
    // Se input_height_ e input_width_ non sono specificati, cerca di dedurli
    if (input_height_ == 0 || input_width_ == 0) {
        // Assumiamo che l'input sia già appiattito
        int total_elements = input.cols();
        if (total_elements % input_channels_ != 0) {
            throw ml_exception::InvalidParameterException(
                "input dimensions", 
                "total elements must be divisible by input_channels", 
                get_type());
        }
        
        int spatial_elements = total_elements / input_channels_;
        // Prova a trovare fattori per height e width
        for (int h = 1; h <= spatial_elements; ++h) {
            if (spatial_elements % h == 0) {
                input_width_ = spatial_elements / h;
                input_height_ = h;
                if (input_width_ >= kernel_size_ && input_height_ >= kernel_size_) {
                    break;
                }
            }
        }
        
        if (input_height_ == 0 || input_width_ == 0) {
            throw ml_exception::InvalidParameterException(
                "input dimensions", 
                "cannot determine valid height and width", 
                get_type());
        }
    }
    
    // Applica im2col
    input_cols_ = im2col(input);
    
    // Calcola dimensioni output
    int output_h = calculate_output_height();
    int output_w = calculate_output_width();
    int output_size = output_channels_ * output_h * output_w;
    
    // Inizializza output
    MatrixXd output = MatrixXd::Zero(batch_size, output_size);
    
    // Convoluzione usando prodotto matriciale
    for (int oc = 0; oc < output_channels_; ++oc) {
        // Kernel per questo canale di output: [input_channels * kernel_elements] x 1
        MatrixXd kernel_flat = kernels_[oc].reshaped(input_channels_ * kernel_size_ * kernel_size_, 1);
        
        // Prodotto matriciale: kernel^T * input_cols
        MatrixXd conv_result = kernel_flat.transpose() * input_cols_;
        
        // Aggiungi bias e riorganizza
        for (int b = 0; b < batch_size; ++b) {
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    int col_index = b * output_h * output_w + oh * output_w + ow;
                    int output_index = oc * output_h * output_w + oh * output_w + ow;
                    
                    output(b, output_index) = conv_result(0, col_index) + biases_(oc);
                }
            }
        }
    }
    
    // Applica funzione di attivazione
    cache_.z = output;
    cache_.output = activation_->forward(output);
    cache_.has_activation = true;
    
    // Salva output_cols_ per backward (se necessario)
    output_cols_ = input_cols_; // Memorizza per backward efficiente
    
    return cache_.output;
}

// Backward propagation
MatrixXd Convolutional::backward(const MatrixXd& gradient, double learning_rate) {
    if (!cache_.has_activation) {
        throw ml_exception::InvalidConfigurationException(
            "Cache not initialized. Call forward() first.", get_type());
    }
    
    int batch_size = cache_.input.rows();
    int output_h = calculate_output_height();
    int output_w = calculate_output_width();
    int kernel_elements = kernel_size_ * kernel_size_;
    
    // Gradiente rispetto a z (pre-attivazione)
    MatrixXd dZ = activation_->backward(gradient, cache_.z);
    
    // Riorganizza dZ in forma colonna
    MatrixXd dZ_cols = MatrixXd::Zero(output_channels_, batch_size * output_h * output_w);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < output_channels_; ++oc) {
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    int col_index = b * output_h * output_w + oh * output_w + ow;
                    int dZ_index = oc * output_h * output_w + oh * output_w + ow;
                    
                    dZ_cols(oc, col_index) = dZ(b, dZ_index);
                }
            }
        }
    }
    
    // Calcola gradienti per i kernels
    for (int oc = 0; oc < output_channels_; ++oc) {
        MatrixXd dW = MatrixXd::Zero(input_channels_, kernel_elements);
        
        for (int ic = 0; ic < input_channels_; ++ic) {
            for (int k = 0; k < kernel_elements; ++k) {
                int row_index = ic * kernel_elements + k;
                
                // dW[oc][ic][k] = sum_over_batches(dZ_cols[oc] * input_cols[row_index])
                double grad = 0.0;
                for (int col = 0; col < dZ_cols.cols(); ++col) {
                    grad += dZ_cols(oc, col) * input_cols_(row_index, col);
                }
                
                dW(ic, k) = grad / batch_size;
            }
        }
        
        // Aggiorna kernel
        kernels_[oc] -= learning_rate * dW;
    }
    
    // Calcola gradienti per i bias
    VectorXd db = VectorXd::Zero(output_channels_);
    for (int oc = 0; oc < output_channels_; ++oc) {
        db(oc) = dZ_cols.row(oc).sum() / batch_size;
    }
    biases_ -= learning_rate * db;
    
    // Calcola gradiente rispetto all'input (per propagare indietro)
    // dA_prev = sum_over_output_channels(kernel^T * dZ)
    MatrixXd dA_prev_cols = MatrixXd::Zero(input_channels_ * kernel_elements, 
                                          batch_size * output_h * output_w);
    
    for (int oc = 0; oc < output_channels_; ++oc) {
        MatrixXd kernel_T = kernels_[oc].transpose(); // [kernel_elements x input_channels]
        dA_prev_cols += kernel_T * dZ_cols.row(oc);
    }
    
    // Converti da formato colonna a formato immagine
    MatrixXd dA_prev = col2im(dA_prev_cols);
    
    return dA_prev;
}

// Padding utilities
MatrixXd Convolutional::apply_padding(const MatrixXd& input) const {
    if (padding_ == 0) return input;
    
    int batch_size = input.rows();
    int padded_h = input_height_ + 2 * padding_;
    int padded_w = input_width_ + 2 * padding_;
    int padded_size = input_channels_ * padded_h * padded_w;
    
    MatrixXd padded = MatrixXd::Zero(batch_size, padded_size);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int ic = 0; ic < input_channels_; ++ic) {
            for (int h = 0; h < input_height_; ++h) {
                for (int w = 0; w < input_width_; ++w) {
                    int input_idx = ic * input_height_ * input_width_ + h * input_width_ + w;
                    int padded_idx = ic * padded_h * padded_w + (h + padding_) * padded_w + (w + padding_);
                    
                    padded(b, padded_idx) = input(b, input_idx);
                }
            }
        }
    }
    
    return padded;
}

MatrixXd Convolutional::remove_padding(const MatrixXd& padded) const {
    if (padding_ == 0) return padded;
    
    int batch_size = padded.rows();
    int padded_h = input_height_ + 2 * padding_;
    int padded_w = input_width_ + 2 * padding_;
    
    MatrixXd unpadded = MatrixXd::Zero(batch_size, input_channels_ * input_height_ * input_width_);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int ic = 0; ic < input_channels_; ++ic) {
            for (int h = 0; h < input_height_; ++h) {
                for (int w = 0; w < input_width_; ++w) {
                    int padded_idx = ic * padded_h * padded_w + (h + padding_) * padded_w + (w + padding_);
                    int unpadded_idx = ic * input_height_ * input_width_ + h * input_width_ + w;
                    
                    unpadded(b, unpadded_idx) = padded(b, padded_idx);
                }
            }
        }
    }
    
    return unpadded;
}

// Metodi di interfaccia
int Convolutional::get_input_size() const {
    return input_channels_ * input_height_ * input_width_;
}

int Convolutional::get_output_size() const {
    int output_h = calculate_output_height();
    int output_w = calculate_output_width();
    return output_channels_ * output_h * output_w;
}

int Convolutional::get_parameter_count() const {
    int kernel_params = output_channels_ * input_channels_ * kernel_size_ * kernel_size_;
    int bias_params = output_channels_;
    return kernel_params + bias_params;
}

std::string Convolutional::get_config() const {
    std::ostringstream oss;
    oss << "Convolutional(input=(" << input_channels_ << ", " 
        << input_height_ << ", " << input_width_ << "), "
        << "output=" << output_channels_ << ", "
        << "kernel=" << kernel_size_ << ", "
        << "stride=" << stride_ << ", "
        << "padding=" << padding_ << ", "
        << "activation=" << activation_->get_type() << ", "
        << "params=" << get_parameter_count() << ")";
    return oss.str();
}

// Setters
void Convolutional::set_weights(const MatrixXd& weights) {
    // Per CNN, i pesi sono organizzati diversamente
    // Questa implementazione semplificata assume input appiattito
    throw ml_exception::NotImplementedException(
        "set_weights not implemented for Convolutional layer", get_type());
}

void Convolutional::set_biases(const VectorXd& biases) {
    if (biases.size() != output_channels_) {
        throw ml_exception::DimensionMismatchException(
            "biases",
            output_channels_, 1,
            biases.size(), 1,
            get_type());
    }
    biases_ = biases;
}

MatrixXd Convolutional::get_weights() const {
    // Appiattisci tutti i kernel in una singola matrice
    int kernel_elements = kernel_size_ * kernel_size_;
    MatrixXd all_weights(output_channels_, input_channels_ * kernel_elements);
    
    for (int oc = 0; oc < output_channels_; ++oc) {
        all_weights.row(oc) = kernels_[oc].reshaped(1, input_channels_ * kernel_elements);
    }
    
    return all_weights;
}

Eigen::VectorXd Convolutional::get_biases() const {
    return biases_;
}

void Convolutional::clear_cache() {
    cache_.input = MatrixXd();
    cache_.z = MatrixXd();
    cache_.output = MatrixXd();
    cache_.has_activation = false;
    input_cols_.clear();
    output_cols_.clear();
}

// Serializzazione
void Convolutional::serialize(std::ostream& out) const {
    using namespace serialization;
    
    // Serializza parametri
    out.write(reinterpret_cast<const char*>(&input_channels_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&input_height_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&input_width_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&output_channels_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&kernel_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&stride_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&padding_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&dilation_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&groups_), sizeof(int));
    
    // Serializza kernels
    int num_kernels = kernels_.size();
    out.write(reinterpret_cast<const char*>(&num_kernels), sizeof(int));
    
    for (const auto& kernel : kernels_) {
        eigen_utils::serialize_eigen(kernel, out);
    }
    
    // Serializza biases
    eigen_utils::serialize_eigen_vector(biases_, out);
    
    // Serializza tipo di attivazione
    std::string act_type = activation_->get_type();
    size_t act_len = act_type.size();
    out.write(reinterpret_cast<const char*>(&act_len), sizeof(size_t));
    out.write(act_type.c_str(), act_len);
}

void Convolutional::deserialize(std::istream& in) {
    using namespace serialization;
    
    // Deserializza parametri
    in.read(reinterpret_cast<char*>(&input_channels_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_height_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_width_), sizeof(int));
    in.read(reinterpret_cast<char*>(&output_channels_), sizeof(int));
    in.read(reinterpret_cast<char*>(&kernel_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&stride_), sizeof(int));
    in.read(reinterpret_cast<char*>(&padding_), sizeof(int));
    in.read(reinterpret_cast<char*>(&dilation_), sizeof(int));
    in.read(reinterpret_cast<char*>(&groups_), sizeof(int));
    
    // Deserializza kernels
    int num_kernels;
    in.read(reinterpret_cast<char*>(&num_kernels), sizeof(int));
    kernels_.resize(num_kernels);
    
    for (int i = 0; i < num_kernels; ++i) {
        eigen_utils::deserialize_eigen(kernels_[i], in);
    }
    
    // Deserializza biases
    eigen_utils::deserialize_eigen_vector(biases_, in);
    
    // Deserializza tipo di attivazione
    size_t act_len;
    in.read(reinterpret_cast<char*>(&act_len), sizeof(size_t));
    std::string act_type(act_len, '\0');
    in.read(&act_type[0], act_len);
    
    // Ricrea funzione di attivazione
    activation_ = activation::create_activation(act_type);
    if (!activation_) {
        throw ml_exception::DeserializationException(
            "layer file", "unknown activation type: " + act_type, get_type());
    }
    
    clear_cache();
}

// Metodi specifici CNN
void Convolutional::set_padding_mode(const std::string& mode) {
    // Implementazione base: solo padding con zeri supportato
    if (mode != "zeros") {
        throw ml_exception::NotImplementedException(
            "Only 'zeros' padding mode is currently supported", get_type());
    }
}

void Convolutional::set_dilation(int dilation) {
    ML_CHECK_PARAM(dilation > 0, "dilation", "must be > 0", get_type());
    dilation_ = dilation;
}

void Convolutional::set_groups(int groups) {
    ML_CHECK_PARAM(groups > 0, "groups", "must be > 0", get_type());
    ML_CHECK_PARAM(input_channels_ % groups == 0, 
                  "groups", "input_channels must be divisible by groups", get_type());
    ML_CHECK_PARAM(output_channels_ % groups == 0,
                  "groups", "output_channels must be divisible by groups", get_type());
    groups_ = groups;
}

std::vector<int> Convolutional::get_output_shape() const {
    return {output_channels_, calculate_output_height(), calculate_output_width()};
}

std::vector<int> Convolutional::get_input_shape() const {
    return {input_channels_, input_height_, input_width_};
}