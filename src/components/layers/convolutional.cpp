#include "components/layers/convolutional.h"
#include "utils/serializable.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;
using namespace layers;

// Costruttore per immagini 2D (input: [batch_size, input_channels * input_height * input_width])
Convolutional::Convolutional(int input_channels, int output_channels,
                           int kernel_size, int stride, int padding,
                           const std::string& activation)
    : input_channels_(input_channels), input_height_(-1), input_width_(-1),
      output_channels_(output_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding), dilation_(1), groups_(1) {
    
    ML_CHECK_PARAM(input_channels > 0, "input_channels", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(output_channels > 0, "output_channels", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(kernel_size > 0, "kernel_size", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(stride > 0, "stride", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(padding >= 0, "padding", "must be >= 0", "Convolutional");
    
    // Inizializza pesi e bias
    initialize_kernels();
    initialize_biases();
    
    // Funzione di attivazione
    activation_ = activation::create_activation(activation);
    if (!activation_) {
        throw ml_exception::InvalidParameterException(
            "activation", "unknown activation function: " + activation, "Convolutional");
    }
    
    clear_cache();
}

// Costruttore per immagini 3D
Convolutional::Convolutional(int input_channels, int input_height, int input_width,
                           int output_channels, int kernel_size,
                           int stride, int padding,
                           const std::string& activation)
    : input_channels_(input_channels), input_height_(input_height), input_width_(input_width),
      output_channels_(output_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding), dilation_(1), groups_(1) {
    
    ML_CHECK_PARAM(input_channels > 0, "input_channels", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(input_height > 0, "input_height", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(input_width > 0, "input_width", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(output_channels > 0, "output_channels", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(kernel_size > 0, "kernel_size", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(stride > 0, "stride", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(padding >= 0, "padding", "must be >= 0", "Convolutional");
    
    initialize_kernels();
    initialize_biases();
    
    activation_ = activation::create_activation(activation);
    if (!activation_) {
        throw ml_exception::InvalidParameterException(
            "activation", "unknown activation function: " + activation, "Convolutional");
    }
    
    clear_cache();
}

// Inizializzazione kernels
void Convolutional::initialize_kernels() {
    kernels_.resize(output_channels_);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // He initialization per CNN
    double stddev = std::sqrt(2.0 / (input_channels_ * kernel_size_ * kernel_size_));
    std::normal_distribution<> dist(0.0, stddev);
    
    for (int oc = 0; oc < output_channels_; ++oc) {
        kernels_[oc] = MatrixXd(input_channels_, kernel_size_ * kernel_size_);
        for (int ic = 0; ic < input_channels_; ++ic) {
            for (int k = 0; k < kernel_size_ * kernel_size_; ++k) {
                kernels_[oc](ic, k) = dist(gen);
            }
        }
    }
}

// Inizializzazione bias
void Convolutional::initialize_biases() {
    biases_ = VectorXd::Zero(output_channels_);
}

// Calcola dimensioni output
int Convolutional::calculate_output_height() const {
    if (input_height_ <= 0) return -1;
    return (input_height_ + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
}

int Convolutional::calculate_output_width() const {
    if (input_width_ <= 0) return -1;
    return (input_width_ + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
}

// Forward pass
MatrixXd Convolutional::forward(const MatrixXd& input) {
    // Determina dimensioni se non specificate
    if (input_height_ <= 0 || input_width_ <= 0) {
        // Assume che input sia flatten: [batch_size, channels * height * width]
        // Per semplicità, assumiamo immagini quadrate
        int spatial_size = static_cast<int>(std::sqrt(input.cols() / input_channels_));
        if (spatial_size * spatial_size * input_channels_ != input.cols()) {
            throw ml_exception::DimensionMismatchException(
                "input dimensions",
                input.cols(), 1,
                input_channels_ * spatial_size * spatial_size, 1,
                "Convolutional");
        }
        input_height_ = spatial_size;
        input_width_ = spatial_size;
    }
    
    int batch_size = input.rows();
    int output_height = calculate_output_height();
    int output_width = calculate_output_width();
    
    if (output_height <= 0 || output_width <= 0) {
        throw ml_exception::InvalidConfigurationException(
            "Invalid output dimensions. Check input size, kernel size, stride and padding.",
            "Convolutional");
    }
    
    // Salva input nella cache
    cache_.input = input;
    
    // Applica padding se necessario
    MatrixXd padded_input = apply_padding(input);
    
    // Converti input in formato colonne (im2col)
    input_cols_.clear();
    for (int b = 0; b < batch_size; ++b) {
        MatrixXd col_matrix = im2col(padded_input.row(b));
        input_cols_.push_back(col_matrix);
    }
    
    // Convoluzione
    MatrixXd output(batch_size, output_channels_ * output_height * output_width);
    output_cols_.clear();
    
    for (int b = 0; b < batch_size; ++b) {
        MatrixXd col_result = convolve(input_cols_[b]);
        output_cols_.push_back(col_result);
        
        // Riassembla output
        for (int oc = 0; oc < output_channels_; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    int idx = oc * output_height * output_width + oh * output_width + ow;
                    int col_idx = oh * output_width + ow;
                    output(b, idx) = col_result(oc, col_idx) + biases_(oc);
                }
            }
        }
    }
    
    // Applica attivazione
    cache_.z = output;
    cache_.output = activation_->forward(output);
    cache_.has_activation = true;
    
    return cache_.output;
}

// Backward pass
MatrixXd Convolutional::backward(const MatrixXd& gradient, double learning_rate) {
    if (!cache_.has_activation) {
        throw ml_exception::InvalidConfigurationException(
            "Cache not initialized. Call forward() first.", "Convolutional");
    }
    
    int batch_size = gradient.rows();
    int output_height = calculate_output_height();
    int output_width = calculate_output_width();
    
    // Gradiente rispetto a z (pre-attivazione)
    MatrixXd dZ = activation_->backward(gradient, cache_.z);
    
    // Inizializza gradienti per kernels e bias
    std::vector<MatrixXd> dKernels(output_channels_);
    for (int oc = 0; oc < output_channels_; ++oc) {
        dKernels[oc] = MatrixXd::Zero(input_channels_, kernel_size_ * kernel_size_);
    }
    VectorXd dBias = VectorXd::Zero(output_channels_);
    
    // Gradiente rispetto all'input
    MatrixXd dInput = MatrixXd::Zero(batch_size, input_channels_ * input_height_ * input_width_);
    
    for (int b = 0; b < batch_size; ++b) {
        // Riassembla dZ in formato colonne
        MatrixXd dZ_col(output_channels_, output_height * output_width);
        for (int oc = 0; oc < output_channels_; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    int idx = oc * output_height * output_width + oh * output_width + ow;
                    int col_idx = oh * output_width + ow;
                    dZ_col(oc, col_idx) = dZ(b, idx);
                }
            }
        }
        
        // Gradiente rispetto ai kernels
        for (int oc = 0; oc < output_channels_; ++oc) {
            dKernels[oc] += dZ_col.row(oc).transpose() * input_cols_[b];
            dBias(oc) += dZ_col.row(oc).sum();
        }
        
        // Gradiente rispetto all'input (da propagare indietro)
        MatrixXd dInput_col = convolve_transpose(dZ_col);
        MatrixXd dInput_unpadded = col2im(dInput_col);
        
        // Rimuovi padding dal gradiente
        MatrixXd dInput_no_padding = remove_padding(dInput_unpadded);
        
        // Somma al gradiente totale dell'input
        dInput.row(b) += dInput_no_padding;
    }
    
    // Aggiorna kernels e bias
    for (int oc = 0; oc < output_channels_; ++oc) {
        kernels_[oc] -= learning_rate * dKernels[oc] / batch_size;
    }
    biases_ -= learning_rate * dBias / batch_size;
    
    return dInput;
}

// Operazione im2col
MatrixXd Convolutional::im2col(const MatrixXd& input) const {
    int padded_height = input_height_ + 2 * padding_;
    int padded_width = input_width_ + 2 * padding_;
    
    // Risistema input come [channels, padded_height, padded_width]
    std::vector<MatrixXd> channels(input_channels_);
    for (int c = 0; c < input_channels_; ++c) {
        channels[c] = Map<const MatrixXd>(input.data() + c * padded_height * padded_width,
                                        padded_height, padded_width);
    }
    
    int output_height = calculate_output_height();
    int output_width = calculate_output_width();
    int col_rows = kernel_size_ * kernel_size_ * input_channels_;
    int col_cols = output_height * output_width;
    
    MatrixXd col_matrix(col_rows, col_cols);
    
    for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
            int col_idx = oh * output_width + ow;
            int row_idx = 0;
            
            for (int c = 0; c < input_channels_; ++c) {
                for (int kh = 0; kh < kernel_size_; ++kh) {
                    for (int kw = 0; kw < kernel_size_; ++kw) {
                        int ih = oh * stride_ + kh - padding_;
                        int iw = ow * stride_ + kw - padding_;
                        
                        if (ih >= 0 && ih < padded_height && iw >= 0 && iw < padded_width) {
                            col_matrix(row_idx, col_idx) = channels[c](ih, iw);
                        } else {
                            col_matrix(row_idx, col_idx) = 0.0; // Padding zero
                        }
                        row_idx++;
                    }
                }
            }
        }
    }
    
    return col_matrix;
}

// Operazione col2im
MatrixXd Convolutional::col2im(const MatrixXd& col_matrix) const {
    int padded_height = input_height_ + 2 * padding_;
    int padded_width = input_width_ + 2 * padding_;
    MatrixXd image = MatrixXd::Zero(input_channels_, padded_height * padded_width);
    
    int output_height = calculate_output_height();
    int output_width = calculate_output_width();
    
    for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
            int col_idx = oh * output_width + ow;
            
            int row_idx = 0;
            for (int c = 0; c < input_channels_; ++c) {
                for (int kh = 0; kh < kernel_size_; ++kh) {
                    for (int kw = 0; kw < kernel_size_; ++kw) {
                        int ih = oh * stride_ + kh - padding_;
                        int iw = ow * stride_ + kw - padding_;
                        
                        if (ih >= 0 && ih < padded_height && iw >= 0 && iw < padded_width) {
                            int pixel_idx = ih * padded_width + iw;
                            image(c, pixel_idx) += col_matrix(row_idx, col_idx);
                        }
                        row_idx++;
                    }
                }
            }
        }
    }
    
    return image;
}

// Convoluzione diretta
MatrixXd Convolutional::convolve(const MatrixXd& col_matrix) const {
    int output_height = calculate_output_height();
    int output_width = calculate_output_width();
    MatrixXd result(output_channels_, output_height * output_width);
    
    for (int oc = 0; oc < output_channels_; ++oc) {
        result.row(oc) = kernels_[oc].reshaped().transpose() * col_matrix;
    }
    
    return result;
}

// Convoluzione trasposta (per backward)
MatrixXd Convolutional::convolve_transpose(const MatrixXd& gradient) const {
    int col_rows = kernel_size_ * kernel_size_ * input_channels_;
    int col_cols = gradient.cols();
    MatrixXd result(col_rows, col_cols);
    
    for (int oc = 0; oc < output_channels_; ++oc) {
        result += kernels_[oc].reshaped() * gradient.row(oc);
    }
    
    return result;
}

// Applica padding
MatrixXd Convolutional::apply_padding(const MatrixXd& input) const {
    if (padding_ == 0) return input;
    
    int batch_size = input.rows();
    int padded_height = input_height_ + 2 * padding_;
    int padded_width = input_width_ + 2 * padding_;
    
    MatrixXd padded(batch_size, input_channels_ * padded_height * padded_width);
    padded.setZero();
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < input_channels_; ++c) {
            for (int h = 0; h < input_height_; ++h) {
                for (int w = 0; w < input_width_; ++w) {
                    int orig_idx = c * input_height_ * input_width_ + h * input_width_ + w;
                    int padded_h = h + padding_;
                    int padded_w = w + padding_;
                    int padded_idx = c * padded_height * padded_width + 
                                    padded_h * padded_width + padded_w;
                    padded(b, padded_idx) = input(b, orig_idx);
                }
            }
        }
    }
    
    return padded;
}

// Rimuovi padding
MatrixXd Convolutional::remove_padding(const MatrixXd& padded) const {
    if (padding_ == 0) return padded;
    
    int batch_size = padded.rows();
    int padded_height = input_height_ + 2 * padding_;
    int padded_width = input_width_ + 2 * padding_;
    
    MatrixXd unpadded(batch_size, input_channels_ * input_height_ * input_width_);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < input_channels_; ++c) {
            for (int h = 0; h < input_height_; ++h) {
                for (int w = 0; w < input_width_; ++w) {
                    int unpadded_idx = c * input_height_ * input_width_ + h * input_width_ + w;
                    int padded_h = h + padding_;
                    int padded_w = w + padding_;
                    int padded_idx = c * padded_height * padded_width + 
                                    padded_h * padded_width + padded_w;
                    unpadded(b, unpadded_idx) = padded(b, padded_idx);
                }
            }
        }
    }
    
    return unpadded;
}

// Informazioni
int Convolutional::get_input_size() const {
    if (input_height_ > 0 && input_width_ > 0) {
        return input_channels_ * input_height_ * input_width_;
    }
    return -1; // Dimensioni non ancora definite
}

int Convolutional::get_output_size() const {
    int output_height = calculate_output_height();
    int output_width = calculate_output_width();
    if (output_height > 0 && output_width > 0) {
        return output_channels_ * output_height * output_width;
    }
    return -1;
}

int Convolutional::get_parameter_count() const {
    int kernel_params = output_channels_ * input_channels_ * kernel_size_ * kernel_size_;
    int bias_params = output_channels_;
    return kernel_params + bias_params;
}

std::string Convolutional::get_config() const {
    std::ostringstream oss;
    oss << "Convolutional(input=(" << input_channels_;
    if (input_height_ > 0) oss << "x" << input_height_ << "x" << input_width_;
    oss << "), output=" << output_channels_;
    oss << ", kernel=" << kernel_size_;
    oss << ", stride=" << stride_;
    oss << ", padding=" << padding_;
    oss << ", activation=" << activation_->get_type();
    oss << ", params=" << get_parameter_count() << ")";
    return oss.str();
}

// Getters e setters
Eigen::MatrixXd Convolutional::get_weights() const {
    // Appiattisce tutti i kernels in una matrice
    int total_kernel_params = input_channels_ * kernel_size_ * kernel_size_;
    MatrixXd all_weights(output_channels_, total_kernel_params);
    for (int oc = 0; oc < output_channels_; ++oc) {
        all_weights.row(oc) = kernels_[oc].reshaped().transpose();
    }
    return all_weights;
}

Eigen::VectorXd Convolutional::get_biases() const {
    return biases_;
}

void Convolutional::set_weights(const Eigen::MatrixXd& weights) {
    if (weights.rows() != output_channels_ || 
        weights.cols() != input_channels_ * kernel_size_ * kernel_size_) {
        throw ml_exception::DimensionMismatchException(
            "weights",
            output_channels_, input_channels_ * kernel_size_ * kernel_size_,
            weights.rows(), weights.cols(),
            "Convolutional");
    }
    
    for (int oc = 0; oc < output_channels_; ++oc) {
        kernels_[oc] = Map<const MatrixXd>(weights.row(oc).data(),
                                         input_channels_, kernel_size_ * kernel_size_);
    }
}

void Convolutional::set_biases(const Eigen::VectorXd& biases) {
    if (biases.size() != output_channels_) {
        throw ml_exception::DimensionMismatchException(
            "biases",
            output_channels_, 1,
            biases.size(), 1,
            "Convolutional");
    }
    biases_ = biases;
}

// Cache management
void Convolutional::clear_cache() {
    cache_.input = MatrixXd();
    cache_.z = MatrixXd();
    cache_.output = MatrixXd();
    cache_.has_activation = false;
    input_cols_.clear();
    output_cols_.clear();
}

// Shape utilities
std::vector<int> Convolutional::get_output_shape() const {
    int output_height = calculate_output_height();
    int output_width = calculate_output_width();
    if (output_height > 0 && output_width > 0) {
        return {output_channels_, output_height, output_width};
    }
    return {};
}

std::vector<int> Convolutional::get_input_shape() const {
    if (input_height_ > 0 && input_width_ > 0) {
        return {input_channels_, input_height_, input_width_};
    }
    return {input_channels_};
}

// Serializzazione
void Convolutional::serialize(std::ostream& out) const {
    using namespace utils;
    
    // Serializza dimensioni
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
    using namespace utils;
    
    // Deserializza dimensioni
    in.read(reinterpret_cast<char*>(&input_channels_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_height_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_width_), sizeof(int));
    in.read(reinterpret_cast<char*>(&output_channels_), sizeof(int));
    in.read(reinterpret_cast<char*>(&kernel_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&stride_), sizeof(int));
    in.read(reinterpret_cast<char*>(&padding_), sizeof(int));
    in.read(reinterpret_cast<char*>(&dilation_), sizeof(int));
    in.read(reinterpret_cast<char*>(&groups_), sizeof(int));
    
    // Re-inizializza kernels
    kernels_.resize(output_channels_);
    for (int oc = 0; oc < output_channels_; ++oc) {
        eigen_utils::deserialize_eigen(kernels_[oc], in);
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
            "layer file", "unknown activation type: " + act_type, "Convolutional");
    }
    
    // Pulisci cache
    clear_cache();
}

// Metodi specifici CNN
void Convolutional::set_padding_mode(const std::string& mode) {
    // Per ora supportiamo solo "zeros"
    if (mode != "zeros") {
        throw ml_exception::InvalidParameterException(
            "padding_mode", "only 'zeros' is supported currently", "Convolutional");
    }
}

void Convolutional::set_dilation(int dilation) {
    ML_CHECK_PARAM(dilation > 0, "dilation", "must be > 0", "Convolutional");
    dilation_ = dilation;
}

void Convolutional::set_groups(int groups) {
    ML_CHECK_PARAM(groups > 0, "groups", "must be > 0", "Convolutional");
    ML_CHECK_PARAM(input_channels_ % groups == 0, "groups", 
                  "must divide input_channels evenly", "Convolutional");
    ML_CHECK_PARAM(output_channels_ % groups == 0, "groups",
                  "must divide output_channels evenly", "Convolutional");
    groups_ = groups;
}