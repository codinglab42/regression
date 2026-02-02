#include "layers/pooling.h"
#include "serialization/serializable.h"
#include <algorithm>
#include <stdexcept>
#include <limits>

using namespace Eigen;
using namespace layers;

// Costruttore
Pooling::Pooling(int pool_size, int stride, PoolType type, int channels)
    : pool_size_(pool_size), stride_(stride), channels_(channels), pool_type_(type) {
    
    ML_CHECK_PARAM(pool_size > 0, "pool_size", "must be > 0", "Pooling");
    ML_CHECK_PARAM(stride > 0, "stride", "must be > 0", "Pooling");
    ML_CHECK_PARAM(channels > 0, "channels", "must be > 0", "Pooling");
    
    clear_cache();
}

// Calcola dimensioni output
std::vector<int> Pooling::calculate_output_shape(const std::vector<int>& input_shape) const {
    // Assume input shape: [channels, height, width] oppure flatten
    if (input_shape.size() == 1) {
        // Flatten: [channels * height * width]
        int total_elements = input_shape[0];
        int spatial_size = static_cast<int>(std::sqrt(total_elements / channels_));
        if (spatial_size * spatial_size * channels_ != total_elements) {
            throw ml_exception::DimensionMismatchException(
                "input dimensions",
                total_elements, 1,
                channels_ * spatial_size * spatial_size, 1,
                "Pooling");
        }
        
        int output_height = (spatial_size - pool_size_) / stride_ + 1;
        int output_width = (spatial_size - pool_size_) / stride_ + 1;
        
        return {channels_ * output_height * output_width};
    }
    
    // Per ora assumiamo input flatten
    return {};
}

// Forward pass
MatrixXd Pooling::forward(const MatrixXd& input) {
    // Assumiamo input flatten: [batch_size, channels * height * width]
    int batch_size = input.rows();
    int total_input_elements = input.cols();
    
    // Calcola dimensioni spaziali
    int spatial_size = static_cast<int>(std::sqrt(total_input_elements / channels_));
    if (spatial_size * spatial_size * channels_ != total_input_elements) {
        throw ml_exception::DimensionMismatchException(
            "input dimensions",
            total_input_elements, 1,
            channels_ * spatial_size * spatial_size, 1,
            "Pooling");
    }
    
    int input_height = spatial_size;
    int input_width = spatial_size;
    
    // Calcola dimensioni output
    int output_height = (input_height - pool_size_) / stride_ + 1;
    int output_width = (input_width - pool_size_) / stride_ + 1;
    int output_size = channels_ * output_height * output_width;
    
    // Salva input nella cache
    cache_.input = input;
    cache_.has_activation = false; // Pooling non ha attivazione
    
    // Risistema input
    std::vector<std::vector<MatrixXd>> input_reshaped(
        batch_size, std::vector<MatrixXd>(channels_));
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels_; ++c) {
            input_reshaped[b][c] = MatrixXd(input_height, input_width);
            for (int h = 0; h < input_height; ++h) {
                for (int w = 0; w < input_width; ++w) {
                    int idx = c * input_height * input_width + h * input_width + w;
                    input_reshaped[b][c](h, w) = input(b, idx);
                }
            }
        }
    }
    
    // Pooling
    MatrixXd output(batch_size, output_size);
    max_indices_.clear();
    max_indices_.resize(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        max_indices_[b].resize(channels_ * output_height * output_width);
        
        for (int c = 0; c < channels_; ++c) {
            MatrixXd pooled = pool_2d(input_reshaped[b][c], c, b);
            
            // Salva output
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    int idx = c * output_height * output_width + oh * output_width + ow;
                    output(b, idx) = pooled(oh, ow);
                }
            }
        }
    }
    
    cache_.output = output;
    return output;
}

// Pooling 2D
MatrixXd Pooling::pool_2d(const MatrixXd& input, int channel, int batch) {
    int input_height = input.rows();
    int input_width = input.cols();
    int output_height = (input_height - pool_size_) / stride_ + 1;
    int output_width = (input_width - pool_size_) / stride_ + 1;
    
    MatrixXd output(output_height, output_width);
    
    for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
            int start_h = oh * stride_;
            int start_w = ow * stride_;
            int end_h = std::min(start_h + pool_size_, input_height);
            int end_w = std::min(start_w + pool_size_, input_width);
            
            if (pool_type_ == MAX) {
                double max_val = -std::numeric_limits<double>::infinity();
                int max_idx = -1;
                
                for (int h = start_h; h < end_h; ++h) {
                    for (int w = start_w; w < end_w; ++w) {
                        if (input(h, w) > max_val) {
                            max_val = input(h, w);
                            max_idx = h * input_width + w;
                        }
                    }
                }
                
                output(oh, ow) = max_val;
                
                // Salva indice per backward
                int output_idx = channel * output_height * output_width + oh * output_width + ow;
                max_indices_[batch][output_idx] = max_idx;
            } else if (pool_type_ == AVERAGE) {
                double sum = 0.0;
                int count = 0;
                
                for (int h = start_h; h < end_h; ++h) {
                    for (int w = start_w; w < end_w; ++w) {
                        sum += input(h, w);
                        count++;
                    }
                }
                
                output(oh, ow) = sum / count;
            }
        }
    }
    
    return output;
}

// Backward pass
MatrixXd Pooling::backward(const MatrixXd& gradient, double learning_rate) {
    // Pooling non ha parametri da aggiornare
    if (gradient.rows() != cache_.input.rows()) {
        throw ml_exception::DimensionMismatchException(
            "gradient rows",
            cache_.input.rows(), 1,
            gradient.rows(), 1,
            "Pooling");
    }
    
    int batch_size = gradient.rows();
    int output_size = gradient.cols();
    
    // Calcola dimensioni
    int total_input_elements = cache_.input.cols();
    int spatial_size = static_cast<int>(std::sqrt(total_input_elements / channels_));
    int input_height = spatial_size;
    int input_width = spatial_size;
    int output_height = (input_height - pool_size_) / stride_ + 1;
    int output_width = (input_width - pool_size_) / stride_ + 1;
    
    // Gradiente rispetto all'input
    MatrixXd dInput = MatrixXd::Zero(batch_size, total_input_elements);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels_; ++c) {
            // Estrai gradiente per questo canale
            MatrixXd grad_2d(output_height, output_width);
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    int idx = c * output_height * output_width + oh * output_width + ow;
                    grad_2d(oh, ow) = gradient(b, idx);
                }
            }
            
            // Calcola gradiente backward
            MatrixXd dInput_2d = pool_backward_2d(grad_2d, c, b);
            
            // Risistema nel formato flatten
            for (int h = 0; h < input_height; ++h) {
                for (int w = 0; w < input_width; ++w) {
                    int idx = c * input_height * input_width + h * input_width + w;
                    dInput(b, idx) += dInput_2d(h, w);
                }
            }
        }
    }
    
    return dInput;
}

// Backward 2D
MatrixXd Pooling::pool_backward_2d(const MatrixXd& gradient, int channel, int batch) {
    int output_height = gradient.rows();
    int output_width = gradient.cols();
    
    int total_input_elements = cache_.input.cols();
    int spatial_size = static_cast<int>(std::sqrt(total_input_elements / channels_));
    int input_height = spatial_size;
    int input_width = spatial_size;
    
    MatrixXd dInput = MatrixXd::Zero(input_height, input_width);
    
    if (pool_type_ == MAX) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                int output_idx = channel * output_height * output_width + oh * output_width + ow;
                int max_idx = max_indices_[batch][output_idx];
                
                if (max_idx >= 0) {
                    int h = max_idx / input_width;
                    int w = max_idx % input_width;
                    dInput(h, w) += gradient(oh, ow);
                }
            }
        }
    } else if (pool_type_ == AVERAGE) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                int start_h = oh * stride_;
                int start_w = ow * stride_;
                int end_h = std::min(start_h + pool_size_, input_height);
                int end_w = std::min(start_w + pool_size_, input_width);
                
                double grad_val = gradient(oh, ow);
                int count = (end_h - start_h) * (end_w - start_w);
                double avg_grad = grad_val / count;
                
                for (int h = start_h; h < end_h; ++h) {
                    for (int w = start_w; w < end_w; ++w) {
                        dInput(h, w) += avg_grad;
                    }
                }
            }
        }
    }
    
    return dInput;
}

// Informazioni
int Pooling::get_input_size() const {
    // Non conosciamo le dimensioni finché non vediamo l'input
    return -1;
}

int Pooling::get_output_size() const {
    // Uguale a get_input_size()
    return -1;
}

std::string Pooling::get_config() const {
    std::ostringstream oss;
    oss << "Pooling(pool_size=" << pool_size_
        << ", stride=" << stride_
        << ", type=" << (pool_type_ == MAX ? "max" : "average")
        << ", channels=" << channels_ << ")";
    return oss.str();
}

// Cache management
void Pooling::clear_cache() {
    cache_.input = MatrixXd();
    cache_.output = MatrixXd();
    cache_.has_activation = false;
    max_indices_.clear();
}

// Utility functions
std::vector<int> Pooling::unravel_index(int flat_index, const std::vector<int>& shape) const {
    std::vector<int> indices(shape.size());
    int product = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        indices[i] = (flat_index / product) % shape[i];
        product *= shape[i];
    }
    return indices;
}

int Pooling::ravel_index(const std::vector<int>& indices, const std::vector<int>& shape) const {
    int index = 0;
    int product = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        index += indices[i] * product;
        product *= shape[i];
    }
    return index;
}

// Serializzazione
void Pooling::serialize(std::ostream& out) const {
    using namespace serialization;
    
    out.write(reinterpret_cast<const char*>(&pool_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&stride_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&channels_), sizeof(int));
    
    int type_int = static_cast<int>(pool_type_);
    out.write(reinterpret_cast<const char*>(&type_int), sizeof(int));
}

void Pooling::deserialize(std::istream& in) {
    using namespace serialization;
    
    in.read(reinterpret_cast<char*>(&pool_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&stride_), sizeof(int));
    in.read(reinterpret_cast<char*>(&channels_), sizeof(int));
    
    int type_int;
    in.read(reinterpret_cast<char*>(&type_int), sizeof(int));
    pool_type_ = static_cast<PoolType>(type_int);
    
    clear_cache();
}