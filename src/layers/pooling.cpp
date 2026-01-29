#include "layers/pooling.h"
#include "serialization/serializable.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

using namespace Eigen;
using namespace layers;

// Costruttore
Pooling::Pooling(int pool_size, int stride, PoolType type, int channels)
    : pool_size_(pool_size), stride_(stride), channels_(channels), pool_type_(type) {
    
    ML_CHECK_PARAM(pool_size > 0, "pool_size", "must be > 0", get_type());
    ML_CHECK_PARAM(stride > 0, "stride", "must be > 0", get_type());
    ML_CHECK_PARAM(channels > 0, "channels", "must be > 0", get_type());
    
    clear_cache();
}

// Calcolo dimensioni output
std::vector<int> Pooling::calculate_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 3) {
        throw ml_exception::InvalidParameterException(
            "input_shape", "must have 3 dimensions (channels, height, width)", get_type());
    }
    
    int channels = input_shape[0];
    int input_h = input_shape[1];
    int input_w = input_shape[2];
    
    int output_h = (input_h - pool_size_) / stride_ + 1;
    int output_w = (input_w - pool_size_) / stride_ + 1;
    
    return {channels, output_h, output_w};
}

// Forward propagation
MatrixXd Pooling::forward(const MatrixXd& input) {
    cache_.input = input;
    
    // Determina dimensioni input
    int batch_size = input.rows();
    int total_elements = input.cols();
    
    if (total_elements % channels_ != 0) {
        throw ml_exception::InvalidParameterException(
            "input dimensions", 
            "total elements must be divisible by channels", 
            get_type());
    }
    
    int spatial_elements = total_elements / channels_;
    
    // Trova altezza e larghezza (assumiamo input quadrato per semplicità)
    int input_size = static_cast<int>(std::sqrt(spatial_elements));
    if (input_size * input_size != spatial_elements) {
        throw ml_exception::InvalidParameterException(
            "input dimensions", 
            "spatial elements must be a perfect square", 
            get_type());
    }
    
    int input_h = input_size;
    int input_w = input_size;
    
    // Calcola dimensioni output
    int output_h = (input_h - pool_size_) / stride_ + 1;
    int output_w = (input_w - pool_size_) / stride_ + 1;
    int output_size = channels_ * output_h * output_w;
    
    // Inizializza output
    MatrixXd output = MatrixXd::Zero(batch_size, output_size);
    
    // Pulisci cache max_indices
    max_indices_.clear();
    max_indices_.resize(batch_size * channels_);
    
    // Applica pooling per ogni batch e canale
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels_; ++c) {
            int batch_channel_idx = b * channels_ + c;
            
            // Estrai il canale corrente
            MatrixXd channel_input = MatrixXd::Zero(input_h, input_w);
            for (int h = 0; h < input_h; ++h) {
                for (int w = 0; w < input_w; ++w) {
                    int input_idx = c * input_h * input_w + h * input_w + w;
                    channel_input(h, w) = input(b, input_idx);
                }
            }
            
            // Applica pooling 2D
            MatrixXd pooled = MatrixXd::Zero(output_h, output_w);
            
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    int start_h = oh * stride_;
                    int start_w = ow * stride_;
                    int end_h = std::min(start_h + pool_size_, input_h);
                    int end_w = std::min(start_w + pool_size_, input_w);
                    
                    if (pool_type_ == MAX) {
                        // Max pooling
                        double max_val = -std::numeric_limits<double>::infinity();
                        int max_idx_h = -1, max_idx_w = -1;
                        
                        for (int ph = start_h; ph < end_h; ++ph) {
                            for (int pw = start_w; pw < end_w; ++pw) {
                                if (channel_input(ph, pw) > max_val) {
                                    max_val = channel_input(ph, pw);
                                    max_idx_h = ph;
                                    max_idx_w = pw;
                                }
                            }
                        }
                        
                        pooled(oh, ow) = max_val;
                        
                        // Memorizza l'indice del massimo per backward
                        if (max_idx_h != -1 && max_idx_w != -1) {
                            int max_idx = max_idx_h * input_w + max_idx_w;
                            max_indices_[batch_channel_idx].push_back(max_idx);
                        }
                    } else {
                        // Average pooling
                        double sum = 0.0;
                        int count = 0;
                        
                        for (int ph = start_h; ph < end_h; ++ph) {
                            for (int pw = start_w; pw < end_w; ++pw) {
                                sum += channel_input(ph, pw);
                                count++;
                            }
                        }
                        
                        pooled(oh, ow) = sum / count;
                    }
                }
            }
            
            // Salva l'output appiattito
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    int output_idx = c * output_h * output_w + oh * output_w + ow;
                    output(b, output_idx) = pooled(oh, ow);
                }
            }
        }
    }
    
    cache_.output = output;
    cache_.has_activation = false; // Pooling non ha attivazione
    
    return output;
}

// Backward propagation
MatrixXd Pooling::backward(const MatrixXd& gradient, double learning_rate) {
    if (cache_.input.rows() == 0) {
        throw ml_exception::InvalidConfigurationException(
            "Cache not initialized. Call forward() first.", get_type());
    }
    
    int batch_size = cache_.input.rows();
    int input_cols = cache_.input.cols();
    int output_cols = gradient.cols();
    
    // Determina dimensioni
    int total_input_elements = input_cols;
    int spatial_input_elements = total_input_elements / channels_;
    int input_size = static_cast<int>(std::sqrt(spatial_input_elements));
    int input_h = input_size;
    int input_w = input_size;
    
    int spatial_output_elements = output_cols / channels_;
    int output_size = static_cast<int>(std::sqrt(spatial_output_elements));
    int output_h = output_size;
    int output_w = output_size;
    
    // Inizializza gradiente rispetto all'input
    MatrixXd dA_prev = MatrixXd::Zero(batch_size, input_cols);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels_; ++c) {
            int batch_channel_idx = b * channels_ + c;
            
            // Estrai il gradiente per questo canale
            MatrixXd channel_grad = MatrixXd::Zero(output_h, output_w);
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    int grad_idx = c * output_h * output_w + oh * output_w + ow;
                    channel_grad(oh, ow) = gradient(b, grad_idx);
                }
            }
            
            // Calcola gradiente rispetto all'input
            MatrixXd input_grad = MatrixXd::Zero(input_h, input_w);
            
            if (pool_type_ == MAX) {
                // Per max pooling, il gradiente va solo al valore massimo
                int idx = 0;
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow < output_w; ++ow) {
                        if (idx < max_indices_[batch_channel_idx].size()) {
                            int max_idx = max_indices_[batch_channel_idx][idx];
                            int max_h = max_idx / input_w;
                            int max_w = max_idx % input_w;
                            
                            input_grad(max_h, max_w) += channel_grad(oh, ow);
                            idx++;
                        }
                    }
                }
            } else {
                // Per average pooling, il gradiente è distribuito uniformemente
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow < output_w; ++ow) {
                        int start_h = oh * stride_;
                        int start_w = ow * stride_;
                        int end_h = std::min(start_h + pool_size_, input_h);
                        int end_w = std::min(start_w + pool_size_, input_w);
                        
                        int pool_area = (end_h - start_h) * (end_w - start_w);
                        double grad_per_element = channel_grad(oh, ow) / pool_area;
                        
                        for (int ph = start_h; ph < end_h; ++ph) {
                            for (int pw = start_w; pw < end_w; ++pw) {
                                input_grad(ph, pw) += grad_per_element;
                            }
                        }
                    }
                }
            }
            
            // Salva il gradiente appiattito
            for (int h = 0; h < input_h; ++h) {
                for (int w = 0; w < input_w; ++w) {
                    int input_idx = c * input_h * input_w + h * input_w + w;
                    dA_prev(b, input_idx) = input_grad(h, w);
                }
            }
        }
    }
    
    return dA_prev;
}

// Metodi di interfaccia
int Pooling::get_input_size() const {
    // Ritorna 0 perché dipende dall'input effettivo
    return 0;
}

int Pooling::get_output_size() const {
    // Ritorna 0 perché dipende dall'input effettivo
    return 0;
}

std::string Pooling::get_config() const {
    std::string type_str = (pool_type_ == MAX) ? "MAX" : "AVERAGE";
    std::ostringstream oss;
    oss << "Pooling(pool=" << pool_size_ 
        << ", stride=" << stride_
        << ", type=" << type_str
        << ", channels=" << channels_ << ")";
    return oss.str();
}

void Pooling::clear_cache() {
    cache_.input = MatrixXd();
    cache_.z = MatrixXd();
    cache_.output = MatrixXd();
    cache_.has_activation = false;
    max_indices_.clear();
}

// Serializzazione
void Pooling::serialize(std::ostream& out) const {
    using namespace serialization;
    
    out.write(reinterpret_cast<const char*>(&pool_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&stride_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&channels_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&pool_type_), sizeof(PoolType));
}

void Pooling::deserialize(std::istream& in) {
    using namespace serialization;
    
    in.read(reinterpret_cast<char*>(&pool_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&stride_), sizeof(int));
    in.read(reinterpret_cast<char*>(&channels_), sizeof(int));
    in.read(reinterpret_cast<char*>(&pool_type_), sizeof(PoolType));
    
    clear_cache();
}

// Utility functions
std::vector<int> Pooling::unravel_index(int flat_index, const std::vector<int>& shape) const {
    std::vector<int> indices(shape.size());
    int product = 1;
    
    for (int i = shape.size() - 1; i >= 0; --i) {
        indices[i] = (flat_index / product) % shape[i];
        product *= shape[i];
    }
    
    return indices;
}

int Pooling::ravel_index(const std::vector<int>& indices, const std::vector<int>& shape) const {
    int flat_index = 0;
    int product = 1;
    
    for (int i = shape.size() - 1; i >= 0; --i) {
        flat_index += indices[i] * product;
        product *= shape[i];
    }
    
    return flat_index;
}