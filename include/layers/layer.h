#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <memory>
#include <string>
#include "exceptions/exception_macros.h"

namespace layers {

    struct LayerCache {
        Eigen::MatrixXd input;    // Input al layer
        Eigen::MatrixXd output;   // Output del layer
        Eigen::MatrixXd z;        // Pre-attivazione (se applicabile)
        bool has_activation;
        
        LayerCache() : has_activation(false) {}
    };

    class Layer {
    public:
        virtual ~Layer() = default;
        
        // Forward propagation
        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
        
        // Backward propagation
        virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                                       double learning_rate) = 0;
        
        // Per serializzazione
        virtual void serialize(std::ostream& out) const = 0;
        virtual void deserialize(std::istream& in) = 0;
        
        // Informazioni
        virtual std::string get_type() const = 0;
        virtual std::string get_config() const = 0;
        virtual int get_input_size() const = 0;
        virtual int get_output_size() const = 0;
        virtual int get_parameter_count() const = 0;
        
        // Cache management
        virtual void clear_cache() = 0;
        virtual const LayerCache& get_cache() const = 0;
        
        // Utility
        virtual bool has_weights() const = 0;
        virtual Eigen::MatrixXd get_weights() const = 0;
        virtual Eigen::VectorXd get_biases() const = 0;
        virtual void set_weights(const Eigen::MatrixXd& weights) = 0;
        virtual void set_biases(const Eigen::VectorXd& biases) = 0;
    };

} // namespace layers

#endif