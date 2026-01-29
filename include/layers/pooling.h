#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "layer.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace layers {

    class Pooling : public Layer {
    public:
        enum PoolType { MAX, AVERAGE };
        
        // Costruttore
        Pooling(int pool_size = 2, int stride = 2, 
                PoolType type = MAX, int channels = 1);
        
        // Layer interface
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                               double learning_rate) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "Pooling"; }
        std::string get_config() const override;
        int get_input_size() const override;
        int get_output_size() const override;
        int get_parameter_count() const override { return 0; } // No parameters
        
        void clear_cache() override;
        const LayerCache& get_cache() const override { return cache_; }
        
        bool has_weights() const override { return false; }
        Eigen::MatrixXd get_weights() const override { return MatrixXd(); }
        Eigen::VectorXd get_biases() const override { return VectorXd(); }
        void set_weights(const Eigen::MatrixXd& weights) override {}
        void set_biases(const Eigen::VectorXd& biases) override {}
        
        // Metodi specifici
        void set_pool_type(PoolType type) { pool_type_ = type; }
        PoolType get_pool_type() const { return pool_type_; }
        
    private:
        int pool_size_;
        int stride_;
        int channels_;
        PoolType pool_type_;
        
        // Cache per backward (indici per max pooling)
        LayerCache cache_;
        std::vector<std::vector<int>> max_indices_; // Memorizza gli indici per max pooling
        
        // Metodi privati
        std::vector<int> calculate_output_shape(const std::vector<int>& input_shape) const;
        MatrixXd pool_2d(const MatrixXd& input, int channel, int batch);
        MatrixXd pool_backward_2d(const MatrixXd& gradient, int channel, int batch);
        
        // Utility
        std::vector<int> unravel_index(int flat_index, const std::vector<int>& shape) const;
        int ravel_index(const std::vector<int>& indices, const std::vector<int>& shape) const;
    };

} // namespace layers

#endif