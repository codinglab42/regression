#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "layer.h"
#include "components/activation/activation.h"
#include <vector>
#include <memory>

namespace layers {

    class Convolutional : public Layer {
    public:
        // Costruttore per immagini 2D
        Convolutional(int input_channels, int output_channels,
                     int kernel_size, int stride = 1, int padding = 0,
                     const std::string& activation = "relu");
        
        // Costruttore per immagini 3D (canali x altezza x larghezza)
        Convolutional(int input_channels, int input_height, int input_width,
                     int output_channels, int kernel_size,
                     int stride = 1, int padding = 0,
                     const std::string& activation = "relu");
        
        // Layer interface
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                               double learning_rate) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "Convolutional"; }
        std::string get_config() const override;
        int get_input_size() const override;
        int get_output_size() const override;
        int get_parameter_count() const override;
        
        void clear_cache() override;
        const LayerCache& get_cache() const override { return cache_; }
        
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;
        Eigen::VectorXd get_biases() const override;
        void set_weights(const Eigen::MatrixXd& weights) override;
        void set_biases(const Eigen::VectorXd& biases) override;
        
        // Metodi specifici CNN
        void set_padding_mode(const std::string& mode); // "zeros", "reflect", "replicate"
        void set_dilation(int dilation);
        void set_groups(int groups); // Per depthwise convolution
        
        // Utility
        std::vector<int> get_output_shape() const;
        std::vector<int> get_input_shape() const;
        
    private:
        // Dimensioni
        int input_channels_;
        int input_height_;
        int input_width_;
        int output_channels_;
        int kernel_size_;
        int stride_;
        int padding_;
        int dilation_;
        int groups_;
        
        // Pesi e bias
        std::vector<Eigen::MatrixXd> kernels_; // [output_channels][input_channels][kernel_size][kernel_size]
        Eigen::VectorXd biases_;
        
        // Attivazione
        std::unique_ptr<activation::Activation> activation_;
        
        // Cache
        LayerCache cache_;
        std::vector<Eigen::MatrixXd> input_cols_; // Per im2col
        std::vector<Eigen::MatrixXd> output_cols_; // Per backward
        
        // Metodi privati
        void initialize_kernels();
        void initialize_biases();
        
        // Operazioni di convoluzione
        Eigen::MatrixXd im2col(const Eigen::MatrixXd& input) const;
        Eigen::MatrixXd col2im(const Eigen::MatrixXd& col_matrix) const;
        Eigen::MatrixXd convolve(const Eigen::MatrixXd& input) const;
        Eigen::MatrixXd convolve_transpose(const Eigen::MatrixXd& gradient) const;
        
        // Calcolo dimensioni output
        int calculate_output_height() const;
        int calculate_output_width() const;
        
        // Padding
        Eigen::MatrixXd apply_padding(const Eigen::MatrixXd& input) const;
        Eigen::MatrixXd remove_padding(const Eigen::MatrixXd& padded) const;
        
        // Utility
        std::vector<int> flatten_indices(int c, int h, int w) const;
    };

} // namespace layers

#endif