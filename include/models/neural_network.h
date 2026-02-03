#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "estimator.h"
#include "components/layers/layer.h"
#include "components/layers/dense.h"
#include "components/layers/pooling.h"
#include "components/activation/activation.h"
#include "components/optimizers/optimizer.h"
#include "components/optimizers/sgd.h"
#include "exceptions/exception_macros.h"
#include <memory>
#include <vector>

namespace models {

    class NeuralNetwork : public Estimator {
    public:
        NeuralNetwork();
        
        // Costruttore con architettura
        NeuralNetwork(const std::vector<int>& layer_sizes,
                     const std::string& activation = "relu",
                     const std::string& output_activation = "sigmoid");
        
        // Configurazione
        void add_layer(std::unique_ptr<layers::Layer> layer);
        void set_optimizer(std::unique_ptr<optimizers::Optimizer> optimizer);
        void set_loss_function(const std::string& loss);
        
        // Metodi Estimator
        void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
        Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
        double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const override;
        
        // Metodi SerializableModel
        std::string to_string() const override;
        void serialize_binary(std::ostream& out) const override;
        void deserialize_binary(std::istream& in) override;
        std::string get_model_type() const override { return "NeuralNetwork"; }
        
        // Metodi specifici
        void fit_batch(const Eigen::MatrixXd& X_batch, const Eigen::VectorXd& y_batch);
        Eigen::MatrixXd predict_proba(const Eigen::MatrixXd& X) const;
        
        // Training configuration
        void set_batch_size(int batch_size);
        void set_epochs(int epochs);
        void set_validation_split(double split);
        void set_verbose(bool verbose);
        
        // Getters
        const std::vector<double>& get_loss_history() const { return loss_history_; }
        const std::vector<double>& get_val_loss_history() const { return val_loss_history_; }
        const std::vector<double>& get_accuracy_history() const { return accuracy_history_; }
        int get_num_layers() const { return static_cast<int>(layers_.size()); }
        int get_num_parameters() const;
		
		// Metodi per aggiungere layer specializzati
		void add_convolutional_layer(int input_channels, int output_channels,
									int kernel_size, int stride = 1, int padding = 0,
									const std::string& activation = "relu");
		
		void add_pooling_layer(int pool_size = 2, int stride = 2,
							  layers::Pooling::PoolType type = layers::Pooling::MAX,
							  int channels = 1);
		
		void add_recurrent_layer(int hidden_size, 
								const std::string& activation = "tanh",
								bool return_sequences = false);
		
		void add_lstm_layer(int hidden_size, bool return_sequences = false);
		void add_gru_layer(int hidden_size, bool return_sequences = false);
			
    private:
        std::vector<std::unique_ptr<layers::Layer>> layers_;
        std::unique_ptr<optimizers::Optimizer> optimizer_;
        std::string loss_function_;
        
        // Training parameters
        int batch_size_;
        int epochs_;
        double validation_split_;
        bool verbose_;
        
        // Training history
        std::vector<double> loss_history_;
        std::vector<double> val_loss_history_;
        std::vector<double> accuracy_history_;
        
        // Cache per training
        mutable std::vector<layers::LayerCache> forward_cache_;
        
        // Metodi privati
        void initialize_weights(const Eigen::MatrixXd& X);
        Eigen::MatrixXd forward_pass(const Eigen::MatrixXd& X, 
                                   bool training = false) const;
        void backward_pass(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                         double learning_rate);
        double compute_loss(const Eigen::MatrixXd& y_pred, 
                          const Eigen::VectorXd& y_true) const;
        Eigen::MatrixXd compute_loss_gradient(const Eigen::MatrixXd& y_pred,
                                            const Eigen::VectorXd& y_true) const;
        std::vector<Eigen::MatrixXd> create_batches(const Eigen::MatrixXd& X,
                                                  const Eigen::VectorXd& y) const;
        void validate_architecture() const;
    };

} // namespace neural_network

#endif