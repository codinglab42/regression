#ifndef RECURRENT_LAYER_H
#define RECURRENT_LAYER_H

#include "layer.h"
#include <vector>
#include <memory>

namespace layers {

    class Recurrent : public Layer {
    public:
        enum RNNType { SIMPLE, LSTM, GRU };
        
        // Costruttore per RNN semplice
        Recurrent(int input_size, int hidden_size, 
                 const std::string& activation = "tanh",
                 bool return_sequences = false);
        
        // Costruttore per LSTM o GRU
        Recurrent(int input_size, int hidden_size,
                 RNNType type, bool return_sequences = false);
        
        // Layer interface
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                               double learning_rate) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "Recurrent"; }
        std::string get_config() const override;
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override;
        int get_parameter_count() const override;
        
        void clear_cache() override;
        const LayerCache& get_cache() const override { return cache_; }
        
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;
        Eigen::VectorXd get_biases() const override;
        void set_weights(const Eigen::MatrixXd& weights) override;
        void set_biases(const Eigen::VectorXd& biases) override;
        
        // Metodi specifici RNN
        void set_dropout(double dropout_rate);
        void set_recurrent_dropout(double recurrent_dropout_rate);
        void set_sequence_length(int seq_length);
        void reset_states(); // Resetta gli stati nascosti
        
        // Per accesso agli stati
        Eigen::MatrixXd get_last_hidden_state() const;
        Eigen::MatrixXd get_last_cell_state() const; // Solo per LSTM
        
    private:
        int input_size_;
        int hidden_size_;
        int sequence_length_;
        RNNType rnn_type_;
        bool return_sequences_;
        
        // Pesi
        Eigen::MatrixXd W_x_; // Pesi per input [hidden_size x input_size]
        Eigen::MatrixXd W_h_; // Pesi ricorrenti [hidden_size x hidden_size]
        Eigen::VectorXd b_;   // Bias [hidden_size]
        
        // Pesi aggiuntivi per LSTM
        Eigen::MatrixXd W_f_, W_i_, W_c_, W_o_; // Pesi per LSTM gates
        Eigen::VectorXd b_f_, b_i_, b_c_, b_o_; // Bias per LSTM gates
        
        // Pesi aggiuntivi per GRU
        Eigen::MatrixXd W_z_, W_r_, W_hh_; // Pesi per GRU gates
        Eigen::VectorXd b_z_, b_r_, b_hh_; // Bias per GRU gates
        
        // Attivazione
        std::unique_ptr<activation::Activation> activation_;
        std::unique_ptr<activation::Activation> gate_activation_; // Per sigmoid in LSTM/GRU
        
        // Cache per training
        LayerCache cache_;
        std::vector<Eigen::MatrixXd> hidden_states_; // Stati nascosti per ogni timestep
        std::vector<Eigen::MatrixXd> cell_states_;   // Stati cella per LSTM
        std::vector<Eigen::MatrixXd> inputs_;        // Input per ogni timestep
        
        // Dropout
        double dropout_rate_;
        double recurrent_dropout_rate_;
        std::vector<Eigen::MatrixXd> dropout_masks_;
        std::vector<Eigen::MatrixXd> recurrent_dropout_masks_;
        
        // Metodi di inizializzazione
        void initialize_weights_simple();
        void initialize_weights_lstm();
        void initialize_weights_gru();
        
        // Metodi forward per diversi tipi RNN
        Eigen::MatrixXd forward_simple(const Eigen::MatrixXd& input);
        Eigen::MatrixXd forward_lstm(const Eigen::MatrixXd& input);
        Eigen::MatrixXd forward_gru(const Eigen::MatrixXd& input);
        
        // Metodi backward
        Eigen::MatrixXd backward_simple(const Eigen::MatrixXd& gradient, double learning_rate);
        Eigen::MatrixXd backward_lstm(const Eigen::MatrixXd& gradient, double learning_rate);
        Eigen::MatrixXd backward_gru(const Eigen::MatrixXd& gradient, double learning_rate);
        
        // Utility
        void apply_dropout(Eigen::MatrixXd& matrix, std::vector<Eigen::MatrixXd>& masks, double rate);
        void apply_dropout_gradient(Eigen::MatrixXd& gradient, const std::vector<Eigen::MatrixXd>& masks);
        
        // Calcolo dimensioni
        int calculate_output_size() const;
    };

} // namespace layers

#endif