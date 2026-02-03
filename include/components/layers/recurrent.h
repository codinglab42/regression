#ifndef RECURRENT_LAYER_H
#define RECURRENT_LAYER_H

#include "layer.h"
#include "components/activation/activation.h"
#include <vector>
#include <memory>

namespace layers {
	
	/*
		RNN semplice con attivazione personalizzabile
		LSTM con 4 gates (input, forget, cell, output)
		GRU con 3 gates (update, reset, candidate)
		Backward propagation completa con BPTT
		Gradient clipping per stabilità
		Caching per performance
		Serializzazione completa
	*/

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
        void set_gradient_clip(double clip_value);
        double get_gradient_clip() const;
        
        // Per accesso agli stati
        Eigen::MatrixXd get_last_hidden_state() const;
        Eigen::MatrixXd get_last_cell_state() const; // Solo per LSTM
        
    private:
        // Dimensioni
        int input_size_;
        int hidden_size_;
        int sequence_length_;
        RNNType rnn_type_;
        bool return_sequences_;
        
        // Pesi COMBINATI
        Eigen::MatrixXd W_x_; // Pesi per input [hidden_size * gates x input_size]
        Eigen::MatrixXd W_h_; // Pesi ricorrenti [hidden_size * gates x hidden_size]
        Eigen::VectorXd b_;   // Bias [hidden_size * gates]
        
        // Attivazione
        std::unique_ptr<activation::Activation> activation_;
        std::unique_ptr<activation::Activation> gate_activation_;
        
        // Cache per training
        LayerCache cache_;
        std::vector<Eigen::MatrixXd> hidden_states_;
        std::vector<Eigen::MatrixXd> cell_states_;   // Solo per LSTM
        std::vector<Eigen::MatrixXd> inputs_;
        
        // Cache ottimizzata per backward
        std::vector<Eigen::MatrixXd> z_gates_;      // Per GRU
        std::vector<Eigen::MatrixXd> r_gates_;      // Per GRU
        std::vector<Eigen::MatrixXd> h_candidate_raw_; // Per GRU
        std::vector<Eigen::MatrixXd> lstm_gates_raw_;  // Per LSTM
        
        // Dropout
        double dropout_rate_;
        double recurrent_dropout_rate_;
        std::vector<Eigen::MatrixXd> dropout_masks_;
        std::vector<Eigen::MatrixXd> recurrent_dropout_masks_;
        
        // Gradient clipping
        double clip_value_;
        
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
        void clip_gradient(Eigen::MatrixXd& gradient, double clip_value);
        void clip_gradient(Eigen::VectorXd& gradient, double clip_value);
        
        // Calcolo dimensioni
        int calculate_output_size() const;
    };

} // namespace layers

#endif