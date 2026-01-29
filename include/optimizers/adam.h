#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include "optimizer.h"

namespace optimizers {

    class Adam : public Optimizer {
    public:
        Adam(double learning_rate = 0.001, 
             double beta1 = 0.9, double beta2 = 0.999, 
             double epsilon = 1e-8);
        
        void update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradients) override;
        void update(Eigen::VectorXd& biases, const Eigen::VectorXd& gradients) override;
        
        void set_learning_rate(double lr) override { learning_rate_ = lr; }
        double get_learning_rate() const override { return learning_rate_; }
        
        void set_betas(double beta1, double beta2);
        void set_epsilon(double epsilon);
        
        std::string get_type() const override { return "Adam"; }
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        void reset(); // Resetta moment estimates
        
    private:
        double learning_rate_;
        double beta1_;
        double beta2_;
        double epsilon_;
        
        // Moment estimates
        Eigen::MatrixXd m_weights_, v_weights_;
        Eigen::VectorXd m_biases_, v_biases_;
        
        // Timestep
        long long t_;
        bool initialized_;
        
        void initialize_moments(const Eigen::MatrixXd& weights, const Eigen::VectorXd& biases);
    };

} // namespace optimizers

#endif