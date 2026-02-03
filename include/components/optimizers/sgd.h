#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include "optimizer.h"

namespace optimizers {

    class SGD : public Optimizer {
    public:
        SGD(double learning_rate = 0.01, double momentum = 0.0);
        
        void update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradients) override;
        void update(Eigen::VectorXd& biases, const Eigen::VectorXd& gradients) override;
        
        void set_learning_rate(double lr) override { learning_rate_ = lr; }
        double get_learning_rate() const override { return learning_rate_; }
        
        void set_momentum(double momentum);
        double get_momentum() const { return momentum_; }
        
        std::string get_type() const override { return "SGD"; }
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
    private:
        double learning_rate_;
        double momentum_;
        
        // Cache per momentum
        Eigen::MatrixXd velocity_weights_;
        Eigen::VectorXd velocity_biases_;
        bool velocity_initialized_;
        
        void initialize_velocity(const Eigen::MatrixXd& weights, const Eigen::VectorXd& biases);
    };

} // namespace optimizers

#endif