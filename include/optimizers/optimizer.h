#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>
#include <string>
#include <memory>

namespace optimizers {

    class Optimizer {
    public:
        virtual ~Optimizer() = default;
        
        virtual void update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradients) = 0;
        virtual void update(Eigen::VectorXd& biases, const Eigen::VectorXd& gradients) = 0;
        
        virtual void set_learning_rate(double lr) = 0;
        virtual double get_learning_rate() const = 0;
        
        virtual std::string get_type() const = 0;
        virtual void serialize(std::ostream& out) const = 0;
        virtual void deserialize(std::istream& in) = 0;
    };

} // namespace optimizers

#endif