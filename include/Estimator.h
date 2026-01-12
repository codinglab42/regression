#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <Eigen/Dense>
#include <string>

namespace regression {

class Estimator {
public:
    virtual ~Estimator() = default;

    // Metodi virtuali puri che ogni modello deve implementare
    virtual void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) = 0;
    
    // Usiamo Eigen::MatrixXd per predizioni batch
    virtual Eigen::VectorXd predict(const Eigen::MatrixXd& X) const = 0;

    virtual double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const = 0;
    
    virtual std::string to_string() const = 0;
    
    virtual void save(const std::string& filename) const = 0;
    virtual void load(const std::string& filename) = 0;
};

} // namespace regression

#endif