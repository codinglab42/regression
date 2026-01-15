// linear_regression.h - CORREGGI
#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "estimator.h"
#include "math_utils.h"  // Aggiungi questa include

namespace regression {
    
    class LinearRegression : public Estimator {
    public:
        enum Solver { GRADIENT_DESCENT, NORMAL_EQUATION, SVD };
    
        LinearRegression();
        LinearRegression(double learning_rate, int max_iter, double lambda = 0.0, Solver solver = GRADIENT_DESCENT);
    
        // Metodi Interface
        void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
        Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
        double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const override;
        void save(const std::string& filename) const override;
        void load(const std::string& filename) override;
        std::string to_string() const override;
    
        // Metodi specifici
        double predict(const Eigen::VectorXd& x) const;
        double mse(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
        double mae(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
        double r2_score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
        static Eigen::VectorXd cross_val_score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int cv = 5, Solver solver = GRADIENT_DESCENT);
    
        // Getters
        const Eigen::VectorXd& coefficients() const { return theta_; }
        double intercept() const { return (theta_.size() > 0) ? theta_(0) : 0.0; }
        const std::vector<double>& cost_history() const { return cost_history_; }
    
    private:
        
        double learning_rate_;
        int max_iter_;
        double lambda_;
        double tolerance_;
        bool verbose_;
        Solver solver_;
        int n_features_;
        int n_iter_;
        Eigen::VectorXd theta_;
        std::vector<double> cost_history_;
    
        struct Scaler {
            Eigen::VectorXd mean, std;
            bool fit = false;
        } scaler_;
    
        void gradient_descent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
        void normal_equation(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
        void svd_solve(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
        double compute_cost(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
        void fit_scaler(const Eigen::MatrixXd& X);
        Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;
    };
}
#endif