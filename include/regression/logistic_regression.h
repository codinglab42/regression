#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "estimator.h"
#include "math_utils.h"
#include "exceptions/exception_macros.h"

namespace regression {

    class LogisticRegression : public Estimator {
    public:
        LogisticRegression();
        LogisticRegression(double learning_rate, int max_iter, 
                          double lambda = 0.0, double tolerance = 1e-4, 
                          bool verbose = false);
    
        // Metodi Interface
        void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
        Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
        double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const override;
        
        // Metodi SerializableModel
        std::string to_string() const override;
        void serialize_binary(std::ostream& out) const override;
        void deserialize_binary(std::istream& in) override;
        std::string get_model_type() const override { return "LogisticRegression"; }
    
        // Metodi specifici
        Eigen::VectorXi predict_class(const Eigen::MatrixXd& X, double threshold = 0.5) const;
        Eigen::Vector3d precision_recall_f1(const Eigen::MatrixXd& X, 
                                          const Eigen::VectorXd& y, 
                                          double threshold = 0.5) const;
        Eigen::MatrixXd confusion_matrix(const Eigen::MatrixXd& X, 
                                       const Eigen::VectorXd& y, 
                                       double threshold = 0.5) const;
        
        // Getters
        const Eigen::VectorXd& coefficients() const { return theta_; }
        double intercept() const { return (theta_.size() > 0) ? theta_(0) : 0.0; }
        const std::vector<double>& cost_history() const { return cost_history_; }
        const std::vector<double>& accuracy_history() const { return accuracy_history_; }
        
        // Setters con validazione
        void set_learning_rate(double rate);
        void set_max_iterations(int max_iter);
        void set_lambda(double lambda);
        void set_tolerance(double tolerance);
        void set_verbose(bool verbose);
    
    private:
        Eigen::VectorXd theta_;
        double learning_rate_;
        int max_iter_;
        double lambda_;
        double tolerance_;
        bool verbose_;
        int n_features_;
        int n_iter_;
        std::vector<double> cost_history_, accuracy_history_;
    
        struct Scaler {
            Eigen::VectorXd mean, std;
            bool fit = false;
            
            void serialize(std::ostream& out) const;
            void deserialize(std::istream& in);
        } scaler_;
    
        double compute_cost(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
        double compute_accuracy(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
        void fit_scaler(const Eigen::MatrixXd& X);
        Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;
        
        // Metodi di training
        void gradient_descent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
        void newton_method(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    };
}

#endif