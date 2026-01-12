#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>

class LinearRegression {
public:
    enum Solver {
        GRADIENT_DESCENT,
        NORMAL_EQUATION,
        SVD
    };
    
    // Costruttori
    LinearRegression();
    LinearRegression(double learning_rate, int max_iter, double lambda = 0.0, 
                     Solver solver = GRADIENT_DESCENT);
    
    // Training
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    
    // Predizioni
    double predict(const Eigen::VectorXd& x) const;
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
    
    // Metrica
    double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
    double mse(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
    double mae(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
    double r2_score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
    
    // Cross-validation
    static Eigen::VectorXd cross_val_score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                          int cv = 5, Solver solver = GRADIENT_DESCENT);
    
    // Getter
    const Eigen::VectorXd& coefficients() const { return theta_; }
    double intercept() const { return theta_.size() > 0 ? theta_(0) : 0.0; }
    Eigen::VectorXd coef() const { return theta_.tail(theta_.size() - 1); }
    
    const std::vector<double>& cost_history() const { return cost_history_; }
    int n_iter() const { return n_iter_; }
    int n_features() const { return n_features_; }
    
    // Salva/carica
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    
    // Utility
    std::string to_string() const;
    
private:
    Eigen::VectorXd theta_;            // Parametri [intercept, coef1, coef2, ...]
    double learning_rate_;             // Tasso di apprendimento
    int max_iter_;                     // Massime iterazioni
    double lambda_;                    // Parametro regolarizzazione
    Solver solver_;                    // Metodo di soluzione
    
    int n_features_;                   // Numero features
    int n_iter_;                       // Iterazioni effettive
    std::vector<double> cost_history_; // Storia costo
    
    // Funzioni interne
    void gradient_descent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    void normal_equation(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    void svd_solve(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    
    double compute_cost(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
    Eigen::VectorXd compute_gradient(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const;
    
    // Feature scaling
    struct Scaler {
        Eigen::VectorXd mean;
        Eigen::VectorXd std;
        bool fit = false;
    };
    
    Scaler scaler_;
    
    void fit_scaler(const Eigen::MatrixXd& X);
    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& X_scaled) const;
    
    // Aggiungi colonna intercetta
    static Eigen::MatrixXd add_intercept(const Eigen::MatrixXd& X);
};

#endif // LINEAR_REGRESSION_H