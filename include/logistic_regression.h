#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>

class LogisticRegression {
public:
    // Costruttori
    LogisticRegression();
    LogisticRegression(double learning_rate, int max_iter, double lambda = 0.0,
                      double tolerance = 1e-4, bool verbose = false);
    
    // Training
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    
    // Predizioni
    double predict_proba(const Eigen::VectorXd& x) const;
    int predict(const Eigen::VectorXd& x, double threshold = 0.5) const;
    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X) const;
    Eigen::VectorXi predict(const Eigen::MatrixXd& X, double threshold = 0.5) const;
    
    // Metriche
    double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, 
                double threshold = 0.5) const;
    Eigen::Vector3d precision_recall_f1(const Eigen::MatrixXd& X, 
                                       const Eigen::VectorXd& y,
                                       double threshold = 0.5) const;
    Eigen::MatrixXd confusion_matrix(const Eigen::MatrixXd& X,
                                    const Eigen::VectorXd& y,
                                    double threshold = 0.5) const;
    
    // Cross-validation
    static Eigen::VectorXd cross_val_score(const Eigen::MatrixXd& X, 
                                          const Eigen::VectorXd& y,
                                          int cv = 5);
    
    // Getter
    const Eigen::VectorXd& coefficients() const { return theta_; }
    double intercept() const { return theta_.size() > 0 ? theta_(0) : 0.0; }
    Eigen::VectorXd coef() const { return theta_.tail(theta_.size() - 1); }
    
    const std::vector<double>& cost_history() const { return cost_history_; }
    const std::vector<double>& accuracy_history() const { return accuracy_history_; }
    int n_iter() const { return n_iter_; }
    int n_features() const { return n_features_; }
    
    // Salva/carica
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    
    // Utility
    std::string to_string() const;
    Eigen::VectorXd decision_function(const Eigen::MatrixXd& X) const;
    
private:
    Eigen::VectorXd theta_;            // Parametri
    double learning_rate_;             // Tasso apprendimento
    int max_iter_;                     // Massime iterazioni
    double lambda_;                    // Regolarizzazione L2
    double tolerance_;                 // Tolleranza convergenza
    bool verbose_;                     // Output dettagliato
    
    int n_features_;                   // Numero features
    int n_iter_;                       // Iterazioni effettive
    std::vector<double> cost_history_; // Storia costo
    std::vector<double> accuracy_history_; // Storia accuracy
    
    // Funzioni matematiche
    static double sigmoid(double z);
    static Eigen::VectorXd sigmoid(const Eigen::VectorXd& z);
    
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
    
    // One-vs-Rest per multi-class (estensione futura)
    std::vector<LogisticRegression> one_vs_rest_models_;
    int n_classes_;
};

#endif // LOGISTIC_REGRESSION_H