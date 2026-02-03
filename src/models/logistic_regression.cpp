#include "models/logistic_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "utils/serializable.h"

using namespace Eigen;
using namespace models;
using namespace utils;

// Costruttori
LogisticRegression::LogisticRegression()
    : learning_rate_(0.1), max_iter_(1000), lambda_(0.0), tolerance_(1e-4),
      verbose_(false), n_features_(0), n_iter_(0) {}

LogisticRegression::LogisticRegression(double learning_rate, int max_iter, 
                                     double lambda, double tolerance, bool verbose)
    : learning_rate_(learning_rate), max_iter_(max_iter), lambda_(lambda), 
      tolerance_(tolerance), verbose_(verbose), n_features_(0), n_iter_(0) {
    
    ML_CHECK_PARAM(learning_rate_ > 0, "learning_rate", "must be > 0", get_model_type());
    ML_CHECK_PARAM(max_iter_ > 0, "max_iter", "must be > 0", get_model_type());
    ML_CHECK_PARAM(lambda_ >= 0, "lambda", "must be >= 0", get_model_type());
    ML_CHECK_PARAM(tolerance_ > 0, "tolerance", "must be > 0", get_model_type());
}

// Setters con validazione
void LogisticRegression::set_learning_rate(double rate) {
    ML_CHECK_PARAM(rate > 0, "learning_rate", "must be > 0", get_model_type());
    learning_rate_ = rate;
}

void LogisticRegression::set_max_iterations(int max_iter) {
    ML_CHECK_PARAM(max_iter > 0, "max_iter", "must be > 0", get_model_type());
    max_iter_ = max_iter;
}

void LogisticRegression::set_lambda(double lambda) {
    ML_CHECK_PARAM(lambda >= 0, "lambda", "must be >= 0", get_model_type());
    lambda_ = lambda;
}

void LogisticRegression::set_tolerance(double tolerance) {
    ML_CHECK_PARAM(tolerance > 0, "tolerance", "must be > 0", get_model_type());
    tolerance_ = tolerance;
}

void LogisticRegression::set_verbose(bool verbose) {
    verbose_ = verbose;
}

// Metodi privati
void LogisticRegression::fit_scaler(const MatrixXd& X) {
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    
    scaler_.mean = X.colwise().mean();
    scaler_.std = ((X.rowwise() - scaler_.mean.transpose()).array().square()
                  .colwise().sum() / static_cast<double>(X.rows())).sqrt();
    
    for (Eigen::Index i = 0; i < scaler_.std.size(); ++i) {
        if (scaler_.std(i) < 1e-9) scaler_.std(i) = 1.0;
    }
    scaler_.fit = true;
}

MatrixXd LogisticRegression::transform(const MatrixXd& X) const {
    if (!scaler_.fit) return X;
    
    ML_CHECK_FEATURES(X.cols(), n_features_, get_model_type());
    return (X.rowwise() - scaler_.mean.transpose()).array().rowwise() 
           / scaler_.std.transpose().array();
}

// Metodo fit principale
void LogisticRegression::fit(const MatrixXd& X, const Eigen::VectorXd& y) {
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    ML_CHECK_NOT_EMPTY(y, "y", get_model_type());
    ML_CHECK_DIMENSIONS(X.rows(), y.size(), X.cols(), 1, 
                       "X and y rows", get_model_type());
    
    // Verifica che y sia binario (0 o 1)
    double y_min = y.minCoeff();
    double y_max = y.maxCoeff();
    if (y_min < 0 || y_max > 1) {
        throw ml_exception::InvalidParameterException(
            "y", "must contain only 0 and 1 values", get_model_type());
    }
    
    // Cast sicuro da Eigen::Index a int
    Eigen::Index cols = X.cols();
    if (cols > std::numeric_limits<int>::max()) {
        throw ml_exception::DimensionMismatchException(
            "Number of features", 
            std::numeric_limits<int>::max(), 1,
            static_cast<int>(cols), 1,
            get_model_type()
        );
    }
    n_features_ = static_cast<int>(cols);
    
    fit_scaler(X);
    MatrixXd X_scaled = transform(X);
    
    gradient_descent(X_scaled, y);
}

void LogisticRegression::gradient_descent(const MatrixXd& X, const VectorXd& y) {
    MatrixXd X_int = MathUtils::add_intercept(X);
    double m = static_cast<double>(X.rows());
    theta_ = VectorXd::Zero(X_int.cols());
    cost_history_.clear();
    accuracy_history_.clear();
    cost_history_.reserve(max_iter_);
    accuracy_history_.reserve(max_iter_);

    for (n_iter_ = 0; n_iter_ < max_iter_; ++n_iter_) {
        // Forward pass
        VectorXd z = X_int * theta_;
        VectorXd h = MathUtils::sigmoid_vec(z);
        
        // Calcolo gradienti
        VectorXd error = h - y;
        VectorXd gradient = (X_int.transpose() * error) / m;
        
        // Regularizzazione L2
        if (lambda_ > 0) {
            VectorXd reg = (lambda_ / m) * theta_;
            reg(0) = 0; // Non regolarizzare l'intercetta
            gradient += reg;
        }
        
        // Aggiornamento parametri
        theta_ -= learning_rate_ * gradient;
        
        // Calcolo metriche
        cost_history_.push_back(compute_cost(X, y));
        accuracy_history_.push_back(compute_accuracy(X, y));
        
        // Early stopping
        if (n_iter_ > 10 && cost_history_.size() > 10) {
            double improvement = cost_history_[cost_history_.size() - 11] 
                               - cost_history_.back();
            if (std::abs(improvement) < tolerance_) {
                if (verbose_) {
                    std::cout << "Early stopping at iteration " << n_iter_ 
                              << ", cost: " << cost_history_.back() 
                              << ", accuracy: " << accuracy_history_.back() 
                              << std::endl;
                }
                break;
            }
        }
        
        // Log progresso
        if (verbose_ && n_iter_ % 100 == 0) {
            std::cout << "Iteration " << n_iter_ 
                      << ", Cost: " << cost_history_.back()
                      << ", Accuracy: " << accuracy_history_.back() 
                      << std::endl;
        }
    }
    
    if (n_iter_ >= max_iter_ && verbose_) {
        std::cout << "Reached maximum iterations: " << max_iter_ 
                  << ", Final cost: " << cost_history_.back()
                  << ", Final accuracy: " << accuracy_history_.back()
                  << std::endl;
    }
}

double LogisticRegression::compute_cost(const MatrixXd& X, const VectorXd& y) const {
    MatrixXd X_int = MathUtils::add_intercept(transform(X));
    double m = static_cast<double>(X.rows());
    
    VectorXd z = X_int * theta_;
    VectorXd h = MathUtils::sigmoid_vec(z);
    
    // Log loss con stabilità numerica
    VectorXd log_h = MathUtils::safe_log(h);
    VectorXd log_1_minus_h = MathUtils::safe_log(VectorXd::Ones(h.size()) - h);
    
    double J = -(y.dot(log_h) + (VectorXd::Ones(y.size()) - y).dot(log_1_minus_h)) / m;
    
    // Regularizzazione L2
    if (lambda_ > 0) {
        J += (lambda_ / (2.0 * m)) * theta_.tail(theta_.size() - 1).squaredNorm();
    }
    
    return J;
}

double LogisticRegression::compute_accuracy(const MatrixXd& X, const VectorXd& y) const {
    VectorXi y_pred_class = predict_class(X, 0.5);
    int correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (y_pred_class(i) == static_cast<int>(y(i))) {
            correct++;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(y.size());
}

// Metodi predict
VectorXd LogisticRegression::predict(const MatrixXd& X) const {
    ML_CHECK_FITTED(theta_.size() > 0, get_model_type());
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    ML_CHECK_FEATURES(X.cols(), n_features_, get_model_type());
    
    MatrixXd X_int = MathUtils::add_intercept(transform(X));
    VectorXd z = X_int * theta_;
    return MathUtils::sigmoid_vec(z);
}

VectorXi LogisticRegression::predict_class(const MatrixXd& X, double threshold) const {
    ML_CHECK_PARAM(threshold >= 0 && threshold <= 1, "threshold", 
                  "must be between 0 and 1", get_model_type());
    
    VectorXd probabilities = predict(X);
    VectorXi classes(probabilities.size());
    
    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        classes(i) = (probabilities(i) >= threshold) ? 1 : 0;
    }
    
    return classes;
}

// Metodi score
double LogisticRegression::score(const MatrixXd& X, const VectorXd& y) const {
    return compute_accuracy(X, y);
}

Vector3d LogisticRegression::precision_recall_f1(const MatrixXd& X, 
                                               const VectorXd& y, 
                                               double threshold) const {
    Eigen::MatrixXd cm = confusion_matrix(X, y, threshold);
    
    double tp = cm(1, 1);
    double fp = cm(0, 1);
    double fn = cm(1, 0);
    
    double precision = (tp + fp > 0) ? tp / (tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? tp / (tp + fn) : 0.0;
    double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
    
    return Vector3d(precision, recall, f1);
}

MatrixXd LogisticRegression::confusion_matrix(const MatrixXd& X, 
                                            const VectorXd& y, 
                                            double threshold) const {
    VectorXi y_pred = predict_class(X, threshold);
    MatrixXd cm = MatrixXd::Zero(2, 2);
    
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        int actual = static_cast<int>(y(i));
        int predicted = y_pred(i);
        cm(actual, predicted) += 1.0;
    }
    
    return cm;
}

// Serializzazione
void LogisticRegression::Scaler::serialize(std::ostream& out) const {
    using namespace utils;
    
    out.write(reinterpret_cast<const char*>(&fit), sizeof(bool));
    if (fit) {
        eigen_utils::serialize_eigen_vector(mean, out);
        eigen_utils::serialize_eigen_vector(std, out);
    }
}

void LogisticRegression::Scaler::deserialize(std::istream& in) {
    using namespace utils;
    
    in.read(reinterpret_cast<char*>(&fit), sizeof(bool));
    if (fit) {
        eigen_utils::deserialize_eigen_vector(mean, in);
        eigen_utils::deserialize_eigen_vector(std, in);
    }
}

void LogisticRegression::serialize_binary(std::ostream& out) const {
    using namespace utils;
    
    // Serializza parametri
    out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&max_iter_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&lambda_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&tolerance_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&verbose_), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&n_iter_), sizeof(int));
    
    // Serializza theta e scaler
    eigen_utils::serialize_eigen_vector(theta_, out);
    scaler_.serialize(out);
    
    // Serializza history
    size_t cost_size = cost_history_.size();
    out.write(reinterpret_cast<const char*>(&cost_size), sizeof(size_t));
    if (cost_size > 0) {
        out.write(reinterpret_cast<const char*>(cost_history_.data()), 
                 cost_size * sizeof(double));
    }
    
    size_t acc_size = accuracy_history_.size();
    out.write(reinterpret_cast<const char*>(&acc_size), sizeof(size_t));
    if (acc_size > 0) {
        out.write(reinterpret_cast<const char*>(accuracy_history_.data()), 
                 acc_size * sizeof(double));
    }
}

void LogisticRegression::deserialize_binary(std::istream& in) {
    using namespace utils;
    
    // Deserializza parametri
    in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(double));
    in.read(reinterpret_cast<char*>(&max_iter_), sizeof(int));
    in.read(reinterpret_cast<char*>(&lambda_), sizeof(double));
    in.read(reinterpret_cast<char*>(&tolerance_), sizeof(double));
    in.read(reinterpret_cast<char*>(&verbose_), sizeof(bool));
    in.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    in.read(reinterpret_cast<char*>(&n_iter_), sizeof(int));
    
    // Deserializza theta e scaler
    eigen_utils::deserialize_eigen_vector(theta_, in);
    scaler_.deserialize(in);
    
    // Deserializza history
    size_t cost_size;
    in.read(reinterpret_cast<char*>(&cost_size), sizeof(size_t));
    cost_history_.resize(cost_size);
    if (cost_size > 0) {
        in.read(reinterpret_cast<char*>(cost_history_.data()), 
               cost_size * sizeof(double));
    }
    
    size_t acc_size;
    in.read(reinterpret_cast<char*>(&acc_size), sizeof(size_t));
    accuracy_history_.resize(acc_size);
    if (acc_size > 0) {
        in.read(reinterpret_cast<char*>(accuracy_history_.data()), 
               acc_size * sizeof(double));
    }
}

std::string LogisticRegression::to_string() const {
    std::ostringstream oss;
    oss << "LogisticRegression [" 
        << "Features: " << n_features_
        << ", Iterations: " << n_iter_
        << ", λ: " << std::fixed << std::setprecision(4) << lambda_
        << ", LR: " << learning_rate_
        << ", Tolerance: " << tolerance_
        << "]";
    return oss.str();
}