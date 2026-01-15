#include "regression/logistic_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>

using namespace Eigen;
using namespace regression;

LogisticRegression::LogisticRegression()
    : learning_rate_(0.1), max_iter_(1000), lambda_(0.0), tolerance_(1e-4), 
      verbose_(false), n_features_(0), n_iter_(0) {}

LogisticRegression::LogisticRegression(double lr, int iter, double lb, double tol, bool verb)
    : learning_rate_(lr), max_iter_(iter), lambda_(lb), tolerance_(tol), 
      verbose_(verb), n_features_(0), n_iter_(0) {
    
    if (learning_rate_ <= 0) throw std::invalid_argument("learning_rate must be > 0");
    if (max_iter_ <= 0) throw std::invalid_argument("max_iter must be > 0");
    if (lambda_ < 0) throw std::invalid_argument("lambda must be >= 0");
    if (tolerance_ <= 0) throw std::invalid_argument("tolerance must be > 0");
}

void LogisticRegression::fit_scaler(const MatrixXd& X) {
    scaler_.mean = X.colwise().mean();
    scaler_.std = ((X.rowwise() - scaler_.mean.transpose()).array().square().colwise().sum() / static_cast<double>(X.rows())).sqrt();
    
    // Gestione feature costanti
    for (Eigen::Index i = 0; i < scaler_.std.size(); ++i) {
        if (scaler_.std(i) < 1e-9) scaler_.std(i) = 1.0;
    }
    
    scaler_.fit = true;
}

MatrixXd LogisticRegression::transform(const MatrixXd& X) const {
    if (!scaler_.fit) return X;
    return (X.rowwise() - scaler_.mean.transpose()).array().rowwise() / scaler_.std.transpose().array();
}

void LogisticRegression::fit(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }
    
    // Controlla che y sia binario (0/1)
    double y_min = y.minCoeff();
    double y_max = y.maxCoeff();
    if (y_min < 0 || y_max > 1) {
        throw std::invalid_argument("y must contain only 0 and 1 values for logistic regression");
    }
    
    n_features_ = static_cast<int>(X.cols());
    fit_scaler(X);
    MatrixXd X_int = MathUtils::add_intercept(transform(X));  // Usa MathUtils
    Eigen::Index m = X.rows();
    theta_ = VectorXd::Zero(X_int.cols());
    cost_history_.clear();

    for (n_iter_ = 0; n_iter_ < max_iter_; ++n_iter_) {
        VectorXd h = MathUtils::sigmoid_vec(X_int * theta_);  // Usa MathUtils
        VectorXd gradient = (X_int.transpose() * (h - y)) / static_cast<double>(m);
        
        if (lambda_ > 0) {
            VectorXd reg = (lambda_ / static_cast<double>(m)) * theta_;
            reg(0) = 0;
            gradient += reg;
        }

        VectorXd prev_theta = theta_;
        theta_ -= learning_rate_ * gradient;

        double cost = compute_cost(X, y);
        cost_history_.push_back(cost);
        
        if (verbose_ && n_iter_ % 100 == 0) {
            std::cout << "Iteration " << n_iter_ << ", Cost: " << cost << std::endl;
        }

        // Early stopping
        if ((prev_theta - theta_).norm() < tolerance_) {
            if (verbose_) {
                std::cout << "Converged at iteration " << n_iter_ << std::endl;
            }
            break;
        }
    }
}

double LogisticRegression::compute_cost(const MatrixXd& X, const VectorXd& y) const {
    MatrixXd X_int = MathUtils::add_intercept(transform(X));  // Usa MathUtils
    VectorXd h = MathUtils::sigmoid_vec(X_int * theta_);  // Usa MathUtils
    double m = static_cast<double>(X.rows());
    
    // Usa log safe per evitare log(0)
    VectorXd log_h = MathUtils::safe_log(h);
    VectorXd log_1_h = MathUtils::safe_log(1.0 - h.array());
    
    double J = -(y.dot(log_h) + (1 - y.array()).matrix().dot(log_1_h)) / m;
    
    if (lambda_ > 0) {
        J += (lambda_ / (2.0 * m)) * theta_.tail(n_features_).squaredNorm();
    }
    
    return J;
}

VectorXd LogisticRegression::predict(const MatrixXd& X) const {
    if (theta_.size() == 0) {
        throw std::runtime_error("Model not fitted yet");
    }
    MatrixXd X_int = MathUtils::add_intercept(transform(X));  // Usa MathUtils
    return MathUtils::sigmoid_vec(X_int * theta_);  // Usa MathUtils
}

VectorXi LogisticRegression::predict_class(const Eigen::MatrixXd& X, double threshold) const {
    VectorXd prob = predict(X);
    return (prob.array() >= threshold).cast<int>();
}

double LogisticRegression::score(const MatrixXd& X, const VectorXd& y) const {
    VectorXi y_pred = predict_class(X);
    return (y_pred.cast<double>().array() == y.array()).cast<double>().mean();
}

// Implementazione metodi mancanti
MatrixXd LogisticRegression::confusion_matrix(const MatrixXd& X, const VectorXd& y, double threshold) const {
    VectorXi y_pred = predict_class(X, threshold);
    MatrixXd cm = MatrixXd::Zero(2, 2);
    
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        int actual = static_cast<int>(y(i));
        int predicted = y_pred(static_cast<Eigen::Index>(i));
        cm(actual, predicted) += 1.0;
    }
    
    return cm;
}

Vector3d LogisticRegression::precision_recall_f1(const MatrixXd& X, const VectorXd& y, double threshold) const {
    MatrixXd cm = confusion_matrix(X, y, threshold);
    
    double tp = cm(1, 1);
    double fp = cm(0, 1);
    double fn = cm(1, 0);
    
    double precision = tp / (tp + fp + 1e-9);
    double recall = tp / (tp + fn + 1e-9);
    double f1 = 2.0 * precision * recall / (precision + recall + 1e-9);
    
    return Vector3d(precision, recall, f1);
}

void LogisticRegression::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
    Eigen::Index ts = theta_.size();
    int ts_int = static_cast<int>(ts);
    file.write(reinterpret_cast<const char*>(&ts_int), sizeof(int));
    std::size_t data_size = static_cast<std::size_t>(ts) * sizeof(double);
    std::streamsize stream_size = static_cast<std::streamsize>(data_size);
    file.write(reinterpret_cast<const char*>(theta_.data()), stream_size);
    
    // Salva lo scaler
    bool has_scaler = scaler_.fit;
    file.write(reinterpret_cast<const char*>(&has_scaler), sizeof(bool));
    if (has_scaler) {
        Eigen::Index mean_size = scaler_.mean.size();
        int mean_size_int = static_cast<int>(mean_size);
        file.write(reinterpret_cast<const char*>(&mean_size_int), sizeof(int));
        std::size_t scaler_data_size = static_cast<std::size_t>(mean_size) * sizeof(double);
        std::streamsize scaler_stream_size = static_cast<std::streamsize>(scaler_data_size);
        file.write(reinterpret_cast<const char*>(scaler_.mean.data()), scaler_stream_size);
        file.write(reinterpret_cast<const char*>(scaler_.std.data()), scaler_stream_size);
    }
    
    file.close();
}

void LogisticRegression::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    int ts_int;
    file.read(reinterpret_cast<char*>(&ts_int), sizeof(int));
    Eigen::Index ts = static_cast<Eigen::Index>(ts_int);
    theta_.resize(ts);
    std::size_t data_size = static_cast<std::size_t>(ts) * sizeof(double);
    std::streamsize stream_size = static_cast<std::streamsize>(data_size);
    file.read(reinterpret_cast<char*>(theta_.data()), stream_size);
    
    // Carica lo scaler
    bool has_scaler;
    file.read(reinterpret_cast<char*>(&has_scaler), sizeof(bool));
    if (has_scaler) {
        int mean_size_int;
        file.read(reinterpret_cast<char*>(&mean_size_int), sizeof(int));
        Eigen::Index mean_size = static_cast<Eigen::Index>(mean_size_int);
        scaler_.mean.resize(mean_size);
        scaler_.std.resize(mean_size);
        scaler_.fit = true;
        std::size_t scaler_data_size = static_cast<std::size_t>(mean_size) * sizeof(double);
        std::streamsize scaler_stream_size = static_cast<std::streamsize>(scaler_data_size);
        file.read(reinterpret_cast<char*>(scaler_.mean.data()), scaler_stream_size);
        file.read(reinterpret_cast<char*>(scaler_.std.data()), scaler_stream_size);
    }
    
    file.close();
}

std::string LogisticRegression::to_string() const {
    std::ostringstream ss;
    ss << "LogisticRegression [Features: " << n_features_ 
       << ", Iterations: " << n_iter_ 
       << ", Cost: " << (cost_history_.empty() ? 0.0 : cost_history_.back()) 
       << "]";
    return ss.str();
}