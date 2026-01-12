#include "logistic_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;

LogisticRegression::LogisticRegression()
    : learning_rate_(0.1), max_iter_(1000), lambda_(0.0),
      tolerance_(1e-4), verbose_(false),
      n_features_(0), n_iter_(0), n_classes_(2) {}

LogisticRegression::LogisticRegression(double learning_rate, int max_iter,
                                       double lambda, double tolerance, bool verbose)
    : learning_rate_(learning_rate), max_iter_(max_iter), lambda_(lambda),
      tolerance_(tolerance), verbose_(verbose),
      n_features_(0), n_iter_(0), n_classes_(2) {}

MatrixXd LogisticRegression::add_intercept(const MatrixXd& X) {
    int m = X.rows();
    MatrixXd X_with_intercept(m, X.cols() + 1);
    X_with_intercept.col(0).setOnes();
    X_with_intercept.rightCols(X.cols()) = X;
    return X_with_intercept;
}

double LogisticRegression::sigmoid(double z) {
    // Implementazione numericamente stabile
    if (z >= 0) {
        return 1.0 / (1.0 + std::exp(-z));
    } else {
        double exp_z = std::exp(z);
        return exp_z / (1.0 + exp_z);
    }
}

VectorXd LogisticRegression::sigmoid(const VectorXd& z) {
    VectorXd result(z.size());
    for (int i = 0; i < z.size(); ++i) {
        result(i) = sigmoid(z(i));
    }
    return result;
}

void LogisticRegression::fit_scaler(const MatrixXd& X) {
    if (X.cols() == 0) return;
    
    scaler_.mean = X.colwise().mean();
    scaler_.std = ((X.rowwise() - scaler_.mean.transpose()).array().square()
                   .colwise().sum() / X.rows()).sqrt();
    
    // Evita divisione per zero
    for (int i = 0; i < scaler_.std.size(); ++i) {
        if (scaler_.std(i) < 1e-8) scaler_.std(i) = 1.0;
    }
    
    scaler_.fit = true;
}

MatrixXd LogisticRegression::transform(const MatrixXd& X) const {
    if (!scaler_.fit || X.cols() != scaler_.mean.size()) {
        return X;
    }
    
    return (X.rowwise() - scaler_.mean.transpose()).array().rowwise() 
           / scaler_.std.transpose().array();
}

MatrixXd LogisticRegression::inverse_transform(const MatrixXd& X_scaled) const {
    if (!scaler_.fit || X_scaled.cols() != scaler_.mean.size()) {
        return X_scaled;
    }
    
    return (X_scaled.array().rowwise() * scaler_.std.transpose().array()).rowwise()
           + scaler_.mean.transpose().array();
}

double LogisticRegression::compute_cost(const MatrixXd& X, const VectorXd& y) const {
    if (X.rows() == 0) return 0.0;
    
    VectorXd h = sigmoid(X * theta_);
    double m = X.rows();
    
    // Log loss con regolarizzazione L2
    double cost = 0.0;
    for (int i = 0; i < m; ++i) {
        double h_i = h(i);
        // Evita log(0)
        h_i = std::max(1e-15, std::min(1.0 - 1e-15, h_i));
        cost += y(i) * std::log(h_i) + (1 - y(i)) * std::log(1 - h_i);
    }
    
    cost = -cost / m;
    
    // Regolarizzazione L2 (esclude intercept)
    if (lambda_ > 0 && theta_.size() > 1) {
        VectorXd theta_no_intercept = theta_.tail(theta_.size() - 1);
        cost += (lambda_ / (2.0 * m)) * theta_no_intercept.squaredNorm();
    }
    
    return cost;
}

VectorXd LogisticRegression::compute_gradient(const MatrixXd& X, const VectorXd& y) const {
    int m = X.rows();
    VectorXd h = sigmoid(X * theta_);
    VectorXd errors = h - y;
    
    VectorXd gradient = (X.transpose() * errors) / m;
    
    // Regolarizzazione L2 (esclude intercept)
    if (lambda_ > 0) {
        VectorXd reg_gradient = theta_;
        reg_gradient(0) = 0.0; // Non regolarizzare intercetta
        gradient += (lambda_ / m) * reg_gradient;
    }
    
    return gradient;
}

void LogisticRegression::fit(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X e y devono avere lo stesso numero di righe");
    }
    
    if (X.rows() == 0) {
        throw std::invalid_argument("X non può essere vuoto");
    }
    
    // Verifica che y sia binario (0 o 1)
    double y_min = y.minCoeff();
    double y_max = y.maxCoeff();
    if (y_min < 0 || y_max > 1) {
        throw std::invalid_argument("y deve contenere solo 0 o 1 per classificazione binaria");
    }
    
    // Scale features
    MatrixXd X_scaled = X;
    if (X.cols() > 0) {
        fit_scaler(X);
        X_scaled = transform(X);
    }
    
    // Aggiungi intercetta
    MatrixXd X_with_intercept = add_intercept(X_scaled);
    
    n_features_ = X.cols();
    
    // Inizializza theta
    theta_ = VectorXd::Zero(X_with_intercept.cols());
    
    // Gradient Descent
    cost_history_.clear();
    accuracy_history_.clear();
    
    double prev_cost = std::numeric_limits<double>::max();
    
    for (n_iter_ = 0; n_iter_ < max_iter_; ++n_iter_) {
        // Calcola gradienti e aggiorna
        VectorXd gradient = compute_gradient(X_with_intercept, y);
        theta_ -= learning_rate_ * gradient;
        
        // Calcola costo corrente
        double current_cost = compute_cost(X_with_intercept, y);
        cost_history_.push_back(current_cost);
        
        // Calcola accuracy corrente
        VectorXi y_pred = predict(X, 0.5);
        double accuracy = (y_pred.cast<double>().array() == y.array())
                         .cast<double>().sum() / y.size();
        accuracy_history_.push_back(accuracy);
        
        // Check convergenza
        if (n_iter_ > 0) {
            double cost_change = std::abs(current_cost - prev_cost);
            if (cost_change < tolerance_) {
                if (verbose_) {
                    std::cout << "Convergenza raggiunta dopo " << n_iter_ 
                              << " iterazioni" << std::endl;
                }
                break;
            }
        }
        
        prev_cost = current_cost;
        
        // Output progresso
        if (verbose_ && n_iter_ % 100 == 0) {
            std::cout << "Iterazione " << n_iter_ 
                      << ", Costo: " << current_cost
                      << ", Accuracy: " << accuracy << std::endl;
        }
    }
    
    // Denormalizza theta se abbiamo scalato
    if (scaler_.fit) {
        double intercept_scaled = theta_(0);
        for (int i = 0; i < n_features_; ++i) {
            intercept_scaled -= theta_(i + 1) * scaler_.mean(i) / scaler_.std(i);
            theta_(i + 1) /= scaler_.std(i);
        }
        theta_(0) = intercept_scaled;
    }
    
    if (verbose_) {
        std::cout << "Training completato in " << n_iter_ << " iterazioni" << std::endl;
        std::cout << "Accuracy finale: " << accuracy_history_.back() << std::endl;
    }
}

double LogisticRegression::predict_proba(const VectorXd& x) const {
    if (x.size() != n_features_) {
        throw std::invalid_argument("Dimensione input errata");
    }
    
    double z = theta_(0); // intercept
    for (int i = 0; i < n_features_; ++i) {
        z += theta_(i + 1) * x(i);
    }
    
    return sigmoid(z);
}

int LogisticRegression::predict(const VectorXd& x, double threshold) const {
    double proba = predict_proba(x);
    return (proba >= threshold) ? 1 : 0;
}

VectorXd LogisticRegression::predict_proba(const MatrixXd& X) const {
    if (X.cols() != n_features_) {
        throw std::invalid_argument("Dimensione input errata");
    }
    
    VectorXd z = (X * theta_.tail(n_features_)).array() + theta_(0);
    
    return sigmoid(z);
}

VectorXi LogisticRegression::predict(const MatrixXd& X, double threshold) const {
    VectorXd probabilities = predict_proba(X);
    VectorXi predictions(X.rows());
    
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = (probabilities(i) >= threshold) ? 1 : 0;
    }
    
    return predictions;
}

// Aggiungi questi metodi alla fine di logistic_regression.cpp

VectorXd LogisticRegression::decision_function(const MatrixXd& X) const {
    if (X.cols() != n_features_) {
        throw std::invalid_argument("Wrong feature dimension");
    }
    
    VectorXd scores(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        scores(i) = theta_(0); // intercept
        for (int j = 0; j < n_features_; ++j) {
            scores(i) += theta_(j + 1) * X(i, j);
        }
    }
    
    return scores;
}

Vector3d LogisticRegression::precision_recall_f1(const MatrixXd& X,
                                                const VectorXd& y,
                                                double threshold) const {
    VectorXi y_pred = predict(X, threshold);
    
    int true_positive = 0, false_positive = 0, false_negative = 0;
    
    for (int i = 0; i < y.size(); ++i) {
        int pred = y_pred(i);
        int actual = static_cast<int>(y(i));
        
        if (pred == 1 && actual == 1) true_positive++;
        else if (pred == 1 && actual == 0) false_positive++;
        else if (pred == 0 && actual == 1) false_negative++;
    }
    
    double precision = 0.0, recall = 0.0, f1 = 0.0;
    
    if (true_positive + false_positive > 0) {
        precision = static_cast<double>(true_positive) / 
                   (true_positive + false_positive);
    }
    
    if (true_positive + false_negative > 0) {
        recall = static_cast<double>(true_positive) / 
                (true_positive + false_negative);
    }
    
    if (precision + recall > 0) {
        f1 = 2.0 * precision * recall / (precision + recall);
    }
    
    return Vector3d(precision, recall, f1);
}

MatrixXd LogisticRegression::confusion_matrix(const MatrixXd& X,
                                             const VectorXd& y,
                                             double threshold) const {
    VectorXi y_pred = predict(X, threshold);
    
    MatrixXd cm = MatrixXd::Zero(2, 2);
    
    for (int i = 0; i < y.size(); ++i) {
        int pred = y_pred(i);
        int actual = static_cast<int>(y(i));
        cm(actual, pred) += 1;
    }
    
    return cm;
}

double LogisticRegression::score(const MatrixXd& X, const VectorXd& y,
                                double threshold) const {
    VectorXi y_pred = predict(X, threshold);
    int correct = 0;
    
    for (int i = 0; i < y.size(); ++i) {
        if (y_pred(i) == static_cast<int>(y(i))) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y.size();
}

VectorXd LogisticRegression::cross_val_score(const MatrixXd& X,
                                            const VectorXd& y,
                                            int cv) {
    if (cv < 2) {
        throw std::invalid_argument("cv must be at least 2");
    }
    
    int m = X.rows();
    int fold_size = m / cv;
    
    VectorXd scores(cv);
    
    for (int i = 0; i < cv; ++i) {
        // Split train/test
        int test_start = i * fold_size;
        int test_end = (i == cv - 1) ? m : (i + 1) * fold_size;
        
        // Create indices
        std::vector<int> train_indices, test_indices;
        for (int j = 0; j < m; ++j) {
            if (j >= test_start && j < test_end) {
                test_indices.push_back(j);
            } else {
                train_indices.push_back(j);
            }
        }
        
        // Create matrices
        MatrixXd X_train(train_indices.size(), X.cols());
        VectorXd y_train(train_indices.size());
        MatrixXd X_test(test_indices.size(), X.cols());
        VectorXd y_test(test_indices.size());
        
        for (size_t j = 0; j < train_indices.size(); ++j) {
            X_train.row(j) = X.row(train_indices[j]);
            y_train(j) = y(train_indices[j]);
        }
        
        for (size_t j = 0; j < test_indices.size(); ++j) {
            X_test.row(j) = X.row(test_indices[j]);
            y_test(j) = y(test_indices[j]);
        }
        
        // Train model
        LogisticRegression model;
        model.fit(X_train, y_train);
        
        // Calculate score
        scores(i) = model.score(X_test, y_test);
    }
    
    return scores;
}

void LogisticRegression::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Save theta size and data
    int theta_size = theta_.size();
    file.write(reinterpret_cast<const char*>(&theta_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(theta_.data()), 
              theta_size * sizeof(double));
    
    // Save metadata
    file.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&max_iter_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&lambda_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&tolerance_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&verbose_), sizeof(bool));
    file.write(reinterpret_cast<const char*>(&n_iter_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&n_classes_), sizeof(int));
    
    // Save scaler
    int has_scaler = scaler_.fit ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&has_scaler), sizeof(int));
    
    if (scaler_.fit) {
        int scaler_size = scaler_.mean.size();
        file.write(reinterpret_cast<const char*>(&scaler_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(scaler_.mean.data()), 
                  scaler_size * sizeof(double));
        file.write(reinterpret_cast<const char*>(scaler_.std.data()), 
                  scaler_size * sizeof(double));
    }
    
    // Save histories
    int cost_history_size = cost_history_.size();
    file.write(reinterpret_cast<const char*>(&cost_history_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(cost_history_.data()), 
              cost_history_size * sizeof(double));
    
    int accuracy_history_size = accuracy_history_.size();
    file.write(reinterpret_cast<const char*>(&accuracy_history_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(accuracy_history_.data()), 
              accuracy_history_size * sizeof(double));
    
    file.close();
}

void LogisticRegression::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Load theta
    int theta_size;
    file.read(reinterpret_cast<char*>(&theta_size), sizeof(int));
    theta_.resize(theta_size);
    file.read(reinterpret_cast<char*>(theta_.data()), theta_size * sizeof(double));
    
    // Load metadata
    file.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(double));
    file.read(reinterpret_cast<char*>(&max_iter_), sizeof(int));
    file.read(reinterpret_cast<char*>(&lambda_), sizeof(double));
    file.read(reinterpret_cast<char*>(&tolerance_), sizeof(double));
    file.read(reinterpret_cast<char*>(&verbose_), sizeof(bool));
    file.read(reinterpret_cast<char*>(&n_iter_), sizeof(int));
    file.read(reinterpret_cast<char*>(&n_classes_), sizeof(int));
    
    // Load scaler
    int has_scaler;
    file.read(reinterpret_cast<char*>(&has_scaler), sizeof(int));
    scaler_.fit = (has_scaler == 1);
    
    if (scaler_.fit) {
        int scaler_size;
        file.read(reinterpret_cast<char*>(&scaler_size), sizeof(int));
        scaler_.mean.resize(scaler_size);
        scaler_.std.resize(scaler_size);
        file.read(reinterpret_cast<char*>(scaler_.mean.data()), 
                 scaler_size * sizeof(double));
        file.read(reinterpret_cast<char*>(scaler_.std.data()), 
                 scaler_size * sizeof(double));
    }
    
    // Load histories
    int cost_history_size;
    file.read(reinterpret_cast<char*>(&cost_history_size), sizeof(int));
    cost_history_.resize(cost_history_size);
    file.read(reinterpret_cast<char*>(cost_history_.data()), 
             cost_history_size * sizeof(double));
    
    int accuracy_history_size;
    file.read(reinterpret_cast<char*>(&accuracy_history_size), sizeof(int));
    accuracy_history_.resize(accuracy_history_size);
    file.read(reinterpret_cast<char*>(accuracy_history_.data()), 
             accuracy_history_size * sizeof(double));
    
    file.close();
}

std::string LogisticRegression::to_string() const {
    std::ostringstream ss;
    ss << "LogisticRegression:\n";
    ss << "  Features: " << n_features_ << "\n";
    ss << "  Intercept: " << intercept() << "\n";
    ss << "  Coefficients: [";
    for (int i = 0; i < std::min(3, n_features_); ++i) {
        ss << theta_(i + 1);
        if (i < std::min(3, n_features_) - 1) ss << ", ";
    }
    if (n_features_ > 3) ss << ", ...";
    ss << "]\n";
    ss << "  Iterations: " << n_iter_ << "\n";
    ss << "  Final accuracy: " << (accuracy_history_.empty() ? 0.0 : accuracy_history_.back());
    return ss.str();
}