#include "linear_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;

LinearRegression::LinearRegression()
    : learning_rate_(0.01), max_iter_(1000), lambda_(0.0), 
      solver_(GRADIENT_DESCENT), n_features_(0), n_iter_(0) {}

LinearRegression::LinearRegression(double learning_rate, int max_iter, 
                                   double lambda, Solver solver)
    : learning_rate_(learning_rate), max_iter_(max_iter), lambda_(lambda),
      solver_(solver), n_features_(0), n_iter_(0) {}

MatrixXd LinearRegression::add_intercept(const MatrixXd& X) {
    int m = X.rows();
    MatrixXd X_with_intercept(m, X.cols() + 1);
    X_with_intercept.col(0).setOnes();
    X_with_intercept.rightCols(X.cols()) = X;
    return X_with_intercept;
}

void LinearRegression::fit_scaler(const MatrixXd& X) {
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

MatrixXd LinearRegression::transform(const MatrixXd& X) const {
    if (!scaler_.fit || X.cols() != scaler_.mean.size()) {
        return X;
    }
    
    return (X.rowwise() - scaler_.mean.transpose()).array().rowwise() 
           / scaler_.std.transpose().array();
}

MatrixXd LinearRegression::inverse_transform(const MatrixXd& X_scaled) const {
    if (!scaler_.fit || X_scaled.cols() != scaler_.mean.size()) {
        return X_scaled;
    }
    
    return (X_scaled.array().rowwise() * scaler_.std.transpose().array()).rowwise()
           + scaler_.mean.transpose().array();
}

double LinearRegression::compute_cost(const MatrixXd& X, const VectorXd& y) const {
    if (X.rows() == 0) return 0.0;
    
    VectorXd predictions = X * theta_;
    VectorXd errors = predictions - y;
    double m = X.rows();
    
    double cost = errors.squaredNorm() / (2.0 * m);
    
    // Regolarizzazione L2 (esclude intercept)
    if (lambda_ > 0 && theta_.size() > 1) {
        VectorXd theta_no_intercept = theta_.tail(theta_.size() - 1);
        cost += (lambda_ / (2.0 * m)) * theta_no_intercept.squaredNorm();
    }
    
    return cost;
}

VectorXd LinearRegression::compute_gradient(const MatrixXd& X, const VectorXd& y) const {
    int m = X.rows();
    VectorXd predictions = X * theta_;
    VectorXd errors = predictions - y;
    
    VectorXd gradient = (X.transpose() * errors) / m;
    
    // Regolarizzazione L2 (esclude intercept)
    if (lambda_ > 0) {
        VectorXd reg_gradient = theta_;
        reg_gradient(0) = 0.0; // Non regolarizzare intercetta
        gradient += (lambda_ / m) * reg_gradient;
    }
    
    return gradient;
}

void LinearRegression::gradient_descent(const MatrixXd& X, const VectorXd& y) {
    int m = X.rows();
    n_features_ = X.cols() - 1; // Escludi intercetta
    
    // Inizializza theta
    theta_ = VectorXd::Zero(X.cols());
    
    cost_history_.clear();
    cost_history_.reserve(max_iter_ / 10);
    
    // Gradient Descent
    for (n_iter_ = 0; n_iter_ < max_iter_; ++n_iter_) {
        VectorXd gradient = compute_gradient(X, y);
        theta_ -= learning_rate_ * gradient;
        
        // Salva costo ogni 10 iterazioni
        if (n_iter_ % 10 == 0) {
            cost_history_.push_back(compute_cost(X, y));
        }
        
        // Convergenza
        if (n_iter_ > 0 && n_iter_ % 100 == 0) {
            double cost_change = std::abs(cost_history_.back() - 
                                         cost_history_[cost_history_.size() - 2]);
            if (cost_change < 1e-6) {
                break;
            }
        }
    }
}

void LinearRegression::normal_equation(const MatrixXd& X, const VectorXd& y) {
    n_features_ = X.cols() - 1;
    
    // Equazione normale: θ = (XᵀX + λI)⁻¹ Xᵀy
    MatrixXd XTX = X.transpose() * X;
    
    // Aggiungi regolarizzazione (escludi intercept)
    if (lambda_ > 0) {
        MatrixXd I = MatrixXd::Identity(X.cols(), X.cols());
        I(0, 0) = 0.0; // Non regolarizzare intercetta
        XTX += lambda_ * I;
    }
    
    // Risolvi sistema lineare
    theta_ = XTX.ldlt().solve(X.transpose() * y);
    
    // Calcola costo iniziale
    cost_history_.push_back(compute_cost(X, y));
    n_iter_ = 1;
}

void LinearRegression::svd_solve(const MatrixXd& X, const VectorXd& y) {
    n_features_ = X.cols() - 1;
    
    // SVD: θ = VΣ⁻¹Uᵀy
    JacobiSVD<MatrixXd> svd(X, ComputeThinU | ComputeThinV);
    
    // Regolarizzazione per valori singolari piccoli
    VectorXd singular_values = svd.singularValues();
    double epsilon = 1e-10;
    
    MatrixXd sigma_inv = MatrixXd::Zero(svd.matrixV().cols(), svd.matrixU().cols());
    for (int i = 0; i < singular_values.size(); ++i) {
        double s = singular_values(i);
        if (s > epsilon) {
            sigma_inv(i, i) = 1.0 / (s + lambda_); // Regolarizzazione Tikhonov
        }
    }
    
    theta_ = svd.matrixV() * sigma_inv * svd.matrixU().transpose() * y;
    
    // Calcola costo
    cost_history_.push_back(compute_cost(X, y));
    n_iter_ = 1;
}

void LinearRegression::fit(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X e y devono avere lo stesso numero di righe");
    }
    
    if (X.rows() == 0) {
        throw std::invalid_argument("X non può essere vuoto");
    }
    
    // Scale features (escludi intercetta dopo)
    MatrixXd X_scaled = X;
    if (X.cols() > 0) {
        fit_scaler(X);
        X_scaled = transform(X);
    }
    
    // Aggiungi intercetta
    MatrixXd X_with_intercept = add_intercept(X_scaled);
    
    // Seleziona solver
    switch (solver_) {
        case GRADIENT_DESCENT:
            gradient_descent(X_with_intercept, y);
            break;
        case NORMAL_EQUATION:
            normal_equation(X_with_intercept, y);
            break;
        case SVD:
            svd_solve(X_with_intercept, y);
            break;
    }
    
    // Denormalizza theta se abbiamo scalato
    if (scaler_.fit) {
        // theta_[0] è l'intercetta per dati scalati
        // Converti a dati originali: y = θ₀ + Σθᵢ*(xᵢ-μᵢ)/σᵢ
        //                         = (θ₀ - Σθᵢ*μᵢ/σᵢ) + Σ(θᵢ/σᵢ)*xᵢ
        
        double intercept_scaled = theta_(0);
        for (int i = 0; i < n_features_; ++i) {
            intercept_scaled -= theta_(i + 1) * scaler_.mean(i) / scaler_.std(i);
            theta_(i + 1) /= scaler_.std(i);
        }
        theta_(0) = intercept_scaled;
    }
}

double LinearRegression::predict(const VectorXd& x) const {
    if (x.size() != n_features_) {
        throw std::invalid_argument("Dimensione input errata");
    }
    
    double prediction = theta_(0); // intercept
    for (int i = 0; i < n_features_; ++i) {
        prediction += theta_(i + 1) * x(i);
    }
    
    return prediction;
}

VectorXd LinearRegression::predict(const MatrixXd& X) const {
    if (X.cols() != n_features_) {
        throw std::invalid_argument("Dimensione input errata");
    }
    
    VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = theta_(0);
        for (int j = 0; j < n_features_; ++j) {
            predictions(i) += theta_(j + 1) * X(i, j);
        }
    }
    
    return predictions;
}

double LinearRegression::score(const MatrixXd& X, const VectorXd& y) const {
    return r2_score(X, y);
}

double LinearRegression::r2_score(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    double ss_res = (y - y_pred).squaredNorm();
    double ss_tot = (y.array() - y.mean()).square().sum();
    
    if (ss_tot < 1e-10) return 1.0;
    return 1.0 - (ss_res / ss_tot);
}

double LinearRegression::mse(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    return (y - y_pred).squaredNorm() / y.size();
}

double LinearRegression::mae(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    return (y - y_pred).array().abs().sum() / y.size();
}

VectorXd LinearRegression::cross_val_score(const MatrixXd& X, const VectorXd& y,
                                          int cv, Solver solver) {
    if (cv < 2) {
        throw std::invalid_argument("cv deve essere almeno 2");
    }
    
    int m = X.rows();
    int fold_size = m / cv;
    
    VectorXd scores(cv);
    
    for (int i = 0; i < cv; ++i) {
        // Split train/test
        int test_start = i * fold_size;
        int test_end = (i == cv - 1) ? m : (i + 1) * fold_size;
        
        // Indici per slicing
        std::vector<int> train_indices, test_indices;
        for (int j = 0; j < m; ++j) {
            if (j >= test_start && j < test_end) {
                test_indices.push_back(j);
            } else {
                train_indices.push_back(j);
            }
        }
        
        // Crea matrici train/test
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
        
        // Addestra modello
        LinearRegression model(0.01, 1000, 0.0, solver);
        model.fit(X_train, y_train);
        
        // Calcola score
        scores(i) = model.score(X_test, y_test);
    }
    
    return scores;
}

void LinearRegression::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Impossibile aprire file: " + filename);
    }
    
    // Salva dimensione theta
    int theta_size = theta_.size();
    file.write(reinterpret_cast<const char*>(&theta_size), sizeof(int));
    
    // Salva theta
    file.write(reinterpret_cast<const char*>(theta_.data()), 
              theta_size * sizeof(double));
    
    // Salva metadati
    file.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&max_iter_), sizeof(int));
    file.write(reinterpret_cast<const char*>(&lambda_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&solver_), sizeof(Solver));
    file.write(reinterpret_cast<const char*>(&n_iter_), sizeof(int));
    
    // Salva scaler
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
    
    // Salva cost history
    int history_size = cost_history_.size();
    file.write(reinterpret_cast<const char*>(&history_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(cost_history_.data()), 
              history_size * sizeof(double));
    
    file.close();
}

void LinearRegression::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Impossibile aprire file: " + filename);
    }
    
    // Carica dimensione theta
    int theta_size;
    file.read(reinterpret_cast<char*>(&theta_size), sizeof(int));
    
    // Carica theta
    theta_.resize(theta_size);
    file.read(reinterpret_cast<char*>(theta_.data()), theta_size * sizeof(double));
    
    // Carica metadati
    file.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(double));
    file.read(reinterpret_cast<char*>(&max_iter_), sizeof(int));
    file.read(reinterpret_cast<char*>(&lambda_), sizeof(double));
    file.read(reinterpret_cast<char*>(&solver_), sizeof(Solver));
    file.read(reinterpret_cast<char*>(&n_iter_), sizeof(int));
    
    // Carica scaler
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
    
    // Carica cost history
    int history_size;
    file.read(reinterpret_cast<char*>(&history_size), sizeof(int));
    cost_history_.resize(history_size);
    file.read(reinterpret_cast<char*>(cost_history_.data()), 
             history_size * sizeof(double));
    
    file.close();
}

std::string LinearRegression::to_string() const {
    std::ostringstream ss;
    ss << "LinearRegression:\n";
    ss << "  Solver: ";
    switch (solver_) {
        case GRADIENT_DESCENT: ss << "Gradient Descent"; break;
        case NORMAL_EQUATION: ss << "Normal Equation"; break;
        case SVD: ss << "SVD"; break;
    }
    ss << "\n";
    ss << "  Features: " << n_features_ << "\n";
    ss << "  Intercept: " << intercept() << "\n";
    ss << "  Coefficients: [";
    for (int i = 0; i < std::min(5, n_features_); ++i) {
        ss << theta_(i + 1);
        if (i < std::min(5, n_features_) - 1) ss << ", ";
    }
    if (n_features_ > 5) ss << ", ...";
    ss << "]\n";
    ss << "  Iterations: " << n_iter_ << "\n";
    ss << "  Final cost: " << (cost_history_.empty() ? 0.0 : cost_history_.back());
    return ss.str();
}