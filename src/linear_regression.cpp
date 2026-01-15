#include "regression/linear_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;
using namespace regression;

LinearRegression::LinearRegression()
    : learning_rate_(0.01), max_iter_(1000), lambda_(0.0), tolerance_(1e-4),
      verbose_(false), solver_(GRADIENT_DESCENT), n_features_(0), n_iter_(0) {}

LinearRegression::LinearRegression(double learning_rate, int max_iter, double lambda, Solver solver)
    : learning_rate_(learning_rate), max_iter_(max_iter), lambda_(lambda), tolerance_(1e-4),
      verbose_(false), solver_(solver), n_features_(0), n_iter_(0) {
    
    if (learning_rate_ <= 0) throw std::invalid_argument("learning_rate must be > 0");
    if (max_iter_ <= 0) throw std::invalid_argument("max_iter must be > 0");
    if (lambda_ < 0) throw std::invalid_argument("lambda must be >= 0");
}

void LinearRegression::fit_scaler(const MatrixXd& X) {
    scaler_.mean = X.colwise().mean();
    // Usato X.rows() che restituisce Index, diviso correttamente
    scaler_.std = ((X.rowwise() - scaler_.mean.transpose()).array().square().colwise().sum() / static_cast<double>(X.rows())).sqrt();
    
    // Corretto il loop usando Eigen::Index per evitare warning di confronto
    for (Eigen::Index i = 0; i < scaler_.std.size(); ++i) {
        if (scaler_.std(i) < 1e-9) scaler_.std(i) = 1.0;
    }
    scaler_.fit = true;
}

MatrixXd LinearRegression::transform(const MatrixXd& X) const {
    if (!scaler_.fit) return X;
    return (X.rowwise() - scaler_.mean.transpose()).array().rowwise() / scaler_.std.transpose().array();
}

void LinearRegression::fit(const MatrixXd& X, const VectorXd& y) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }
    
    // Cast esplicito da Eigen::Index a int con controllo
    Eigen::Index cols = X.cols();
    if (cols > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Number of features exceeds int limit");
    }
    n_features_ = static_cast<int>(cols);
    
    fit_scaler(X);
    MatrixXd X_scaled = transform(X);

    if (solver_ == GRADIENT_DESCENT) {
        gradient_descent(X_scaled, y);
    } else if (solver_ == NORMAL_EQUATION) {
        normal_equation(X_scaled, y);
    } else {
        svd_solve(X_scaled, y);
    }
}

void LinearRegression::normal_equation(const MatrixXd& X, const VectorXd& y) {
    MatrixXd X_int = MathUtils::add_intercept(X);
    MatrixXd XtX = X_int.transpose() * X_int;
    VectorXd Xty = X_int.transpose() * y;

    if (lambda_ > 0) {
        MatrixXd I = MatrixXd::Identity(XtX.rows(), XtX.cols());
        I(0, 0) = 0; 
        XtX += lambda_ * I;
    }
    theta_ = XtX.ldlt().solve(Xty);
    
    if (verbose_) {
        std::cout << "Normal equation solved. Theta size: " << theta_.size() << std::endl;
    }
}

void LinearRegression::svd_solve(const MatrixXd& X, const VectorXd& y) {
    MatrixXd X_int = MathUtils::add_intercept(X);
    theta_ = X_int.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
    
    if (verbose_) {
        std::cout << "SVD solved. Theta size: " << theta_.size() << std::endl;
    }
}

void LinearRegression::gradient_descent(const MatrixXd& X, const VectorXd& y) {
    MatrixXd X_int = MathUtils::add_intercept(X);
    double m = static_cast<double>(X.rows()); // m come double per calcoli
    theta_ = VectorXd::Zero(X_int.cols());
    cost_history_.clear();

    for (n_iter_ = 0; n_iter_ < max_iter_; ++n_iter_) {
        VectorXd error = (X_int * theta_) - y;
        VectorXd gradient = (X_int.transpose() * error) / m;
        
        if (lambda_ > 0) {
            VectorXd reg = (lambda_ / m) * theta_;
            reg(0) = 0;
            gradient += reg;
        }

        theta_ -= learning_rate_ * gradient;
        cost_history_.push_back(compute_cost(X, y));
        
        // Risolto confronto tra int e size_t (cost_history_.size())
        if (n_iter_ > 10 && cost_history_.size() > 10) {
            double improvement = cost_history_[cost_history_.size() - 11] - cost_history_.back();
            if (std::abs(improvement) < tolerance_) {
                if (verbose_) {
                    std::cout << "Early stopping at iteration " << n_iter_ << std::endl;
                }
                break;
            }
        }
    }
}

double LinearRegression::compute_cost(const MatrixXd& X, const VectorXd& y) const {
    MatrixXd X_int = MathUtils::add_intercept(transform(X));
    double m = static_cast<double>(X.rows());
    double J = (X_int * theta_ - y).squaredNorm() / (2.0 * m);
    if (lambda_ > 0) {
        // Corretto calcolo indici per tail
        J += (lambda_ / (2.0 * m)) * theta_.tail(theta_.size() - 1).squaredNorm();
    }
    return J;
}

VectorXd LinearRegression::predict(const MatrixXd& X) const {
    if (theta_.size() == 0) {
        throw std::runtime_error("Model not fitted yet");
    }
    MatrixXd X_int = MathUtils::add_intercept(transform(X));
    return X_int * theta_;
}

double LinearRegression::score(const MatrixXd& X, const VectorXd& y) const {
    return r2_score(X, y);
}

double LinearRegression::r2_score(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    double ss_res = (y - y_pred).squaredNorm();
    double ss_tot = (y.array() - y.mean()).square().sum();
    return 1.0 - (ss_res / ss_tot);
}

double LinearRegression::predict(const VectorXd& x) const {
    // Confronto tra Eigen::Index e int con cast
    if (x.size() != static_cast<Eigen::Index>(n_features_)) {
        throw std::invalid_argument("Dimension mismatch in predict()");
    }
    MatrixXd X_row(1, n_features_);
    X_row.row(0) = x;
    return predict(X_row)(0);
}

double LinearRegression::mse(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    return (y - y_pred).squaredNorm() / static_cast<double>(X.rows());
}

double LinearRegression::mae(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    return (y - y_pred).cwiseAbs().sum() / static_cast<double>(X.rows());
}

VectorXd LinearRegression::cross_val_score(const MatrixXd& X, const VectorXd& y, int cv, Solver solver) {
    if (cv <= 1) {
        throw std::invalid_argument("cv must be > 1");
    }
    
    // Usiamo Eigen::Index per tutto il loop di partizionamento
    Eigen::Index n_samples = X.rows();
    Eigen::Index fold_size = n_samples / static_cast<Eigen::Index>(cv);
    VectorXd scores(cv);
    
    for (int i = 0; i < cv; ++i) {
        Eigen::Index start = static_cast<Eigen::Index>(i) * fold_size;
        Eigen::Index end = (i == cv-1) ? n_samples : static_cast<Eigen::Index>(i+1) * fold_size;
        
        MatrixXd X_train(n_samples - (end-start), X.cols());
        VectorXd y_train(n_samples - (end-start));
        MatrixXd X_test(end-start, X.cols());
        VectorXd y_test(end-start);
        
        Eigen::Index train_idx = 0, test_idx = 0;
        for (Eigen::Index j = 0; j < n_samples; ++j) {
            if (j >= start && j < end) {
                X_test.row(test_idx) = X.row(j);
                y_test(test_idx) = y(j);
                test_idx++;
            } else {
                X_train.row(train_idx) = X.row(j);
                y_train(train_idx) = y(j);
                train_idx++;
            }
        }
        
        LinearRegression lr(0.01, 1000, 0.0, solver);
        lr.fit(X_train, y_train);
        scores(i) = lr.score(X_test, y_test);
    }
    
    return scores;
}

void LinearRegression::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
    
    // Cast sicuro per dimensioni Eigen
    Eigen::Index theta_size_idx = theta_.size();
    if (theta_size_idx > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Theta size exceeds int limit for serialization");
    }
    int theta_size = static_cast<int>(theta_size_idx);
    file.write(reinterpret_cast<const char*>(&theta_size), sizeof(int));
    
    // CORREZIONE: Usa size_t per il calcolo dei byte, poi cast a streamsize
    auto theta_bytes_size_t = static_cast<size_t>(theta_size_idx) * sizeof(double);
    if (theta_bytes_size_t > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Theta byte size exceeds streamsize limit");
    }
    std::streamsize theta_bytes = static_cast<std::streamsize>(theta_bytes_size_t);
    
    file.write(reinterpret_cast<const char*>(theta_.data()), theta_bytes);
    
    bool has_scaler = scaler_.fit;
    file.write(reinterpret_cast<const char*>(&has_scaler), sizeof(bool));
    if (has_scaler) {
        Eigen::Index mean_size_idx = scaler_.mean.size();
        if (mean_size_idx > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Mean size exceeds int limit for serialization");
        }
        int mean_size = static_cast<int>(mean_size_idx);
        file.write(reinterpret_cast<const char*>(&mean_size), sizeof(int));
        
        // CORREZIONE: Stessa logica per mean_bytes
        auto mean_bytes_size_t = static_cast<size_t>(mean_size_idx) * sizeof(double);
        if (mean_bytes_size_t > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("Mean byte size exceeds streamsize limit");
        }
        std::streamsize mean_bytes = static_cast<std::streamsize>(mean_bytes_size_t);
        
        file.write(reinterpret_cast<const char*>(scaler_.mean.data()), mean_bytes);
        file.write(reinterpret_cast<const char*>(scaler_.std.data()), mean_bytes);
    }
}

void LinearRegression::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    int theta_size;
    file.read(reinterpret_cast<char*>(&theta_size), sizeof(int));
    theta_.resize(theta_size);
    
    // CORREZIONE: Cast esplicito attraverso size_t
    auto theta_bytes_size_t = static_cast<size_t>(theta_size) * sizeof(double);
    if (theta_bytes_size_t > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Theta byte size exceeds streamsize limit");
    }
    std::streamsize theta_bytes = static_cast<std::streamsize>(theta_bytes_size_t);
    
    file.read(reinterpret_cast<char*>(theta_.data()), theta_bytes);
    
    bool has_scaler;
    file.read(reinterpret_cast<char*>(&has_scaler), sizeof(bool));
    if (has_scaler) {
        int mean_size;
        file.read(reinterpret_cast<char*>(&mean_size), sizeof(int));
        scaler_.mean.resize(mean_size);
        scaler_.std.resize(mean_size);
        scaler_.fit = true;
        
        // CORREZIONE: Stessa logica per mean_bytes
        auto mean_bytes_size_t = static_cast<size_t>(mean_size) * sizeof(double);
        if (mean_bytes_size_t > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("Mean byte size exceeds streamsize limit");
        }
        std::streamsize mean_bytes = static_cast<std::streamsize>(mean_bytes_size_t);
        
        file.read(reinterpret_cast<char*>(scaler_.mean.data()), mean_bytes);
        file.read(reinterpret_cast<char*>(scaler_.std.data()), mean_bytes);
    }
}

std::string LinearRegression::to_string() const {
    std::string solver_str;
    switch(solver_) {
        case GRADIENT_DESCENT: solver_str = "Gradient Descent"; break;
        case NORMAL_EQUATION: solver_str = "Normal Equation"; break;
        case SVD: solver_str = "SVD"; break;
    }
    
    return "LinearRegression [Solver: " + solver_str + 
           ", Features: " + std::to_string(n_features_) + 
           ", Iterations: " + std::to_string(n_iter_) + "]";
}