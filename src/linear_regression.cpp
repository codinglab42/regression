#include "linear_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;
using namespace regression;

LinearRegression::LinearRegression()
    : learning_rate_(0.01), max_iter_(1000), lambda_(0.0), 
      solver_(GRADIENT_DESCENT), n_features_(0), n_iter_(0) {}

LinearRegression::LinearRegression(double learning_rate, int max_iter, double lambda, Solver solver)
    : learning_rate_(learning_rate), max_iter_(max_iter), lambda_(lambda),
      solver_(solver), n_features_(0), n_iter_(0) {}

MatrixXd LinearRegression::add_intercept(const MatrixXd& X) {
    MatrixXd X_with_intercept(X.rows(), X.cols() + 1);
    X_with_intercept.col(0).setOnes();
    X_with_intercept.rightCols(X.cols()) = X;
    return X_with_intercept;
}

void LinearRegression::fit_scaler(const MatrixXd& X) {
    scaler_.mean = X.colwise().mean();
    scaler_.std = ((X.rowwise() - scaler_.mean.transpose()).array().square().colwise().sum() / X.rows()).sqrt();
    for (int i = 0; i < scaler_.std.size(); ++i) {
        if (scaler_.std(i) < 1e-9) scaler_.std(i) = 1.0;
    }
    scaler_.fit = true;
}

MatrixXd LinearRegression::transform(const MatrixXd& X) const {
    if (!scaler_.fit) return X;
    return (X.rowwise() - scaler_.mean.transpose()).array().rowwise() / scaler_.std.transpose().array();
}

void LinearRegression::fit(const MatrixXd& X, const VectorXd& y) {
    n_features_ = X.cols();
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
    MatrixXd X_int = add_intercept(X);
    MatrixXd XtX = X_int.transpose() * X_int;
    VectorXd Xty = X_int.transpose() * y;

    if (lambda_ > 0) {
        MatrixXd I = MatrixXd::Identity(XtX.rows(), XtX.cols());
        I(0, 0) = 0; 
        XtX += lambda_ * I;
    }
    theta_ = XtX.ldlt().solve(Xty); // Solutore stabile
}

void LinearRegression::svd_solve(const MatrixXd& X, const VectorXd& y) {
    MatrixXd X_int = add_intercept(X);
    theta_ = X_int.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
}

void LinearRegression::gradient_descent(const MatrixXd& X, const VectorXd& y) {
    MatrixXd X_int = add_intercept(X);
    int m = X.rows();
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
    }
}

double LinearRegression::compute_cost(const MatrixXd& X, const VectorXd& y) const {
    MatrixXd X_int = add_intercept(transform(X));
    double m = X.rows();
    double J = (X_int * theta_ - y).squaredNorm() / (2 * m);
    if (lambda_ > 0) {
        J += (lambda_ / (2 * m)) * theta_.tail(theta_.size() - 1).squaredNorm();
    }
    return J;
}

VectorXd LinearRegression::predict(const MatrixXd& X) const {
    MatrixXd X_int = add_intercept(transform(X));
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

void LinearRegression::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
    int theta_size = theta_.size();
    file.write(reinterpret_cast<const char*>(&theta_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(theta_.data()), theta_size * sizeof(double));
    file.close();
}

void LinearRegression::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    file.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    int theta_size;
    file.read(reinterpret_cast<char*>(&theta_size), sizeof(int));
    theta_.resize(theta_size);
    file.read(reinterpret_cast<char*>(theta_.data()), theta_size * sizeof(double));
    file.close();
}

std::string LinearRegression::to_string() const {
    return "LinearRegression Model [Features: " + std::to_string(n_features_) + "]";
}