#include "logistic_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>

using namespace Eigen;
using namespace regression;

LogisticRegression::LogisticRegression()
    : learning_rate_(0.1), max_iter_(1000), lambda_(0.0), tolerance_(1e-4), verbose_(false), n_features_(0) {}

LogisticRegression::LogisticRegression(double lr, int iter, double lb, double tol, bool verb)
    : learning_rate_(lr), max_iter_(iter), lambda_(lb), tolerance_(tol), verbose_(verb), n_features_(0) {}

double LogisticRegression::sigmoid(double z) {
    if (z >= 0) return 1.0 / (1.0 + std::exp(-z));
    double ez = std::exp(z);
    return ez / (1.0 + ez);
}

VectorXd LogisticRegression::sigmoid_vec(const VectorXd& z) {
    return z.unaryExpr(&sigmoid);
}

MatrixXd LogisticRegression::add_intercept(const MatrixXd& X) {
    MatrixXd X_int(X.rows(), X.cols() + 1);
    X_int.col(0).setOnes();
    X_int.rightCols(X.cols()) = X;
    return X_int;
}

void LogisticRegression::fit_scaler(const MatrixXd& X) {
    scaler_.mean = X.colwise().mean();
    scaler_.std = ((X.rowwise() - scaler_.mean.transpose()).array().square().colwise().sum() / X.rows()).sqrt();
    scaler_.fit = true;
}

MatrixXd LogisticRegression::transform(const MatrixXd& X) const {
    if (!scaler_.fit) return X;
    return (X.rowwise() - scaler_.mean.transpose()).array().rowwise() / scaler_.std.transpose().array();
}

void LogisticRegression::fit(const MatrixXd& X, const VectorXd& y) {
    n_features_ = X.cols();
    fit_scaler(X);
    MatrixXd X_int = add_intercept(transform(X));
    int m = X.rows();
    theta_ = VectorXd::Zero(X_int.cols());
    cost_history_.clear();

    for (n_iter_ = 0; n_iter_ < max_iter_; ++n_iter_) {
        VectorXd h = sigmoid_vec(X_int * theta_);
        VectorXd gradient = (X_int.transpose() * (h - y)) / m;
        
        if (lambda_ > 0) {
            VectorXd reg = (lambda_ / m) * theta_;
            reg(0) = 0;
            gradient += reg;
        }

        VectorXd prev_theta = theta_;
        theta_ -= learning_rate_ * gradient;

        double cost = compute_cost(X, y);
        cost_history_.push_back(cost);

        if ((prev_theta - theta_).norm() < tolerance_) break;
    }
}

double LogisticRegression::compute_cost(const MatrixXd& X, const VectorXd& y) const {
    MatrixXd X_int = add_intercept(transform(X));
    VectorXd h = sigmoid_vec(X_int * theta_);
    double m = X.rows();
    double J = -(y.array() * h.array().log() + (1 - y.array()) * (1 - h.array()).log()).mean();
    if (lambda_ > 0) J += (lambda_ / (2 * m)) * theta_.tail(n_features_).squaredNorm();
    return J;
}

VectorXd LogisticRegression::predict(const MatrixXd& X) const {
    MatrixXd X_int = add_intercept(transform(X));
    return sigmoid_vec(X_int * theta_);
}

VectorXi LogisticRegression::predict_class(const MatrixXd& X, double threshold) const {
    VectorXd prob = predict(X);
    return (prob.array() >= threshold).cast<int>();
}

double LogisticRegression::score(const MatrixXd& X, const VectorXd& y) const {
    VectorXi y_pred = predict_class(X);
    return (y_pred.cast<double>().array() == y.array()).cast<double>().mean();
}

void LogisticRegression::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
    int ts = theta_.size();
    file.write(reinterpret_cast<const char*>(&ts), sizeof(int));
    file.write(reinterpret_cast<const char*>(theta_.data()), ts * sizeof(double));
    file.close();
}

void LogisticRegression::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    file.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    int ts;
    file.read(reinterpret_cast<char*>(&ts), sizeof(int));
    theta_.resize(ts);
    file.read(reinterpret_cast<char*>(theta_.data()), ts * sizeof(double));
    file.close();
}

std::string LogisticRegression::to_string() const {
    std::ostringstream ss;
    ss << "LogisticRegression [Features: " << n_features_ << ", Iterations: " << n_iter_ << "]";
    return ss.str();
}