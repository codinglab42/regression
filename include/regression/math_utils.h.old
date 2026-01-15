#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace regression {

    class MathUtils {
    public:
        /**
         * @brief Sigmoide numericamente stabile
         */
        static double sigmoid(double z) {
            if (z >= 0) {
                return 1.0 / (1.0 + std::exp(-z));
            } else {
                double ez = std::exp(z);
                return ez / (1.0 + ez);
            }
        }

        /**
         * @brief Versione vettorizzata della sigmoide
         */
        static Eigen::VectorXd sigmoid_vec(const Eigen::VectorXd& z) {
            return z.unaryExpr([](double v) { return sigmoid(v); });
        }

        /**
         * @brief Aggiunge una colonna di 1 (intercetta)
         */
        static Eigen::MatrixXd add_intercept(const Eigen::MatrixXd& X) {
            if (X.rows() == 0) return X;
            Eigen::MatrixXd X_int(X.rows(), X.cols() + 1);
            X_int.col(0).setOnes();
            X_int.rightCols(X.cols()) = X;
            return X_int;
        }

        /**
         * @brief Logaritmo sicuro per evitare log(0)
         */
        static Eigen::VectorXd safe_log(const Eigen::VectorXd& v) {
            const double eps = 1e-15;
            return v.array().max(eps).log();
        }

        /**
         * @brief Calcola il gradiente per regressione lineare
         */
        static Eigen::VectorXd compute_gradient_linear(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, 
                                                       const Eigen::VectorXd& theta, double lambda = 0.0) {
            Eigen::Index m = X.rows();
            Eigen::VectorXd gradient = (X.transpose() * (X * theta - y)) / m;
                                                    
            if (lambda > 0) {
                Eigen::VectorXd reg = (lambda / static_cast<double>(m)) * theta;
                reg(0) = 0; // Non regolarizzare l'intercetta
                gradient += reg;
            }

            return gradient;
        }

        /**
         * @brief Calcola il gradiente per regressione logistica
         */
        static Eigen::VectorXd compute_gradient_logistic(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                                         const Eigen::VectorXd& theta, double lambda = 0.0) {
            Eigen::Index m = X.rows();
            Eigen::VectorXd h = sigmoid_vec(X * theta);
            Eigen::VectorXd gradient = (X.transpose() * (h - y)) / m;
                                                        
            if (lambda > 0) {
                Eigen::VectorXd reg = (lambda / static_cast<double>(m)) * theta;
                reg(0) = 0;
                gradient += reg;
            }

            return gradient;
        }

        /**
         * @brief Standardizza le feature
         */
        static void standardize_features(Eigen::MatrixXd& X, Eigen::VectorXd& mean, Eigen::VectorXd& std) {
            if (X.rows() == 0) return;

            mean = X.colwise().mean();
            std = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() / X.rows()).sqrt();

            // Gestione feature costanti
            for (int i = 0; i < std.size(); ++i) {
                if (std(i) < 1e-9) std(i) = 1.0;
            }

            X = (X.rowwise() - mean.transpose()).array().rowwise() / std.transpose().array();
        }

        /**
         * @brief Split random per cross-validation
         */
        static std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> train_test_split(
            const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double test_size = 0.2, int random_state = 42) {
            
            std::srand(static_cast<unsigned int>(random_state));
            Eigen::Index n_samples = X.rows();
            auto n_test = static_cast<int>(static_cast<double>(n_samples) * test_size);
            
            std::vector<int> indices(static_cast<typename std::vector<int>::size_type>(n_samples));
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            
            Eigen::MatrixXd X_train(n_samples - n_test, X.cols());
            Eigen::VectorXd y_train(n_samples - n_test);
            Eigen::MatrixXd X_test(n_test, X.cols());
            Eigen::VectorXd y_test(n_test);
            
            for (int i = 0; i < n_samples - n_test; ++i) {
                X_train.row(i) = X.row(indices[i]);
                y_train(i) = y(indices[i]);
            }

            for (int i = 0; i < n_test; ++i) {
                X_test.row(i) = X.row(indices[n_samples - n_test + i]);
                y_test(i) = y(indices[n_samples - n_test + i]);
            }

            return {{X_train, y_train}, {X_test, y_test}};
        }
    };

} // namespace regression

#endif