#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include "exceptions/exception_macros.h"

namespace utils {

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
         * @brief Derivata della sigmoide
         */
        static double sigmoid_derivative(double z) {
            double s = sigmoid(z);
            return s * (1 - s);
        }
        
        /**
         * @brief Versione vettorizzata della derivata della sigmoide
         */
        static Eigen::VectorXd sigmoid_derivative_vec(const Eigen::VectorXd& z) {
            return z.unaryExpr([](double v) { 
                double s = sigmoid(v);
                return s * (1 - s);
            });
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
         * @brief Logaritmo sicuro per matrici
         */
        static Eigen::MatrixXd safe_log_matrix(const Eigen::MatrixXd& m) {
            const double eps = 1e-15;
            return m.array().max(eps).log();
        }

        /**
         * @brief Calcola il gradiente per regressione lineare
         */
        static Eigen::VectorXd compute_gradient_linear(const Eigen::MatrixXd& X, 
                                                     const Eigen::VectorXd& y, 
                                                     const Eigen::VectorXd& theta, 
                                                     double lambda = 0.0) {
            Eigen::Index m = X.rows();
            Eigen::VectorXd gradient = (X.transpose() * (X * theta - y)) / static_cast<double>(m);
                                                    
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
        static Eigen::VectorXd compute_gradient_logistic(const Eigen::MatrixXd& X, 
                                                       const Eigen::VectorXd& y,
                                                       const Eigen::VectorXd& theta, 
                                                       double lambda = 0.0) {
            Eigen::Index m = X.rows();
            Eigen::VectorXd h = sigmoid_vec(X * theta);
            Eigen::VectorXd gradient = (X.transpose() * (h - y)) / static_cast<double>(m);
                                                        
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
        static void standardize_features(Eigen::MatrixXd& X, 
                                       Eigen::VectorXd& mean, 
                                       Eigen::VectorXd& std,
                                       const std::string& model_type = "") {
            if (X.rows() == 0) {
                throw ml_exception::EmptyDatasetException("X", model_type);
            }

            mean = X.colwise().mean();
            std = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() 
                  / static_cast<double>(X.rows())).sqrt();

            // Gestione feature costanti
            for (Eigen::Index i = 0; i < std.size(); ++i) {
                if (std(i) < 1e-9) std(i) = 1.0;
            }

            X = (X.rowwise() - mean.transpose()).array().rowwise() / std.transpose().array();
        }

        /**
         * @brief Split random per cross-validation
         */
        static std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> 
        train_test_split(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, 
                        double test_size = 0.2, int random_state = 42,
                        const std::string& model_type = "") {
            
            ML_CHECK_NOT_EMPTY(X, "X", model_type);
            ML_CHECK_NOT_EMPTY(y, "y", model_type);
            ML_CHECK_DIMENSIONS(X.rows(), y.size(), X.cols(), 1, 
                              "X and y rows", model_type);
            ML_CHECK_PARAM(test_size > 0 && test_size < 1, "test_size", 
                          "must be between 0 and 1", model_type);
            
            Eigen::Index n_samples = X.rows();
            auto n_test = static_cast<Eigen::Index>(static_cast<double>(n_samples) * test_size);
            Eigen::Index n_train = n_samples - n_test;
            
            if (n_test == 0 || n_train == 0) {
                throw ml_exception::InvalidParameterException(
                    "test_size", "results in empty train or test set", model_type);
            }
            
            // Usiamo size_t per indici di std::vector
            std::vector<size_t> indices(static_cast<size_t>(n_samples));
            std::iota(indices.begin(), indices.end(), 0);
            
            // Per riproducibilità
            std::mt19937 g(static_cast<unsigned int>(random_state));
            std::shuffle(indices.begin(), indices.end(), g);
            
            Eigen::MatrixXd X_train(n_train, X.cols());
            Eigen::VectorXd y_train(n_train);
            Eigen::MatrixXd X_test(n_test, X.cols());
            Eigen::VectorXd y_test(n_test);
            
            for (Eigen::Index i = 0; i < n_train; ++i) {
                X_train.row(i) = X.row(static_cast<Eigen::Index>(indices[static_cast<size_t>(i)]));
                y_train(i) = y(static_cast<Eigen::Index>(indices[static_cast<size_t>(i)]));
            }

            for (Eigen::Index i = 0; i < n_test; ++i) {
                size_t idx = static_cast<size_t>(n_train + i);
                X_test.row(i) = X.row(static_cast<Eigen::Index>(indices[idx]));
                y_test(i) = y(static_cast<Eigen::Index>(indices[idx]));
            }

            return {{X_train, y_train}, {X_test, y_test}};
        }
        
        /**
         * @brief Calcola accuracy per classificazione
         */
        static double accuracy_score(const Eigen::VectorXi& y_true, 
                                   const Eigen::VectorXi& y_pred,
                                   const std::string& model_type = "") {
            if (y_true.size() != y_pred.size()) {
                throw ml_exception::DimensionMismatchException(
                    "y_true and y_pred", 
                    y_true.size(), 1,
                    y_pred.size(), 1,
                    model_type);
            }
            
            if (y_true.size() == 0) {
                throw ml_exception::EmptyDatasetException("labels", model_type);
            }
            
            int correct = 0;
            for (Eigen::Index i = 0; i < y_true.size(); ++i) {
                if (y_true(i) == y_pred(i)) {
                    correct++;
                }
            }
            
            return static_cast<double>(correct) / static_cast<double>(y_true.size());
        }
        
        /**
         * @brief Inizializzazione pesi He (per ReLU)
         */
        static Eigen::MatrixXd he_initialization(int input_size, int output_size) {
            double stddev = std::sqrt(2.0 / static_cast<double>(input_size));
            Eigen::MatrixXd weights = Eigen::MatrixXd::Random(output_size, input_size);
            return weights * stddev;
        }
        
        /**
         * @brief Inizializzazione pesi Xavier (per sigmoid/tanh)
         */
        static Eigen::MatrixXd xavier_initialization(int input_size, int output_size) {
            double limit = std::sqrt(6.0 / (input_size + output_size));
            Eigen::MatrixXd weights = Eigen::MatrixXd::Random(output_size, input_size);
            return weights * limit;
        }
        
        /**
         * @brief One-hot encoding
         */
        static Eigen::MatrixXd one_hot_encode(const Eigen::VectorXi& labels, int num_classes) {
            Eigen::MatrixXd encoded = Eigen::MatrixXd::Zero(labels.size(), num_classes);
            for (Eigen::Index i = 0; i < labels.size(); ++i) {
                if (labels(i) >= 0 && labels(i) < num_classes) {
                    encoded(i, labels(i)) = 1.0;
                }
            }
            return encoded;
        }
        
        /**
         * @brief Normalizza i dati tra 0 e 1
         */
        static void normalize_minmax(Eigen::MatrixXd& X, 
                                   Eigen::VectorXd& min_vals,
                                   Eigen::VectorXd& max_vals,
                                   const std::string& model_type = "") {
            if (X.rows() == 0) {
                throw ml_exception::EmptyDatasetException("X", model_type);
            }
            
            min_vals = X.colwise().minCoeff();
            max_vals = X.colwise().maxCoeff();
            
            for (Eigen::Index i = 0; i < min_vals.size(); ++i) {
                if (std::abs(max_vals(i) - min_vals(i)) < 1e-9) {
                    max_vals(i) = min_vals(i) + 1.0; // Evita divisione per zero
                }
            }
            
            X = (X.rowwise() - min_vals.transpose()).array().rowwise() 
                / (max_vals - min_vals).transpose().array();
        }

    };

} // namespace regression

#endif