#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "regression/estimator.h"
#include "regression/linear_regression.h"
#include "regression/logistic_regression.h"
#include "regression/math_utils.h"
#include "neural_network/neural_network.h"
#include "exceptions/ml_exception.h"

namespace py = pybind11;

// Convertitore per eccezioni C++ → Python
void translate_exception(const ml_exception::MLException& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

PYBIND11_MODULE(regression_module, m) {
    m.doc() = R"pbdoc(
        Machine Learning Library Python Bindings
        =========================================
        
        A comprehensive C++ ML library with Python bindings using pybind11.
        
        Features:
        - Linear Regression with multiple solvers
        - Logistic Regression with regularization
        - Neural Networks with multiple layers
        - Cross-validation support
        - Model serialization
        - Comprehensive exception handling
        
        Examples
        --------
        >>> import numpy as np
        >>> import regression_module as ml
        >>> 
        >>> # Linear Regression
        >>> X = np.random.rand(100, 3)
        >>> y = np.random.rand(100)
        >>> 
        >>> model = ml.LinearRegression()
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> 
        >>> # Logistic Regression
        >>> X_log = np.random.rand(100, 3)
        >>> y_log = np.random.randint(0, 2, 100)
        >>> 
        >>> log_model = ml.LogisticRegression()
        >>> log_model.fit(X_log, y_log)
        >>> probabilities = log_model.predict(X_log)
        >>> 
        >>> # Neural Network
        >>> nn = ml.NeuralNetwork([3, 64, 32, 1])
        >>> nn.fit(X, y)
        >>> nn_predictions = nn.predict(X)
    )pbdoc";
    
    // Version info
    m.attr("__version__") = "2.0.0";
    m.attr("__author__") = "Your Name";
    m.attr("__email__") = "your.email@example.com";
    
    // Registra il traduttore di eccezioni
    py::register_exception_translator<ml_exception::MLException>(&translate_exception);
    
    // Bind LinearRegression::Solver enum
    py::enum_<regression::LinearRegression::Solver> solver_enum(m, "LinearSolver");
    solver_enum.value("GRADIENT_DESCENT", regression::LinearRegression::Solver::GRADIENT_DESCENT)
               .value("NORMAL_EQUATION", regression::LinearRegression::Solver::NORMAL_EQUATION)
               .value("SVD", regression::LinearRegression::Solver::SVD)
               .export_values();
    
    // Bind Estimator base class
    py::class_<regression::Estimator, std::shared_ptr<regression::Estimator>>(m, "Estimator")
        .def("fit", &regression::Estimator::fit,
             py::arg("X"), py::arg("y"),
             "Fit the model to the data")
        .def("predict", &regression::Estimator::predict,
             py::arg("X"),
             "Make predictions")
        .def("score", &regression::Estimator::score,
             py::arg("X"), py::arg("y"),
             "Compute model score")
        .def("save", &regression::Estimator::save,
             py::arg("filename"),
             "Save model to file")
        .def("load", &regression::Estimator::load,
             py::arg("filename"),
             "Load model from file")
        .def("to_string", &regression::Estimator::to_string,
             "String representation of the model")
        .def("__repr__", &regression::Estimator::to_string)
        .def("__str__", &regression::Estimator::to_string);
    
    // Bind LinearRegression
    py::class_<regression::LinearRegression, regression::Estimator, 
               std::shared_ptr<regression::LinearRegression>>(m, "LinearRegression")
        .def(py::init<>())
        .def(py::init<double, int, double, regression::LinearRegression::Solver>(),
             py::arg("learning_rate") = 0.01,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("solver") = regression::LinearRegression::Solver::GRADIENT_DESCENT)
        
        .def("fit", &regression::LinearRegression::fit,
             py::arg("X"), py::arg("y"),
             "Fit the linear regression model")
        
        .def("predict", py::overload_cast<const Eigen::MatrixXd&>(&regression::LinearRegression::predict, py::const_),
             py::arg("X"),
             "Predict using the linear model")
        
        .def("predict", py::overload_cast<const Eigen::VectorXd&>(&regression::LinearRegression::predict, py::const_),
             py::arg("x"),
             "Predict a single sample")
        
        .def("score", &regression::LinearRegression::score,
             py::arg("X"), py::arg("y"),
             "Return the R² score")
        
        .def("mse", &regression::LinearRegression::mse,
             py::arg("X"), py::arg("y"),
             "Compute Mean Squared Error")
        
        .def("mae", &regression::LinearRegression::mae,
             py::arg("X"), py::arg("y"),
             "Compute Mean Absolute Error")
        
        .def("r2_score", &regression::LinearRegression::r2_score,
             py::arg("X"), py::arg("y"),
             "Compute R² score")
        
        .def("save", &regression::LinearRegression::save,
             py::arg("filename"),
             "Save model to file")
        
        .def("load", &regression::LinearRegression::load,
             py::arg("filename"),
             "Load model from file")
        
        .def("to_string", &regression::LinearRegression::to_string,
             "String representation of the model")
        
        .def_static("cross_val_score", &regression::LinearRegression::cross_val_score,
                    py::arg("X"), py::arg("y"),
                    py::arg("cv") = 5,
                    py::arg("solver") = regression::LinearRegression::Solver::GRADIENT_DESCENT,
                    "Cross-validation scores")
        
        .def_property_readonly("coefficients", &regression::LinearRegression::coefficients,
                               "Model coefficients (theta)")
        
        .def_property_readonly("intercept", &regression::LinearRegression::intercept,
                               "Model intercept")
        
        .def_property_readonly("cost_history", &regression::LinearRegression::cost_history,
                               "History of cost values during training")
        
        // Setters
        .def("set_learning_rate", &regression::LinearRegression::set_learning_rate,
             py::arg("rate"),
             "Set learning rate")
        
        .def("set_max_iterations", &regression::LinearRegression::set_max_iterations,
             py::arg("max_iter"),
             "Set maximum iterations")
        
        .def("set_lambda", &regression::LinearRegression::set_lambda,
             py::arg("lambda"),
             "Set regularization parameter")
        
        .def("__repr__", &regression::LinearRegression::to_string)
        .def("__str__", &regression::LinearRegression::to_string);
    
    // Bind LogisticRegression
    py::class_<regression::LogisticRegression, regression::Estimator,
               std::shared_ptr<regression::LogisticRegression>>(m, "LogisticRegression")
        .def(py::init<>())
        .def(py::init<double, int, double, double, bool>(),
             py::arg("learning_rate") = 0.1,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("tolerance") = 1e-4,
             py::arg("verbose") = false)
        
        .def("fit", &regression::LogisticRegression::fit,
             py::arg("X"), py::arg("y"),
             "Fit the logistic regression model")
        
        .def("predict", &regression::LogisticRegression::predict,
             py::arg("X"),
             "Predict probabilities")
        
        .def("predict_class", &regression::LogisticRegression::predict_class,
             py::arg("X"), py::arg("threshold") = 0.5,
             "Predict class labels")
        
        .def("score", &regression::LogisticRegression::score,
             py::arg("X"), py::arg("y"),
             "Return the accuracy score")
        
        .def("precision_recall_f1", &regression::LogisticRegression::precision_recall_f1,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5,
             "Compute precision, recall and F1 score")
        
        .def("confusion_matrix", &regression::LogisticRegression::confusion_matrix,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5,
             "Compute confusion matrix")
        
        .def("save", &regression::LogisticRegression::save,
             py::arg("filename"),
             "Save model to file")
        
        .def("load", &regression::LogisticRegression::load,
             py::arg("filename"),
             "Load model from file")
        
        .def("to_string", &regression::LogisticRegression::to_string,
             "String representation of the model")
        
        .def_property_readonly("coefficients", &regression::LogisticRegression::coefficients,
                               "Model coefficients (theta)")
        
        .def_property_readonly("intercept", &regression::LogisticRegression::intercept,
                               "Model intercept")
        
        .def_property_readonly("cost_history", &regression::LogisticRegression::cost_history,
                               "History of cost values during training")
        
        .def_property_readonly("accuracy_history", &regression::LogisticRegression::accuracy_history,
                               "History of accuracy values during training")
        
        // Setters
        .def("set_learning_rate", &regression::LogisticRegression::set_learning_rate,
             py::arg("rate"),
             "Set learning rate")
        
        .def("set_max_iterations", &regression::LogisticRegression::set_max_iterations,
             py::arg("max_iter"),
             "Set maximum iterations")
        
        .def("set_lambda", &regression::LogisticRegression::set_lambda,
             py::arg("lambda"),
             "Set regularization parameter")
        
        .def("set_tolerance", &regression::LogisticRegression::set_tolerance,
             py::arg("tolerance"),
             "Set convergence tolerance")
        
        .def("set_verbose", &regression::LogisticRegression::set_verbose,
             py::arg("verbose"),
             "Set verbose mode")
        
        .def("__repr__", &regression::LogisticRegression::to_string)
        .def("__str__", &regression::LogisticRegression::to_string);
    
    // Bind NeuralNetwork
    py::class_<neural_network::NeuralNetwork, regression::Estimator,
               std::shared_ptr<neural_network::NeuralNetwork>>(m, "NeuralNetwork")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, const std::string&, const std::string&>(),
             py::arg("layer_sizes"),
             py::arg("activation") = "relu",
             py::arg("output_activation") = "sigmoid",
             "Create a neural network with specified architecture")
        
        .def("fit", &neural_network::NeuralNetwork::fit,
             py::arg("X"), py::arg("y"),
             "Fit the neural network")
        
        .def("predict", &neural_network::NeuralNetwork::predict,
             py::arg("X"),
             "Make predictions")
        
        .def("predict_proba", &neural_network::NeuralNetwork::predict_proba,
             py::arg("X"),
             "Predict probabilities")
        
        .def("score", &neural_network::NeuralNetwork::score,
             py::arg("X"), py::arg("y"),
             "Compute model score")
        
        .def("save", &neural_network::NeuralNetwork::save,
             py::arg("filename"),
             "Save model to file")
        
        .def("load", &neural_network::NeuralNetwork::load,
             py::arg("filename"),
             "Load model from file")
        
        .def("to_string", &neural_network::NeuralNetwork::to_string,
             "String representation of the model")
        
        // Configuration methods
        .def("set_batch_size", &neural_network::NeuralNetwork::set_batch_size,
             py::arg("batch_size"),
             "Set batch size for training")
        
        .def("set_epochs", &neural_network::NeuralNetwork::set_epochs,
             py::arg("epochs"),
             "Set number of training epochs")
        
        .def("set_validation_split", &neural_network::NeuralNetwork::set_validation_split,
             py::arg("split"),
             "Set validation split ratio")
        
        .def("set_verbose", &neural_network::NeuralNetwork::set_verbose,
             py::arg("verbose"),
             "Set verbose mode")
        
        .def("set_loss_function", &neural_network::NeuralNetwork::set_loss_function,
             py::arg("loss"),
             "Set loss function")
        
        // Getters
        .def_property_readonly("loss_history", &neural_network::NeuralNetwork::get_loss_history,
                               "History of loss values during training")
        
        .def_property_readonly("val_loss_history", &neural_network::NeuralNetwork::get_val_loss_history,
                               "History of validation loss values")
        
        .def_property_readonly("accuracy_history", &neural_network::NeuralNetwork::get_accuracy_history,
                               "History of accuracy values")
        
        .def_property_readonly("num_layers", &neural_network::NeuralNetwork::get_num_layers,
                               "Number of layers in the network")
        
        .def_property_readonly("num_parameters", &neural_network::NeuralNetwork::get_num_parameters,
                               "Total number of parameters in the network")
        
        .def("__repr__", &neural_network::NeuralNetwork::to_string)
        .def("__str__", &neural_network::NeuralNetwork::to_string);
    
    // Bind MathUtils as a utility module
    py::class_<regression::MathUtils>(m, "MathUtils")
        .def_static("sigmoid", py::overload_cast<double>(&regression::MathUtils::sigmoid),
                    py::arg("z"),
                    "Compute sigmoid function")
        
        .def_static("sigmoid_vec", &regression::MathUtils::sigmoid_vec,
                    py::arg("z"),
                    "Compute sigmoid for a vector")
        
        .def_static("add_intercept", &regression::MathUtils::add_intercept,
                    py::arg("X"),
                    "Add intercept column to matrix")
        
        .def_static("train_test_split", &regression::MathUtils::train_test_split,
                    py::arg("X"), py::arg("y"),
                    py::arg("test_size") = 0.2,
                    py::arg("random_state") = 42,
                    py::arg("model_type") = "",
                    "Split data into train and test sets")
        
        .def_static("accuracy_score", &regression::MathUtils::accuracy_score,
                    py::arg("y_true"), py::arg("y_pred"),
                    py::arg("model_type") = "",
                    "Compute accuracy score")
        
        .def_static("one_hot_encode", &regression::MathUtils::one_hot_encode,
                    py::arg("labels"), py::arg("num_classes"),
                    "One-hot encode labels");
    
    // Convenience functions
    m.def("test_library", []() {
        return "Machine Learning library v2.0 is working correctly!";
    });
    
    // Version check
    m.def("check_version", []() {
        return std::string("v2.0 - Neural Networks, Exceptions, Serialization");
    });
    
    // Register Eigen matrix converters
    py::implicitly_convertible<py::array_t<double>, Eigen::MatrixXd>();
    py::implicitly_convertible<py::array_t<double>, Eigen::VectorXd>();
}