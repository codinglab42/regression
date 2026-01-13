#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "regression/estimator.h"
#include "regression/linear_regression.h"
#include "regression/logistic_regression.h"
#include "regression/math_utils.h"

namespace py = pybind11;

// Convertitore per gli enum
template <typename T>
py::enum_<T> bind_enum(py::module &m, const std::string &name) {
    py::enum_<T> enum_type(m, name.c_str());
    return enum_type;
}

PYBIND11_MODULE(regression, m) {
    m.doc() = R"pbdoc(
        Regression Library Python Bindings
        ==================================
        
        A C++ regression library with Python bindings using pybind11.
        
        Features:
        - Linear Regression with Gradient Descent / Normal Equation / SVD
        - Logistic Regression with L2 regularization
        - Cross-validation support
        - Model serialization
        
        Examples
        --------
        >>> import numpy as np
        >>> import regression
        >>> 
        >>> # Linear Regression
        >>> X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
        >>> y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        >>> 
        >>> model = regression.LinearRegression()
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> 
        >>> # Logistic Regression
        >>> X_log = np.array([[1, 1], [2, 1], [3, 1]], dtype=np.float64)
        >>> y_log = np.array([0, 0, 1], dtype=np.float64)
        >>> 
        >>> log_model = regression.LogisticRegression()
        >>> log_model.fit(X_log, y_log)
        >>> probabilities = log_model.predict(X_log)
    )pbdoc";
    
    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Your Name";
    m.attr("__email__") = "your.email@example.com";
    
    // Bind LinearRegression::Solver enum
    py::enum_<regression::LinearRegression::Solver> solver_enum(m, "LinearSolver");
    solver_enum.value("GRADIENT_DESCENT", regression::LinearRegression::Solver::GRADIENT_DESCENT)
               .value("NORMAL_EQUATION", regression::LinearRegression::Solver::NORMAL_EQUATION)
               .value("SVD", regression::LinearRegression::Solver::SVD)
               .export_values();
    
    // Bind LinearRegression class
    py::class_<regression::LinearRegression, regression::Estimator>(m, "LinearRegression")
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
        
        .def("__repr__", &regression::LinearRegression::to_string)
        .def("__str__", &regression::LinearRegression::to_string);
    
    // Bind LogisticRegression class
    py::class_<regression::LogisticRegression, regression::Estimator>(m, "LogisticRegression")
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
        
        .def("__repr__", &regression::LogisticRegression::to_string)
        .def("__str__", &regression::LogisticRegression::to_string);
    
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
                    "Split data into train and test sets");
    
    // Convenience function for users
    m.def("test_model", []() {
        return "Regression library is working correctly!";
    });
    
    // Register Eigen matrix converters
    py::implicitly_convertible<py::array, Eigen::MatrixXd>();
    py::implicitly_convertible<py::array, Eigen::VectorXd>();
    
    // Add some numpy type converters
    m.def("as_matrix", [](py::array_t<double> arr) {
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("Number of dimensions must be two");
        
        return Eigen::Map<Eigen::MatrixXd>(
            static_cast<double*>(buf.ptr),
            buf.shape[0],
            buf.shape[1]
        );
    }, py::arg("array"), "Convert numpy array to Eigen matrix");
    
    m.def("as_vector", [](py::array_t<double> arr) {
        auto buf = arr.request();
        if (buf.ndim != 1)
            throw std::runtime_error("Number of dimensions must be one");
        
        return Eigen::Map<Eigen::VectorXd>(
            static_cast<double*>(buf.ptr),
            buf.shape[0]
        );
    }, py::arg("array"), "Convert numpy array to Eigen vector");
}