#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>  // Fondamentale per il supporto automatico a Eigen/NumPy
#include "linear_regression.h"
#include "logistic_regression.h"

namespace py = pybind11;

PYBIND11_MODULE(pyregression, m) {
    m.doc() = "Libreria C++ di Machine Learning con Eigen e Pybind11";

    // ========== LINEAR REGRESSION ==========
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def(py::init<double, int, double, LinearRegression::Solver>(),
             py::arg("learning_rate") = 0.01,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("solver") = LinearRegression::GRADIENT_DESCENT)
        
        .def("fit", &LinearRegression::fit, py::arg("X"), py::arg("y"), 
             "Addestra il modello di regressione lineare")
        
        // Overload per predizioni (singolo vettore o matrice)
        .def("predict", py::overload_cast<const Eigen::VectorXd&>(&LinearRegression::predict, py::const_),
             py::arg("x"), "Predice un singolo valore")
        .def("predict", py::overload_cast<const Eigen::MatrixXd&>(&LinearRegression::predict, py::const_),
             py::arg("X"), "Predice un batch di valori")
        
        // Metriche
        .def("score", &LinearRegression::score, py::arg("X"), py::arg("y"), "R2 Score")
        .def("mse", &LinearRegression::mse, py::arg("X"), py::arg("y"))
        .def("r2_score", &LinearRegression::r2_score, py::arg("X"), py::arg("y"))
        
        // Accesso ai parametri
        .def("get_coefficients", &LinearRegression::coefficients)
        .def("get_intercept", &LinearRegression::intercept)
        .def("get_cost_history", &LinearRegression::cost_history)
        
        // Salvataggio
        .def("save", &LinearRegression::save, py::arg("filename"))
        .def("load", &LinearRegression::load, py::arg("filename"))
        .def("__repr__", [](const LinearRegression &mod) { return mod.to_string(); });

    // Registrazione dell'enum Solver per LinearRegression
    py::enum_<LinearRegression::Solver>(m, "Solver")
        .value("GRADIENT_DESCENT", LinearRegression::GRADIENT_DESCENT)
        .value("NORMAL_EQUATION", LinearRegression::NORMAL_EQUATION)
        .value("SVD", LinearRegression::SVD)
        .export_values();


    // ========== LOGISTIC REGRESSION ==========
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<>())
        .def(py::init<double, int, double, double, bool>(),
             py::arg("learning_rate") = 0.1,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("tolerance") = 1e-4,
             py::arg("verbose") = false)
        
        .def("fit", &LogisticRegression::fit, py::arg("X"), py::arg("y"),
             "Addestra il modello di regressione logistica")
        
        // Probabilità (Overload)
        .def("predict_proba", py::overload_cast<const Eigen::VectorXd&>(&LogisticRegression::predict_proba, py::const_),
             py::arg("x"))
        .def("predict_proba", py::overload_cast<const Eigen::MatrixXd&>(&LogisticRegression::predict_proba, py::const_),
             py::arg("X"))
        
        // Classi (Overload)
        .def("predict", py::overload_cast<const Eigen::VectorXd&, double>(&LogisticRegression::predict, py::const_),
             py::arg("x"), py::arg("threshold") = 0.5)
        .def("predict", py::overload_cast<const Eigen::MatrixXd&, double>(&LogisticRegression::predict, py::const_),
             py::arg("X"), py::arg("threshold") = 0.5)
        
        // Metriche e Utility
        .def("score", &LogisticRegression::score, py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5)
        .def("precision_recall_f1", &LogisticRegression::precision_recall_f1, py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5)
        .def("confusion_matrix", &LogisticRegression::confusion_matrix, py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5)
        
        .def("get_coefficients", &LogisticRegression::coefficients)
        .def("get_intercept", &LogisticRegression::intercept)
        .def("get_cost_history", &LogisticRegression::cost_history)
        .def("get_accuracy_history", &LogisticRegression::accuracy_history)
        
        .def("save", &LogisticRegression::save, py::arg("filename"))
        .def("load", &LogisticRegression::load, py::arg("filename"))
        .def("__repr__", [](const LogisticRegression &mod) { return mod.to_string(); });
}