#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "linear_regression.h"
#include "logistic_regression.h"

namespace py = pybind11;

// Utilizziamo il namespace per pulizia nel binding
using namespace regression;

PYBIND11_MODULE(pyregression, m) {
    m.doc() = "Libreria C++ di Machine Learning ottimizzata per Python";

    // ========================================================================
    // LINEAR REGRESSION BINDINGS
    // ========================================================================
    
    // Esponiamo l'Enum Solver all'interno del modulo
    py::enum_<LinearRegression::Solver>(m, "Solver")
        .value("GRADIENT_DESCENT", LinearRegression::GRADIENT_DESCENT)
        .value("NORMAL_EQUATION", LinearRegression::NORMAL_EQUATION)
        .value("SVD", LinearRegression::SVD)
        .export_values();

    py::class_<LinearRegression>(m, "LinearRegression")
        // Costruttori
        .def(py::init<>())
        .def(py::init<double, int, double, LinearRegression::Solver>(),
             py::arg("learning_rate") = 0.01,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("solver") = LinearRegression::GRADIENT_DESCENT)
        
        // Training
        .def("fit", &LinearRegression::fit, 
             py::arg("X"), py::arg("y"),
             "Addestra il modello usando il solver specificato")
        
        // Predizioni (Overload per singolo vettore e matrice)
        .def("predict", py::overload_cast<const Eigen::VectorXd&>(&LinearRegression::predict, py::const_),
             py::arg("x"), "Predizione per un singolo campione")
        .def("predict", py::overload_cast<const Eigen::MatrixXd&>(&LinearRegression::predict, py::const_),
             py::arg("X"), "Predizione batch per una matrice di campioni")
        
        // Metriche e Valutazione
        .def("score", &LinearRegression::score, py::arg("X"), py::arg("y"), "Ritorna il coefficiente R2")
        .def("mse", &LinearRegression::mse, py::arg("X"), py::arg("y"), "Mean Squared Error")
        .def("mae", &LinearRegression::mae, py::arg("X"), py::arg("y"), "Mean Absolute Error")
        .def("r2_score", &LinearRegression::r2_score, py::arg("X"), py::arg("y"), "R2 Score")
        
        // Metodi Statici (Cross Validation)
        .def_static("cross_val_score", &LinearRegression::cross_val_score,
                    py::arg("X"), py::arg("y"), py::arg("cv") = 5, 
                    py::arg("solver") = LinearRegression::GRADIENT_DESCENT)
        
        // Accesso ai dati interni (Getters)
        .def("get_coefficients", &LinearRegression::coefficients, "Vettore dei pesi (theta)")
        .def("get_intercept", &LinearRegression::intercept, "Valore dell'intercetta (bias)")
        .def("get_cost_history", &LinearRegression::cost_history, "Storia del costo durante GD")
        
        // Persistenza e Utility
        .def("save", &LinearRegression::save, py::arg("filename"), "Salva il modello in formato binario")
        .def("load", &LinearRegression::load, py::arg("filename"), "Carica un modello da file")
        .def("__repr__", &LinearRegression::to_string);


    // ========================================================================
    // LOGISTIC REGRESSION BINDINGS
    // ========================================================================
    
    py::class_<LogisticRegression>(m, "LogisticRegression")
        // Costruttori
        .def(py::init<>())
        .def(py::init<double, int, double, double, bool>(),
             py::arg("learning_rate") = 0.1,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("tolerance") = 1e-4,
             py::arg("verbose") = false)
        
        // Training
        .def("fit", &LogisticRegression::fit, py::arg("X"), py::arg("y"),
             "Addestra il modello di classificazione logistica")
        
        // Probabilità (Implementazione predict dell'interfaccia Estimator)
        .def("predict_proba", &LogisticRegression::predict, py::arg("X"),
             "Ritorna le probabilità sigmoidee (0-1)")
        
        // Classificazione (Classi 0 o 1)
        .def("predict", &LogisticRegression::predict_class, 
             py::arg("X"), py::arg("threshold") = 0.5,
             "Ritorna le classi predette (0 o 1) in base alla soglia")
        
        // Metriche avanzate
        .def("score", &LogisticRegression::score, 
             py::arg("X"), py::arg("y"), "Accuracy del modello")
        .def("precision_recall_f1", &LogisticRegression::precision_recall_f1,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5,
             "Ritorna un vettore con [Precision, Recall, F1-Score]")
        .def("confusion_matrix", &LogisticRegression::confusion_matrix,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5,
             "Ritorna la matrice di confusione 2x2")
        
        // Accesso ai dati interni
        .def("get_coefficients", &LogisticRegression::coefficients)
        .def("get_intercept", &LogisticRegression::intercept)
        .def("get_cost_history", &LogisticRegression::cost_history)
        
        // Persistenza e Utility
        .def("save", &LogisticRegression::save, py::arg("filename"))
        .def("load", &LogisticRegression::load, py::arg("filename"))
        .def("__repr__", &LogisticRegression::to_string);
}