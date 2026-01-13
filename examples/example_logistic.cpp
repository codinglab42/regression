/**
 * @file example_logistic.cpp
 * @brief Esempio completo di utilizzo della Logistic Regression
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "regression/logistic_regression.h"
#include "regression/math_utils.h"

using namespace regression;

void example_simple_classification() {
    std::cout << "========================================" << std::endl;
    std::cout << "EXAMPLE 1: Simple Binary Classification" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Dataset: studenti che passano l'esame basato su ore di studio
    // Features: [ore_studio, ore_sonno]
    // Target: 0 = fallito, 1 = passato
    Eigen::MatrixXd X(10, 2);
    Eigen::VectorXd y(10);
    
    // Dati: ogni riga è [ore_studio, ore_sonno]
    X << 1.0, 7.0,   // Studente 1: 1 ora studio, 7 ore sonno
         2.0, 6.5,   // Studente 2: 2 ore studio, 6.5 ore sonno
         3.0, 6.0,
         4.0, 5.5,
         5.0, 5.0,
         6.0, 4.5,
         7.0, 4.0,
         8.0, 3.5,
         9.0, 3.0,
         10.0, 2.5;
    
    // Risultati: più si studia, più probabilità di passare
    y << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1;
    
    std::cout << "\nDataset (10 studenti):" << std::endl;
    std::cout << "Colonne: [Ore Studio, Ore Sonno] -> Passato (1) o Fallito (0)" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    for(int i = 0; i < X.rows(); ++i) {
        std::cout << "  Studente " << std::setw(2) << i+1 << ": [" 
                  << std::setw(4) << X(i, 0) << ", " 
                  << std::setw(4) << X(i, 1) << "] -> " 
                  << static_cast<int>(y(i)) << std::endl;
    }
    
    // Creazione e addestramento del modello
    std::cout << "\nTraining Logistic Regression model..." << std::endl;
    LogisticRegression model(0.1, 2000, 0.01, 1e-6, true);
    model.fit(X, y);
    
    std::cout << "\nModel trained successfully!" << std::endl;
    std::cout << "Coefficients (theta): " << model.coefficients().transpose() << std::endl;
    std::cout << "Intercept: " << model.intercept() << std::endl;
    
    // Predizioni sul training set
    std::cout << "\nPredictions on training data:" << std::endl;
    std::cout << std::setw(5) << "ID" << std::setw(12) << "Study" 
              << std::setw(8) << "Sleep" << std::setw(12) << "Actual" 
              << std::setw(12) << "Prob" << std::setw(12) << "Pred" 
              << std::setw(12) << "Correct" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    Eigen::VectorXd probabilities = model.predict(X);
    Eigen::VectorXi predictions = model.predict_class(X);
    
    int correct_count = 0;
    for(int i = 0; i < X.rows(); ++i) {
        bool correct = (predictions(i) == static_cast<int>(y(i)));
        if(correct) correct_count++;
        
        std::cout << std::setw(5) << i+1 
                  << std::setw(12) << X(i, 0) 
                  << std::setw(8) << X(i, 1)
                  << std::setw(12) << static_cast<int>(y(i))
                  << std::setw(12) << std::setprecision(3) << probabilities(i)
                  << std::setw(12) << predictions(i)
                  << std::setw(12) << (correct ? "✓" : "✗") << std::endl;
    }
    
    // Metriche di valutazione
    std::cout << "\nModel Evaluation:" << std::endl;
    std::cout << "  Accuracy: " << std::setprecision(4) << model.score(X, y) * 100 << "%" << std::endl;
    
    Eigen::Vector3d metrics = model.precision_recall_f1(X, y);
    std::cout << "  Precision: " << metrics[0] << std::endl;
    std::cout << "  Recall: " << metrics[1] << std::endl;
    std::cout << "  F1-Score: " << metrics[2] << std::endl;
    
    // Matrice di confusione
    Eigen::MatrixXd cm = model.confusion_matrix(X, y);
    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "           Predicted 0  Predicted 1" << std::endl;
    std::cout << "Actual 0: " << std::setw(10) << cm(0,0) << std::setw(12) << cm(0,1) << std::endl;
    std::cout << "Actual 1: " << std::setw(10) << cm(1,0) << std::setw(12) << cm(1,1) << std::endl;
    
    // Predizione su nuovi dati
    std::cout << "\n========================================" << std::endl;
    std::cout << "Making predictions on new students:" << std::endl;
    std::cout << "========================================" << std::endl;
    
    Eigen::MatrixXd new_students(4, 2);
    new_students << 2.5, 6.0,   // Studente A: studio moderato
                    5.5, 5.0,   // Studente B: borderline
                    8.5, 3.0,   // Studente C: molto studio
                    12.0, 2.0;  // Studente D: studio estremo
    
    std::cout << "\nNew students to classify:" << std::endl;
    for(int i = 0; i < new_students.rows(); ++i) {
        Eigen::VectorXd student = new_students.row(i);
        double prob = model.predict(student)(0);
        int pred_class = (prob > 0.5) ? 1 : 0;
        
        std::cout << "  Student " << char('A' + i) << ": [" 
                  << student(0) << "h study, " << student(1) << "h sleep]" << std::endl;
        std::cout << "    -> Probability of passing: " << std::setprecision(3) << prob * 100 << "%" << std::endl;
        std::cout << "    -> Predicted class: " << pred_class 
                  << " (" << (pred_class == 1 ? "PASS" : "FAIL") << ")" << std::endl;
        std::cout << std::endl;
    }
    
    // Salvataggio del modello
    std::cout << "\nSaving model to 'student_pass_predictor.bin'..." << std::endl;
    model.save("student_pass_predictor.bin");
    std::cout << "Model saved successfully!" << std::endl;
}

void example_decision_boundary() {
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "EXAMPLE 2: Decision Boundary Visualization" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Dataset sintetico 2D per visualizzare il confine decisionale
    Eigen::MatrixXd X(20, 2);
    Eigen::VectorXd y(20);
    
    // Genera punti in due cerchi separati
    int idx = 0;
    for(double angle = 0; angle < 2*M_PI; angle += M_PI/5) {
        // Classe 0: cerchio interno (raggio 1-2)
        X(idx, 0) = 1.5 + cos(angle);
        X(idx, 1) = 1.5 + sin(angle);
        y(idx) = 0;
        idx++;
        
        // Classe 1: cerchio esterno (raggio 3-4)
        X(idx, 0) = 3.5 + 2*cos(angle);
        X(idx, 1) = 3.5 + 2*sin(angle);
        y(idx) = 1;
        idx++;
    }
    
    // Addestra modello
    LogisticRegression model(0.1, 3000, 0.1);
    model.fit(X, y);
    
    std::cout << "\nGenerating decision boundary points..." << std::endl;
    
    // Genera griglia per confine decisionale
    const int grid_size = 20;
    Eigen::MatrixXd grid(grid_size * grid_size, 2);
    
    for(int i = 0; i < grid_size; ++i) {
        for(int j = 0; j < grid_size; ++j) {
            int index = i * grid_size + j;
            grid(index, 0) = i * 0.3;  // da 0 a 6
            grid(index, 1) = j * 0.3;  // da 0 a 6
        }
    }
    
    // Predici probabilità sulla griglia
    Eigen::VectorXd grid_probs = model.predict(grid);
    
    // Trova punti vicini al confine decisionale (probabilità ~0.5)
    std::cout << "\nPoints near decision boundary (probability ≈ 0.5):" << std::endl;
    int boundary_count = 0;
    
    for(int i = 0; i < grid_probs.size(); ++i) {
        if(std::abs(grid_probs(i) - 0.5) < 0.05) {
            boundary_count++;
            if(boundary_count <= 5) {  // Mostra solo i primi 5
                std::cout << "  Point [" << std::setw(4) << grid(i, 0) 
                          << ", " << std::setw(4) << grid(i, 1) 
                          << "]: p = " << std::setprecision(3) << grid_probs(i) << std::endl;
            }
        }
    }
    
    std::cout << "  ... and " << (boundary_count - 5) << " more points" << std::endl;
    
    // Calcola accuratezza
    double accuracy = model.score(X, y);
    std::cout << "\nModel accuracy on synthetic data: " 
              << std::setprecision(4) << accuracy * 100 << "%" << std::endl;
}

void example_threshold_tuning() {
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "EXAMPLE 3: Threshold Tuning" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Dataset medico: test per malattia
    // Scenario: malattia rara, preferiamo alta precisione
    Eigen::MatrixXd X(15, 3);
    Eigen::VectorXd y(15);
    
    // Features: [età, pressione_sanguigna, colesterolo]
    // Target: 0 = sano, 1 = malato
    X << 25, 120, 180,
         30, 125, 190,
         35, 130, 200,
         40, 135, 210,
         45, 140, 220,
         50, 145, 230,
         55, 150, 240,
         60, 155, 250,
         65, 160, 260,
         70, 165, 270,
         75, 170, 280,
         80, 175, 290,
         85, 180, 300,
         90, 185, 310,
         95, 190, 320;
    
    y << 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1;
    
    LogisticRegression model(0.05, 1500, 0.05);
    model.fit(X, y);
    
    std::cout << "\nTesting different classification thresholds:" << std::endl;
    std::cout << std::setw(10) << "Threshold" 
              << std::setw(12) << "Precision" 
              << std::setw(10) << "Recall" 
              << std::setw(10) << "F1" 
              << std::setw(12) << "Positives" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    for(double threshold = 0.3; threshold <= 0.8; threshold += 0.1) {
        Eigen::Vector3d metrics = model.precision_recall_f1(X, y, threshold);
        Eigen::VectorXi predictions = model.predict_class(X, threshold);
        int positives = predictions.sum();
        
        std::cout << std::setw(10) << std::setprecision(2) << threshold
                  << std::setw(12) << std::setprecision(3) << metrics[0]
                  << std::setw(10) << metrics[1]
                  << std::setw(10) << metrics[2]
                  << std::setw(12) << positives << std::endl;
    }
    
    std::cout << "\nInterpretation:" << std::endl;
    std::cout << "  - Lower threshold (0.3): More sensitive, catches more cases" << std::endl;
    std::cout << "  - Higher threshold (0.8): More specific, fewer false positives" << std::endl;
    std::cout << "  - For rare diseases: prefer higher threshold to avoid false alarms" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "LOGISTIC REGRESSION EXAMPLES" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // Esempio 1: Classificazione semplice
        example_simple_classification();
        
        // Esempio 2: Confine decisionale
        example_decision_boundary();
        
        // Esempio 3: Tuning della soglia
        example_threshold_tuning();
        
        // Pulisci file temporanei
        std::remove("student_pass_predictor.bin");
        
        std::cout << "\n\n========================================" << std::endl;
        std::cout << "ALL EXAMPLES COMPLETED SUCCESSFULLY!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
}