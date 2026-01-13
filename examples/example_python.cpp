/**
 * @file example_python.cpp
 * @brief Esempio di integrazione tra C++ e Python per regression
 * 
 * Questo esempio mostra come:
 * 1. Addestrare modelli in C++
 * 2. Salvarli su disco
 * 3. Caricarli in Python per inferenza
 * 4. Scambiare dati tra C++ e Python
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Python.h>
#include "regression/linear_regression.h"
#include "regression/logistic_regression.h"
#include "regression/math_utils.h"

using namespace regression;

void initialize_python() {
    std::cout << "Initializing Python interpreter..." << std::endl;
    Py_Initialize();
    
    // Aggiungi directory corrente al path di Python
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('.')");
    PyRun_SimpleString("sys.path.append('./build')");
    
    std::cout << "Python " << Py_GetVersion() << " initialized" << std::endl;
}

void finalize_python() {
    Py_Finalize();
    std::cout << "Python interpreter finalized" << std::endl;
}

void run_python_script(const std::string& script) {
    std::cout << "\nExecuting Python script:" << std::endl;
    std::cout << "------------------------" << std::endl;
    
    FILE* python_script = fopen("temp_script.py", "w");
    if (!python_script) {
        std::cerr << "Error creating temporary Python script" << std::endl;
        return;
    }
    
    fprintf(python_script, "%s", script.c_str());
    fclose(python_script);
    
    // Esegui lo script
    FILE* fp = fopen("temp_script.py", "r");
    if (fp) {
        PyRun_SimpleFile(fp, "temp_script.py");
        fclose(fp);
    }
    
    std::remove("temp_script.py");
}

void example_cpp_to_python_workflow() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "EXAMPLE 1: C++ Training -> Python Inference" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Fase 1: Addestra modello in C++
    std::cout << "\n[PHASE 1: Training in C++]" << std::endl;
    
    // Dataset: prezzo case basato su metratura e numero stanze
    Eigen::MatrixXd X_house(6, 2);
    Eigen::VectorXd y_house(6);
    
    // Dati: [metratura_mq, num_stanze] -> prezzo (in migliaia)
    X_house << 50, 2,
               75, 3,
               100, 3,
               120, 4,
               150, 4,
               200, 5;
    
    y_house << 100,  // 100,000 €
               150,
               200,
               250,
               300,
               400;
    
    std::cout << "Training Linear Regression on housing data..." << std::endl;
    LinearRegression house_model(0.01, 1000, 0.0, LinearRegression::NORMAL_EQUATION);
    house_model.fit(X_house, y_house);
    
    std::cout << "Model coefficients: " << house_model.coefficients().transpose() << std::endl;
    std::cout << "R² score: " << house_model.r2_score(X_house, y_house) << std::endl;
    
    // Salva modello
    std::cout << "\nSaving model to 'house_price_model.bin'..." << std::endl;
    house_model.save("house_price_model.bin");
    
    // Fase 2: Usa Python per caricare e usare il modello
    std::cout << "\n[PHASE 2: Inference in Python]" << std::endl;
    
    std::string python_script = R"(
import sys
import numpy as np

print("Loading C++ trained model in Python...")

# Aggiungi percorso della libreria
sys.path.insert(0, './build')

try:
    import regression
    
    # Carica il modello salvato da C++
    model = regression.LinearRegression()
    model.load('house_price_model.bin')
    
    print("Model loaded successfully!")
    print(f"Coefficients: {model.coefficients()}")
    print(f"Intercept: {model.intercept()}")
    
    # Crea nuovi dati per la predizione
    new_houses = np.array([
        [80, 3],    # 80 mq, 3 stanze
        [130, 4],   # 130 mq, 4 stanze
        [180, 5]    # 180 mq, 5 stanze
    ], dtype=np.float64)
    
    print("\nMaking predictions in Python:")
    predictions = model.predict(new_houses)
    
    for i, (house, price) in enumerate(zip(new_houses, predictions)):
        print(f"House {i+1}: {house[0]:.0f} m², {int(house[1])} rooms -> €{price:.0f},000")
    
    # Calcola statistiche aggiuntive in Python
    print("\nAdditional analysis in Python:")
    
    # Simula dati di test
    X_test = np.array([[60, 2], [90, 3], [110, 3]], dtype=np.float64)
    y_test = np.array([80, 170, 210], dtype=np.float64)
    
    mse = model.mse(X_test, y_test)
    r2 = model.r2_score(X_test, y_test)
    
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R²: {r2:.4f}")
    
    # Visualizza risultati (semplice)
    print("\nSimple visualization:")
    print("House Size (m²)  |  Predicted Price (€000)")
    print("-" * 40)
    
    for size in range(50, 210, 30):
        sample = np.array([[size, 3]], dtype=np.float64)
        price = model.predict(sample)[0]
        print(f"{size:^15} | {price:^20.0f}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
)";
    
    run_python_script(python_script);
    
    // Pulisci
    std::remove("house_price_model.bin");
}

void example_python_to_cpp_workflow() {
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "EXAMPLE 2: Python Data -> C++ Training" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n[PHASE 1: Python generates and saves data]" << std::endl;
    
    std::string python_data_gen = R"(
import numpy as np
import json

print("Generating customer churn dataset in Python...")

# Genera dati sintetici per churn prediction
np.random.seed(42)
n_samples = 100

# Features: [durata_abbonamento_mesi, utilizzo_settimanale_ore, soddisfazione]
X = np.zeros((n_samples, 3))
X[:, 0] = np.random.uniform(1, 24, n_samples)  # 1-24 mesi
X[:, 1] = np.random.uniform(0.5, 20, n_samples)  # 0.5-20 ore/settimana
X[:, 2] = np.random.uniform(1, 10, n_samples)   # 1-10 soddisfazione

# Target: probabilità di churn basata su features
# Più lungo l'abbonamento e più alto l'utilizzo → meno churn
churn_prob = 1 / (1 + np.exp(-(-0.5 + 0.1*X[:,0] - 0.05*X[:,1] - 0.2*X[:,2])))
y = (np.random.rand(n_samples) < churn_prob).astype(np.float64)

print(f"Generated {n_samples} samples")
print(f"Churn rate: {y.mean():.1%}")

# Salva dati in formato CSV per C++
np.savetxt('churn_data_features.csv', X, delimiter=',', fmt='%.6f')
np.savetxt('churn_data_target.csv', y, delimiter=',', fmt='%.6f')

print("\nFirst 5 samples:")
for i in range(5):
    print(f"Sample {i+1}: months={X[i,0]:.1f}, hours={X[i,1]:.1f}, "
          f"satisfaction={X[i,2]:.1f} -> churn={int(y[i])}")

# Salva anche in formato JSON per metadati
metadata = {
    "features": ["subscription_months", "weekly_usage_hours", "satisfaction_score"],
    "target": "churn",
    "samples": n_samples,
    "churn_rate": float(y.mean())
}

with open('churn_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nData saved to:")
print("  - churn_data_features.csv")
print("  - churn_data_target.csv")
print("  - churn_metadata.json")
)";
    
    run_python_script(python_data_gen);
    
    // Fase 2: C++ carica i dati e addestra il modello
    std::cout << "\n[PHASE 2: C++ loads data and trains model]" << std::endl;
    
    try {
        // Carica dati dal CSV generato da Python
        std::ifstream features_file("churn_data_features.csv");
        std::ifstream target_file("churn_data_target.csv");
        
        if (!features_file.is_open() || !target_file.is_open()) {
            throw std::runtime_error("Could not open data files generated by Python");
        }
        
        // Leggi dati
        std::vector<std::vector<double>> features_data;
        std::vector<double> target_data;
        
        std::string line;
        while (std::getline(features_file, line)) {
            std::vector<double> row;
            std::stringstream ss(line);
            std::string value;
            
            while (std::getline(ss, value, ',')) {
                row.push_back(std::stod(value));
            }
            features_data.push_back(row);
        }
        
        while (std::getline(target_file, line)) {
            target_data.push_back(std::stod(line));
        }
        
        // Converti in Eigen matrices
        int n_samples = features_data.size();
        int n_features = features_data[0].size();
        
        Eigen::MatrixXd X(n_samples, n_features);
        Eigen::VectorXd y(n_samples);
        
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                X(i, j) = features_data[i][j];
            }
            y(i) = target_data[i];
        }
        
        std::cout << "Data loaded successfully:" << std::endl;
        std::cout << "  Samples: " << n_samples << std::endl;
        std::cout << "  Features: " << n_features << std::endl;
        std::cout << "  Churn rate: " << (y.sum() / y.size()) * 100 << "%" << std::endl;
        
        // Addestra modello logistic regression
        std::cout << "\nTraining Logistic Regression model in C++..." << std::endl;
        LogisticRegression churn_model(0.1, 2000, 0.1, 1e-6, true);
        churn_model.fit(X, y);
        
        std::cout << "Model trained!" << std::endl;
        std::cout << "Coefficients: " << churn_model.coefficients().transpose() << std::endl;
        std::cout << "Training accuracy: " << churn_model.score(X, y) * 100 << "%" << std::endl;
        
        // Salva modello
        churn_model.save("churn_predictor_model.bin");
        std::cout << "\nModel saved to 'churn_predictor_model.bin'" << std::endl;
        
        // Fase 3: Python carica il modello addestrato in C++
        std::cout << "\n[PHASE 3: Python loads C++ trained model for deployment]" << std::endl;
        
        std::string python_deploy = R"(
import numpy as np
import json

print("Deploying C++ trained model in Python production environment...")

try:
    import regression
    
    # Carica il modello addestrato in C++
    model = regression.LogisticRegression()
    model.load('churn_predictor_model.bin')
    
    print("Model deployed successfully!")
    
    # Carica metadati
    with open('churn_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"\nModel info:")
    print(f"  Features: {metadata['features']}")
    print(f"  Target: {metadata['target']}")
    print(f"  Trained on: {metadata['samples']} samples")
    
    # Simula nuovi clienti
    print("\nPredicting churn for new customers:")
    
    new_customers = np.array([
        [6, 5.0, 7.5],   # Cliente nuovo, uso medio, soddisfazione media
        [18, 15.0, 9.0], # Cliente fedele, uso alto, molto soddisfatto
        [3, 2.0, 3.0],   # Cliente nuovo, uso basso, poco soddisfatto
        [12, 8.0, 6.0]   # Cliente di 1 anno, uso moderato
    ], dtype=np.float64)
    
    # Fai predizioni
    probabilities = model.predict(new_customers)
    predictions = model.predict_class(new_customers, threshold=0.5)
    
    for i, (customer, prob, pred) in enumerate(zip(new_customers, probabilities, predictions)):
        status = "CHURN" if pred == 1 else "RETAIN"
        print(f"\nCustomer {i+1}:")
        print(f"  - Subscription: {customer[0]:.0f} months")
        print(f"  - Usage: {customer[1]:.1f} hours/week")
        print(f"  - Satisfaction: {customer[2]:.1f}/10")
        print(f"  - Churn probability: {prob:.1%}")
        print(f"  - Prediction: {status}")
        
        # Suggerimenti basati sulla predizione
        if pred == 1:
            print(f"  - Action: Consider retention offer!")
    
    # Analisi batch
    print("\n" + "="*50)
    print("Batch analysis for 1000 simulated customers:")
    
    # Genera 1000 clienti simulati
    np.random.seed(123)
    batch_size = 1000
    batch_customers = np.random.randn(batch_size, 3)
    batch_customers[:, 0] = batch_customers[:, 0] * 6 + 12  # mesi: μ=12, σ=6
    batch_customers[:, 1] = np.abs(batch_customers[:, 1] * 3 + 8)  # ore: μ=8, σ=3
    batch_customers[:, 2] = np.clip(batch_customers[:, 2] * 1.5 + 6.5, 1, 10)  # soddisfazione
    
    batch_probs = model.predict(batch_customers)
    batch_preds = model.predict_class(batch_customers)
    
    churn_rate = batch_preds.mean()
    avg_prob = batch_probs.mean()
    
    print(f"  Predicted churn rate: {churn_rate:.1%}")
    print(f"  Average churn probability: {avg_prob:.1%}")
    
    # Segmentazione per mesi di abbonamento
    print("\nChurn rate by subscription length:")
    segments = [(0, 6), (6, 12), (12, 24), (24, 100)]
    
    for low, high in segments:
        mask = (batch_customers[:, 0] >= low) & (batch_customers[:, 0] < high)
        if mask.any():
            segment_churn = batch_preds[mask].mean()
            print(f"  {low}-{high} months: {segment_churn:.1%}")
    
except Exception as e:
    print(f"Error in deployment: {e}")
    import traceback
    traceback.print_exc()
)";
        
        run_python_script(python_deploy);
        
        // Pulisci file temporanei
        std::remove("churn_data_features.csv");
        std::remove("churn_data_target.csv");
        std::remove("churn_metadata.json");
        std::remove("churn_predictor_model.bin");
        
    } catch (const std::exception& e) {
        std::cerr << "Error in C++ phase: " << e.what() << std::endl;
    }
}

void example_real_time_integration() {
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "EXAMPLE 3: Real-time C++/Python Integration" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\nThis example shows real-time data exchange between C++ and Python" << std::endl;
    
    std::string real_time_script = R"(
import numpy as np
import time

print("Real-time integration demo starting...")

# Simula stream di dati in tempo reale
def generate_realtime_data(num_points=10):
    """Genera dati in tempo reale (simulato)"""
    timestamps = []
    features_list = []
    
    current_time = time.time()
    
    for i in range(num_points):
        # Simula misurazioni di sensori
        temperature = 20 + 5 * np.sin(i * 0.5) + np.random.normal(0, 0.5)
        humidity = 50 + 10 * np.sin(i * 0.3) + np.random.normal(0, 2)
        pressure = 1013 + 5 * np.sin(i * 0.2) + np.random.normal(0, 1)
        
        timestamps.append(current_time + i)
        features_list.append([temperature, humidity, pressure])
    
    return np.array(timestamps), np.array(features_list)

try:
    import regression
    
    # Crea un modello semplice in Python
    print("\n1. Creating and training a simple model in Python...")
    
    # Dati di training per predire qualità dell'aria
    X_train = np.array([
        [18, 45, 1010],
        [22, 50, 1012],
        [25, 55, 1013],
        [28, 60, 1011],
        [30, 65, 1009],
        [32, 70, 1008],
        [35, 75, 1007]
    ], dtype=np.float64)
    
    y_train = np.array([1, 1, 0.8, 0.6, 0.4, 0.2, 0], dtype=np.float64)  # Qualità aria (1=ottima, 0=pessima)
    
    model = regression.LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Model trained with R²: {model.r2_score(X_train, y_train):.3f}")
    
    # Simula ricezione dati in tempo reale
    print("\n2. Simulating real-time data stream...")
    
    timestamps, realtime_data = generate_realtime_data(20)
    
    print(f"Received {len(realtime_data)} real-time samples")
    print("First 3 samples:")
    for i in range(3):
        print(f"  Sample {i+1}: T={realtime_data[i,0]:.1f}°C, "
              f"H={realtime_data[i,1]:.1f}%, P={realtime_data[i,2]:.1f}hPa")
    
    # Predizioni in tempo reale
    print("\n3. Making real-time predictions...")
    