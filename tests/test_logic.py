import sys
from pathlib import Path
import numpy as np

# Configurazione percorsi
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "build" / "python_module"))
sys.path.insert(0, str(project_root / "build" / "lib"))

try:
    import machine_learning_module as ml
    print("✓ Import riuscito")

    # DATI DI TEST (y = 2x + 1)
    X = np.array([[1], [2], [3], [4]], dtype=np.float64)
    y = np.array([3, 5, 7, 9], dtype=np.float64)

    # Inizializzazione modello
    model = ml.LinearRegression(0.01, 1000, 0.0, ml.LinearSolver.GRADIENT_DESCENT)
    
    print("Inizio training...")
    model.fit(X, y)
    
    # Verifica risultati
    coef = model.coefficients
    intercept = model.intercept
    print(f"Coefficienti: {coef.flatten()}")
    print(f"Intercetta: {intercept}")

    # Test predizione
    test_X = np.array([[5]], dtype=np.float64)
    pred = model.predict(test_X)
    print(f"Predizione per x=5 (atteso ~11): {pred.flatten()}")
   
    # Test Linear Regression
    print("\n" + "="*50)
    print("Testing Linear Regression...")
    
    # Create sample data
    X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    
    # Create and fit model
    lr = regression.LinearRegression(learning_rate=0.01, max_iter=1000)
    lr.fit(X, y)
    
    print(f"  Coefficients: {lr.coefficients}")
    print(f"  Intercept: {lr.intercept}")
    print(f"  R2 Score: {lr.score(X, y):.6f}")
    
    # Test prediction
    X_test = np.array([[6], [7]], dtype=np.float64)
    predictions = lr.predict(X_test)
    print(f"  Predictions for [6, 7]: {predictions}")
    
    # Test Logistic Regression
    print("\n" + "="*50)
    print("Testing Logistic Regression...")
    
    X_log = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]], dtype=np.float64)
    y_log = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    
    logr = regression.LogisticRegression(learning_rate=0.1, max_iter=1000)
    logr.fit(X_log, y_log)
    
    print(f"  Coefficients: {logr.coefficients}")
    print(f"  Accuracy: {logr.score(X_log, y_log):.6f}")
    
    # Test prediction
    test_sample = np.array([[3.5, 1]], dtype=np.float64)
    prob = logr.predict(test_sample)[0]
    print(f"  Probability for [3.5, 1]: {prob:.6f}")
    
    print("\n" + "="*50)
    print("✓ All Python binding tests passed!")

except Exception as e:
    print(f"✗ Errore durante il test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)