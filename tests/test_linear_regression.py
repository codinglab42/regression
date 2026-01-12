#!/usr/bin/env python3
# tests/test_linear_regression.py

import sys
import os
import numpy as np


"""Setup del path del progetto - da importare in tutti gli script"""
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)  # Torna alla root
    
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
print(f"🚀 Project root added to path: {project_root}")

import setup_project

# Importa la funzione di setup dal test_import
#from test_import import setup_paths, test_import
from utils.helpers.import_lib import lib_pymlalgorithms_import

def test_linear_regression():
    """Test della regressione lineare con dati sintetici"""
    print("\n" + "="*60)
    print("TEST REGRESSIONE LINEARE")
    print("="*60)
    
    # Configura path prima di tutto
    # project_root, lib_dir = setup_lib_paths()
    
    # Importa la libreria
    ml = lib_pymlalgorithms_import()
    if ml is None:
        return False
    
    # Crea dati di test: y = 2 + 3x
    np.random.seed(42)
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 8.0, 11.0, 14.0, 17.0])  # 2 + 3*x
    
    print("\nDati di test:")
    print(f"X: {X}")
    print(f"y: {y}")
    print("Formula attesa: y = 2 + 3x")
    
    try:
        # Crea modello
        model = ml.LinearRegressionOneVar(learning_rate=0.01, iterations=1000)
        
        # Allena il modello
        print("\nTraining del modello...")
        # Converti numpy array in liste per pybind11
        model.fit(X.tolist(), y.tolist())
        
        # Stampa risultati
        print(f"\nRisultati:")
        print(f"Theta0 (intercept): {model.theta0:.4f} (atteso: 2.0)")
        print(f"Theta1 (slope):     {model.theta1:.4f} (atteso: 3.0)")
        
        # Predizioni
        test_x = 6.0
        prediction = model.predict_single(test_x)
        print(f"\nPredizione per x={test_x}: {prediction:.4f}")
        print(f"Valore atteso per x={test_x}: {2 + 3*test_x}")
        
        # Test con array
        test_x_array = [6.0, 7.0, 8.0]
        predictions = model.predict(test_x_array)
        print(f"\nPredizioni per {test_x_array}:")
        for x, pred in zip(test_x_array, predictions):
            print(f"  x={x}: {pred:.4f} (atteso: {2 + 3*x})")
        
        # Test costo
        cost = model.compute_cost([test_x], [2 + 3*test_x])
        print(f"\nCosto per predizione perfetta: {cost:.6f} (atteso: ~0)")
        
        # Verifica storie
        cost_history = model.cost_history
        print(f"\nStoria costi (ultimi 5): {cost_history[-5:] if len(cost_history) > 5 else cost_history}")
        
        return True
        
    except Exception as e:
        print(f"✗ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Prima configura i path
    #project_root , lib_dir = setup_lib_paths()
    #print(f"Project root: {project_root}")
    #print(f"Lib directory: {lib_dir}")
    
    # Testa l'import
    ml = lib_pymlalgorithms_import()
    if ml is None:
        print("❌ Test fallito: impossibile importare la libreria")
        sys.exit(1)
    
    # Testa la regressione lineare
    success = test_linear_regression()
    
    if success:
        print("\n" + "="*60)
        print("✅ TUTTI I TEST PASSATI!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ TEST FALLITI")
        print("="*60)
        sys.exit(1)