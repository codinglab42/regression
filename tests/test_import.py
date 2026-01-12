#!/usr/bin/env python3
# tests/test_import.py

import sys
import os

def setup_paths():
    """Configura i path per importare la libreria"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    lib_dir = os.path.join(project_root, 'lib')
    
    # Aggiungi al path
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    
    return project_root, lib_dir

def test_import():
    """Testa l'import della libreria"""
    print(f"Python path: {sys.path[:2]}...")
    
    try:
        import pymlalgorithms as ml
        print("✓ Libreria importata con successo!")
        print(f"✓ Modulo: {ml}")
        
        # Test creazione oggetto
        lr = ml.LinearRegressionOneVar()
        print(f"✓ Oggetto creato: {lr}")
        print(f"✓ Theta0: {lr.theta0}")
        print(f"✓ Theta1: {lr.theta1}")
        
        return ml
        
    except ImportError as e:
        print(f"✗ Errore import: {e}")
        print("Suggerimenti:")
        print("1. Assicurati di aver compilato con: cd build && cmake .. && make")
        print("2. Verifica che il file .so sia in lib/")
        return None
    except Exception as e:
        print(f"✗ Errore: {e}")
        return None

if __name__ == "__main__":
    # Configura path
    setup_paths()
    
    # Esegui test
    result = test_import()
    if result:
        print("\n✅ TEST IMPORT PASSATO")
    else:
        print("\n❌ TEST IMPORT FALLITO")
        sys.exit(1)