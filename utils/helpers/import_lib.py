#!/usr/bin/env python3

import sys
import os

def setup_lib_paths():
    """Configura i path per importare la libreria C++"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))


    lib_dir = os.path.join(project_root, 'lib')
    
    # Aggiungi al path
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    
    return project_root, lib_dir

def lib_pymlalgorithms_import():
    """import della libreria"""

    setup_lib_paths()
    # print(f"Python path: {sys.path[:2]}...")
    
    try:
        import pymlalgorithms as ml
        return ml
        
    except ImportError as e:
        print(f"✗ Errore import: {e}")
        print("Suggerimenti:")
        print("2. Verifica che il file .so sia in lib/")
        return None
    except Exception as e:
        print(f"✗ Errore: {e}")
        return None
