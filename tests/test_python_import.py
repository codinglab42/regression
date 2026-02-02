import sys
import os
from pathlib import Path

# 1. Calcola i percorsi relativi
# Siamo in 'tests/', quindi saliamo di uno e entriamo in 'build/...'
current_dir = Path(__file__).parent
project_root = current_dir.parent
module_dir = project_root / "build" / "python_module"
lib_dir = project_root / "build" / "lib"

# 2. Aggiungi i percorsi a sys.path
sys.path.insert(0, str(module_dir))
sys.path.insert(0, str(lib_dir))

print(f"Ricerca modulo in: {module_dir}")

try:
    # Usa il nome definito nel tuo PYBIND11_MODULE
    import machine_learning_module as ml
    print("✓ Test Import Python: SUCCESS")
    print(f"✓ Percorso file: {ml.__file__}")
except ImportError as e:
    print(f"✗ Test Import Python: FAILED")
    print(f"  Errore: {e}")
    sys.exit(1)