import sys
import os

import numpy as np
import matplotlib.pyplot as plt

"""Setup del path del progetto - da importare in tutti gli script"""
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)  # Torna alla root
    
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
print(f"🚀 Project root added to path: {project_root}")

import setup_project

from utils.helpers.import_lib import lib_pymlalgorithms_import

# Importa la libreria
ml = lib_pymlalgorithms_import()
if ml is None:
    print("❌ impossibile importare la libreria")
    sys.exit(1)

# Esempio identico al corso di Andrew Ng
X = [1.0, 2.0, 3.0, 4.0, 5.0]  # Feature (una variabile)
y = [1.0, 2.0, 3.0, 4.0, 5.0]  # Target

# Crea e addestra il modello
model = ml.LinearRegressionOneVar(learning_rate=0.01, iterations=1000)
model.fit(X, y)

# Previsioni
new_X = [6.0, 7.0, 8.0]
predictions = model.predict(new_X)
print(f"Predictions for {new_X}: {predictions}")

# Parametri finali
print(f"Final parameters: theta0 = {model.theta0}, theta1 = {model.theta1}")

# Grafico della cost function
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(model.cost_history)
plt.title('Cost Function J(θ)')
plt.xlabel('Iterations')
plt.ylabel('Cost')

plt.subplot(1, 3, 2)
plt.plot(model.theta0_history, model.cost_history)
plt.title('Theta0 vs Cost')
plt.xlabel('Theta0')
plt.ylabel('Cost')

plt.subplot(1, 3, 3)
plt.plot(model.theta1_history, model.cost_history)
plt.title('Theta1 vs Cost')
plt.xlabel('Theta1')
plt.ylabel('Cost')

plt.tight_layout()
plt.show()