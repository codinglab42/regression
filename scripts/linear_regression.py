#!/usr/bin/env python3
"""
Visualizzazione Regressione Lineare stile Andrew Ng
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Aggiungi path alla libreria
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import pymlalgorithms as ml

def create_sample_data():
    """Crea dati di esempio come nel corso di Andrew Ng"""
    # y = 3 + 2x + rumore
    np.random.seed(42)
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = 3 + 2 * X + np.random.randn(10) * 2  # Rumore gaussiano
    
    return X.tolist(), y.tolist()

def plot_data_and_model(X, y, model, title="Regressione Lineare"):
    """Plot dati e linea di regressione"""
    plt.figure(figsize=(12, 8))
    
    # Plot dati originali
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, color='red', marker='x', label='Dati training')
    
    # Plot linea di regressione
    X_min, X_max = min(X), max(X)
    X_line = np.linspace(X_min - 1, X_max + 1, 100)
    y_pred = [model.predict_single(x) for x in X_line]
    
    plt.plot(X_line, y_pred, color='blue', linewidth=2, 
             label=f'y = {model.theta0:.2f} + {model.theta1:.2f}x')
    
    plt.xlabel('x (feature)')
    plt.ylabel('y (target)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Plot funzione costo durante training
    plt.subplot(2, 2, 2)
    cost_history = model.cost_history
    plt.plot(range(len(cost_history)), cost_history, color='green')
    plt.xlabel('Numero iterazioni')
    plt.ylabel('Costo J(θ₀, θ₁)')
    plt.title('Funzione Costo durante Gradient Descent')
    plt.grid(True)
    
    # Plot contorno della funzione costo (stile Andrew Ng)
    plt.subplot(2, 2, 3)
    if len(model.theta0_history) > 0:
        # Mostra il percorso di gradient descent
        plt.plot(model.theta0_history, model.theta1_history, 'rx-', markersize=5, linewidth=1)
        plt.xlabel('θ₀')
        plt.ylabel('θ₁')
        plt.title('Percorso Gradient Descent nello spazio dei parametri')
        plt.grid(True)
    
    # Plot predizioni vs valori reali
    plt.subplot(2, 2, 4)
    y_pred = model.predict(X)
    plt.scatter(y, y_pred, color='purple')
    
    # Linea ideale y = x
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Valori reali (y)')
    plt.ylabel('Predizioni (ŷ)')
    plt.title('Predizioni vs Valori Reali')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_cost_function_3d(model, X, y):
    """Visualizzazione 3D della funzione costo (come Andrew Ng)"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Calcola costo per un range di valori di theta0 e theta1
    theta0_vals = np.linspace(-10, 10, 50)
    theta1_vals = np.linspace(-5, 5, 50)
    
    # Meshgrid per superficie 3D
    theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)
    cost_grid = np.zeros_like(theta0_grid)
    
    # Calcola costo per ogni combinazione
    m = len(X)
    for i in range(theta0_grid.shape[0]):
        for j in range(theta0_grid.shape[1]):
            theta0_temp = theta0_grid[i, j]
            theta1_temp = theta1_grid[i, j]
            cost = 0
            for k in range(m):
                prediction = theta0_temp + theta1_temp * X[k]
                error = prediction - y[k]
                cost += error * error
            cost_grid[i, j] = cost / (2 * m)
    
    # Plot 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Superficie
    surf = ax.plot_surface(theta0_grid, theta1_grid, cost_grid, 
                          cmap='viridis', alpha=0.8)
    
    # Percorso gradient descent
    if len(model.theta0_history) > 0:
        # Calcola costo per ogni punto del percorso
        path_costs = []
        for t0, t1 in zip(model.theta0_history, model.theta1_history):
            cost = 0
            for k in range(m):
                prediction = t0 + t1 * X[k]
                error = prediction - y[k]
                cost += error * error
            path_costs.append(cost / (2 * m))
        
        ax.plot(model.theta0_history, model.theta1_history, path_costs, 
                'r-', linewidth=3, label='Gradient Descent Path')
        ax.scatter([model.theta0], [model.theta1], [path_costs[-1] if path_costs else 0], 
                  color='red', s=100, label='Soluzione finale')
    
    ax.set_xlabel('θ₀')
    ax.set_ylabel('θ₁')
    ax.set_zlabel('Costo J(θ₀, θ₁)')
    ax.set_title('Superficie della Funzione Costo (3D) - Stile Andrew Ng')
    ax.legend()
    
    plt.show()

def demo_regression():
    """Dimostrazione completa come Andrew Ng"""
    print("=" * 60)
    print("REGRESSIONE LINEARE - STILE ANDREW NG")
    print("=" * 60)
    
    # 1. Crea dati
    print("\n1. Creazione dati di esempio...")
    X, y = create_sample_data()
    print(f"   X = {X}")
    print(f"   y = {y}")
    
    # 2. Crea e allena modello
    print("\n2. Creazione modello...")
    model = ml.LinearRegressionOneVar(learning_rate=0.01, iterations=1000)
    
    print("\n3. Training con Gradient Descent...")
    print("   (simile alla spiegazione di Andrew Ng)")
    model.fit(X, y)
    
    # 3. Risultati
    print("\n4. Risultati del modello:")
    print(f"   θ₀ (intercetta): {model.theta0:.4f}")
    print(f"   θ₁ (pendenza): {model.theta1:.4f}")
    print(f"   Formula: y = {model.theta0:.4f} + {model.theta1:.4f}x")
    
    # 4. Predizioni
    print("\n5. Esempi di predizione:")
    test_values = [4.5, 7.5, 11.0]
    for x in test_values:
        pred = model.predict_single(x)
        print(f"   Per x = {x}: ŷ = {pred:.4f}")
    
    # 5. Visualizzazioni
    print("\n6. Generazione visualizzazioni...")
    
    # Plot 2D
    plot_data_and_model(X, y, model, "Regressione Lineare - Andrew Ng Style")
    
    # Plot 3D (opzionale, richiede matplotlib 3D)
    try:
        plot_cost_function_3d(model, X, y)
    except ImportError:
        print("   Nota: Plot 3D non disponibile (mpl_toolkits necessario)")
    
    # 6. Salva modello
    print("\n7. Salvataggio modello...")
    model.save_model("linear_regression_model.txt")
    print("   Modello salvato in 'linear_regression_model.txt'")
    
    print("\n" + "=" * 60)
    print("DIMOSTRAZIONE COMPLETATA!")
    print("=" * 60)

def interactive_example():
    """Esempio interattivo"""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    # Dati semplici
    X = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.35)
    
    # Plot dati
    ax.scatter(X, y, color='red', s=100, label='Dati')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Regressione Lineare Interattiva')
    ax.grid(True)
    ax.legend()
    
    # Linea iniziale
    line, = ax.plot([], [], 'b-', linewidth=2, label='Modello')
    
    # Sliders
    ax_theta0 = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_theta1 = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_cost = plt.axes([0.25, 0.1, 0.65, 0.03])
    
    slider_theta0 = Slider(ax_theta0, 'θ₀', -5.0, 5.0, valinit=0)
    slider_theta1 = Slider(ax_theta1, 'θ₁', -2.0, 2.0, valinit=0)
    
    # Calcola costo iniziale
    model = ml.LinearRegressionOneVar()
    
    def update(val):
        theta0 = slider_theta0.val
        theta1 = slider_theta1.val
        
        # Aggiorna modello
        # Nota: nella nostra implementazione attuale non possiamo settare
        # direttamente theta0/theta1, ma possiamo creare predizioni manuali
        
        # Predizioni
        X_line = np.linspace(0, 6, 100)
        y_line = [theta0 + theta1 * x for x in X_line]
        line.set_data(X_line, y_line)
        
        # Calcola costo
        cost = 0
        for xi, yi in zip(X, y):
            prediction = theta0 + theta1 * xi
            error = prediction - yi
            cost += error * error
        cost = cost / (2 * len(X))
        
        ax.set_title(f'Regressione Lineare: y = {theta0:.2f} + {theta1:.2f}x, Costo = {cost:.4f}')
        fig.canvas.draw_idle()
    
    slider_theta0.on_changed(update)
    slider_theta1.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    # Dimostrazione principale
    # demo_regression()
    
    # Per l'esempio interattivo, decommenta:
    interactive_example()