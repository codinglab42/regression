#!/usr/bin/env python3
"""
Regressione Lineare Multi-Variabile
Esempio: Prezzo casa in funzione di dimensione, numero camere, età
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Aggiungi path alla libreria
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

try:
    import pymlalgorithms as ml
    print("✅ Libreria C++ importata con successo")
except ImportError as e:
    print(f"❌ Errore import: {e}")
    print("Prima compila: cd build && cmake .. && make")
    sys.exit(1)

def generate_sample_data(n_samples=1000):
    """Genera dati di esempio per regressione multi-variabile"""
    np.random.seed(42)
    
    # Features: [dimensione_mq, num_camere, distanza_dal_centro]
    dimensione = np.random.uniform(50, 200, n_samples)
    camere = np.random.randint(1, 6, n_samples)
    distanza = np.random.uniform(1, 30, n_samples)
    
    # Target: prezzo = base + coeff*dimensione + coeff*camere - coeff*distanza + rumore
    prezzo = (50000 + 2000 * dimensione + 15000 * camere - 5000 * distanza + 
              np.random.randn(n_samples) * 20000)
    
    # Crea dataset 2D per plotting
    X_2d = np.column_stack([dimensione, camere])
    y_2d = prezzo
    
    # Dataset 3D per esempio completo
    X_3d = np.column_stack([dimensione, camere, distanza])
    y_3d = prezzo
    
    return X_2d, y_2d, X_3d, y_3d

def convert_to_cpp_format(X_numpy, y_numpy):
    """Converte numpy array al formato C++ (lista di liste)"""
    X_list = X_numpy.tolist()
    y_list = y_numpy.tolist()
    return X_list, y_list

def demo_2d_regression():
    """Regressione con 2 features (per visualizzazione 3D)"""
    print("\n" + "="*60)
    print("REGRESSIONE LINEARE 2D")
    print("(Prezzo casa vs Dimensione + Numero Camere)")
    print("="*60)
    
    # Genera dati
    X_np, y_np, _, _ = generate_sample_data(500)
    X, y = convert_to_cpp_format(X_np, y_np)
    
    print(f"Dati generati: {len(X)} campioni, {len(X[0])} features")
    print(f"Feature 1 (dimensione): {X_np[:,0].min():.1f}-{X_np[:,0].max():.1f} mq")
    print(f"Feature 2 (camere): {X_np[:,1].min():.0f}-{X_np[:,1].max():.0f}")
    print(f"Target (prezzo): €{y_np.min():,.0f}-€{y_np.max():,.0f}")
    
    # Crea e allena modello
    model = ml.LinearRegressionMultiVar(learning_rate=0.01, iterations=2000)
    print("\nTraining in corso...")
    model.fit(X, y)
    
    # Risultati
    print(f"\n✅ Modello allenato!")
    print(f"Formula: {model.get_formula()}")
    print(f"Numero features: {model.num_features}")
    print(f"Parametri theta: {model.theta}")
    
    # Predizioni
    test_samples = [
        [100, 3],  # 100 mq, 3 camere
        [150, 4],  # 150 mq, 4 camere  
        [80, 2]    # 80 mq, 2 camere
    ]
    
    print("\n📊 Esempi di predizione:")
    for x in test_samples:
        pred = model.predict(x)
        print(f"  Casa {x[0]} mq, {x[1]} camere: €{pred:,.0f}")
    
    # Metriche
    y_pred = model.predict_batch(X)
    r2 = model.r2_score(X, y)
    mse = model.mse(X, y)
    
    print(f"\n📈 Metriche sul training set:")
    print(f"  R² score: {r2:.4f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: €{np.sqrt(mse):,.0f}")
    
    # Visualizzazione 3D
    plot_3d_regression(X_np, y_np, model)

def demo_3d_regression():
    """Regressione con 3 features"""
    print("\n" + "="*60)
    print("REGRESSIONE LINEARE 3D")
    print("(Prezzo casa vs Dimensione + Camere + Distanza)")
    print("="*60)
    
    # Genera dati con 3 features
    _, _, X_np, y_np = generate_sample_data(1000)
    X, y = convert_to_cpp_format(X_np, y_np)
    
    print(f"Dati generati: {len(X)} campioni, {len(X[0])} features")
    
    # Crea e allena modello
    model = ml.LinearRegressionMultiVar(learning_rate=0.005, iterations=3000)
    print("\nTraining in corso...")
    model.fit(X, y)
    
    # Risultati
    print(f"\n✅ Modello allenato!")
    print(f"Formula: {model.get_formula()}")
    
    # Analizza coefficienti
    print("\n📊 Analisi coefficienti:")
    features = ["Intercetta (€)", "Prezzo per mq (€/mq)", 
                "Valore per camera (€)", "Sconto per km dal centro (€/km)"]
    
    for i, (feat, theta) in enumerate(zip(features, model.theta)):
        print(f"  {feat:30s}: {theta:12,.2f}")
    
    # Importanza features
    print("\n📈 Importanza relativa delle features:")
    # Stima l'impatto medio di ogni feature
    feature_means = X_np.mean(axis=0)
    feature_effects = model.theta[1:] * feature_means
    
    total_effect = abs(feature_effects).sum()
    for i, effect in enumerate(feature_effects):
        importance = abs(effect) / total_effect * 100
        feature_names = ["Dimensione (mq)", "Numero camere", "Distanza dal centro (km)"]
        print(f"  {feature_names[i]:25s}: {importance:5.1f}%")
    
    # Predizioni esempio
    print("\n🏠 Esempi di predizione:")
    example_houses = [
        [120, 3, 10],  # 120 mq, 3 camere, 10 km
        [90, 2, 5],    # 90 mq, 2 camere, 5 km
        [200, 4, 25]   # 200 mq, 4 camere, 25 km
    ]
    
    for house in example_houses:
        pred = model.predict(house)
        print(f"  Casa {house[0]} mq, {house[1]} camere, {house[2]} km: €{pred:,.0f}")
    
    # Plot convergenza
    plot_convergence(model)

def plot_3d_regression(X_np, y_np, model):
    """Visualizzazione 3D della regressione"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Dati e piano di regressione
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Scatter dati
    ax1.scatter(X_np[:,0], X_np[:,1], y_np, 
                c=y_np, cmap='viridis', alpha=0.6, s=20)
    
    # Crea meshgrid per piano
    x1_range = np.linspace(X_np[:,0].min(), X_np[:,0].max(), 20)
    x2_range = np.linspace(X_np[:,1].min(), X_np[:,1].max(), 20)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    
    # Predizioni per piano
    y_grid = np.zeros_like(x1_grid)
    for i in range(x1_grid.shape[0]):
        for j in range(x1_grid.shape[1]):
            y_grid[i,j] = model.predict([x1_grid[i,j], x2_grid[i,j]])
    
    # Plot piano
    ax1.plot_surface(x1_grid, x2_grid, y_grid, 
                     alpha=0.5, cmap='coolwarm')
    
    ax1.set_xlabel('Dimensione (mq)')
    ax1.set_ylabel('Numero camere')
    ax1.set_zlabel('Prezzo (€)')
    ax1.set_title('Piano di Regressione 3D')
    
    # Plot 2: Predizioni vs Reali
    ax2 = fig.add_subplot(132)
    y_pred = model.predict_batch(X_np.tolist())
    ax2.scatter(y_np, y_pred, alpha=0.5)
    
    # Linea y = x
    min_val = min(y_np.min(), min(y_pred))
    max_val = max(y_np.max(), max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    ax2.set_xlabel('Valori Reali (€)')
    ax2.set_ylabel('Predizioni (€)')
    ax2.set_title('Predizioni vs Reali')
    ax2.grid(True)
    
    # Plot 3: Convergenza
    ax3 = fig.add_subplot(133)
    cost_history = model.cost_history
    ax3.plot(cost_history)
    ax3.set_xlabel('Iterazioni (x100)')
    ax3.set_ylabel('Costo J(θ)')
    ax3.set_title('Convergenza Gradient Descent')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_convergence(model):
    """Plot convergenza del costo"""
    plt.figure(figsize=(10, 6))
    
    cost_history = model.cost_history
    plt.plot(cost_history, linewidth=2)
    
    plt.xlabel('Iterazioni (ogni 100)', fontsize=12)
    plt.ylabel('Costo J(θ)', fontsize=12)
    plt.title('Convergenza Gradient Descent - Regressione Multi-Variabile', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Aggiungi annotazioni
    if len(cost_history) > 10:
        initial_cost = cost_history[0]
        final_cost = cost_history[-1]
        reduction = (initial_cost - final_cost) / initial_cost * 100
        
        plt.annotate(f'Costo iniziale: {initial_cost:.2e}', 
                    xy=(0, initial_cost), xytext=(10, initial_cost*1.1),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.annotate(f'Costo finale: {final_cost:.2e}\n'
                    f'Riduzione: {reduction:.1f}%', 
                    xy=(len(cost_history)-1, final_cost), 
                    xytext=(len(cost_history)-50, final_cost*0.9),
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.show()

def main():
    """Menu principale"""
    print("="*60)
    print("REGRESSIONE LINEARE MULTI-VARIABILE")
    print("Implementazione C++ con binding Python")
    print("="*60)
    
    while True:
        print("\nMenu:")
        print("  1. Regressione 2D (2 features + visualizzazione 3D)")
        print("  2. Regressione 3D (3 features + analisi completa)")
        print("  3. Test prestazioni")
        print("  4. Esci")
        
        choice = input("\nScelta: ").strip()
        
        if choice == "1":
            demo_2d_regression()
        elif choice == "2":
            demo_3d_regression()
        elif choice == "3":
            test_performance()
        elif choice == "4":
            print("Arrivederci!")
            break
        else:
            print("Scelta non valida")

def test_performance():
    """Test prestazioni su dataset di diverse dimensioni"""
    print("\n" + "="*60)
    print("TEST PRESTAZIONI")
    print("="*60)
    
    import time
    
    sizes = [100, 1000, 10000]
    n_features = 5  # Test con 5 features
    
    for size in sizes:
        print(f"\nTest con {size} campioni, {n_features} features...")
        
        # Genera dati
        np.random.seed(42)
        X_np = np.random.randn(size, n_features)
        y_np = np.random.randn(size)
        
        X, y = convert_to_cpp_format(X_np, y_np)
        
        # Timing
        model = ml.LinearRegressionMultiVar(learning_rate=0.01, iterations=100)
        
        start_time = time.time()
        model.fit(X, y)
        training_time = time.time() - start_time
        
        start_time = time.time()
        predictions = model.predict_batch(X)
        prediction_time = time.time() - start_time
        
        print(f"  Training time: {training_time:.3f}s")
        print(f"  Prediction time (batch): {prediction_time:.3f}s")
        print(f"  R² score: {model.r2_score(X, y):.4f}")

if __name__ == "__main__":
    main()