#!/usr/bin/env python3
"""
Logistic Regression - Implementazione come Andrew Ng
Esempi: Classificazione binaria
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Aggiungi path alla libreria
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

try:
    import pymlalgorithms as ml
    print("✅ Libreria C++ importata con successo")
except ImportError as e:
    print(f"❌ Errore import: {e}")
    print("Prima compila: cd build && cmake .. && make")
    sys.exit(1)

def generate_linear_separable_data(n_samples=300):
    """Genera dati linearmente separabili"""
    np.random.seed(42)
    
    # Classe 0
    mean0 = [2, 2]
    cov0 = [[1, 0.5], [0.5, 1]]
    X0 = np.random.multivariate_normal(mean0, cov0, n_samples//2)
    y0 = np.zeros(n_samples//2)
    
    # Classe 1
    mean1 = [6, 6]
    cov1 = [[1, -0.5], [-0.5, 1]]
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples//2)
    y1 = np.ones(n_samples//2)
    
    # Combina
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Mescola
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

def generate_nonlinear_data(n_samples=300):
    """Genera dati non linearmente separabili (per testare limiti)"""
    np.random.seed(42)
    
    t = np.linspace(0, 4*np.pi, n_samples)
    
    # Spirale
    r = 0.5 + 0.1*t
    X0 = np.column_stack([r*np.cos(t) + 2, r*np.sin(t) + 2])
    X1 = np.column_stack([r*np.cos(t + np.pi) + 2, r*np.sin(t + np.pi) + 2])
    
    y0 = np.zeros(len(X0))
    y1 = np.ones(len(X1))
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

def convert_to_cpp_format(X_numpy, y_numpy):
    """Converte numpy array al formato C++"""
    X_list = X_numpy.tolist()
    y_list = y_numpy.tolist()
    return X_list, y_list

def demo_binary_classification():
    """Classificazione binaria semplice"""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION - CLASSIFICAZIONE BINARIA")
    print("="*60)
    
    # Genera dati
    X_np, y_np = generate_linear_separable_data(400)
    X, y = convert_to_cpp_format(X_np, y_np)
    
    print(f"Dati generati: {len(X)} campioni, {len(X[0])} features")
    print(f"Classe 0: {np.sum(y_np == 0)} campioni")
    print(f"Classe 1: {np.sum(y_np == 1)} campioni")
    
    # Split train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train_np, X_test_np = X_np[:split_idx], X_np[split_idx:]
    y_train_np, y_test_np = y_np[:split_idx], y_np[split_idx:]
    
    # Crea e allena modello
    print("\nTraining Logistic Regression...")
    model = ml.LogisticRegression(
        learning_rate=0.1,
        iterations=3000,
        lambda_=0.1  # Regolarizzazione L2
    )
    
    model.fit(X_train, y_train)
    
    # Risultati
    print(f"\n✅ Modello allenato!")
    print(f"Formula: {model.get_formula()}")
    print(f"Numero features: {model.num_features}")
    print(f"Parametri theta: {model.theta}")
    
    # Metriche sul test set
    accuracy = model.accuracy(X_test, y_test)
    precision, recall, f1 = model.precision_recall_f1(X_test, y_test)
    
    print(f"\n📊 Metriche sul test set:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    # Esempi di predizione
    print(f"\n🎯 Esempi di predizione:")
    test_samples = [
        [2.0, 2.5],  # Probabilmente classe 0
        [5.5, 5.0],  # Probabilmente classe 1
        [4.0, 4.0],  # Vicino al confine decisionale
    ]
    
    for x in test_samples:
        prob = model.predict_probability(x)
        pred_class = model.predict_class(x)
        print(f"  Punto {x}: Probabilità = {prob:.4f}, Classe = {pred_class}")
    
    # Decision boundary
    try:
        boundary_params = model.get_decision_boundary_2d()
        print(f"\n📐 Decision boundary:")
        print(f"  Equazione: x2 = {boundary_params[0]:.4f} + {boundary_params[1]:.4f} * x1")
    except Exception as e:
        print(f"\n⚠️  Non è possibile calcolare il decision boundary: {e}")
    
    # Visualizzazione
    plot_classification_results(X_train_np, y_train_np, X_test_np, y_test_np, model)
    plot_learning_curve(model)

def plot_classification_results(X_train, y_train, X_test, y_test, model):
    """Visualizza risultati classificazione"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Training data
    ax = axes[0, 0]
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                        cmap='coolwarm', alpha=0.6, edgecolors='k', s=50)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Training Data')
    plt.colorbar(scatter, ax=ax, label='Classe')
    ax.grid(True, alpha=0.3)
    
    # 2. Test data
    ax = axes[0, 1]
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                        cmap='coolwarm', alpha=0.6, edgecolors='k', s=50)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Test Data')
    plt.colorbar(scatter, ax=ax, label='Classe')
    ax.grid(True, alpha=0.3)
    
    # 3. Decision boundary
    ax = axes[0, 2]
    
    # Plot data
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
              cmap='coolwarm', alpha=0.3, edgecolors='k', s=30)
    
    # Create mesh for decision boundary
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict probabilities for mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()].tolist()
    try:
        Z = np.array(model.predict_probabilities(mesh_points))
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Decision Boundary (threshold = 0.5)')
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting\nboundary:\n{e}", 
                ha='center', va='center', transform=ax.transAxes)
    
    # 4. Probability distribution
    ax = axes[1, 0]
    predictions = model.predict_probabilities(X_train.tolist())
    ax.hist(predictions, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    ax.set_xlabel('Probabilità predetta')
    ax.set_ylabel('Frequenza')
    ax.set_title('Distribuzione Probabilità Predette')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Confusion matrix (semplice)
    ax = axes[1, 1]
    y_pred = model.predict_classes(X_test.tolist())
    y_true = y_test
    
    # Calcola confusion matrix
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Plot
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['True 0', 'True 1'])
    
    # Aggiungi valori
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 6. ROC Curve (semplice)
    ax = axes[1, 2]
    
    # Calcola TPR e FPR per diverse soglie
    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []
    
    for thresh in thresholds:
        y_pred = model.predict_classes(X_test.tolist(), threshold=thresh)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(model):
    """Plot curva di apprendimento"""
    cost_history = model.cost_history
    
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, linewidth=2)
    
    plt.xlabel('Iterazioni (ogni 100)', fontsize=12)
    plt.ylabel('Costo J(θ)', fontsize=12)
    plt.title('Curva di Apprendimento - Logistic Regression', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if len(cost_history) > 10:
        plt.annotate(f'Costo iniziale: {cost_history[0]:.4f}', 
                    xy=(0, cost_history[0]), xytext=(10, cost_history[0]*1.1),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.annotate(f'Costo finale: {cost_history[-1]:.4f}', 
                    xy=(len(cost_history)-1, cost_history[-1]), 
                    xytext=(len(cost_history)-50, cost_history[-1]*0.9),
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.show()

def demo_multiclass_one_vs_all():
    """Estensione One-vs-All per classificazione multi-classe"""
    print("\n" + "="*60)
    print("ONE-VS-ALL MULTI-CLASS CLASSIFICATION")
    print("="*60)
    
    # Genera dati per 3 classi
    np.random.seed(42)
    n_classes = 3
    n_samples = 100
    
    X_list = []
    y_list = []
    
    for class_idx in range(n_classes):
        mean = [class_idx * 3, class_idx * 3]
        cov = [[1, 0.5], [0.5, 1]]
        X_class = np.random.multivariate_normal(mean, cov, n_samples)
        y_class = np.ones(n_samples) * class_idx
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    X_np = np.vstack(X_list)
    y_np = np.hstack(y_list)
    
    # Mescola
    indices = np.random.permutation(len(X_np))
    X_np = X_np[indices]
    y_np = y_np[indices]
    
    print(f"Dati generati: {len(X_np)} campioni, {n_classes} classi")
    
    # One-vs-All: addestra un classificatore binario per ogni classe
    classifiers = []
    
    for class_idx in range(n_classes):
        print(f"\nTraining classifier per classe {class_idx}...")
        
        # Crea etichette binarie: 1 per la classe corrente, 0 per le altre
        y_binary = (y_np == class_idx).astype(float)
        X, y = convert_to_cpp_format(X_np, y_binary)
        
        # Addestra modello logistico
        model = ml.LogisticRegression(learning_rate=0.1, iterations=2000, lambda_=0.1)
        model.fit(X, y)
        
        classifiers.append(model)
        
        accuracy = model.accuracy(X, y)
        print(f"  Accuracy sul training set: {accuracy:.4f}")
    
    # Predizione multi-classe: scegli la classe con probabilità più alta
    print(f"\n📊 Predizione multi-classe:")
    
    test_points = [
        [0.5, 0.5],   # Probabilmente classe 0
        [3.5, 3.5],   # Probabilmente classe 1
        [6.5, 6.5],   # Probabilmente classe 2
        [2.0, 2.0],   # Tra classe 0 e 1
    ]
    
    for point in test_points:
        probabilities = []
        for class_idx, model in enumerate(classifiers):
            prob = model.predict_probability(point)
            probabilities.append(prob)
        
        predicted_class = np.argmax(probabilities)
        print(f"  Punto {point}:")
        print(f"    Probabilità: {probabilities}")
        print(f"    Classe predetta: {predicted_class}")
    
    # Visualizzazione
    plot_multiclass_results(X_np, y_np, classifiers)

def plot_multiclass_results(X, y, classifiers):
    """Visualizza risultati multi-classe"""
    plt.figure(figsize=(15, 5))
    
    # 1. Dati originali
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', 
                         alpha=0.6, edgecolors='k', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Dati Multi-Classe Originali')
    plt.colorbar(scatter, label='Classe')
    plt.grid(True, alpha=0.3)
    
    # 2. Decision boundaries
    plt.subplot(1, 3, 2)
    
    # Plot data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', 
               alpha=0.3, edgecolors='k', s=30)
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict for each point
    mesh_points = np.c_[xx.ravel(), yy.ravel()].tolist()
    
    predictions = []
    for point in mesh_points:
        probs = [model.predict_probability(point) for model in classifiers]
        predictions.append(np.argmax(probs))
    
    Z = np.array(predictions).reshape(xx.shape)
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Regioni di Decisione (One-vs-All)')
    plt.grid(True, alpha=0.3)
    
    # 3. Probabilità per ogni classificatore
    plt.subplot(1, 3, 3)
    
    # Prendi un campione rappresentativo
    sample_idx = 0
    point = X[sample_idx]
    true_class = int(y[sample_idx])
    
    probabilities = []
    for model in classifiers:
        prob = model.predict_probability(point.tolist())
        probabilities.append(prob)
    
    plt.bar(range(len(probabilities)), probabilities, 
           color=['red' if i == true_class else 'blue' for i in range(len(probabilities))])
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Classe')
    plt.ylabel('Probabilità')
    plt.title(f'Probabilità per punto {point} (vera classe: {true_class})')
    plt.xticks(range(len(probabilities)))
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def main():
    """Menu principale"""
    print("="*60)
    print("LOGISTIC REGRESSION - IMPLEMENTAZIONE C++")
    print("Come spiegato da Andrew Ng")
    print("="*60)
    
    while True:
        print("\nMenu:")
        print("  1. Classificazione Binaria (dati linearmente separabili)")
        print("  2. One-vs-All Multi-Class Classification (3 classi)")
        print("  3. Test con dati non lineari (mostra limiti)")
        print("  4. Esci")
        
        choice = input("\nScelta: ").strip()
        
        if choice == "1":
            demo_binary_classification()
        elif choice == "2":
            demo_multiclass_one_vs_all()
        elif choice == "3":
            demo_nonlinear_data()
        elif choice == "4":
            print("Arrivederci!")
            break
        else:
            print("Scelta non valida")

def demo_nonlinear_data():
    """Mostra limiti della Logistic Regression lineare"""
    print("\n" + "="*60)
    print("LIMITI: DATI NON LINEARMENTE SEPARABILI")
    print("="*60)
    
    # Genera dati non lineari
    X_np, y_np = generate_nonlinear_data(500)
    X, y = convert_to_cpp_format(X_np, y_np)
    
    print("Logistic Regression lineare non può separare dati a spirale")
    print("Servirebbero feature polinomiali o kernel")
    
    # Prova comunque
    model = ml.LogisticRegression(learning_rate=0.1, iterations=2000)
    model.fit(X, y)
    
    accuracy = model.accuracy(X, y)
    print(f"\nAccuracy su dati di training: {accuracy:.4f}")
    print("(Si aspettava accuracy bassa per dati non lineari)")
    
    # Visualizza
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()].tolist()
    Z = np.array(model.predict_probabilities(mesh_points))
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot data
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, 
               cmap='coolwarm', alpha=0.6, edgecolors='k', s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Limiti Logistic Regression Lineare\n(Dati a spirale non separabili linearmente)')
    plt.colorbar(label='Classe')
    plt.grid(True, alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    main()