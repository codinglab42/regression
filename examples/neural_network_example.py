import numpy as np
import machine_learning_module as ml
import matplotlib.pyplot as plt

def regression_example():
    """Esempio di regressione con Neural Network"""
    print("=== Regression Example ===")
    
    # Genera dati sintetici
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    true_weights = np.array([1.0, -0.5, 2.0, 0.3, -1.2])
    y = X @ true_weights + np.random.randn(n_samples) * 0.1
    
    # Crea e addestra rete neurale
    nn = ml.models.NeuralNetwork([5, 64, 32, 1], activation="relu", output_activation="linear")
    nn.set_epochs(100)
    nn.set_batch_size(32)
    nn.set_validation_split(0.2)
    nn.set_verbose(True)
    
    print(f"Training Neural Network with {nn.num_parameters:,} parameters...")
    nn.fit(X, y)
    
    # Predizioni
    y_pred = nn.predict(X)
    
    # Metriche
    mse = np.mean((y - y_pred) ** 2)
    r2 = nn.score(X, y)
    
    print(f"\nResults:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²: {r2:.6f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(nn.loss_history, label='Training Loss')
    if nn.val_loss_history:
        plt.plot(nn.val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_results.png')
    print("\nPlot saved to 'regression_results.png'")

def classification_example():
    """Esempio di classificazione con Neural Network"""
    print("\n=== Classification Example ===")
    
    # Genera dati sintetici non-lineari
    np.random.seed(42)
    n_samples = 2000
    X = np.random.randn(n_samples, 10)
    
    # Funzione decisione non-lineare
    decision = (X[:, 0] ** 2 + np.sin(X[:, 1]) + 
                X[:, 2] * X[:, 3] + np.exp(X[:, 4]))
    y = (decision > np.median(decision)).astype(np.float64)
    
    # Split train-test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crea e addestra rete neurale
    nn = ml.models.NeuralNetwork([10, 128, 64, 32, 1], 
                         activation="relu", 
                         output_activation="sigmoid")
    nn.set_epochs(150)
    nn.set_batch_size(64)
    nn.set_validation_split=0.2
    nn.set_loss_function("binary_crossentropy")
    nn.set_verbose(True)
    
    print(f"Training Neural Network with {nn.num_parameters:,} parameters...")
    nn.fit(X_train, y_train)
    
    # Predizioni e metriche
    y_pred_prob = nn.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = np.mean(y_pred == y_test)
    precision_recall_f1 = ml.MathUtils.precision_recall_f1(
        y_test, y_pred, model_type="NeuralNetwork"
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision_recall_f1[0]:.4f}")
    print(f"  Recall: {precision_recall_f1[1]:.4f}")
    print(f"  F1-Score: {precision_recall_f1[2]:.4f}")
    
    # Confronto con Logistic Regression
    logreg = ml.LogisticRegression(learning_rate=0.1, max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_acc = logreg.score(X_test, y_test)
    
    print(f"\nComparison with Logistic Regression:")
    print(f"  Neural Network Accuracy: {accuracy:.4f}")
    print(f"  Logistic Regression Accuracy: {logreg_acc:.4f}")
    print(f"  Improvement: {accuracy - logreg_acc:.4f}")

def model_persistence_example():
    """Esempio di salvataggio e caricamento modello"""
    print("\n=== Model Persistence Example ===")
    
    # Crea dati semplici
    X = np.random.randn(100, 3)
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2]
    
    # Crea e addestra modello
    model = ml.NeuralNetwork([3, 16, 1])
    model.set_epochs(10)
    model.set_verbose(False)
    model.fit(X, y)
    
    # Salva modello
    model.save("my_model.bin")
    print("Model saved to 'my_model.bin'")
    
    # Carica modello
    loaded_model = ml.NeuralNetwork()
    loaded_model.load("my_model.bin")
    print("Model loaded from 'my_model.bin'")
    
    # Confronta predizioni
    y_orig = model.predict(X)
    y_loaded = loaded_model.predict(X)
    
    diff = np.mean(np.abs(y_orig - y_loaded))
    print(f"Prediction difference: {diff:.10f}")
    
    if diff < 1e-10:
        print("✓ Model persistence test PASSED")
    else:
        print("✗ Model persistence test FAILED")
    
    # Pulisci
    import os
    os.remove("my_model.bin")

if __name__ == "__main__":
    print("ML Library v2.0 - Examples")
    print("===========================\n")
    
    try:
        regression_example()
        classification_example()
        model_persistence_example()
        
        print("\n=== All examples completed successfully ===")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()