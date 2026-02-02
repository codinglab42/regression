# test_ml_module.py
import sys
import os

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build/pybinding'))

try:
    import machine_learning_module as ml
    print("✅ SUCCESS: Python module imported")
    print(f"   Module: {ml.__name__}")
    print(f"   Version: {ml.__version__}")
    
    # Basic test
    import numpy as np
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([1, 2], dtype=np.float64)
    
    lr = ml.LinearRegression()
    lr.fit(X, y)
    pred = lr.predict(X)
    print("   LinearRegression test: ✓")
    
    # Test Neural Network
    nn = ml.NeuralNetwork([2, 4, 1])
    nn.fit(X, y)
    nn_pred = nn.predict(X)
    print("   NeuralNetwork test: ✓")
    
    print("✅ All Python tests passed")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)