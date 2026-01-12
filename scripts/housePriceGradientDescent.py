import os
import sys

import math, copy
import numpy as np
import matplotlib.pyplot as plt



def setup_project():
    """Setup del path del progetto - da importare in tutti gli script"""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)  # Torna alla root
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"🚀 Project root added to path: {project_root}")
    return project_root

# Esegui automaticamente
project_root = setup_project()

from utils.helpers.lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
from utils.models.computeCostFunction import compute_cost
from utils.models.computeGradient import compute_gradient

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    w = copy.deepcopy(w_in) # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing


def main():

    print(f"🚀 Inizio del main gradient descent of house price prediction")

    x = np.array([1.0, 2.0])
    y = np.array([300.0, 500.0])

    
    plt_gradients(x, y, compute_cost, compute_gradient)
    plt.show()


    w_init = 0
    b_init = 0
    iterations = 10000
    tmp_alpha = 1.0e-2

    #X = x.reshape(1, -1)
    w_final, b_final, J_hist, p_hist = gradient_descent(x, y, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

if __name__ == "__main__":
    main()
