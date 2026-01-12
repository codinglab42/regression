import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

from utils.helpers.lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
from utils.helpers.lab_utils_common import compute_cost

#plt.style.use('./deepleaarning.mplstyle')



#def old_main():
#
#    print(f"🚀 Inizio del main cost function of house price prediction")
#
#
#    DATASET = os.path.join(project_root, "dataset", "housePrice.csv")
#    COLS=["size", "price"]
#
#    # Read housing dataset
#    housingData = pd.read_csv(DATASET, usecols=COLS)
#
#    # print(housingData.head(5))
#
#    # Axis
#    x = housingData["size"]
#    y = housingData["price"]
#
#
#    rng = np.array([200-200,200+200])
#    tmp_b = 100
#
#    ar = np.arange(*rng, 5)
#    cost = np.zeros_like(ar)
#
#    for i in range(len(ar)):
#        tmp_w = ar[i]
#        cost [i]= compute_cost(x, y, tmp_w, tmp_b)
#
#    @interact(w=(rng[0], rng[1], 10), continuous_update=False)
#    def func(w=150):
#
#        f_wb = np.dot(x, y) + tmp_b
#
#        fig, ax = plt.subplot(1, 2, constrayned_layout = True, figsize = (8,4))
#        fig.canvas.toolbar_position = 'bottom'



def main():

    #x = np.array([1.0, 2.0])
    #y = np.array([300.0, 500.0])

    #x = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
    #y = np.array([250, 300, 480, 430, 630, 730])

    DATASET = os.path.join(project_root, "dataset", "housePrice.csv")
    COLS=["size", "price"]

    # Read housing dataset
    housingData = pd.read_csv(DATASET, usecols=COLS)

    # print(housingData.head(5))

    # Axis
    x = housingData["size"]
    y = housingData["price"]

    plt_intuition(x, y)

    plt.close('all')
    fig, ax, dyn_items = plt_stationary(x, y)
    updater = plt_update_onclick(fig, ax, x, y, dyn_items)

    soup_bowl()


        


if __name__ == "__main__":
    main()






