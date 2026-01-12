import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""Setup del path del progetto - da importare in tutti gli script"""
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)  # Torna alla root
    
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
print(f"🚀 Project root added to path: {project_root}")

import setup_project

from utils.helpers.import_lib import lib_pymlalgorithms_import


def prediction():
    # Importa la libreria
    ml = lib_pymlalgorithms_import()
    if ml is None:
        print("❌ impossibile importare la libreria")
        sys.exit(1)

    DATASET = os.path.join(project_root, "dataset", "housePrice.csv")
    COLS=["size", "price"]

    # Read housing dataset
    housingData = pd.read_csv(DATASET, usecols=COLS)

    # print(housingData.head(5))

    # Axis
    x = housingData["size"]
    y = housingData["price"]
    

    # Crea e addestra il modello
    model = ml.LinearRegressionOneVar(learning_rate=0.01, iterations=1000)
    model.fit(x, y)

    return model

def main():

    print(f"🚀 Inizio del main house price prediction")
#from ... import ml_util.models.LinearRegresssion import LinearRegressionOneVariable



    tmp_f_wb = prediction()


    print("sto per iniziare il plot")


    #plt_house_x(x, y, tmp_f_wb, plt)


    # Plot
    plt.title("Housing Price")
    plt.ylabel("Price")
    plt.xlabel("Size")

    # Plot the prediction line
    plt.plot(x, tmp_f_wb, c='b', label='prediction')

    # Plot the data
    plt.scatter(x,y, marker='x', c='r')


    # Plot one prediction point: 1200 size
    #tmp_f_wb = LinearRegressionOneVariable(1200, w, b)
    x_pred = 1200
    y_pred = w * x_pred + b

    # Point prediction
    plt.scatter(x_pred, y_pred, color="green", s = 100, label="Prediction for size=1200")
    plt.plot([x_pred, x_pred], [0, y_pred], color="green", linestyle="--")
    plt.plot([0, x_pred], [y_pred, y_pred], color="green", linestyle="--")

    # Disegno gli assi
    plt.xlim(0, x.max() + 200)
    plt.ylim(0, tmp_f_wb.max() + 200)
    #
    plt.axvline(0, color="green", linestyle="--", alpha=0.5)
    plt.axhline(0, color="green", linestyle="--", alpha=0.5)

    plt.text(x_pred + 200, y_pred - 120000, "Prediction:\nsize = 1200\nPrice = " + str(y_pred), fontsize = 6, bbox = dict(facecolor = 'white', alpha = 0.8))

    plt.show(block = True)


if __name__ == "__main__":
    main()