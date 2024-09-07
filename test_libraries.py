# Librerías necesarias
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn as sns

def test_libraries():
    print("Verificación de las librerías necesarias para el proyecto:\n")

    # Verificar TensorFlow y Keras
    try:
        print(f"TensorFlow versión: {tf.__version__}")
        print(f"Keras versión: {keras.__version__}")
    except Exception as e:
        print(f"Error con TensorFlow o Keras: {e}")

    # Verificar pandas
    try:
        print(f"Pandas versión: {pd.__version__}")
    except Exception as e:
        print(f"Error con Pandas: {e}")

    # Verificar numpy
    try:
        print(f"NumPy versión: {np.__version__}")
    except Exception as e:
        print(f"Error con NumPy: {e}")

    # Verificar scikit-learn
    try:
        print(f"Scikit-learn versión: {sklearn.__version__}")
    except Exception as e:
        print(f"Error con Scikit-learn: {e}")

    # Verificar matplotlib
    try:
        print(f"Matplotlib versión: {matplotlib.__version__}")
    except Exception as e:
        print(f"Error con Matplotlib: {e}")

    # Verificar seaborn
    try:
        print(f"Seaborn versión: {sns.__version__}")
    except Exception as e:
        print(f"Error con Seaborn: {e}")

if __name__ == "__main__":
    test_libraries()