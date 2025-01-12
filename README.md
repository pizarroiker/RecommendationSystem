# Sistema de Recomendación de Libros

Este proyecto desarrolla un sistema de recomendación de libros basado en el conjunto de datos Goodbooks-10k, utilizando modelos de recomendación basados en contenido y redes neuronales. A continuación, se detallan los pasos para configurar, ejecutar y utilizar el sistema.

---

## Estructura del Proyecto

1. **Notebooks**:
   - `preprocessing_data.ipynb`: Realiza el preprocesamiento de datos y el análisis exploratorio.
   - `model_implementation.ipynb`: Desarrolla y exporta el modelo de recomendación.

2. **backend**:
   - Carpeta que contiene el backend desarrollado en Flask
   - `app.py`: Servidor backend desarrollado en Flask que expone el modelo para la interacción con el frontend.

4. **book-recommender**:
   - Carpeta con código React para la interfaz de usuario.

5. **Dataset**:
   - Carpeta que contiene el dataset Goodbooks-10k obtenido en la web de Kaggle
   - Referencia: https://www.kaggle.com/datasets/zygmunt/goodbooks-10k

---

## Requisitos Previos

Antes de empezar, asegúrate de tener instalados:

- **Python** (versión 3.8 o superior)
- **Node.js** (versión 14 o superior)
- **pip** (gestor de paquetes de Python)
- **npm** (gestor de paquetes de Node.js)

---

## Instalación y Funcionamiento

### 1. Clonar el Repositorio

- git clone [https://github.com/pizarroiker/RecommendationSystem.git](https://github.com/pizarroiker/RecommendationSystem.git)
- cd tu_repositorio

### 2. Instalar Librerías

- Para las librerías que no tengamos instaladas realizar: pip install nombre_libreria==número_version
- El número de versión es opcional pero se recomienda utilizar Anaconda como entorno ya que al instalar librerías es capaz de resolver dependencias y cuadrar versiones entre las diferentes librerías instaladas.

### 3. Ejecutar Notebooks

- Una vez instaladas las librerías ejecutar los notebooks en el orden:
      1. `preprocessing_data.ipynb`
      2. `model_implementation.ipynb`

### 4. Ejecutar Aplicación Web

- Ejecutar Backend
    - Ejecutar archivo app.py dentro de la carpeta backend
 
- Ejecutar Frontend (Recomendado: ejecutar desde Git Bash)
    - Asumimos que nos situamos en la ubicación del repositorio
    - cd book-recommender/
    - npm install
    - npm start
