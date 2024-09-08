import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cargar los datos de prueba
test_data = pd.read_csv('test_dataset.csv')

# Separar los datos de prueba
X_test_user = test_data['user_index']
X_test_book = test_data['book_index']
X_test_features = test_data.iloc[:, 6:106]  # Características adicionales
y_test = test_data['normalized_rating']

# Cargar el modelo entrenado
model = keras.models.load_model('recommender_model.keras')
model.load_weights('recommender.weights.h5')

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict([X_test_user, X_test_book, X_test_features])

# Calcular las métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
