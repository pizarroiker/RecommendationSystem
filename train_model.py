import pandas as pd
from tensorflow import keras

# Cargar los datos de entrenamiento
train_data = pd.read_csv('train_dataset.csv')

# Separar los datos de entrenamiento
X_train_user = train_data['user_index']
X_train_book = train_data['book_index']
X_train_features = train_data.iloc[:, 6:106]  # Características adicionales
y_train = train_data['normalized_rating']

# Cargar el modelo guardado
model = keras.models.load_model('recommender_model.keras')

# Entrenar el modelo
model.fit([X_train_user, X_train_book, X_train_features], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Guardar los pesos entrenados (opcional)
model.save_weights('recommender.weights.h5')
print("Modelo entrenado y pesos guardados como recommender_weights.keras")

