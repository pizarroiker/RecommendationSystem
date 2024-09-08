import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Cargar los archivos de entrenamiento y evaluación
train_path = 'train_dataset.csv'
test_path = 'test_dataset.csv'

# Leer los archivos
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Definir algunas variables clave
num_users = train_data['user_index'].nunique()
num_books = train_data['book_index'].nunique()
embedding_size = 50  # Tamaño de los embeddings para usuarios y libros

# Características adicionales (del libro)
num_features = 100  # columnas de 0 a 99 son las características adicionales

# Input para el filtrado colaborativo (usuarios y libros)
user_input = keras.Input(shape=(1,), name='user_input')
book_input = keras.Input(shape=(1,), name='book_input')

# Embeddings para usuarios y libros
user_embedding = layers.Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
book_embedding = layers.Embedding(input_dim=num_books, output_dim=embedding_size, name='book_embedding')(book_input)

# Aplanar los embeddings
user_vector = layers.Flatten()(user_embedding)
book_vector = layers.Flatten()(book_embedding)

# Concatenación de los embeddings (filtrado colaborativo)
collaborative_vector = layers.Concatenate()([user_vector, book_vector])

# Input para características adicionales del libro
book_features_input = keras.Input(shape=(num_features,), name='book_features_input')

# Red densa para procesar las características del contenido del libro
x = layers.Dense(128, activation='relu')(book_features_input)
x = layers.Dense(64, activation='relu')(x)

# Combinar las representaciones colaborativa y de características de contenido
combined_vector = layers.Concatenate()([collaborative_vector, x])

# Pasar por capas densas adicionales
x = layers.Dense(64, activation='relu')(combined_vector)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)

# Definir el modelo final
model = keras.Model(inputs=[user_input, book_input, book_features_input], outputs=output)

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Resumen del modelo
model.summary()

model.save('recommender_model.keras')
