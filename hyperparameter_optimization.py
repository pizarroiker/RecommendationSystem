import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# Cargar los datos de entrenamiento
train_data = pd.read_csv('train_dataset.csv')

X_train_user = train_data['user_index']
X_train_book = train_data['book_index']
X_train_features = train_data.iloc[:, 6:106]  # Características adicionales
y_train = train_data['normalized_rating']
num_users = train_data['user_index'].max() + 1
num_books = train_data['book_index'].max() + 1

# Función para construir el modelo con hiperparámetros
def build_model(hp):

    # Tamaño de los embeddings
    embedding_size = hp.Int('embedding_size', min_value=32, max_value=128, step=16)

    # Input para el filtrado colaborativo (usuarios y libros)
    user_input = keras.Input(shape=(1,), name='user_input')
    book_input = keras.Input(shape=(1,), name='book_input')

    # Embeddings para usuarios y libros
    user_embedding = layers.Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    book_embedding = layers.Embedding(input_dim=num_books, output_dim=embedding_size, name='book_embedding')(book_input)

    # Aplanar los embeddings
    user_vector = layers.Flatten()(user_embedding)
    book_vector = layers.Flatten()(book_embedding)

    # Concatenación de los embeddings
    collaborative_vector = layers.Concatenate()([user_vector, book_vector])

    # Input para características adicionales del libro
    book_features_input = keras.Input(shape=(100,), name='book_features_input')

    # Red densa para procesar las características del contenido del libro
    x = layers.Dense(hp.Int('units_1', min_value=64, max_value=256, step=32), activation='relu')(book_features_input)
    x = layers.Dense(hp.Int('units_2', min_value=32, max_value=128, step=16), activation='relu')(x)

    # Combinar las representaciones colaborativa y de características de contenido
    combined_vector = layers.Concatenate()([collaborative_vector, x])

    # Capas adicionales
    for i in range(hp.Int('num_layers', 1, 3)):
        combined_vector = layers.Dense(hp.Int(f'units_{i+3}', min_value=32, max_value=128, step=16), activation='relu')(combined_vector)

    output = layers.Dense(1, activation='sigmoid')(combined_vector)

    # Compilar el modelo
    model = keras.Model(inputs=[user_input, book_input, book_features_input], outputs=output)

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    return model

# Definir la búsqueda de hiperparámetros
tuner = kt.Hyperband(
    build_model,
    objective='val_mean_absolute_error',
    max_epochs=10,
    factor=3,
    directory='hyperparam_tuning',
    project_name='book_recommender_advanced'
)

# Callback para detener el entrenamiento temprano si la validación no mejora
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Correr la búsqueda de los mejores hiperparámetros
tuner.search([X_train_user, X_train_book, X_train_features], y_train,
             epochs=10,
             validation_split=0.2,
             callbacks=[stop_early])

# Obtener los mejores hiperparámetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Mejor tamaño de embedding: {best_hps.get('embedding_size')}")
print(f"Mejor número de unidades en la primera capa: {best_hps.get('units_1')}")
print(f"Mejor número de capas: {best_hps.get('num_layers')}")
print(f"Mejor tasa de aprendizaje: {best_hps.get('learning_rate')}")
