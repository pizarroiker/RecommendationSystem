from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# Inicializar la aplicación Flask
app = Flask(__name__)
CORS(app)

# Cargar los datos desde los CSVs
books_df = pd.read_csv('clean_datasets/books.csv')
ratings_df = pd.read_csv('clean_datasets/ratings.csv')

# Variable para el modelo de recomendación
model = None


def load_model():
    """
    Carga el modelo Siamese previamente entrenado desde un archivo .h5.
    Este modelo es utilizado para generar recomendaciones de libros.
    """
    global model
    model = tf.keras.models.load_model('recommender_model.h5')


def preprocess_data(books_df, tag_weight=1.4):
    """
    Preprocesa los datos de libros y crea una matriz de características que será utilizada
    para el modelo de recomendaciones. Este proceso incluye embeddings de autores, 
    transformación TF-IDF de etiquetas, y normalización de las fechas de publicación.

    Args:
        books_df (pd.DataFrame): DataFrame que contiene la información de los libros.
        tag_weight (float): Factor de ponderación para ajustar la importancia de las etiquetas.

    Returns:
        X (np.array): Matriz de características de los libros.
    """
    # Crear embeddings de autores
    unique_authors = books_df['authors'].unique()
    author_to_index = {author: idx for idx, author in enumerate(unique_authors)}
    author_embeddings = np.zeros((len(books_df), len(unique_authors)))
    
    for i, author in enumerate(books_df['authors']):
        author_embeddings[i, author_to_index[author]] = 1.0
        
    # Procesar etiquetas con TF-IDF
    books_df['tag_name'].fillna('', inplace=True)
    tfidf = TfidfVectorizer(stop_words=None)
    tags_tfidf_matrix = tfidf.fit_transform(books_df['tag_name']).toarray() * tag_weight
    
    # Normalizar las fechas de publicación
    books_df['year_normalized'] = (books_df['original_publication_year'] - books_df['original_publication_year'].min()) / (
        books_df['original_publication_year'].max() - books_df['original_publication_year'].min())
    
    # Concatenar todas las características (embeddings de autores, calificaciones, fechas y TF-IDF de etiquetas)
    X = np.hstack([
        author_embeddings,
        books_df[['average_rating', 'year_normalized']].values,
        tags_tfidf_matrix
    ])
    
    return X


@app.route('/user/<int:user_id>/ratings', methods=['GET'])
def get_rated_books(user_id):
    """
    Devuelve una lista de los libros valorados por el usuario identificado por el `user_id`.

    Args:
        user_id (int): ID del usuario.

    Returns:
        dict: JSON con los detalles de los libros valorados por el usuario, incluyendo título, autor y calificación.
    """
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    if user_ratings.empty:
        return jsonify({"ratedBooks": []})

    rated_books = pd.merge(user_ratings, books_df, on='book_id')[['book_id', 'title', 'authors', 'rating', 'small_image_url']]
    rated_books_list = rated_books.to_dict(orient='records')

    return jsonify({"ratedBooks": rated_books_list})


@app.route('/user/<int:user_id>/recommendations', methods=['GET'])
def get_recommendations(user_id):
    """
    Genera y devuelve una lista de recomendaciones de libros para el usuario con el ID proporcionado, 
    basándose en las valoraciones previas del usuario.

    Args:
        user_id (int): ID del usuario.

    Returns:
        dict: JSON con los libros recomendados, incluyendo título, autor y URL de la imagen.
    """
    n = 10
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    if user_ratings.empty:
        # Si el usuario no ha valorado libros, devolver los libros más populares
        print(f"User {user_id} has no ratings, returning popular books.")
        popular_books = get_popular_books(n)
        return jsonify({"recommendedBooks": popular_books})

    rated_books = user_ratings[['book_id', 'rating']]

    # Preprocesar los datos de los libros
    X = preprocess_data(books_df, 1.4)
    
    recommended_books = []

    # Obtener recomendaciones basadas en los libros valorados
    for _,row in rated_books.iterrows():
        book_id = row['book_id']
        rating = row['rating']
        recommended_indices = get_recommendations_with_batches(book_id, X, model, books_df, 1048, rating, n)
        recommended_books.extend(recommended_indices)
    
    # Preparar las recomendaciones en el formato correcto
    recommended_books = pd.merge(pd.DataFrame(recommended_books, columns=['book_id', 'distance']), books_df, on='book_id')
    recommended_books = recommended_books.sort_values('distance').drop_duplicates(subset=['book_id'], keep='first')
    recommended_books = recommended_books[~recommended_books['book_id'].isin(user_ratings['book_id'])]
    recommended_books = recommended_books[['book_id', 'title', 'authors', 'small_image_url']].to_dict(orient='records')

    return jsonify({"recommendedBooks": recommended_books[:n]})


def is_first_book_in_series(title):
    """
    Verifica si un libro es el primero de una serie basándose en su título mediante expresiones regulares.

    Args:
        title (str): Título del libro.

    Returns:
        bool: True si es el primer libro de una serie, False en caso contrario.
    """

    if re.search(r'#1\b(?![\.-])', title):
        return True
    elif re.search(r'#\d+', title):
        return False
    return True


def filter_duplicate_titles(recommended_indices, books_df):
    """
    Filtra títulos que son duplicados o libros que pertenecen a colecciones (boxsets, trilogías, etc.).

    Args:
        recommended_indices (list): Índices de los libros recomendados.
        books_df (pd.DataFrame): DataFrame con la información de los libros.

    Returns:
        list: Índices filtrados, sin duplicados.
    """
    palabras_clave = ["box set", "boxset", "complete collection", "boxed set", "omnibus", "trilogy", "quartet", "quintet"]
    filtered_recommendations = []

    for idx in recommended_indices:
        title = books_df.iloc[idx]['title']
        if not any(keyword in title.lower() for keyword in palabras_clave):
            filtered_recommendations.append(idx)
      
    return filtered_recommendations


def apply_diversity_filter(recommended_indices, books_df, book_id):
    """
    Aplica un filtro para evitar múltiples recomendaciones de libros del mismo autor,
    y da prioridad a los primeros libros de una serie.

    Args:
        recommended_indices (list): Índices de los libros recomendados.
        books_df (pd.DataFrame): DataFrame con la información de los libros.
        book_id (int): ID del libro base para evitar su inclusión en las recomendaciones.

    Returns:
        list: Índices filtrados para mantener la diversidad.
    """
    filtered_recommendations = []
    authors_seen = {}

    # Evitar recomendar el libro ya valorado
    recommended_indices = [idx for idx in recommended_indices if idx != book_id]

    for idx in recommended_indices:
        author = books_df.iloc[idx]['authors']
        title = books_df.iloc[idx]['title']

        if authors_seen.get(author, 0) < 3:
            if is_first_book_in_series(title):
                filtered_recommendations.append(idx)
                authors_seen[author] = authors_seen.get(author, 0) + 1

    return filtered_recommendations


def get_book_index_by_id(book_id, books_df):
    """
    Dado un book_id, devuelve el índice correspondiente en el DataFrame de libros.

    Args:
        book_id (int): ID del libro.
        books_df (pd.DataFrame): DataFrame con la información de los libros.

    Returns:
        int: Índice del libro en el DataFrame.

    Raises:
        ValueError: Si el book_id no se encuentra en el DataFrame.
    """
    try:
        return books_df.loc[books_df['book_id'] == book_id].index[0]
    except IndexError:
        raise ValueError(f"El book_id {book_id} no se encontró en el DataFrame")


def get_recommendations_with_batches(book_id, X, model, books_df, batch_size, rating, top_n):
    """
    Calcula las distancias entre el libro base y los demás libros en lotes (batches),
    y devuelve una lista de las mejores recomendaciones basadas en esas distancias.

    Args:
        book_id (int): ID del libro base.
        X (np.array): Matriz de características de los libros.
        model (tf.keras.Model): Modelo Siamese para calcular las distancias.
        books_df (pd.DataFrame): DataFrame con la información de los libros.
        batch_size (int): Tamaño del lote para el procesamiento en batches.
        rating (float): Calificación del usuario al libro base, usada como peso.
        top_n (int): Número máximo de libros recomendados a devolver.

    Returns:
        list: Lista de tuplas con el book_id y la distancia de los libros recomendados.
    """
    book_index = get_book_index_by_id(book_id, books_df)
    book_vector = X[book_index].reshape(1, -1)
    distances = []
    weight = 1.0/rating

    # Calcular distancias por lotes
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_size_actual = len(batch)
        book_batch = np.tile(book_vector, (batch_size_actual, 1))
        batch_distances = model.predict([book_batch, batch])
        distances.extend(batch_distances*weight)

    distances = np.array(distances).flatten()
    recommended_indices = distances.argsort()

    # Aplicar filtros de duplicados y diversidad
    filtered_recommendations = filter_duplicate_titles(recommended_indices, books_df)
    final_recommendations = apply_diversity_filter(filtered_recommendations, books_df, book_id=book_index)

    # Crear lista de tuplas (book_id, distancia)
    recommendations_with_distances = [(books_df.iloc[idx]['book_id'], distances[idx]) for idx in final_recommendations]

    return recommendations_with_distances[:top_n]


def get_popular_books(n=10):
    """
    Devuelve una lista de los libros más populares, basados en el número de valoraciones,
    aplicando filtros de duplicados y diversidad.

    Args:
        n (int): Número de libros populares a devolver.

    Returns:
        list: Lista de diccionarios con la información de los libros populares (book_id, título, autores, etc.).
    """
    popular_books_idx = filter_duplicate_titles(books_df.index, books_df)
    final_recommendations = apply_diversity_filter(popular_books_idx, books_df, book_id=None)
    popular_books_df = books_df.iloc[final_recommendations].sort_values('ratings_count', ascending=False)
    popular_books_info = popular_books_df[['book_id', 'title', 'authors', 'small_image_url']].to_dict(orient='records')

    return popular_books_info[:n]


if __name__ == '__main__':
    """
    Punto de entrada de la aplicación Flask. 
    - Se carga el modelo Siamese.
    - Se inicia el servidor Flask en modo debug en el puerto 5000.
    """
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
