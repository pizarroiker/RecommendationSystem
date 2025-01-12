import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import BookList from '../BookList';
import { Spinner } from 'react-bootstrap';

const UserPage = () => {
  const { userId } = useParams(); // Para obtener el ID del usuario de la URL
  const navigate = useNavigate(); // Para manejar la navegación
  const [ratedBooks, setRatedBooks] = useState([]); // Estado para los libros valorados
  const [recommendedBooks, setRecommendedBooks] = useState([]); // Estado para los libros recomendados
  const [isLoadingRated, setIsLoadingRated] = useState(true); // Estado de carga para libros valorados
  const [isLoadingRecommended, setIsLoadingRecommended] = useState(true); // Estado de carga para recomendaciones
  const [errorRated, setErrorRated] = useState(null); // Estado para manejar errores en libros valorados
  const [errorRecommended, setErrorRecommended] = useState(null); // Estado para manejar errores en recomendaciones

  // useEffect para obtener los libros valorados
  useEffect(() => {
    const fetchRatedBooks = async () => {
      setIsLoadingRated(true); // Activar pantalla de carga
      try {
        const ratedResponse = await fetch(`http://localhost:5000/user/${userId}/ratings`);
        if (!ratedResponse.ok) {
          throw new Error("Error fetching rated books");
        }
        const ratedData = await ratedResponse.json();
        setRatedBooks(ratedData.ratedBooks);
      } catch (error) {
        setErrorRated(error.message);
      } finally {
        setIsLoadingRated(false); // Desactivar pantalla de carga
      }
    };

    fetchRatedBooks(); // Ejecuta solo cuando cambia el userId
  }, [userId]);

  // useEffect para obtener las recomendaciones
  useEffect(() => {
    const fetchRecommendedBooks = async () => {
      setIsLoadingRecommended(true); // Activar pantalla de carga
      try {
        const recommendedResponse = await fetch(`http://localhost:5000/user/${userId}/recommendations`);
        if (!recommendedResponse.ok) {
          throw new Error("Error fetching recommended books");
        }
        const recommendedData = await recommendedResponse.json();
        setRecommendedBooks(recommendedData.recommendedBooks);
      } catch (error) {
        setErrorRecommended(error.message);
      } finally {
        setIsLoadingRecommended(false); // Desactivar pantalla de carga
      }
    };

    fetchRecommendedBooks(); // Ejecuta solo cuando cambia el userId
  }, [userId]);

  // Si todavía están cargando los libros valorados o las recomendaciones
  if (isLoadingRated || isLoadingRecommended) {
    return (
      <div className="d-flex justify-content-center align-items-center" style={{ height: '100vh' }}>
        <Spinner animation="border" role="status">
          <span className="sr-only">Loading...</span>
        </Spinner>
      </div>
    );
  }

  // Manejar los errores en ambas peticiones
  if (errorRated || errorRecommended) {
    return <p>Error: {errorRated || errorRecommended}</p>;
  }

  return (
    <div className="container mt-5">
      <div className="d-flex justify-content-between">
        <button className="btn btn-secondary" onClick={() => navigate('/')}>
          Volver atrás
        </button>
      </div>

      <h2 className="text-center mt-4">Bienvenido, Usuario {userId}</h2>

      <div className="mt-4">
        <h3 className="text-center">Rated Books</h3>
        {/* Mostrar lista de libros valorados */}
        <BookList books={ratedBooks} canRate={false} />
      </div>

      <div className="mt-5">
        <h3 className="text-center">Recommended Books</h3>
        {/* Mostrar lista de libros recomendados */}
        <BookList books={recommendedBooks} canRate={true} userId={userId} />
      </div>
    </div>
  );
};

export default UserPage;

