import React, { useState } from 'react';

const BookCard = ({ book, canRate, userId }) => {
  const [rating, setRating] = useState(book.rating || 0);

  const handleRating = async (newRating) => {
    setRating(newRating);
    if (canRate) {
      await fetch(`/user/${userId}/rate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ book_id: book.book_id, rating: newRating }),
      });
      window.location.reload();
    }
  };

  return (
    <div className="card mb-4" style={{ width: '15rem' }}>
      <img
        src={book.small_image_url}  // Imagen del libro
        className="card-img-top"
        alt={book.title}
        style={{ objectFit: 'contain', height: '250px' }}
      />
      <div className="card-body">
        <h5 className="card-title">{book.title}</h5>
        <p className="card-text">Autor(es): {book.authors}</p>
        <div className="star-rating">
          {[1, 2, 3, 4, 5].map((star) => (
            <i
              key={star}
              className={`fa-star ${star <= rating ? 'fas' : 'far'}`}  // 'fas' = estrella llena, 'far' = estrella vacía
              onClick={() => handleRating(star)}
              style={{ cursor: 'pointer', marginRight: '5px', color: 'orange' }}  // Ajustamos el cursor y el color
            ></i>
          ))}
        </div>
      </div>
    </div>
  );
};

export default BookCard;



