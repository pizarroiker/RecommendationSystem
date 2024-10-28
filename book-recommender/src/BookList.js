import React from 'react';
import BookCard from './BookCard';

const BookList = ({ books, canRate, userId }) => {
  if (books.length === 0) {
    return <p className="text-center">No books found</p>;
  }

  return (
    <div className="row justify-content-center">  {/* Asegura que los libros estén alineados al centro */}
      {books.map((book) => (
        <div key={book.book_id} className="col-lg-3 col-md-4 col-sm-6 mb-4 d-flex justify-content-center">
          <BookCard book={book} canRate={canRate} userId={userId} />
        </div>
      ))}
    </div>
  );
};

export default BookList;
