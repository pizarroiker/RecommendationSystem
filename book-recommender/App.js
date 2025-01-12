import React, { useState } from "react";
import BookList from "./Booklist";

function App() {
  const [userId, setUserId] = useState("");
  const [ratedBooks, setRatedBooks] = useState([]);
  const [recommendedBooks, setRecommendedBooks] = useState([]);
  const [error, setError] = useState("");

  const handleUserIdChange = (e) => {
    setUserId(e.target.value);
  };

  const fetchRatedBooks = async () => {
    try {
      const response = await fetch(`http://localhost:5000/user/${userId}/ratings`);
      if (!response.ok) throw new Error("Error fetching rated books");
      const data = await response.json();
      setRatedBooks(data.ratedBooks);
    } catch (err) {
      setError("Could not fetch rated books.");
    }
  };

  const fetchRecommendedBooks = async () => {
    try {
      const response = await fetch(`http://localhost:5000/user/${userId}/recommendations`);
      if (!response.ok) throw new Error("Error fetching recommended books");
      const data = await response.json();
      setRecommendedBooks(data.recommendedBooks);
    } catch (err) {
      setError("Could not fetch recommended books.");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    await fetchRatedBooks();
    await fetchRecommendedBooks();
  };

  return (
    <div className="container">
      <h1 className="mt-5">Book Recommender System</h1>
      
      <form onSubmit={handleSubmit} className="mt-3">
        <div className="form-group">
          <label htmlFor="userId">User ID:</label>
          <input
            type="text"
            className="form-control"
            id="userId"
            value={userId}
            onChange={handleUserIdChange}
            placeholder="Enter User ID"
          />
        </div>
        <button type="submit" className="btn btn-primary">Get Recommendations</button>
      </form>

      {error && <div className="alert alert-danger mt-3">{error}</div>}

      <div className="mt-5">
        <h2>Rated Books</h2>
        <BookList books={ratedBooks} />
      </div>

      <div className="mt-5">
        <h2>Recommended Books</h2>
        <BookList books={recommendedBooks} />
      </div>
    </div>
  );
}

export default App;
