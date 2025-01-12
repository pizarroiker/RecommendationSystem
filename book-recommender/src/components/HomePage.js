import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const HomePage = () => {
  const [userId, setUserId] = useState('');
  const navigate = useNavigate();

  const handleInputChange = (e) => {
    setUserId(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (userId) {
      // Redirigir a la página de usuario con el ID del usuario
      navigate(`/user/${userId}`);
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center">Book Recommender System</h1>
      <form onSubmit={handleSubmit} className="mt-4">
        <div className="form-group">
          <label htmlFor="userId">User ID:</label>
          <input
            type="text"
            className="form-control"
            id="userId"
            value={userId}
            onChange={handleInputChange}
            placeholder="Enter your user ID"
            required
          />
        </div>
        <button type="submit" className="btn btn-primary mt-3">
          Get Recommendations
        </button>
      </form>
    </div>
  );
};

export default HomePage;
