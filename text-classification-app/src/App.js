import React, { useState } from 'react';
import './App.css';

function App() {
  const [textInput, setTextInput] = useState('');
  const [result, setResult] = useState(null);

  const handleInputChange = (e) => {
    setTextInput(e.target.value);
  };

  const handleSubmit = async () => {
    if (!textInput) {
      alert('Please enter text to classify.');
      return;
    }

    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ new_texts: [textInput] })
    });

    if (!response.ok) {
      alert('Failed to fetch the prediction');
      return;
    }

    const result = await response.json();
    setResult(result[0]);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="container">
      <div className="instruction">
        <p>
          Please copy and paste from these examples, or provide an error to test the capability of the BERT ML trainer when it comes to escalating an error to the mapped department. We have 3 departments that the trainer is picking from: <span style={{ color: 'blue', fontWeight: 'bold' }}>data, infra, and engineering.</span> Here are the examples of the errors:
        </p>
        <ul>
          <li>"database connection issues" (data)</li>
          <li>"firewall configuration problem" (infra)</li>
          <li>"network latency affecting all servers" (infra)</li>
          <li>"data extraction errors from database" (data)</li>
          <li>"VPN access not working for remote employees" (infra)</li>
          <li>"database schema mismatch" (data)</li>
          <li>"security breach detected in the network" (infra)</li>
          <li>"inconsistent data across multiple databases" (data)</li>
          <li>"Pipeline expired" (engineering)</li>
          <li>"out of memory" (engineering)</li>
        </ul>
      </div>
      <div className="form-group">
        <label htmlFor="text-input">Enter Text:</label>
        <input
          type="text"
          id="text-input"
          value={textInput}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
        />
      </div>
      <button className="btn" onClick={handleSubmit}>Predict</button>
      {result && (
          <div className="result">
              <p><strong><span style={{ color: 'blue', fontWeight: 'bold' }}>Text:</span></strong> <span style={{ color: 'black', fontWeight: 'bold' }}>{result.text}</span></p>
              <p><strong><span style={{ color: 'green', fontWeight: 'bold' }}>Predicted Department:</span></strong> <span style={{ color: 'black', fontWeight: 'bold' }}>{result.predicted_department}</span></p>
          </div>
      )}
    </div>
  );
}

export default App;