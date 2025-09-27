import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [systemStatus, setSystemStatus] = useState('Loading...');
  const [apiData, setApiData] = useState(null);
  const [isTrading, setIsTrading] = useState(false);

  useEffect(() => {
    checkSystemHealth();
    fetchApiSummary();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await axios.get('/health');
      setSystemStatus('System Healthy ?');
    } catch (error) {
      setSystemStatus('System Error ?');
    }
  };

  const fetchApiSummary = async () => {
    try {
      const response = await axios.get('/api/summary');
      setApiData(response.data);
    } catch (error) {
      setApiData({ error: 'API not available' });
    }
  };

  const handleStartTrading = async () => {
    try {
      await axios.post('/api/trading/start');
      setIsTrading(true);
    } catch (error) {
      alert('Failed to start trading');
    }
  };

  const handleStopTrading = async () => {
    try {
      await axios.post('/api/trading/stop');
      setIsTrading(false);
    } catch (error) {
      alert('Failed to stop trading');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>?? AI/ML Trader Bot</h1>
        
        <div className="status-section">
          <h2>System Status</h2>
          <p className="status">{systemStatus}</p>
        </div>

        <div className="api-section">
          <h2>Portfolio Overview</h2>
          {apiData ? (
            <div className="portfolio-stats">
              <div className="stat">
                <span className="label">Total Accounts:</span>
                <span className="value">{apiData.total_accounts || 0}</span>
              </div>
              <div className="stat">
                <span className="label">Active Strategies:</span>
                <span className="value">{apiData.active_strategies || 0}</span>
              </div>
              <div className="stat">
                <span className="label">Total P&L:</span>
                <span className="value">${apiData.total_pnl || 0}</span>
              </div>
              <div className="stat">
                <span className="label">Status:</span>
                <span className="value">{apiData.status || 'Demo Mode'}</span>
              </div>
            </div>
          ) : (
            <p>Loading portfolio data...</p>
          )}
        </div>

        <div className="trading-controls">
          <h2>Trading Controls</h2>
          <div className="buttons">
            <button 
              onClick={handleStartTrading}
              disabled={isTrading}
              className={`btn ${isTrading ? 'btn-disabled' : 'btn-start'}`}
            >
              {isTrading ? 'Trading Active' : 'Start Trading'}
            </button>
            <button 
              onClick={handleStopTrading}
              disabled={!isTrading}
              className={`btn ${!isTrading ? 'btn-disabled' : 'btn-stop'}`}
            >
              Stop Trading
            </button>
          </div>
        </div>

        <div className="system-info">
          <h2>System Information</h2>
          <p>AI/ML Trading Bot v1.0.0 - FINAL FIXED VERSION</p>
          <p>Multi-broker algorithmic trading platform</p>
          <p>React Frontend + Flask Backend + Celery Workers</p>
        </div>
      </header>
    </div>
  );
}

export default App;
