import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Brokers from './pages/Brokers';
import Strategies from './pages/Strategies';
import Settings from './pages/Settings';
import { ApiService } from './services/apiService';
import './App.css';

function App() {
  const [systemStatus, setSystemStatus] = useState('healthy');
  const [isTrading, setIsTrading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const health = await ApiService.getHealth();
      setSystemStatus(health.status);
      const status = await ApiService.getTradingStatus();
      setIsTrading(status.is_running);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch system status:', error);
      setSystemStatus('error');
    }
  };

  return (
    <Router>
      <div className="App">
        <header className="app-header">
          <div className="header-content">
            <div className="logo-section">
              <h1>🤖 AI/ML Trader Bot</h1>
              <div className="system-status">
                <span className={`status-indicator ${systemStatus}`}></span>
                <span className="status-text">
                  {systemStatus === 'healthy' ? 'System Operational' : 'System Issues'}
                </span>
              </div>
            </div>
            
            <div className="trading-status">
              <div className={`trading-indicator ${isTrading ? 'active' : 'inactive'}`}>
                {isTrading ? '🟢 Trading Active' : '🔴 Trading Stopped'}
              </div>
              <div className="last-update">
                Last Update: {lastUpdate.toLocaleTimeString()}
              </div>
            </div>
          </div>
          
          <nav className="main-navigation">
            <NavLink 
              to="/" 
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
              end
            >
              📊 Dashboard
            </NavLink>
            <NavLink 
              to="/brokers" 
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
            >
              🏦 Brokerzy / Konta
            </NavLink>
            <NavLink 
              to="/strategies" 
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
            >
              🧠 Strategie AI/ML
            </NavLink>
            <NavLink 
              to="/settings" 
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
            >
              ⚙️ Ustawienia
            </NavLink>
          </nav>
        </header>

        <main className="app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/brokers" element={<Brokers />} />
            <Route path="/strategies" element={<Strategies />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>

        <footer className="app-footer">
          <div className="footer-content">
            <p>AI/ML Trader Bot v1.0.0 - Professional Trading Platform</p>
            <div className="tech-stack">
              <span>React + Flask + Celery + Docker + AI/ML</span>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;