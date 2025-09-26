import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isTrading, setIsTrading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('connecting');

  useEffect(() => {
    // Check backend health first
    checkBackendHealth();
    
    // Then start polling for updates
    const interval = setInterval(() => {
      if (connectionStatus === 'connected') {
        fetchSummary();
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [connectionStatus]);

  const checkBackendHealth = async () => {
    try {
      const response = await axios.get('/health');
      if (response.data.status === 'healthy') {
        setConnectionStatus('connected');
        fetchSummary();
      }
    } catch (err) {
      setConnectionStatus('error');
      setError('Cannot connect to backend server');
      setLoading(false);
    }
  };

  const fetchSummary = async () => {
    try {
      const response = await axios.get('/api/summary');
      setSummary(response.data);
      setError(null);
      setConnectionStatus('connected');
    } catch (err) {
      if (err.response?.status === 404) {
        // API endpoint not found, use mock data
        setSummary({
          total_equity: 0.0,
          total_pnl: 0.0,
          accounts: [],
          active_strategies: 0,
          last_updated: new Date().toISOString()
        });
        setError(null);
      } else {
        setError('Failed to fetch trading summary');
        setConnectionStatus('error');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleStartTrading = async () => {
    try {
      await axios.post('/api/trading/start');
      setIsTrading(true);
      setError(null);
      fetchSummary();
    } catch (err) {
      setError('Failed to start trading');
    }
  };

  const handleStopTrading = async () => {
    try {
      await axios.post('/api/trading/stop');
      setIsTrading(false);
      setError(null);
      fetchSummary();
    } catch (err) {
      setError('Failed to stop trading');
    }
  };

  if (loading) {
    return (
      <div className="app">
        <div className="loading">
          <div className="logo">
            🤖 AI/ML Trader Bot
          </div>
          <h2>Loading system...</h2>
          <div className="spinner"></div>
          <p>Connecting to backend server...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo-section">
          <h1>🤖 AI/ML Trading Bot</h1>
          <span className="version">v1.0.0</span>
        </div>
        
        <div className="header-controls">
          <div className={`connection-status ${connectionStatus}`}>
            {connectionStatus === 'connected' && '🟢 Backend Connected'}
            {connectionStatus === 'connecting' && '🟡 Connecting...'}
            {connectionStatus === 'error' && '🔴 Connection Error'}
          </div>
          
          <div className={`status-indicator ${isTrading ? 'active' : 'inactive'}`}>
            {isTrading ? '🟢 Trading Active' : '⏸️ Trading Stopped'}
          </div>
          
          {!isTrading ? (
            <button className="btn btn-start" onClick={handleStartTrading}>
              ▶️ Start Trading
            </button>
          ) : (
            <button className="btn btn-stop" onClick={handleStopTrading}>
              ⏹️ Stop Trading
            </button>
          )}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          ⚠️ {error}
          <button className="retry-btn" onClick={checkBackendHealth}>
            🔄 Retry Connection
          </button>
        </div>
      )}

      <main className="main-content">
        {/* System Status Overview */}
        <div className="status-section">
          <h2>📊 System Overview</h2>
          <div className="status-grid">
            <div className="status-card">
              <div className="status-icon">💼</div>
              <div className="status-info">
                <h3>Portfolio Value</h3>
                <div className="value">
                  ${summary?.total_equity?.toLocaleString('en-US', {minimumFractionDigits: 2}) || '0.00'}
                </div>
              </div>
            </div>
            
            <div className="status-card">
              <div className="status-icon">📈</div>
              <div className="status-info">
                <h3>Total P&L</h3>
                <div className={`value ${(summary?.total_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
                  {(summary?.total_pnl || 0) >= 0 ? '+' : ''}${summary?.total_pnl?.toLocaleString('en-US', {minimumFractionDigits: 2}) || '0.00'}
                </div>
              </div>
            </div>
            
            <div className="status-card">
              <div className="status-icon">🏦</div>
              <div className="status-info">
                <h3>Connected Brokers</h3>
                <div className="value">
                  {summary?.accounts?.length || 0}
                </div>
              </div>
            </div>
            
            <div className="status-card">
              <div className="status-icon">🤖</div>
              <div className="status-info">
                <h3>Active Strategies</h3>
                <div className="value">
                  {summary?.active_strategies || 0}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Trading Accounts */}
        {summary?.accounts?.length > 0 && (
          <div className="section">
            <h2>💳 Trading Accounts</h2>
            <div className="accounts-grid">
              {summary.accounts.map((account, index) => (
                <div key={index} className="account-card">
                  <div className="account-header">
                    <h4>{account.account_id}</h4>
                    <span className="broker-badge">{account.broker}</span>
                  </div>
                  <div className="account-stats">
                    <div className="stat">
                      <span className="label">Balance:</span>
                      <span className="value">${account.equity?.toFixed(2) || '0.00'}</span>
                    </div>
                    <div className="stat">
                      <span className="label">P&L:</span>
                      <span className={`value ${(account.pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
                        {(account.pnl || 0) >= 0 ? '+' : ''}${account.pnl?.toFixed(2) || '0.00'}
                      </span>
                    </div>
                    <div className="stat">
                      <span className="label">Positions:</span>
                      <span className="value">{account.positions_count || 0}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Getting Started */}
        {(!summary?.accounts || summary.accounts.length === 0) && (
          <div className="section">
            <h2>🚀 Getting Started</h2>
            <div className="getting-started">
              <div className="step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h3>Connect Your Broker</h3>
                  <p>Add your trading accounts from supported brokers (MetaTrader, CCXT crypto exchanges, Interactive Brokers)</p>
                </div>
              </div>
              
              <div className="step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h3>Configure Strategies</h3>
                  <p>Set up AI/ML trading strategies including SMC analysis, DOM analysis, and custom ML models</p>
                </div>
              </div>
              
              <div className="step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h3>Start Trading</h3>
                  <p>Enable automated trading and monitor performance in real-time</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* System Info */}
        <div className="section">
          <h2>⚙️ System Information</h2>
          <div className="system-info">
            <div className="info-grid">
              <div className="info-item">
                <span className="label">Last Updated:</span>
                <span className="value">
                  {summary?.last_updated ? 
                    new Date(summary.last_updated).toLocaleString() : 
                    'Never'
                  }
                </span>
              </div>
              
              <div className="info-item">
                <span className="label">Backend Status:</span>
                <span className={`value ${connectionStatus === 'connected' ? 'positive' : 'negative'}`}>
                  {connectionStatus === 'connected' ? '🟢 Online' : '🔴 Offline'}
                </span>
              </div>
              
              <div className="info-item">
                <span className="label">Platform:</span>
                <span className="value">Docker Container</span>
              </div>
              
              <div className="info-item">
                <span className="label">Features:</span>
                <span className="value">Multi-Broker + AI/ML + Real-time</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-left">
            <p>© 2025 AI/ML Trading Bot | Advanced Algorithmic Trading Platform</p>
          </div>
          <div className="footer-right">
            <span className="tech-stack">React + Flask + Docker + AI/ML</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;