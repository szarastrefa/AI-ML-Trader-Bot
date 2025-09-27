import React, { useState, useEffect } from 'react';
import { ApiService } from '../services/apiService';

function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isTrading, setIsTrading] = useState(false);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [summaryData, fullDashboard] = await Promise.all([
        ApiService.getSummary(),
        ApiService.getDashboard()
      ]);
      setSummary(summaryData);
      setDashboardData(fullDashboard);
      setIsTrading(summaryData.status === 'Live Trading');
      setError(null);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setError('Nie można połączyć się z systemem tradingowym');
    } finally {
      setLoading(false);
    }
  };

  const handleStartTrading = async () => {
    try {
      await ApiService.startTrading();
      setIsTrading(true);
      fetchDashboardData();
    } catch (error) {
      alert('Błąd przy uruchamianiu tradingu');
    }
  };

  const handleStopTrading = async () => {
    try {
      await ApiService.stopTrading();
      setIsTrading(false);
      fetchDashboardData();
    } catch (error) {
      alert('Błąd przy zatrzymywaniu tradingu');
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('pl-PL', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatPercentage = (value) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Ładowanie danych dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="error-container">
          <h2>⚠️ Błąd Połączenia</h2>
          <p>{error}</p>
          <button onClick={fetchDashboardData} className="btn btn-retry">
            Spróbuj Ponownie
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container dashboard-page">
      <div className="page-header">
        <h1>📊 Dashboard - Strona Główna</h1>
        <p>Skumulowana wartość wszystkich połączonych kont</p>
      </div>

      {/* Portfolio Summary Cards */}
      <div className="summary-section">
        <div className="summary-cards">
          <div className="summary-card total-value">
            <div className="card-icon">💰</div>
            <div className="card-content">
              <h3>Łączna Wartość Portfela</h3>
              <div className="card-value">{formatCurrency(summary?.portfolio_value || 0)}</div>
              <div className="card-change positive">
                Equity: {formatCurrency(summary?.total_equity || 0)}
              </div>
            </div>
          </div>

          <div className="summary-card total-pnl">
            <div className="card-icon">📈</div>
            <div className="card-content">
              <h3>Łączny Zysk/Strata</h3>
              <div className={`card-value ${(summary?.total_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
                {formatCurrency(summary?.total_pnl || 0)}
              </div>
              <div className="card-change">
                Dziś: {formatCurrency(summary?.daily_pnl || 0)}
              </div>
            </div>
          </div>

          <div className="summary-card accounts-count">
            <div className="card-icon">🏦</div>
            <div className="card-content">
              <h3>Podłączone Konta</h3>
              <div className="card-value">{summary?.total_accounts || 0}</div>
              <div className="card-change">
                Aktywne konta brokerskie
              </div>
            </div>
          </div>

          <div className="summary-card strategies-count">
            <div className="card-icon">🧠</div>
            <div className="card-content">
              <h3>Aktywne Strategie</h3>
              <div className="card-value">{summary?.active_strategies || 0}</div>
              <div className="card-change">
                AI/ML modele w użyciu
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Trading Controls */}
      <div className="trading-controls-section">
        <div className="controls-header">
          <h2>🎮 Kontrola Tradingu</h2>
          <div className="trading-status">
            Status: <span className={isTrading ? 'status-active' : 'status-inactive'}>
              {isTrading ? 'AKTYWNY' : 'ZATRZYMANY'}
            </span>
          </div>
        </div>
        <div className="controls-container">
          <div className="control-buttons">
            <button 
              onClick={handleStartTrading}
              disabled={isTrading}
              className={`btn btn-lg ${isTrading ? 'btn-disabled' : 'btn-start'}`}
            >
              {isTrading ? '🟢 Trading Aktywny' : '▶️ Uruchom Trading'}
            </button>
            <button 
              onClick={handleStopTrading}
              disabled={!isTrading}
              className={`btn btn-lg ${!isTrading ? 'btn-disabled' : 'btn-stop'}`}
            >
              {!isTrading ? '⏹️ Trading Zatrzymany' : '⏹️ Zatrzymaj Trading'}
            </button>
          </div>
          <div className="trading-stats">
            <div className="stat">
              <span className="label">Otwarte Pozycje:</span>
              <span className="value">{summary?.open_positions || 0}</span>
            </div>
            <div className="stat">
              <span className="label">Transakcje Dziś:</span>
              <span className="value">{summary?.trades_today || 0}</span>
            </div>
            <div className="stat">
              <span className="label">Wskaźnik Wygranych:</span>
              <span className="value">{summary?.win_rate || 0}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Connected Accounts Overview */}
      {dashboardData?.accounts && dashboardData.accounts.length > 0 && (
        <div className="accounts-overview">
          <h2>🏦 Przegląd Kont</h2>
          <div className="accounts-grid">
            {dashboardData.accounts.map(account => (
              <div key={account.account_id} className="account-card">
                <div className="account-header">
                  <h3>{account.broker}</h3>
                  <span className={`status ${account.status.toLowerCase()}`}>
                    {account.status}
                  </span>
                </div>
                <div className="account-details">
                  <div className="detail-row">
                    <span>Saldo:</span>
                    <span>{formatCurrency(account.balance)}</span>
                  </div>
                  <div className="detail-row">
                    <span>Equity:</span>
                    <span>{formatCurrency(account.equity)}</span>
                  </div>
                  <div className="detail-row">
                    <span>P&L:</span>
                    <span className={account.profit >= 0 ? 'positive' : 'negative'}>
                      {formatCurrency(account.profit)} ({formatPercentage(account.profit_percentage)})
                    </span>
                  </div>
                  <div className="detail-row">
                    <span>Ping:</span>
                    <span>{account.ping}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Active Strategies Overview */}
      {dashboardData?.strategies && dashboardData.strategies.length > 0 && (
        <div className="strategies-overview">
          <h2>🧠 Aktywne Strategie AI/ML</h2>
          <div className="strategies-grid">
            {dashboardData.strategies.filter(s => s.status === 'Active').map(strategy => (
              <div key={strategy.id} className="strategy-card active">
                <div className="strategy-header">
                  <h3>{strategy.name}</h3>
                  <span className="strategy-type">{strategy.type}</span>
                </div>
                <div className="strategy-stats">
                  <div className="stat-row">
                    <span>Symbol:</span>
                    <span className="symbol">{strategy.symbol}</span>
                  </div>
                  <div className="stat-row">
                    <span>P&L:</span>
                    <span className={strategy.profit >= 0 ? 'positive' : 'negative'}>
                      {formatCurrency(strategy.profit)}
                    </span>
                  </div>
                  <div className="stat-row">
                    <span>Confidence:</span>
                    <span>{strategy.confidence}%</span>
                  </div>
                  <div className="stat-row">
                    <span>Sygnał:</span>
                    <span className={`signal ${strategy.last_signal?.toLowerCase()}`}>
                      {strategy.last_signal}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Open Positions */}
      {dashboardData?.positions && dashboardData.positions.length > 0 && (
        <div className="positions-section">
          <h2>📈 Otwarte Pozycje</h2>
          <div className="positions-table">
            <div className="table-header">
              <span>Symbol</span>
              <span>Strona</span>
              <span>Wielkość</span>
              <span>Cena Wejścia</span>
              <span>Aktualna Cena</span>
              <span>P&L</span>
              <span>Strategia</span>
            </div>
            {dashboardData.positions.map(position => (
              <div key={position.id} className="table-row">
                <span className="symbol">{position.symbol}</span>
                <span className={`side ${position.side.toLowerCase()}`}>
                  {position.side}
                </span>
                <span>{position.size}</span>
                <span>{position.entry_price}</span>
                <span>{position.current_price}</span>
                <span className={position.profit >= 0 ? 'positive' : 'negative'}>
                  {formatCurrency(position.profit)} ({formatPercentage(position.profit_percentage)})
                </span>
                <span className="strategy-name">{position.strategy}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Market Overview */}
      {dashboardData?.market && (
        <div className="market-section">
          <h2>🌍 Przegląd Rynku</h2>
          <div className="market-grid">
            <div className="market-group">
              <h3>📈 Forex Majors</h3>
              <div className="market-items">
                {dashboardData.market.major_pairs?.map(pair => (
                  <div key={pair.symbol} className="market-item">
                    <span className="symbol">{pair.symbol}</span>
                    <span className="price">{pair.price}</span>
                    <span className={`change ${pair.change >= 0 ? 'positive' : 'negative'}`}>
                      {formatPercentage(pair.change_pct)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="market-group">
              <h3>₿ Kryptowaluty</h3>
              <div className="market-items">
                {dashboardData.market.crypto_pairs?.map(pair => (
                  <div key={pair.symbol} className="market-item">
                    <span className="symbol">{pair.symbol}</span>
                    <span className="price">{formatCurrency(pair.price)}</span>
                    <span className={`change ${pair.change >= 0 ? 'positive' : 'negative'}`}>
                      {formatPercentage(pair.change_pct)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="market-group">
              <h3>📉 Akcje</h3>
              <div className="market-items">
                {dashboardData.market.stocks?.map(pair => (
                  <div key={pair.symbol} className="market-item">
                    <span className="symbol">{pair.symbol}</span>
                    <span className="price">{formatCurrency(pair.price)}</span>
                    <span className={`change ${pair.change >= 0 ? 'positive' : 'negative'}`}>
                      {formatPercentage(pair.change_pct)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* System Information */}
      <div className="system-info-section">
        <h2>⚙️ Informacje o Systemie</h2>
        <div className="system-grid">
          <div className="system-card">
            <h3>AI/ML Trading Bot v1.0.0</h3>
            <p>Profesjonalna platforma do tradingu algorytmicznego</p>
            <div className="tech-stack">
              <span className="tech">React</span>
              <span className="tech">Flask</span>
              <span className="tech">Celery</span>
              <span className="tech">Docker</span>
              <span className="tech">PostgreSQL</span>
              <span className="tech">Redis</span>
            </div>
          </div>
          <div className="system-card">
            <h3>Status Systemu</h3>
            <div className="status-items">
              <div className="status-item">
                <span>Uptime:</span>
                <span>{summary?.uptime_hours || 0}h</span>
              </div>
              <div className="status-item">
                <span>Ostatnia Aktualizacja:</span>
                <span>{new Date(summary?.last_update || Date.now()).toLocaleTimeString()}</span>
              </div>
              <div className="status-item">
                <span>Status:</span>
                <span className="operational">{summary?.system_status || 'Operational'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;