import React, { useState, useEffect } from 'react';
import { ApiService } from '../services/apiService';

function Settings() {
  const [connectedAccounts, setConnectedAccounts] = useState([]);
  const [systemInfo, setSystemInfo] = useState(null);
  const [logs, setLogs] = useState([]);
  const [supportedBrokers, setSupportedBrokers] = useState([]);
  const [showAddAccount, setShowAddAccount] = useState(false);
  const [loading, setLoading] = useState(true);
  const [newAccountForm, setNewAccountForm] = useState({
    broker: '',
    login: '',
    password: '',
    server: '',
    api_key: '',
    api_secret: ''
  });

  useEffect(() => {
    fetchSettingsData();
  }, []);

  const fetchSettingsData = async () => {
    try {
      const [accounts, sysInfo, brokers, systemLogs] = await Promise.all([
        ApiService.getAccounts(),
        ApiService.getSystemInfo(),
        ApiService.getSupportedBrokers(),
        ApiService.getSystemLogs()
      ]);
      setConnectedAccounts(accounts);
      setSystemInfo(sysInfo);
      setSupportedBrokers(brokers);
      setLogs(systemLogs);
    } catch (error) {
      console.error('Failed to fetch settings data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddAccount = async (e) => {
    e.preventDefault();
    try {
      await ApiService.addAccount(newAccountForm);
      setNewAccountForm({
        broker: '',
        login: '',
        password: '',
        server: '',
        api_key: '',
        api_secret: ''
      });
      setShowAddAccount(false);
      fetchSettingsData(); // Refresh data
      alert('Konto zostało pomyślnie dodane!');
    } catch (error) {
      console.error('Failed to add account:', error);
      alert('Błąd podczas dodawania konta');
    }
  };

  const handleRemoveAccount = async (accountId) => {
    if (window.confirm('Czy na pewno chcesz usunąć to konto?')) {
      try {
        await ApiService.removeAccount(accountId);
        fetchSettingsData(); // Refresh data
      } catch (error) {
        console.error('Failed to remove account:', error);
        alert('Błąd podczas usuwania konta');
      }
    }
  };

  const getLogLevelClass = (level) => {
    switch (level.toLowerCase()) {
      case 'error': return 'log-error';
      case 'warning': return 'log-warning';
      case 'info': return 'log-info';
      case 'debug': return 'log-debug';
      default: return '';
    }
  };

  const handleExportLogs = () => {
    const logsText = logs.map(log => 
      `[${log.timestamp}] ${log.level.toUpperCase()}: ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logsText], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trading-bot-logs-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Ładowanie ustawień...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container settings-page">
      <div className="page-header">
        <h1>⚙️ Ustawienia</h1>
        <p>Zarządzanie kontami, konfiguracja systemu i dokumentacja</p>
      </div>

      <div className="settings-sections">
        {/* Connected Accounts Section */}
        <section className="settings-section">
          <div className="section-header">
            <h2>🏦 Konta Podłączone</h2>
            <button 
              className="btn btn-primary"
              onClick={() => setShowAddAccount(!showAddAccount)}
            >
              {showAddAccount ? 'Anuluj' : '+ Dodaj Konto'}
            </button>
          </div>

          {showAddAccount && (
            <div className="add-account-form">
              <h3>Dodaj Nowe Konto Brokerskie</h3>
              <form onSubmit={handleAddAccount}>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Broker:</label>
                    <select
                      value={newAccountForm.broker}
                      onChange={(e) => setNewAccountForm({...newAccountForm, broker: e.target.value})}
                      required
                    >
                      <option value="">Wybierz brokera...</option>
                      {supportedBrokers.map(broker => (
                        <option key={broker.name} value={broker.name}>
                          {broker.display_name} ({broker.category})
                        </option>
                      ))}
                    </select>
                  </div>

                  {newAccountForm.broker === 'MetaTrader 5' && (
                    <>
                      <div className="form-group">
                        <label>Login:</label>
                        <input
                          type="text"
                          value={newAccountForm.login}
                          onChange={(e) => setNewAccountForm({...newAccountForm, login: e.target.value})}
                          placeholder="Numer konta MT5"
                          required
                        />
                      </div>
                      <div className="form-group">
                        <label>Hasło:</label>
                        <input
                          type="password"
                          value={newAccountForm.password}
                          onChange={(e) => setNewAccountForm({...newAccountForm, password: e.target.value})}
                          placeholder="Hasło do konta"
                          required
                        />
                      </div>
                      <div className="form-group">
                        <label>Serwer:</label>
                        <input
                          type="text"
                          value={newAccountForm.server}
                          onChange={(e) => setNewAccountForm({...newAccountForm, server: e.target.value})}
                          placeholder="np. Demo-MetaQuotes"
                          required
                        />
                      </div>
                    </>
                  )}

                  {['Binance', 'Coinbase Pro', 'Kraken', 'Alpaca'].includes(newAccountForm.broker) && (
                    <>
                      <div className="form-group">
                        <label>API Key:</label>
                        <input
                          type="text"
                          value={newAccountForm.api_key}
                          onChange={(e) => setNewAccountForm({...newAccountForm, api_key: e.target.value})}
                          placeholder="Klucz API"
                          required
                        />
                      </div>
                      <div className="form-group">
                        <label>API Secret:</label>
                        <input
                          type="password"
                          value={newAccountForm.api_secret}
                          onChange={(e) => setNewAccountForm({...newAccountForm, api_secret: e.target.value})}
                          placeholder="Sekret API"
                          required
                        />
                      </div>
                    </>
                  )}

                  {newAccountForm.broker === 'Interactive Brokers' && (
                    <>
                      <div className="form-group">
                        <label>Username:</label>
                        <input
                          type="text"
                          value={newAccountForm.login}
                          onChange={(e) => setNewAccountForm({...newAccountForm, login: e.target.value})}
                          placeholder="Nazwa użytkownika IBKR"
                          required
                        />
                      </div>
                      <div className="form-group">
                        <label>Hasło:</label>
                        <input
                          type="password"
                          value={newAccountForm.password}
                          onChange={(e) => setNewAccountForm({...newAccountForm, password: e.target.value})}
                          placeholder="Hasło do konta"
                          required
                        />
                      </div>
                    </>
                  )}
                </div>

                <div className="form-actions">
                  <button type="submit" className="btn btn-primary">
                    Dodaj Konto
                  </button>
                  <button 
                    type="button" 
                    className="btn btn-secondary"
                    onClick={() => setShowAddAccount(false)}
                  >
                    Anuluj
                  </button>
                </div>
              </form>
            </div>
          )}

          <div className="accounts-list">
            {connectedAccounts.length === 0 ? (
              <div className="no-accounts">
                <p>Brak podłączonych kont</p>
                <p>Użyj przycisku "+ Dodaj Konto" aby dodać pierwsze konto</p>
              </div>
            ) : (
              connectedAccounts.map(account => (
                <div key={account.account_id} className="account-item">
                  <div className="account-info">
                    <div className="account-header">
                      <h4>{account.broker}</h4>
                      <span className={`status ${account.status.toLowerCase()}`}>
                        {account.status}
                      </span>
                    </div>
                    <div className="account-details">
                      <span>ID: {account.account_id}</span>
                      <span>Equity: ${account.equity?.toFixed(2)}</span>
                      <span>Ping: {account.ping}</span>
                      <span>Waluta: {account.currency}</span>
                    </div>
                  </div>
                  <div className="account-actions">
                    <button 
                      className="btn btn-danger btn-sm"
                      onClick={() => handleRemoveAccount(account.account_id)}
                    >
                      Odłącz
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Documentation Section */}
        <section className="settings-section">
          <div className="section-header">
            <h2>📚 Dokumentacja</h2>
          </div>
          
          <div className="documentation-links">
            <div className="doc-category">
              <h3>Instrukcje Brokerów</h3>
              <div className="doc-links">
                <div className="doc-link-item">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <strong>Interactive Brokers - Konfiguracja TWS API</strong>
                    <p>Przewodnik po konfiguracji połączenia z TWS API dla IBKR</p>
                  </div>
                  <button className="btn btn-outline btn-sm" onClick={() => alert('Dokumentacja w przygotowaniu')}>
                    Czytaj
                  </button>
                </div>
                <div className="doc-link-item">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <strong>Binance - Jak uzyskać API Key</strong>
                    <p>Instrukcja tworzenia i konfiguracji kluczy API dla Binance</p>
                  </div>
                  <button className="btn btn-outline btn-sm" onClick={() => alert('Dokumentacja w przygotowaniu')}>
                    Czytaj
                  </button>
                </div>
                <div className="doc-link-item">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <strong>MetaTrader 5 - Połączenie z Pythonem</strong>
                    <p>Konfiguracja połączenia MT5 z systemem tradingowym</p>
                  </div>
                  <button className="btn btn-outline btn-sm" onClick={() => alert('Dokumentacja w przygotowaniu')}>
                    Czytaj
                  </button>
                </div>
                <div className="doc-link-item">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <strong>Kraken - Konfiguracja API</strong>
                    <p>Przewodnik po ustawieniu kluczy API w Kraken</p>
                  </div>
                  <button className="btn btn-outline btn-sm" onClick={() => alert('Dokumentacja w przygotowaniu')}>
                    Czytaj
                  </button>
                </div>
              </div>
            </div>

            <div className="doc-category">
              <h3>System i Strategie</h3>
              <div className="doc-links">
                <div className="doc-link-item">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <strong>Architektura Systemu</strong>
                    <p>Opis architektury AI/ML Trader Bot i jego komponentów</p>
                  </div>
                  <button className="btn btn-outline btn-sm" onClick={() => alert('Dokumentacja w przygotowaniu')}>
                    Czytaj
                  </button>
                </div>
                <div className="doc-link-item">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <strong>Strategie AI/ML - Smart Money Concept</strong>
                    <p>Przewodnik po strategiach uczenia maszynowego</p>
                  </div>
                  <button className="btn btn-outline btn-sm" onClick={() => alert('Dokumentacja w przygotowaniu')}>
                    Czytaj
                  </button>
                </div>
                <div className="doc-link-item">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <strong>Wdrożenie z Docker</strong>
                    <p>Instrukcja wdrażania systemu przy użyciu kontenerów Docker</p>
                  </div>
                  <button className="btn btn-outline btn-sm" onClick={() => alert('Dokumentacja w przygotowaniu')}>
                    Czytaj
                  </button>
                </div>
                <div className="doc-link-item">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <strong>Dokumentacja API</strong>
                    <p>Pełna dokumentacja REST API systemu tradingowego</p>
                  </div>
                  <button className="btn btn-outline btn-sm" onClick={() => alert('Dokumentacja w przygotowaniu')}>
                    Czytaj
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* System Information */}
        <section className="settings-section">
          <div className="section-header">
            <h2>ℹ️ Informacje o Systemie</h2>
          </div>
          
          {systemInfo && (
            <div className="system-info">
              <div className="info-grid">
                <div className="info-item">
                  <span>Nazwa:</span>
                  <span><strong>{systemInfo.name}</strong></span>
                </div>
                <div className="info-item">
                  <span>Wersja:</span>
                  <span>{systemInfo.version}</span>
                </div>
                <div className="info-item">
                  <span>Autor:</span>
                  <span>{systemInfo.author}</span>
                </div>
                <div className="info-item">
                  <span>Środowisko:</span>
                  <span className="environment">{systemInfo.environment}</span>
                </div>
                <div className="info-item">
                  <span>Data Buildu:</span>
                  <span>{systemInfo.build_date}</span>
                </div>
                <div className="info-item">
                  <span>GitHub:</span>
                  <span>
                    <a href={systemInfo.github} target="_blank" rel="noopener noreferrer" className="github-link">
                      szarastrefa/AI-ML-Trader-Bot
                    </a>
                  </span>
                </div>
              </div>

              <div className="supported-features">
                <h4>Obsługiwane Funkcje:</h4>
                <div className="features-list">
                  {systemInfo.features?.map((feature, index) => (
                    <span key={index} className="feature-badge">{feature}</span>
                  ))}
                </div>
              </div>

              <div className="supported-brokers-info">
                <h4>Obsługiwani Brokerzy:</h4>
                <div className="brokers-list">
                  {systemInfo.supported_brokers?.map((broker, index) => (
                    <span key={index} className="broker-badge">{broker}</span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* System Logs */}
        <section className="settings-section">
          <div className="section-header">
            <h2>📏 Logi Operacyjne</h2>
            <button className="btn btn-outline" onClick={handleExportLogs}>
              <span>📁</span> Eksportuj Logi
            </button>
          </div>
          
          <div className="logs-container">
            <div className="logs-header">
              <span>Ostatnie wpisy logów serwera (debug, info, warning, error)</span>
              <div className="logs-legend">
                <span className="legend-item error">ERROR</span>
                <span className="legend-item warning">WARNING</span>
                <span className="legend-item info">INFO</span>
                <span className="legend-item debug">DEBUG</span>
              </div>
            </div>
            <div className="logs-content">
              {logs.length === 0 ? (
                <div className="no-logs">
                  <p>Brak logów do wyświetlenia</p>
                </div>
              ) : (
                logs.map((log, index) => (
                  <div key={index} className={`log-entry ${getLogLevelClass(log.level)}`}>
                    <span className="log-time">
                      {new Date(log.timestamp).toLocaleTimeString('pl-PL')}
                    </span>
                    <span className="log-level">[{log.level.toUpperCase()}]</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default Settings;