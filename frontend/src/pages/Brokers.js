import React, { useState, useEffect } from 'react';
import { ApiService } from '../services/apiService';

function Brokers() {
  const [accounts, setAccounts] = useState([]);
  const [selectedAccount, setSelectedAccount] = useState(null);
  const [availableStrategies, setAvailableStrategies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddAccount, setShowAddAccount] = useState(false);
  const [supportedBrokers, setSupportedBrokers] = useState([]);
  const [newAccountForm, setNewAccountForm] = useState({
    broker: '',
    login: '',
    password: '',
    server: '',
    api_key: '',
    api_secret: ''
  });

  useEffect(() => {
    fetchBrokersData();
  }, []);

  const fetchBrokersData = async () => {
    try {
      const [accountsData, strategiesData, brokersData] = await Promise.all([
        ApiService.getAccounts(),
        ApiService.getStrategies(),
        ApiService.getSupportedBrokers()
      ]);
      setAccounts(accountsData);
      setAvailableStrategies(strategiesData.all || []);
      setSupportedBrokers(brokersData);
    } catch (error) {
      console.error('Failed to fetch brokers data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAccountSelect = (account) => {
    setSelectedAccount(account);
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
      fetchBrokersData();
      alert('Konto zostało pomyślnie dodane!');
    } catch (error) {
      console.error('Failed to add account:', error);
      alert('Błąd podczas dodawania konta');
    }
  };

  const handleRemoveAccount = async (accountId) => {
    if (window.confirm('Czy na pewno chcesz odłączyć to konto?')) {
      try {
        await ApiService.removeAccount(accountId);
        if (selectedAccount?.account_id === accountId) {
          setSelectedAccount(null);
        }
        fetchBrokersData();
      } catch (error) {
        console.error('Failed to remove account:', error);
        alert('Błąd podczas odłączania konta');
      }
    }
  };

  const formatCurrency = (amount, currency = 'USD') => {
    return new Intl.NumberFormat('pl-PL', {
      style: 'currency',
      currency: currency
    }).format(amount);
  };

  const formatPercentage = (value) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const groupAccountsByBroker = (accounts) => {
    return accounts.reduce((groups, account) => {
      const broker = account.broker;
      if (!groups[broker]) {
        groups[broker] = [];
      }
      groups[broker].push(account);
      return groups;
    }, {});
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Ładowanie danych brokerów...</p>
        </div>
      </div>
    );
  }

  const brokerGroups = groupAccountsByBroker(accounts);

  return (
    <div className="page-container brokers-page">
      <div className="page-header">
        <h1>🏦 Brokerzy i Konta</h1>
        <p>Zarządzanie kontami brokerskimi i przypisywanie strategii</p>
      </div>

      {/* Add Account Button */}
      <div className="page-actions">
        <button 
          className="btn btn-primary"
          onClick={() => setShowAddAccount(!showAddAccount)}
        >
          {showAddAccount ? 'Anuluj' : '+ Dodaj Konto'}
        </button>
      </div>

      {/* Add Account Form */}
      {showAddAccount && (
        <div className="add-account-section">
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
        </div>
      )}

      <div className="brokers-content">
        <div className="brokers-sidebar">
          <h2>Podłączone Brokerzy</h2>
          {Object.keys(brokerGroups).length === 0 ? (
            <div className="no-brokers">
              <p>Brak podłączonych kont</p>
              <p>Użyj przycisku "+ Dodaj Konto" powyżej</p>
            </div>
          ) : (
            Object.entries(brokerGroups).map(([brokerName, brokerAccounts]) => (
              <div key={brokerName} className="broker-group">
                <div className="broker-header">
                  <h3>{brokerName}</h3>
                  <span className="accounts-count">{brokerAccounts.length} kont</span>
                </div>
                
                <div className="accounts-list">
                  {brokerAccounts.map(account => (
                    <div 
                      key={account.account_id}
                      className={`account-item ${selectedAccount?.account_id === account.account_id ? 'selected' : ''}`}
                      onClick={() => handleAccountSelect(account)}
                    >
                      <div className="account-info">
                        <div className="account-id">{account.account_id}</div>
                        <div className={`account-status ${account.status.toLowerCase()}`}>
                          {account.status}
                        </div>
                      </div>
                      <div className="account-balance">
                        <div className="equity">{formatCurrency(account.equity, account.currency)}</div>
                        <div className={`pnl ${account.profit >= 0 ? 'positive' : 'negative'}`}>
                          {account.profit >= 0 ? '+' : ''}{formatCurrency(account.profit, account.currency)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>

        <div className="account-details">
          {selectedAccount ? (
            <>
              <div className="account-header">
                <h2>Szczegóły Konta: {selectedAccount.account_id}</h2>
                <div className={`status-badge ${selectedAccount.status.toLowerCase()}`}>
                  {selectedAccount.status}
                </div>
              </div>

              <div className="account-stats">
                <div className="stat-card">
                  <h4>Saldo</h4>
                  <div className="stat-value">
                    {formatCurrency(selectedAccount.balance, selectedAccount.currency)}
                  </div>
                </div>
                <div className="stat-card">
                  <h4>Equity</h4>
                  <div className="stat-value">
                    {formatCurrency(selectedAccount.equity, selectedAccount.currency)}
                  </div>
                </div>
                <div className="stat-card">
                  <h4>Marża</h4>
                  <div className="stat-value">
                    {formatCurrency(selectedAccount.margin, selectedAccount.currency)}
                  </div>
                </div>
                <div className="stat-card">
                  <h4>Wolna Marża</h4>
                  <div className="stat-value">
                    {formatCurrency(selectedAccount.free_margin, selectedAccount.currency)}
                  </div>
                </div>
                <div className="stat-card">
                  <h4>Leverage</h4>
                  <div className="stat-value">
                    {selectedAccount.leverage}
                  </div>
                </div>
                <div className="stat-card">
                  <h4>Ping</h4>
                  <div className="stat-value">
                    {selectedAccount.ping}
                  </div>
                </div>
              </div>

              <div className="strategy-assignment">
                <h3>Przypisz Strategię AI/ML</h3>
                <div className="strategy-selector">
                  <select 
                    onChange={(e) => {
                      if (e.target.value) {
                        alert(`Strategia ${e.target.value} przypisana do konta ${selectedAccount.account_id}`);
                      }
                    }}
                    defaultValue=""
                  >
                    <option value="">Wybierz strategię...</option>
                    {availableStrategies.map(strategy => (
                      <option key={strategy.id} value={strategy.id}>
                        {strategy.name} ({strategy.type})
                      </option>
                    ))}
                  </select>
                </div>
                <div className="strategy-config">
                  <label>
                    <input type="checkbox" defaultChecked />
                    Propagacja sygnałów BUY/SELL
                  </label>
                  <label>
                    <input type="checkbox" defaultChecked />
                    Automatyczne zarządzanie ryzykiem
                  </label>
                  <label>
                    <input type="checkbox" />
                    Limit dzienny strat (Stop Loss)
                  </label>
                </div>
              </div>

              <div className="account-actions">
                <button 
                  className="btn btn-danger"
                  onClick={() => handleRemoveAccount(selectedAccount.account_id)}
                >
                  🗑️ Odłącz Konto
                </button>
              </div>
            </>
          ) : (
            <div className="no-account-selected">
              <div className="placeholder-icon">🏦</div>
              <h3>Wybierz Konto</h3>
              <p>Kliknij na konto z listy po lewej, aby zobaczyć szczegóły</p>
              {accounts.length === 0 && (
                <div className="no-accounts-hint">
                  <p>Nie masz jeszcze żadnych podłączonych kont.</p>
                  <p>Użyj przycisku "+ Dodaj Konto" aby dodać pierwsze konto.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Brokers;