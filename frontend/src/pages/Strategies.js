import React, { useState, useEffect } from 'react';
import { ApiService } from '../services/apiService';

function Strategies() {
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [uploadingModel, setUploadingModel] = useState(false);
  const [modelFile, setModelFile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStrategies();
  }, []);

  const fetchStrategies = async () => {
    try {
      const data = await ApiService.getStrategies();
      setStrategies(data.all || []);
    } catch (error) {
      console.error('Failed to fetch strategies:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && (file.name.endsWith('.pkl') || file.name.endsWith('.onnx') || file.name.endsWith('.h5'))) {
      setModelFile(file);
    } else {
      alert('Proszę wybrać plik .pkl, .onnx lub .h5');
    }
  };

  const handleModelUpload = async () => {
    if (!modelFile) return;
    
    setUploadingModel(true);
    try {
      // Simulate upload for demo
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Create new strategy from uploaded model
      const newStrategy = {
        id: `UPLOADED_${Date.now()}`,
        name: `Uploaded Model: ${modelFile.name.split('.')[0]}`,
        type: 'UPLOADED_MODEL',
        symbol: 'MULTIPLE',
        status: 'Paused',
        profit: 0,
        profit_percentage: 0,
        trades_today: 0,
        win_rate: 0,
        max_drawdown: 0,
        sharpe_ratio: 0,
        confidence: 85.0,
        created_at: new Date().toISOString(),
        last_signal: 'NONE'
      };
      
      setStrategies(prev => [...prev, newStrategy]);
      setModelFile(null);
      alert('Model został pomyślnie wgrany!');
    } catch (error) {
      console.error('Failed to upload model:', error);
      alert('Błąd podczas wgrywania modelu');
    } finally {
      setUploadingModel(false);
    }
  };

  const handleModelExport = async (strategy) => {
    try {
      // Simulate export for demo
      const blob = new Blob([JSON.stringify(strategy, null, 2)], {
        type: 'application/json'
      });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${strategy.id}.json`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export model:', error);
      alert('Błąd podczas eksportu modelu');
    }
  };

  const handleModelDelete = async (strategyId) => {
    if (window.confirm('Czy na pewno chcesz usunąć tą strategię?')) {
      try {
        setStrategies(prev => prev.filter(s => s.id !== strategyId));
        if (selectedStrategy?.id === strategyId) {
          setSelectedStrategy(null);
        }
      } catch (error) {
        console.error('Failed to delete strategy:', error);
        alert('Błąd podczas usuwania strategii');
      }
    }
  };

  const handleStrategyToggle = async (strategyId, currentStatus) => {
    try {
      const newStatus = currentStatus === 'Active' ? 'Paused' : 'Active';
      
      setStrategies(prev => prev.map(s => 
        s.id === strategyId ? { ...s, status: newStatus } : s
      ));
      
      if (selectedStrategy?.id === strategyId) {
        setSelectedStrategy(prev => ({ ...prev, status: newStatus }));
      }
      
      if (newStatus === 'Active') {
        await ApiService.activateStrategy(strategyId);
      } else {
        await ApiService.deactivateStrategy(strategyId);
      }
    } catch (error) {
      console.error('Failed to toggle strategy:', error);
      // Revert state on error
      fetchStrategies();
    }
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Ładowanie strategii AI/ML...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container strategies-page">
      <div className="page-header">
        <h1>🧠 Strategie AI/ML</h1>
        <p>Zarządzanie modelami uczenia maszynowego</p>
      </div>

      {/* Model Upload Section */}
      <div className="model-upload-section">
        <div className="upload-card">
          <h3>📄 Wgraj Nowy Model</h3>
          <div className="upload-area">
            <input
              type="file"
              id="modelFile"
              accept=".pkl,.onnx,.h5"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
            <label htmlFor="modelFile" className="upload-button">
              {modelFile ? (
                <div className="file-selected">
                  <span className="file-icon">📁</span>
                  <span className="file-name">{modelFile.name}</span>
                  <span className="file-size">({(modelFile.size / 1024 / 1024).toFixed(2)} MB)</span>
                </div>
              ) : (
                <div className="upload-placeholder">
                  <span className="upload-icon">📄</span>
                  <span>Wybierz plik modelu</span>
                  <span className="file-types">(.pkl, .onnx, .h5)</span>
                </div>
              )}
            </label>
            {modelFile && (
              <button 
                className="upload-submit"
                onClick={handleModelUpload}
                disabled={uploadingModel}
              >
                {uploadingModel ? (
                  <>
                    <span className="loading-icon">⌛</span>
                    Wgrywanie...
                  </>
                ) : (
                  <>
                    <span>⬆️</span>
                    Wgraj Model
                  </>
                )}
              </button>
            )}
          </div>
          <div className="upload-help">
            <h4>Obsługiwane formaty:</h4>
            <ul>
              <li><strong>.pkl</strong> - scikit-learn, pandas models</li>
              <li><strong>.onnx</strong> - PyTorch, TensorFlow (ONNX format)</li>
              <li><strong>.h5</strong> - Keras, TensorFlow models</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Strategies Overview */}
      <div className="strategies-overview">
        <div className="strategies-header">
          <h2>Dostępne Strategie ({strategies.length})</h2>
          <div className="strategies-stats">
            <div className="stat">
              <span className="stat-label">Aktywne:</span>
              <span className="stat-value active">
                {strategies.filter(s => s.status === 'Active').length}
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Wstrzymane:</span>
              <span className="stat-value paused">
                {strategies.filter(s => s.status === 'Paused').length}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Strategies Grid */}
      <div className="strategies-grid">
        {strategies.length === 0 ? (
          <div className="no-strategies">
            <div className="placeholder-icon">🧠</div>
            <h3>Brak Strategii</h3>
            <p>Nie masz jeszcze żadnych strategii AI/ML.</p>
            <p>Wgraj swój pierwszy model powyżej!</p>
          </div>
        ) : (
          strategies.map(strategy => (
            <div 
              key={strategy.id} 
              className={`strategy-card ${selectedStrategy?.id === strategy.id ? 'selected' : ''} ${strategy.status.toLowerCase()}`}
              onClick={() => setSelectedStrategy(strategy)}
            >
              <div className="strategy-header">
                <h3>{strategy.name}</h3>
                <div className={`strategy-status ${strategy.status.toLowerCase()}`}>
                  <span className="status-dot"></span>
                  {strategy.status}
                </div>
              </div>

              <div className="strategy-type">
                <span className="type-badge">{strategy.type}</span>
                <span className="symbol-badge">{strategy.symbol}</span>
              </div>

              <div className="strategy-stats">
                <div className="stat-row">
                  <span>P&L:</span>
                  <span className={strategy.profit >= 0 ? 'positive' : 'negative'}>
                    {strategy.profit >= 0 ? '+' : ''}${strategy.profit?.toFixed(2)}
                  </span>
                </div>
                <div className="stat-row">
                  <span>Win Rate:</span>
                  <span>{strategy.win_rate}%</span>
                </div>
                <div className="stat-row">
                  <span>Confidence:</span>
                  <span className="confidence">{strategy.confidence}%</span>
                </div>
                <div className="stat-row">
                  <span>Trades Today:</span>
                  <span>{strategy.trades_today}</span>
                </div>
              </div>

              <div className="strategy-actions">
                <button 
                  className={`btn btn-sm ${strategy.status === 'Active' ? 'btn-pause' : 'btn-play'}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleStrategyToggle(strategy.id, strategy.status);
                  }}
                >
                  {strategy.status === 'Active' ? (
                    <><span>⏸️</span> Zatrzymaj</>
                  ) : (
                    <><span>▶️</span> Uruchom</>
                  )}
                </button>
                <button 
                  className="btn btn-sm btn-export"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleModelExport(strategy);
                  }}
                >
                  <span>📁</span> Eksport
                </button>
                <button 
                  className="btn btn-sm btn-delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleModelDelete(strategy.id);
                  }}
                >
                  <span>🗑️</span> Usuń
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Strategy Details Panel */}
      {selectedStrategy && (
        <div className="strategy-details">
          <div className="details-header">
            <h2>Szczegóły Strategii: {selectedStrategy.name}</h2>
            <div className={`status-badge ${selectedStrategy.status.toLowerCase()}`}>
              <span className="status-dot"></span>
              {selectedStrategy.status}
            </div>
          </div>

          <div className="details-content">
            <div className="detail-section">
              <h4>Informacje Podstawowe</h4>
              <div className="detail-grid">
                <div className="detail-item">
                  <span>Typ:</span>
                  <span><strong>{selectedStrategy.type}</strong></span>
                </div>
                <div className="detail-item">
                  <span>Symbol:</span>
                  <span className="symbol">{selectedStrategy.symbol}</span>
                </div>
                <div className="detail-item">
                  <span>Utworzono:</span>
                  <span>{new Date(selectedStrategy.created_at).toLocaleDateString('pl-PL')}</span>
                </div>
                <div className="detail-item">
                  <span>Ostatni Sygnał:</span>
                  <span className={`signal ${selectedStrategy.last_signal?.toLowerCase()}`}>
                    {selectedStrategy.last_signal || 'BRAK'}
                  </span>
                </div>
              </div>
            </div>

            <div className="detail-section">
              <h4>Wydajność i Metryki</h4>
              <div className="performance-metrics">
                <div className="metric">
                  <span className="metric-label">Zysk/Strata</span>
                  <span className={`metric-value ${selectedStrategy.profit >= 0 ? 'positive' : 'negative'}`}>
                    ${selectedStrategy.profit?.toFixed(2)}
                  </span>
                  <span className="metric-subtitle">
                    {selectedStrategy.profit_percentage?.toFixed(2)}% ROI
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Win Rate</span>
                  <span className="metric-value">{selectedStrategy.win_rate}%</span>
                  <span className="metric-subtitle">
                    {selectedStrategy.trades_today} transakcji dziś
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Max Drawdown</span>
                  <span className="metric-value warning">{selectedStrategy.max_drawdown}%</span>
                  <span className="metric-subtitle">Maksymalna strata</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Sharpe Ratio</span>
                  <span className="metric-value">{selectedStrategy.sharpe_ratio}</span>
                  <span className="metric-subtitle">Ryzyko/Zysk</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Confidence</span>
                  <span className="metric-value confidence">{selectedStrategy.confidence}%</span>
                  <span className="metric-subtitle">Pewność modelu</span>
                </div>
              </div>
            </div>

            <div className="detail-section">
              <h4>Ustawienia Strategii</h4>
              <div className="strategy-settings">
                <label className="setting-item">
                  <input type="checkbox" defaultChecked={selectedStrategy.status === 'Active'} />
                  <span>Automatyczne wykonywanie transakcji</span>
                </label>
                <label className="setting-item">
                  <input type="checkbox" defaultChecked />
                  <span>Stop Loss na poziomie 2%</span>
                </label>
                <label className="setting-item">
                  <input type="checkbox" defaultChecked />
                  <span>Take Profit na poziomie 5%</span>
                </label>
                <label className="setting-item">
                  <input type="checkbox" />
                  <span>Notyfikacje o sygnałach</span>
                </label>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Strategies;