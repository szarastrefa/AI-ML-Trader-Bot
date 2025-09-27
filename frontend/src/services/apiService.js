class ApiService {
  static baseURL = process.env.REACT_APP_API_URL || '';

  static async request(endpoint, options = {}) {
    const url = `${this.baseURL}/api${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    const response = await fetch(url, config);
    
    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return response.json();
  }

  // Dashboard endpoints
  static async getSummary() {
    return this.request('/summary');
  }

  static async getDashboard() {
    return this.request('/dashboard');
  }

  // Health and system
  static async getHealth() {
    const response = await fetch(`${this.baseURL}/health`);
    return response.json();
  }

  static async getTradingStatus() {
    return this.request('/trading/status');
  }

  static async getSystemInfo() {
    return this.request('/system/info');
  }

  static async getSystemLogs() {
    return this.request('/system/logs');
  }

  // Accounts and brokers
  static async getAccounts() {
    return this.request('/accounts');
  }

  static async addAccount(accountData) {
    // Simulate API call for demo
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({ success: true, message: 'Account added successfully' });
      }, 1000);
    });
  }

  static async removeAccount(accountId) {
    // Simulate API call for demo
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({ success: true, message: 'Account removed successfully' });
      }, 500);
    });
  }

  static async getSupportedBrokers() {
    return this.request('/brokers');
  }

  // Strategies
  static async getStrategies() {
    return this.request('/strategies');
  }

  static async activateStrategy(strategyId) {
    return this.request(`/strategies/${strategyId}/start`, {
      method: 'POST',
    });
  }

  static async deactivateStrategy(strategyId) {
    return this.request(`/strategies/${strategyId}/stop`, {
      method: 'POST',
    });
  }

  static async uploadModel(formData) {
    // Simulate upload for demo
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (Math.random() > 0.1) { // 90% success rate
          resolve({ success: true, message: 'Model uploaded successfully' });
        } else {
          reject(new Error('Upload failed'));
        }
      }, 2000);
    });
  }

  static async exportModel(strategyId) {
    // Simulate export for demo
    return new Promise((resolve) => {
      setTimeout(() => {
        const demoData = {
          strategy_id: strategyId,
          model_type: 'ML_MOMENTUM',
          version: '1.0',
          exported_at: new Date().toISOString()
        };
        const blob = new Blob([JSON.stringify(demoData, null, 2)], {
          type: 'application/json'
        });
        resolve(blob);
      }, 1000);
    });
  }

  static async deleteStrategy(strategyId) {
    // Simulate API call for demo
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({ success: true, message: 'Strategy deleted successfully' });
      }, 500);
    });
  }

  // Trading controls
  static async startTrading() {
    return this.request('/trading/start', {
      method: 'POST',
    });
  }

  static async stopTrading() {
    return this.request('/trading/stop', {
      method: 'POST',
    });
  }

  // Market data
  static async getMarketData() {
    return this.request('/market');
  }

  // Positions
  static async getPositions() {
    return this.request('/positions');
  }
}

export { ApiService };