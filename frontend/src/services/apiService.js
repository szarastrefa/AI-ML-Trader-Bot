class ApiService {
  static baseURL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  static async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint.startsWith('/api') ? endpoint : '/api' + endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error [${url}]:`, error);
      throw error;
    }
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
    try {
      const response = await fetch(`${this.baseURL}/health`);
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
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
    return this.request('/accounts', {
      method: 'POST',
      body: JSON.stringify(accountData)
    });
  }

  static async removeAccount(accountId) {
    return this.request(`/accounts/${accountId}`, {
      method: 'DELETE'
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
    try {
      const response = await fetch(`${this.baseURL}/api/strategies/import`, {
        method: 'POST',
        body: formData // Don't set Content-Type for FormData
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Model upload error:', error);
      throw error;
    }
  }

  static async exportModel(strategyId) {
    try {
      const response = await fetch(`${this.baseURL}/api/strategies/${strategyId}/export`);
      
      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }
      
      return await response.blob();
    } catch (error) {
      console.error('Model export error:', error);
      throw error;
    }
  }

  static async deleteStrategy(strategyId) {
    return this.request(`/strategies/${strategyId}`, {
      method: 'DELETE'
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

  // Test connectivity
  static async testConnection() {
    try {
      const start = Date.now();
      const health = await this.getHealth();
      const responseTime = Date.now() - start;
      
      return {
        connected: true,
        health: health,
        responseTime: responseTime,
        baseURL: this.baseURL
      };
    } catch (error) {
      return {
        connected: false,
        error: error.message,
        baseURL: this.baseURL
      };
    }
  }
}

export { ApiService };