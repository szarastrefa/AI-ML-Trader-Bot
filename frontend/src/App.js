import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Layout, ConfigProvider, theme, Spin } from 'antd';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import io from 'socket.io-client';

// Components
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Portfolio from './pages/Portfolio';
import Strategies from './pages/Strategies';
import Brokers from './pages/Brokers';
import Settings from './pages/Settings';
import Login from './pages/Login';

// Services
import { authService } from './services/authService';
import { socketService } from './services/socketService';

// Styles
import './App.css';

const { Header, Content, Sider } = Layout;

// React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [socket, setSocket] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  // Initialize authentication check
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('token');
        if (token) {
          const isValid = await authService.validateToken(token);
          setIsAuthenticated(isValid);
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        localStorage.removeItem('token');
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  // Initialize WebSocket connection when authenticated
  useEffect(() => {
    if (isAuthenticated && !socket) {
      const newSocket = io(process.env.REACT_APP_WS_URL || 'http://localhost:5000', {
        auth: {
          token: localStorage.getItem('token')
        },
        transports: ['websocket', 'polling']
      });

      newSocket.on('connect', () => {
        console.log('Connected to WebSocket');
        setConnectionStatus('connected');
      });

      newSocket.on('disconnect', () => {
        console.log('Disconnected from WebSocket');
        setConnectionStatus('disconnected');
      });

      newSocket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setConnectionStatus('error');
      });

      // Initialize socket service
      socketService.setSocket(newSocket);
      setSocket(newSocket);

      return () => {
        newSocket.close();
        setSocket(null);
        setConnectionStatus('disconnected');
      };
    }
  }, [isAuthenticated, socket]);

  // Theme configuration
  const themeConfig = {
    algorithm: darkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
    token: {
      colorPrimary: '#1890ff',
      borderRadius: 6,
      wireframe: false,
    },
  };

  // Handle login
  const handleLogin = (token) => {
    localStorage.setItem('token', token);
    setIsAuthenticated(true);
  };

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    if (socket) {
      socket.close();
      setSocket(null);
    }
    setConnectionStatus('disconnected');
  };

  // Toggle theme
  const toggleTheme = () => {
    setDarkMode(!darkMode);
    localStorage.setItem('darkMode', !darkMode);
  };

  // Load theme preference
  useEffect(() => {
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme) {
      setDarkMode(JSON.parse(savedTheme));
    }
  }, []);

  if (loading) {
    return (
      <div className="loading-container">
        <Spin size="large" tip="Loading AI Trader Bot..." />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <ConfigProvider theme={themeConfig}>
        <Login onLogin={handleLogin} />
        <Toaster position="top-right" />
      </ConfigProvider>
    );
  }

  return (
    <ConfigProvider theme={themeConfig}>
      <QueryClientProvider client={queryClient}>
        <Router>
          <Layout className="app-layout" style={{ minHeight: '100vh' }}>
            <Sider 
              trigger={null} 
              collapsible 
              collapsed={collapsed}
              className="app-sidebar"
              theme={darkMode ? 'dark' : 'light'}
            >
              <Sidebar 
                collapsed={collapsed}
                onCollapse={setCollapsed}
                darkMode={darkMode}
                connectionStatus={connectionStatus}
              />
            </Sider>
            
            <Layout className="site-layout">
              <Header className="app-header">
                <div className="header-content">
                  <div className="header-left">
                    <h2>AI/ML Trader Bot</h2>
                    <span className={`connection-status ${connectionStatus}`}>
                      {connectionStatus === 'connected' && '🟢 Connected'}
                      {connectionStatus === 'disconnected' && '🔴 Disconnected'}
                      {connectionStatus === 'error' && '🟡 Connection Error'}
                    </span>
                  </div>
                  
                  <div className="header-right">
                    <button 
                      className="theme-toggle"
                      onClick={toggleTheme}
                      title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
                    >
                      {darkMode ? '☀️' : '🌙'}
                    </button>
                    
                    <button 
                      className="logout-button"
                      onClick={handleLogout}
                      title="Logout"
                    >
                      🚪 Logout
                    </button>
                  </div>
                </div>
              </Header>
              
              <Content className="app-content">
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard socket={socket} />} />
                  <Route path="/portfolio" element={<Portfolio socket={socket} />} />
                  <Route path="/strategies" element={<Strategies socket={socket} />} />
                  <Route path="/brokers" element={<Brokers socket={socket} />} />
                  <Route path="/settings" element={<Settings socket={socket} onThemeChange={toggleTheme} darkMode={darkMode} />} />
                </Routes>
              </Content>
            </Layout>
          </Layout>
        </Router>
        
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: darkMode ? '#2f2f2f' : '#fff',
              color: darkMode ? '#fff' : '#333',
            },
          }}
        />
      </QueryClientProvider>
    </ConfigProvider>
  );
}

export default App;