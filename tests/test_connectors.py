#!/usr/bin/env python3
"""
Broker Connector Tests
Unit and integration tests for broker connectivity modules
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

# Import the connector modules
try:
    from backend.connectors.base_connector import (
        BaseConnector, Position, Order, MarketData, AccountInfo,
        OrderType, OrderSide, OrderStatus
    )
    from backend.connectors.mt5_connector import MT5Connector
    from backend.connectors.ccxt_connector import CCXTConnector
    from backend.connectors.ibkr_connector import IBKRConnector
except ImportError:
    # Mock classes for testing when modules are not available
    class BaseConnector:
        pass
    
    class MT5Connector(BaseConnector):
        pass
    
    class CCXTConnector(BaseConnector):
        pass
    
    class IBKRConnector(BaseConnector):
        pass


class MockConnector(BaseConnector):
    """Mock connector for testing base functionality"""
    
    def __init__(self, broker_name: str, config: Dict[str, Any] = None):
        super().__init__(broker_name, config)
        self.mock_account = AccountInfo(
            account_id="TEST_001",
            broker=broker_name,
            balance=10000.0,
            equity=10000.0,
            margin_used=0.0,
            margin_free=10000.0
        )
        self.mock_positions = []
        self.mock_orders = []
        self.mock_market_data = {}
    
    async def connect(self) -> bool:
        self.is_connected = True
        return True
    
    async def disconnect(self) -> bool:
        self.is_connected = False
        return True
    
    async def get_account_info(self) -> AccountInfo:
        return self.mock_account
    
    async def get_positions(self) -> list:
        return self.mock_positions
    
    async def get_orders(self, symbol: str = None) -> list:
        if symbol:
            return [order for order in self.mock_orders if order.symbol == symbol]
        return self.mock_orders
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         size: float, price: float = None, 
                         stop_price: float = None) -> Order:
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            stop_price=stop_price,
            order_id=f"ORDER_{len(self.mock_orders) + 1}",
            status=OrderStatus.PENDING
        )
        self.mock_orders.append(order)
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        for order in self.mock_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    async def get_market_data(self, symbol: str) -> MarketData:
        return self.mock_market_data.get(symbol, MarketData(
            symbol=symbol,
            bid=1.0000,
            ask=1.0001,
            last_price=1.00005,
            volume=1000.0
        ))
    
    async def get_historical_data(self, symbol: str, timeframe: str,
                                 start_date: datetime, end_date: datetime = None):
        # Return mock historical data
        return [
            {
                'timestamp': start_date + timedelta(hours=i),
                'open': 1.0000 + i * 0.0001,
                'high': 1.0010 + i * 0.0001,
                'low': 0.9990 + i * 0.0001,
                'close': 1.0005 + i * 0.0001,
                'volume': 1000 + i * 100
            }
            for i in range(10)
        ]
    
    def get_supported_symbols(self) -> list:
        return ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]


@pytest.fixture
def mock_connector():
    """Fixture providing a mock connector instance"""
    return MockConnector("MockBroker")


@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data"""
    return {
        "EURUSD": MarketData(
            symbol="EURUSD",
            bid=1.0850,
            ask=1.0851,
            last_price=1.08505,
            volume=150000.0
        ),
        "BTCUSD": MarketData(
            symbol="BTCUSD",
            bid=45000.0,
            ask=45001.0,
            last_price=45000.5,
            volume=10.5
        )
    }


class TestBaseConnector:
    """Tests for BaseConnector abstract class"""
    
    def test_connector_initialization(self, mock_connector):
        """Test connector initialization"""
        assert mock_connector.broker_name == "MockBroker"
        assert mock_connector.is_connected == False
        assert mock_connector.config == {}
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, mock_connector):
        """Test connect and disconnect functionality"""
        # Test initial state
        assert not mock_connector.is_connected
        
        # Test connection
        result = await mock_connector.connect()
        assert result == True
        assert mock_connector.is_connected == True
        
        # Test disconnection
        result = await mock_connector.disconnect()
        assert result == True
        assert mock_connector.is_connected == False
    
    @pytest.mark.asyncio
    async def test_account_info_retrieval(self, mock_connector):
        """Test account information retrieval"""
        await mock_connector.connect()
        account = await mock_connector.get_account_info()
        
        assert isinstance(account, AccountInfo)
        assert account.account_id == "TEST_001"
        assert account.broker == "MockBroker"
        assert account.balance == 10000.0
        assert account.equity == 10000.0
    
    @pytest.mark.asyncio
    async def test_position_management(self, mock_connector):
        """Test position retrieval"""
        await mock_connector.connect()
        positions = await mock_connector.get_positions()
        
        assert isinstance(positions, list)
        # Mock connector starts with no positions
        assert len(positions) == 0
    
    @pytest.mark.asyncio
    async def test_order_management(self, mock_connector):
        """Test order placement and management"""
        await mock_connector.connect()
        
        # Test order placement
        order = await mock_connector.place_order(
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=1000.0,
            price=1.0850
        )
        
        assert isinstance(order, Order)
        assert order.symbol == "EURUSD"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.size == 1000.0
        assert order.price == 1.0850
        assert order.status == OrderStatus.PENDING
        
        # Test order retrieval
        orders = await mock_connector.get_orders()
        assert len(orders) == 1
        assert orders[0].order_id == order.order_id
        
        # Test order cancellation
        cancel_result = await mock_connector.cancel_order(order.order_id)
        assert cancel_result == True
        
        # Verify order status changed
        orders = await mock_connector.get_orders()
        assert orders[0].status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_market_data_retrieval(self, mock_connector):
        """Test market data retrieval"""
        await mock_connector.connect()
        market_data = await mock_connector.get_market_data("EURUSD")
        
        assert isinstance(market_data, MarketData)
        assert market_data.symbol == "EURUSD"
        assert market_data.bid > 0
        assert market_data.ask > market_data.bid
        assert market_data.spread == market_data.ask - market_data.bid
    
    @pytest.mark.asyncio
    async def test_historical_data_retrieval(self, mock_connector):
        """Test historical data retrieval"""
        await mock_connector.connect()
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow()
        
        historical_data = await mock_connector.get_historical_data(
            "EURUSD", "1h", start_date, end_date
        )
        
        assert isinstance(historical_data, list)
        assert len(historical_data) > 0
        
        # Check data structure
        bar = historical_data[0]
        required_keys = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for key in required_keys:
            assert key in bar
    
    def test_symbol_support_check(self, mock_connector):
        """Test symbol support checking"""
        supported_symbols = mock_connector.get_supported_symbols()
        assert isinstance(supported_symbols, list)
        assert len(supported_symbols) > 0
        
        # Test specific symbol checks
        assert mock_connector.is_symbol_supported("EURUSD")
        assert mock_connector.is_symbol_supported("eurusd")  # Case insensitive
        assert not mock_connector.is_symbol_supported("INVALID_SYMBOL")
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_connector):
        """Test connector health check"""
        # Test health check when disconnected
        health = await mock_connector.health_check()
        assert health['status'] in ['healthy', 'unhealthy']
        assert health['broker'] == "MockBroker"
        assert 'timestamp' in health
        
        # Test health check when connected
        await mock_connector.connect()
        health = await mock_connector.health_check()
        assert health['status'] == 'healthy'
        assert health['connected'] == True
        assert 'account_id' in health


class TestMT5Connector:
    """Tests for MetaTrader 5 connector"""
    
    def test_mt5_connector_initialization(self):
        """Test MT5 connector initialization"""
        connector = MT5Connector("MT5_Test")
        assert connector.broker_name == "MT5_Test"
        assert isinstance(connector, BaseConnector)
    
    @patch('MetaTrader5.initialize')
    @patch('MetaTrader5.login')
    def test_mt5_connection_mock(self, mock_login, mock_initialize):
        """Test MT5 connection with mocked MT5 library"""
        # Mock successful initialization and login
        mock_initialize.return_value = True
        mock_login.return_value = True
        
        connector = MT5Connector("MT5_Test")
        config = {
            'login': 12345678,
            'password': 'test_password',
            'server': 'TestServer-Demo'
        }
        
        # This would test the actual connection logic
        # Implementation depends on actual MT5Connector code
        assert connector.broker_name == "MT5_Test"


class TestCCXTConnector:
    """Tests for CCXT (cryptocurrency) connector"""
    
    def test_ccxt_connector_initialization(self):
        """Test CCXT connector initialization"""
        connector = CCXTConnector("Binance")
        assert connector.broker_name == "Binance"
        assert isinstance(connector, BaseConnector)
    
    @patch('ccxt.binance')
    def test_ccxt_connection_mock(self, mock_binance):
        """Test CCXT connection with mocked ccxt library"""
        # Mock exchange instance
        mock_exchange = MagicMock()
        mock_binance.return_value = mock_exchange
        
        connector = CCXTConnector("Binance")
        config = {
            'api_key': 'test_api_key',
            'api_secret': 'test_api_secret',
            'sandbox': True
        }
        
        # This would test the actual connection logic
        assert connector.broker_name == "Binance"


class TestIBKRConnector:
    """Tests for Interactive Brokers connector"""
    
    def test_ibkr_connector_initialization(self):
        """Test IBKR connector initialization"""
        connector = IBKRConnector("IBKR_Test")
        assert connector.broker_name == "IBKR_Test"
        assert isinstance(connector, BaseConnector)


class TestConnectorIntegration:
    """Integration tests for multiple connectors"""
    
    @pytest.mark.asyncio
    async def test_multiple_connector_management(self):
        """Test managing multiple connectors simultaneously"""
        connectors = {
            'mock1': MockConnector("MockBroker1"),
            'mock2': MockConnector("MockBroker2")
        }
        
        # Connect all connectors
        for name, connector in connectors.items():
            result = await connector.connect()
            assert result == True
            assert connector.is_connected == True
        
        # Test account info from each
        for name, connector in connectors.items():
            account = await connector.get_account_info()
            assert account.broker == connector.broker_name
        
        # Disconnect all
        for name, connector in connectors.items():
            result = await connector.disconnect()
            assert result == True
            assert connector.is_connected == False
    
    @pytest.mark.asyncio
    async def test_cross_broker_arbitrage_scenario(self):
        """Test scenario involving multiple brokers for arbitrage"""
        broker1 = MockConnector("Broker1")
        broker2 = MockConnector("Broker2")
        
        await broker1.connect()
        await broker2.connect()
        
        # Set different prices on each broker
        broker1.mock_market_data["BTCUSD"] = MarketData(
            symbol="BTCUSD",
            bid=45000.0,
            ask=45001.0,
            last_price=45000.5
        )
        
        broker2.mock_market_data["BTCUSD"] = MarketData(
            symbol="BTCUSD",
            bid=45010.0,
            ask=45011.0,
            last_price=45010.5
        )
        
        # Get market data from both
        data1 = await broker1.get_market_data("BTCUSD")
        data2 = await broker2.get_market_data("BTCUSD")
        
        # Check for arbitrage opportunity
        price_diff = data2.bid - data1.ask
        assert price_diff == 9.0  # 45010 - 45001
        
        # Place orders to exploit arbitrage
        if price_diff > 0:
            buy_order = await broker1.place_order(
                "BTCUSD", OrderSide.BUY, OrderType.LIMIT, 0.1, data1.ask
            )
            sell_order = await broker2.place_order(
                "BTCUSD", OrderSide.SELL, OrderType.LIMIT, 0.1, data2.bid
            )
            
            assert buy_order.symbol == "BTCUSD"
            assert sell_order.symbol == "BTCUSD"
            assert buy_order.side == OrderSide.BUY
            assert sell_order.side == OrderSide.SELL


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self):
        """Test handling of connection failures"""
        class FailingConnector(MockConnector):
            async def connect(self):
                raise ConnectionError("Failed to connect")
        
        connector = FailingConnector("FailingBroker")
        
        with pytest.raises(ConnectionError):
            await connector.connect()
        
        assert not connector.is_connected
    
    @pytest.mark.asyncio
    async def test_invalid_order_handling(self, mock_connector):
        """Test handling of invalid orders"""
        await mock_connector.connect()
        
        # Test with invalid symbol
        with pytest.raises(ValueError):
            await mock_connector.place_order(
                "", OrderSide.BUY, OrderType.LIMIT, 1000.0, 1.0850
            )
        
        # Test with invalid size
        with pytest.raises(ValueError):
            await mock_connector.place_order(
                "EURUSD", OrderSide.BUY, OrderType.LIMIT, -1000.0, 1.0850
            )
    
    @pytest.mark.asyncio
    async def test_market_data_timeout(self):
        """Test market data retrieval timeout"""
        class SlowConnector(MockConnector):
            async def get_market_data(self, symbol: str):
                await asyncio.sleep(0.1)  # Simulate slow response
                return await super().get_market_data(symbol)
        
        connector = SlowConnector("SlowBroker")
        await connector.connect()
        
        # This should complete within reasonable time
        start_time = datetime.now()
        data = await connector.get_market_data("EURUSD")
        end_time = datetime.now()
        
        assert isinstance(data, MarketData)
        assert (end_time - start_time).total_seconds() < 1.0


class TestPerformanceMetrics:
    """Test performance-related functionality"""
    
    @pytest.mark.asyncio
    async def test_order_execution_speed(self, mock_connector):
        """Test order execution performance"""
        await mock_connector.connect()
        
        start_time = datetime.now()
        
        # Place multiple orders
        orders = []
        for i in range(10):
            order = await mock_connector.place_order(
                "EURUSD", OrderSide.BUY, OrderType.LIMIT, 1000.0, 1.0850 + i * 0.0001
            )
            orders.append(order)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert execution_time < 1.0
        assert len(orders) == 10
        
        # Verify all orders were created
        all_orders = await mock_connector.get_orders()
        assert len(all_orders) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_connector):
        """Test concurrent connector operations"""
        await mock_connector.connect()
        
        # Create multiple concurrent tasks
        tasks = []
        
        # Market data retrieval tasks
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            tasks.append(mock_connector.get_market_data(symbol))
        
        # Order placement tasks
        for i in range(3):
            tasks.append(mock_connector.place_order(
                "EURUSD", OrderSide.BUY, OrderType.LIMIT, 1000.0, 1.0850 + i * 0.0001
            ))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert len(results) == 6
        
        # Check market data results
        for i in range(3):
            assert isinstance(results[i], MarketData)
        
        # Check order results
        for i in range(3, 6):
            assert isinstance(results[i], Order)


if __name__ == "__main__":
    # Run tests
    pytest.main(["-v", __file__])