#!/usr/bin/env python3
"""
Base Connector Abstract Class
Defines the common interface for all broker connectors
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    """Trading position data structure"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    profit: float
    profit_percentage: float
    timestamp: datetime
    broker_position_id: str = ""


@dataclass
class Order:
    """Order data structure"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: float = 0.0
    timestamp: datetime = None
    broker_order_id: str = ""


@dataclass
class Balance:
    """Account balance structure"""
    total: float
    available: float
    used: float
    currency: str = "USD"
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0


@dataclass
class Ticker:
    """Market ticker data"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    change: float = 0.0
    change_percentage: float = 0.0


@dataclass
class OrderBookEntry:
    """Order book entry (bid/ask)"""
    price: float
    amount: float


@dataclass
class OrderBook:
    """Market depth / order book"""
    symbol: str
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]
    timestamp: datetime


@dataclass
class HistoricalBar:
    """Historical price bar (OHLCV)"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BaseConnector(ABC):
    """Abstract base class for all broker connectors"""
    
    def __init__(self, broker_name: str):
        self.broker_name = broker_name
        self.is_connected = False
        self.account_info = {}
        self.config = {}
        
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to the broker
        
        Args:
            config: Connection configuration (API keys, credentials, etc.)
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker"""
        pass
    
    @abstractmethod
    def get_balance(self) -> Balance:
        """Get account balance
        
        Returns:
            Balance: Current account balance information
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all open positions
        
        Returns:
            List[Position]: List of current positions
        """
        pass
    
    @abstractmethod
    def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List[Order]: List of open orders
        """
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place a new order
        
        Args:
            order: Order to place
            
        Returns:
            str: Order ID assigned by broker
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancelled successfully
        """
        pass
    
    @abstractmethod
    def modify_order(self, order_id: str, **kwargs) -> bool:
        """Modify an existing order
        
        Args:
            order_id: Order ID to modify
            **kwargs: Order parameters to modify
            
        Returns:
            bool: True if modified successfully
        """
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Ticker:
        """Get current market ticker
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker: Current market data
        """
        pass
    
    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 10) -> OrderBook:
        """Get market depth (order book)
        
        Args:
            symbol: Trading symbol
            limit: Number of levels to retrieve
            
        Returns:
            OrderBook: Current market depth
        """
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_time: datetime, end_time: datetime) -> List[HistoricalBar]:
        """Get historical price data
        
        Args:
            symbol: Trading symbol
            timeframe: Time interval (1m, 5m, 1h, 1d, etc.)
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            List[HistoricalBar]: Historical price bars
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols
        
        Returns:
            List[str]: Available symbols
        """
        pass
    
    def update_market_data(self) -> None:
        """Update market data (optional implementation)"""
        pass
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information
        
        Returns:
            Dict: Account details
        """
        return self.account_info.copy()
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading
        
        Args:
            symbol: Trading symbol
            
        Returns:
            bool: True if market is open
        """
        # Default implementation - assume always open
        # Override in specific connectors for accurate market hours
        return True
    
    def get_minimum_order_size(self, symbol: str) -> float:
        """Get minimum order size for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            float: Minimum order size
        """
        # Default implementation
        return 0.01
    
    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict: Trading fees (maker, taker)
        """
        # Default implementation
        return {"maker": 0.001, "taker": 0.001}
    
    def validate_order(self, order: Order) -> bool:
        """Validate order before placing
        
        Args:
            order: Order to validate
            
        Returns:
            bool: True if order is valid
        """
        # Basic validation
        if order.amount <= 0:
            return False
        
        if order.order_type == OrderType.LIMIT and order.price is None:
            return False
            
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return False
            
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.broker_name})"
    
    def __repr__(self) -> str:
        return self.__str__()
