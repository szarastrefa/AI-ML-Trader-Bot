#!/usr/bin/env python3
"""
Broker Connectors Module
Initialization for all broker integrations
"""

from typing import Dict, Optional, Type
from abc import ABC, abstractmethod

# Import all connector implementations
from .base_connector import BaseConnector
from .mt5_connector import MT5Connector
from .ccxt_connector import CCXTConnector
from .ibkr_connector import IBKRConnector


class BrokerManager:
    """Manager for all broker connections"""
    
    def __init__(self):
        self.connectors: Dict[str, Type[BaseConnector]] = {
            'mt5': MT5Connector,
            'binance': CCXTConnector,
            'kraken': CCXTConnector,
            'coinbase_pro': CCXTConnector,
            'bitfinex': CCXTConnector,
            'huobi': CCXTConnector,
            'okx': CCXTConnector,
            'bybit': CCXTConnector,
            'kucoin': CCXTConnector,
            'bittrex': CCXTConnector,
            'interactive_brokers': IBKRConnector,
            'ibkr': IBKRConnector,
        }
        self.active_connections: Dict[str, BaseConnector] = {}
    
    def get_connector(self, broker_name: str) -> Optional[BaseConnector]:
        """Get a connector instance for the specified broker"""
        broker_name = broker_name.lower()
        
        if broker_name in self.active_connections:
            return self.active_connections[broker_name]
            
        if broker_name in self.connectors:
            connector_class = self.connectors[broker_name]
            connector = connector_class(broker_name)
            return connector
            
        return None
    
    def register_connector(self, broker_name: str, connector_class: Type[BaseConnector]):
        """Register a new connector class"""
        self.connectors[broker_name.lower()] = connector_class
    
    def list_supported_brokers(self) -> list:
        """Get list of all supported brokers"""
        return list(self.connectors.keys())
    
    def is_broker_supported(self, broker_name: str) -> bool:
        """Check if broker is supported"""
        return broker_name.lower() in self.connectors


__all__ = [
    'BaseConnector',
    'MT5Connector', 
    'CCXTConnector',
    'IBKRConnector',
    'BrokerManager'
]
