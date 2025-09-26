#!/usr/bin/env python3
"""
Trading Strategies Module
Implements various trading strategies including Smart Money Concept,
Depth of Market Analysis, and Machine Learning strategies
"""

from typing import Dict, List, Optional, Type, Any
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass

# Import strategy implementations
try:
    from .smc_strategy import SMCStrategy
except ImportError:
    SMCStrategy = None

try:
    from .dom_analysis import DOMAnalysis
except ImportError:
    DOMAnalysis = None

try:
    from .ml_strategy import MLStrategy
except ImportError:
    MLStrategy = None


@dataclass
class Signal:
    """Trading signal data structure"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    strategy_name: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_active = False
        self.last_signals = []
        self.performance_metrics = {
            'total_signals': 0,
            'profitable_signals': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0
        }
        
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on market data
        
        Args:
            market_data: Dict containing market data (prices, volume, etc.)
            
        Returns:
            List[Signal]: List of trading signals
        """
        pass
    
    @abstractmethod
    def update_parameters(self, new_params: Dict[str, Any]) -> bool:
        """Update strategy parameters
        
        Args:
            new_params: Dictionary of new parameter values
            
        Returns:
            bool: True if update successful
        """
        pass
    
    def activate(self):
        """Activate the strategy"""
        self.is_active = True
        
    def deactivate(self):
        """Deactivate the strategy"""
        self.is_active = False
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics"""
        return self.performance_metrics.copy()
    
    def update_performance(self, signal: Signal, pnl: float):
        """Update performance metrics after trade execution
        
        Args:
            signal: The executed signal
            pnl: Profit/loss from the trade
        """
        self.performance_metrics['total_signals'] += 1
        self.performance_metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.performance_metrics['profitable_signals'] += 1
            
        # Calculate win rate
        if self.performance_metrics['total_signals'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['profitable_signals'] / 
                self.performance_metrics['total_signals']
            )
            
        # Calculate average profit
        self.performance_metrics['avg_profit'] = (
            self.performance_metrics['total_pnl'] / 
            self.performance_metrics['total_signals']
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration"""
        return self.config.copy()
    
    def validate_signal(self, signal: Signal) -> bool:
        """Validate a trading signal
        
        Args:
            signal: Signal to validate
            
        Returns:
            bool: True if signal is valid
        """
        if not signal.symbol:
            return False
            
        if signal.action not in ['buy', 'sell', 'hold']:
            return False
            
        if not (0.0 <= signal.confidence <= 1.0):
            return False
            
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class StrategyManager:
    """Manager for all trading strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        self.active_strategies: Dict[str, BaseStrategy] = {}
        
        # Register available strategies
        if SMCStrategy:
            self.strategies['smc'] = SMCStrategy
            self.strategies['smart_money_concept'] = SMCStrategy
            
        if DOMAnalysis:
            self.strategies['dom'] = DOMAnalysis
            self.strategies['depth_of_market'] = DOMAnalysis
            
        if MLStrategy:
            self.strategies['ml'] = MLStrategy
            self.strategies['machine_learning'] = MLStrategy
    
    def create_strategy(self, strategy_type: str, name: str, config: Dict[str, Any] = None) -> Optional[BaseStrategy]:
        """Create a new strategy instance
        
        Args:
            strategy_type: Type of strategy to create
            name: Name for the strategy instance
            config: Strategy configuration
            
        Returns:
            BaseStrategy: Created strategy instance or None
        """
        strategy_type = strategy_type.lower()
        
        if strategy_type not in self.strategies:
            return None
            
        strategy_class = self.strategies[strategy_type]
        strategy = strategy_class(name, config)
        
        return strategy
    
    def add_strategy(self, strategy: BaseStrategy) -> bool:
        """Add a strategy to active strategies
        
        Args:
            strategy: Strategy instance to add
            
        Returns:
            bool: True if added successfully
        """
        if strategy.name in self.active_strategies:
            return False
            
        self.active_strategies[strategy.name] = strategy
        return True
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy from active strategies
        
        Args:
            strategy_name: Name of strategy to remove
            
        Returns:
            bool: True if removed successfully
        """
        if strategy_name in self.active_strategies:
            self.active_strategies[strategy_name].deactivate()
            del self.active_strategies[strategy_name]
            return True
        return False
    
    def get_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name
        
        Args:
            strategy_name: Name of strategy to get
            
        Returns:
            BaseStrategy: Strategy instance or None
        """
        return self.active_strategies.get(strategy_name)
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all available strategy types
        
        Returns:
            List[Dict]: List of strategy information
        """
        strategies = []
        for strategy_type, strategy_class in self.strategies.items():
            strategies.append({
                'type': strategy_type,
                'class_name': strategy_class.__name__,
                'description': strategy_class.__doc__ or 'No description available'
            })
        return strategies
    
    def list_active_strategies(self) -> List[Dict[str, Any]]:
        """List all active strategy instances
        
        Returns:
            List[Dict]: List of active strategy information
        """
        active = []
        for name, strategy in self.active_strategies.items():
            active.append({
                'name': name,
                'type': strategy.__class__.__name__,
                'is_active': strategy.is_active,
                'performance': strategy.get_performance_metrics()
            })
        return active
    
    def generate_all_signals(self, market_data: Dict[str, Any]) -> Dict[str, List[Signal]]:
        """Generate signals from all active strategies
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Dict[str, List[Signal]]: Signals grouped by strategy name
        """
        all_signals = {}
        
        for name, strategy in self.active_strategies.items():
            if strategy.is_active:
                try:
                    signals = strategy.generate_signals(market_data)
                    # Validate signals
                    valid_signals = [signal for signal in signals if strategy.validate_signal(signal)]
                    all_signals[name] = valid_signals
                    strategy.last_signals = valid_signals
                except Exception as e:
                    print(f"Error generating signals for {name}: {e}")
                    all_signals[name] = []
        
        return all_signals
    
    def get_combined_signals(self, market_data: Dict[str, Any], 
                           symbol: str = None) -> List[Signal]:
        """Get combined signals from all strategies for a symbol
        
        Args:
            market_data: Market data to analyze
            symbol: Optional symbol filter
            
        Returns:
            List[Signal]: Combined signals
        """
        all_signals = self.generate_all_signals(market_data)
        combined = []
        
        for strategy_signals in all_signals.values():
            for signal in strategy_signals:
                if symbol is None or signal.symbol == symbol:
                    combined.append(signal)
        
        return combined
    
    def register_strategy(self, strategy_type: str, strategy_class: Type[BaseStrategy]):
        """Register a new strategy type
        
        Args:
            strategy_type: Name of the strategy type
            strategy_class: Strategy class
        """
        self.strategies[strategy_type.lower()] = strategy_class
    
    def is_strategy_supported(self, strategy_type: str) -> bool:
        """Check if strategy type is supported
        
        Args:
            strategy_type: Strategy type to check
            
        Returns:
            bool: True if supported
        """
        return strategy_type.lower() in self.strategies


__all__ = [
    'BaseStrategy',
    'Signal', 
    'StrategyManager',
    'SMCStrategy',
    'DOMAnalysis',
    'MLStrategy'
]