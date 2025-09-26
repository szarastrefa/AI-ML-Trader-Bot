#!/usr/bin/env python3
"""
Risk Management System
Comprehensive risk management with position sizing, portfolio protection, and automated controls
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    portfolio_value: float
    total_exposure: float
    available_margin: float
    used_margin: float
    margin_level: float
    unrealized_pnl: float
    daily_pnl: float
    max_drawdown: float
    var_1d: float  # Value at Risk 1 day
    var_1w: float  # Value at Risk 1 week
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    risk_level: RiskLevel
    last_updated: datetime


class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.position_history = []
        self.pnl_history = []
        self.risk_metrics_history = []
        self.is_emergency_stop = False
        self.daily_loss = 0.0
        self.max_daily_loss = self.config['max_daily_loss']
        self.circuit_breaker_triggered = False
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk management configuration"""
        return {
            # Position sizing
            'max_risk_per_trade': 0.02,      # 2% of capital per trade
            'max_portfolio_risk': 0.10,      # 10% total portfolio risk
            'min_position_size': 0.01,       # Minimum position size
            'max_position_size': 1.0,        # Maximum position size
            
            # Portfolio protection
            'max_drawdown': 0.15,            # 15% maximum drawdown
            'max_daily_loss': 0.05,          # 5% maximum daily loss
            'max_correlation': 0.7,          # Maximum correlation between positions
            'max_positions': 10,             # Maximum number of positions
            
            # Risk calculations
            'confidence_level': 0.95,        # VaR confidence level
            'lookback_days': 252,            # Trading days for calculations
            'volatility_window': 20,         # Days for volatility calculation
            
            # Circuit breakers
            'enable_circuit_breaker': True,
            'circuit_breaker_threshold': 0.08,  # 8% loss triggers circuit breaker
            'emergency_stop_threshold': 0.12,   # 12% loss triggers emergency stop
            
            # Kelly Criterion
            'use_kelly_criterion': True,
            'kelly_fraction': 0.25,          # Use 25% of Kelly optimal
            'min_kelly_size': 0.001,         # Minimum Kelly position size
            'max_kelly_size': 0.05,          # Maximum Kelly position size
            
            # Dynamic adjustments
            'volatility_adjustment': True,
            'correlation_adjustment': True,
            'performance_adjustment': True,
        }
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                              stop_loss: float, confidence: float = 1.0,
                              symbol: str = None) -> float:
        """Calculate optimal position size
        
        Args:
            account_balance: Account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Signal confidence (0-1)
            symbol: Trading symbol (for symbol-specific adjustments)
            
        Returns:
            Optimal position size
        """
        try:
            # Calculate risk per share/unit
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit == 0:
                logger.warning("Risk per unit is zero, using minimum position size")
                return self.config['min_position_size']
            
            # Base position size (fixed fractional method)
            max_risk_amount = account_balance * self.config['max_risk_per_trade']
            base_position_size = max_risk_amount / risk_per_unit
            
            # Apply Kelly Criterion if enabled
            if self.config['use_kelly_criterion']:
                kelly_size = self._calculate_kelly_position_size(
                    account_balance, entry_price, stop_loss, symbol
                )
                # Use fraction of Kelly optimal
                kelly_adjusted = kelly_size * self.config['kelly_fraction']
                base_position_size = min(base_position_size, kelly_adjusted)
            
            # Apply confidence adjustment
            position_size = base_position_size * confidence
            
            # Apply volatility adjustment if enabled
            if self.config['volatility_adjustment'] and symbol:
                volatility_multiplier = self._get_volatility_adjustment(symbol)
                position_size *= volatility_multiplier
            
            # Apply portfolio correlation adjustment
            if self.config['correlation_adjustment']:
                correlation_multiplier = self._get_correlation_adjustment(symbol)
                position_size *= correlation_multiplier
            
            # Ensure position size is within limits
            position_size = max(position_size, self.config['min_position_size'])
            position_size = min(position_size, self.config['max_position_size'])
            
            # Check against portfolio limits
            max_portfolio_size = account_balance * self.config['max_portfolio_risk']
            if position_size * entry_price > max_portfolio_size:
                position_size = max_portfolio_size / entry_price
            
            logger.info(f"Calculated position size: {position_size:.6f} for {symbol}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.config['min_position_size']
    
    def _calculate_kelly_position_size(self, balance: float, entry_price: float, 
                                     stop_loss: float, symbol: str = None) -> float:
        """Calculate Kelly Criterion optimal position size"""
        try:
            # Estimate win probability and average win/loss from historical data
            # This is a simplified version - in practice, use historical strategy performance
            
            # Default estimates (should be calculated from actual strategy performance)
            win_probability = 0.55  # 55% win rate
            avg_win = 0.015        # 1.5% average win
            avg_loss = 0.02        # 2% average loss (risk per trade)
            
            # Kelly formula: f = (bp - q) / b
            # where: b = odds (avg_win/avg_loss), p = win_prob, q = loss_prob
            b = avg_win / avg_loss
            p = win_probability
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Ensure Kelly fraction is positive and reasonable
            kelly_fraction = max(0, min(kelly_fraction, self.config['max_kelly_size']))
            kelly_fraction = max(kelly_fraction, self.config['min_kelly_size'])
            
            # Calculate position size based on Kelly fraction
            kelly_position_size = (balance * kelly_fraction) / entry_price
            
            return kelly_position_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}")
            return balance * 0.01 / entry_price  # 1% fallback
    
    def _get_volatility_adjustment(self, symbol: str) -> float:
        """Get volatility-based position size adjustment
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Volatility adjustment multiplier (0.5 to 2.0)
        """
        try:
            # This should use actual market data to calculate volatility
            # For now, return neutral multiplier
            base_volatility = 0.02  # 2% daily volatility baseline
            current_volatility = 0.02  # Should be calculated from recent price data
            
            # Adjust position size inversely to volatility
            volatility_ratio = base_volatility / current_volatility if current_volatility > 0 else 1.0
            
            # Limit adjustment range
            return max(0.5, min(2.0, volatility_ratio))
            
        except Exception:
            return 1.0  # Neutral adjustment
    
    def _get_correlation_adjustment(self, symbol: str) -> float:
        """Get correlation-based position size adjustment
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Correlation adjustment multiplier (0.5 to 1.0)
        """
        try:
            # This should calculate correlation with existing positions
            # For now, return neutral multiplier
            max_correlation = 0.0  # Should be calculated from position correlations
            
            if max_correlation > self.config['max_correlation']:
                # Reduce position size for highly correlated positions
                adjustment = 1.0 - (max_correlation - self.config['max_correlation'])
                return max(0.5, adjustment)
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def validate_signals(self, signals: List[Any], strategy_config: Dict[str, Any]) -> List[Any]:
        """Validate trading signals through risk filters
        
        Args:
            signals: List of trading signals
            strategy_config: Strategy configuration
            
        Returns:
            List of validated signals
        """
        validated_signals = []
        
        try:
            for signal in signals:
                if self._validate_individual_signal(signal, strategy_config):
                    validated_signals.append(signal)
                else:
                    logger.info(f"Signal rejected by risk management: {signal.symbol} {signal.action}")
            
            return validated_signals
            
        except Exception as e:
            logger.error(f"Error validating signals: {e}")
            return []
    
    def _validate_individual_signal(self, signal: Any, strategy_config: Dict[str, Any]) -> bool:
        """Validate individual trading signal
        
        Args:
            signal: Trading signal to validate
            strategy_config: Strategy configuration
            
        Returns:
            bool: True if signal passes validation
        """
        try:
            # Check if emergency stop is active
            if self.is_emergency_stop:
                logger.warning("Emergency stop active, rejecting all signals")
                return False
            
            # Check if circuit breaker is triggered
            if self.circuit_breaker_triggered:
                logger.warning("Circuit breaker triggered, rejecting signals")
                return False
            
            # Check daily loss limit
            if abs(self.daily_loss) >= self.max_daily_loss:
                logger.warning(f"Daily loss limit exceeded: {self.daily_loss:.2%}")
                return False
            
            # Check signal confidence
            min_confidence = strategy_config.get('min_confidence', 0.5)
            if signal.confidence < min_confidence:
                logger.debug(f"Signal confidence too low: {signal.confidence:.2f} < {min_confidence:.2f}")
                return False
            
            # Check maximum positions limit
            current_positions = len(self.position_history)
            if current_positions >= self.config['max_positions']:
                logger.warning(f"Maximum positions limit reached: {current_positions}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def update_risk_metrics(self, portfolio_summary: Dict[str, Any]):
        """Update risk metrics based on portfolio summary
        
        Args:
            portfolio_summary: Current portfolio summary
        """
        try:
            # Extract portfolio data
            total_equity = portfolio_summary.get('total_equity', 0.0)
            total_pnl = portfolio_summary.get('total_pnl', 0.0)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_summary)
            
            # Update daily P&L
            self._update_daily_pnl(total_pnl)
            
            # Check risk thresholds
            self._check_risk_thresholds(risk_metrics)
            
            # Store metrics history
            self.risk_metrics_history.append({
                'metrics': risk_metrics,
                'timestamp': datetime.now()
            })
            
            # Keep only recent history (last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            self.risk_metrics_history = [
                entry for entry in self.risk_metrics_history 
                if entry['timestamp'] > cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def _calculate_risk_metrics(self, portfolio_summary: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            total_equity = portfolio_summary.get('total_equity', 0.0)
            total_pnl = portfolio_summary.get('total_pnl', 0.0)
            
            # Calculate portfolio exposure
            total_exposure = 0.0
            used_margin = 0.0
            
            for account in portfolio_summary.get('accounts', []):
                total_exposure += account.get('equity', 0.0)
                used_margin += account.get('balance', {}).get('used', 0.0)
            
            # Calculate margin level
            margin_level = (total_equity / used_margin * 100) if used_margin > 0 else 100.0
            
            # Calculate drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Calculate VaR (simplified)
            var_1d = self._calculate_var(1)  # 1 day VaR
            var_1w = self._calculate_var(7)  # 1 week VaR
            
            # Calculate performance ratios
            sharpe_ratio = self._calculate_sharpe_ratio()
            sortino_ratio = self._calculate_sortino_ratio()
            calmar_ratio = self._calculate_calmar_ratio(max_drawdown)
            
            # Determine risk level
            risk_level = self._determine_risk_level(max_drawdown, margin_level, var_1d)
            
            return RiskMetrics(
                portfolio_value=total_equity,
                total_exposure=total_exposure,
                available_margin=total_equity - used_margin,
                used_margin=used_margin,
                margin_level=margin_level,
                unrealized_pnl=total_pnl,
                daily_pnl=self.daily_loss,
                max_drawdown=max_drawdown,
                var_1d=var_1d,
                var_1w=var_1w,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                risk_level=risk_level,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                portfolio_value=0.0, total_exposure=0.0, available_margin=0.0,
                used_margin=0.0, margin_level=100.0, unrealized_pnl=0.0,
                daily_pnl=0.0, max_drawdown=0.0, var_1d=0.0, var_1w=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                risk_level=RiskLevel.LOW, last_updated=datetime.now()
            )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity history"""
        try:
            if len(self.pnl_history) < 2:
                return 0.0
            
            # Convert to numpy array for calculation
            equity_curve = np.array([entry['equity'] for entry in self.pnl_history])
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(equity_curve)
            
            # Calculate drawdown
            drawdown = (equity_curve - running_max) / running_max
            
            # Return maximum drawdown (negative value)
            return abs(np.min(drawdown))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_var(self, days: int = 1, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk
        
        Args:
            days: Number of days
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            VaR value
        """
        try:
            if len(self.pnl_history) < 30:
                return 0.0
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(self.pnl_history)):
                prev_equity = self.pnl_history[i-1]['equity']
                curr_equity = self.pnl_history[i]['equity']
                if prev_equity > 0:
                    daily_return = (curr_equity - prev_equity) / prev_equity
                    returns.append(daily_return)
            
            if not returns:
                return 0.0
            
            # Calculate VaR using historical simulation
            returns_array = np.array(returns)
            percentile = (1 - confidence_level) * 100
            var_1d = np.percentile(returns_array, percentile)
            
            # Scale to requested time horizon
            var = var_1d * np.sqrt(days)
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.pnl_history) < 30:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, len(self.pnl_history)):
                prev_equity = self.pnl_history[i-1]['equity']
                curr_equity = self.pnl_history[i]['equity']
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            
            # Annualize returns and volatility
            annual_return = np.mean(returns_array) * 252  # 252 trading days
            annual_volatility = np.std(returns_array) * np.sqrt(252)
            
            if annual_volatility == 0:
                return 0.0
            
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if len(self.pnl_history) < 30:
                return 0.0
            
            returns = []
            for i in range(1, len(self.pnl_history)):
                prev_equity = self.pnl_history[i-1]['equity']
                curr_equity = self.pnl_history[i]['equity']
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            
            # Calculate downside deviation (only negative returns)
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) == 0:
                return float('inf')  # No downside risk
            
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            annual_return = np.mean(returns_array) * 252
            
            if downside_deviation == 0:
                return float('inf')
            
            sortino = (annual_return - risk_free_rate) / downside_deviation
            return sortino
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        try:
            if len(self.pnl_history) < 30 or max_drawdown == 0:
                return 0.0
            
            # Calculate annualized return
            if len(self.pnl_history) >= 2:
                initial_equity = self.pnl_history[0]['equity']
                final_equity = self.pnl_history[-1]['equity']
                
                if initial_equity > 0:
                    total_return = (final_equity - initial_equity) / initial_equity
                    days = len(self.pnl_history)
                    annual_return = total_return * (252 / days)
                    
                    calmar = annual_return / max_drawdown
                    return calmar
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def _determine_risk_level(self, max_drawdown: float, margin_level: float, var_1d: float) -> RiskLevel:
        """Determine overall portfolio risk level"""
        try:
            risk_score = 0
            
            # Drawdown risk
            if max_drawdown > 0.10:  # > 10%
                risk_score += 3
            elif max_drawdown > 0.05:  # > 5%
                risk_score += 2
            elif max_drawdown > 0.02:  # > 2%
                risk_score += 1
            
            # Margin risk
            if margin_level < 150:  # < 150%
                risk_score += 3
            elif margin_level < 300:  # < 300%
                risk_score += 2
            elif margin_level < 500:  # < 500%
                risk_score += 1
            
            # VaR risk
            if var_1d > 0.05:  # > 5%
                risk_score += 3
            elif var_1d > 0.03:  # > 3%
                risk_score += 2
            elif var_1d > 0.01:  # > 1%
                risk_score += 1
            
            # Daily loss risk
            daily_loss_percent = abs(self.daily_loss)
            if daily_loss_percent > 0.04:  # > 4%
                risk_score += 3
            elif daily_loss_percent > 0.02:  # > 2%
                risk_score += 2
            elif daily_loss_percent > 0.01:  # > 1%
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 8:
                return RiskLevel.CRITICAL
            elif risk_score >= 5:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return RiskLevel.MEDIUM
    
    def _update_daily_pnl(self, current_pnl: float):
        """Update daily P&L tracking"""
        try:
            # Reset daily loss at start of new day
            now = datetime.now()
            if (hasattr(self, '_last_pnl_reset') and 
                self._last_pnl_reset.date() < now.date()):
                self.daily_loss = 0.0
            
            # Update daily loss (accumulative)
            if hasattr(self, '_previous_pnl'):
                pnl_change = current_pnl - self._previous_pnl
                if pnl_change < 0:  # Only count losses
                    self.daily_loss += abs(pnl_change)
            
            self._previous_pnl = current_pnl
            self._last_pnl_reset = now
            
        except Exception as e:
            logger.error(f"Error updating daily P&L: {e}")
    
    def _check_risk_thresholds(self, risk_metrics: RiskMetrics):
        """Check risk thresholds and trigger protective actions"""
        try:
            # Check circuit breaker threshold
            if (self.config['enable_circuit_breaker'] and
                risk_metrics.max_drawdown >= self.config['circuit_breaker_threshold']):
                
                if not self.circuit_breaker_triggered:
                    logger.critical(f"Circuit breaker triggered! Max drawdown: {risk_metrics.max_drawdown:.2%}")
                    self.circuit_breaker_triggered = True
                    self._trigger_circuit_breaker()
            
            # Check emergency stop threshold
            if risk_metrics.max_drawdown >= self.config['emergency_stop_threshold']:
                if not self.is_emergency_stop:
                    logger.critical(f"Emergency stop triggered! Max drawdown: {risk_metrics.max_drawdown:.2%}")
                    self.is_emergency_stop = True
                    self._trigger_emergency_stop()
            
            # Check margin level
            if risk_metrics.margin_level < 150:  # 150% margin level
                logger.warning(f"Low margin level: {risk_metrics.margin_level:.1f}%")
            
            # Check daily loss limit
            if abs(self.daily_loss) >= self.max_daily_loss * 0.8:  # 80% of limit
                logger.warning(f"Daily loss approaching limit: {self.daily_loss:.2%}")
                
        except Exception as e:
            logger.error(f"Error checking risk thresholds: {e}")
    
    def _trigger_circuit_breaker(self):
        """Trigger circuit breaker (stop new positions)"""
        logger.critical("CIRCUIT BREAKER TRIGGERED - No new positions allowed")
        # Additional actions could be added here (notifications, etc.)
    
    def _trigger_emergency_stop(self):
        """Trigger emergency stop (close all positions)"""
        logger.critical("EMERGENCY STOP TRIGGERED - All positions should be closed")
        # Additional actions could be added here (force close positions, notifications, etc.)
    
    def get_portfolio_risk_metrics(self) -> Dict[str, Any]:
        """Get current portfolio risk metrics
        
        Returns:
            Dictionary of current risk metrics
        """
        try:
            if not self.risk_metrics_history:
                return {'error': 'No risk metrics available'}
            
            latest_metrics = self.risk_metrics_history[-1]['metrics']
            
            return {
                'portfolio_value': latest_metrics.portfolio_value,
                'total_exposure': latest_metrics.total_exposure,
                'margin_level': latest_metrics.margin_level,
                'max_drawdown': latest_metrics.max_drawdown,
                'daily_pnl': latest_metrics.daily_pnl,
                'var_1d': latest_metrics.var_1d,
                'var_1w': latest_metrics.var_1w,
                'sharpe_ratio': latest_metrics.sharpe_ratio,
                'sortino_ratio': latest_metrics.sortino_ratio,
                'calmar_ratio': latest_metrics.calmar_ratio,
                'risk_level': latest_metrics.risk_level.value,
                'is_emergency_stop': self.is_emergency_stop,
                'circuit_breaker_triggered': self.circuit_breaker_triggered,
                'last_updated': latest_metrics.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {'error': str(e)}
    
    def reset_circuit_breaker(self) -> bool:
        """Reset circuit breaker (manual override)
        
        Returns:
            bool: True if reset successful
        """
        try:
            self.circuit_breaker_triggered = False
            logger.info("Circuit breaker reset manually")
            return True
        except Exception as e:
            logger.error(f"Error resetting circuit breaker: {e}")
            return False
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop (manual override)
        
        Returns:
            bool: True if reset successful
        """
        try:
            self.is_emergency_stop = False
            logger.info("Emergency stop reset manually")
            return True
        except Exception as e:
            logger.error(f"Error resetting emergency stop: {e}")
            return False
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update risk management configuration
        
        Args:
            new_config: New configuration values
            
        Returns:
            bool: True if update successful
        """
        try:
            self.config.update(new_config)
            self.max_daily_loss = self.config['max_daily_loss']
            logger.info("Risk management configuration updated")
            return True
        except Exception as e:
            logger.error(f"Error updating risk config: {e}")
            return False


__all__ = [
    'RiskManager',
    'RiskMetrics',
    'RiskLevel'
]