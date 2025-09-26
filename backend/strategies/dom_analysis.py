#!/usr/bin/env python3
"""
Depth of Market (DOM) Analysis Strategy
Real-time order book analysis for institutional order flow detection
Includes imbalance detection, iceberg orders, and volume profile analysis
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from . import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class DOMAnalysis(BaseStrategy):
    """Depth of Market analysis strategy for order flow trading"""
    
    def __init__(self, name: str = "DOM_Analysis", config: Dict[str, Any] = None):
        default_config = {
            'order_book_levels': 20,              # Levels of order book to analyze
            'imbalance_threshold': 2.0,           # Ratio threshold for buy/sell imbalance
            'volume_profile_periods': 100,        # Periods for volume profile
            'iceberg_detection': True,            # Enable iceberg order detection
            'flow_analysis_window': 10,           # Minutes for order flow analysis
            'min_order_size': 1000,               # Minimum order size to consider (in base currency)
            'large_order_threshold': 10000,       # Large order threshold
            'price_level_aggregation': 0.0001,    # Price level aggregation (0.01%)
            'time_decay_factor': 0.95,            # Decay factor for historical flow
            'confidence_threshold': 0.6,          # Minimum confidence for signals
            'min_liquidity_ratio': 1.5,           # Minimum liquidity ratio for trading
            'momentum_lookback': 5,               # Minutes to look back for momentum
            'support_resistance_strength': 3,     # Minimum touches for S/R levels
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name, default_config)
        
        # Internal state for DOM analysis
        self.order_book_history = defaultdict(lambda: deque(maxlen=1000))
        self.order_flow_history = defaultdict(lambda: deque(maxlen=500))
        self.volume_profile = defaultdict(dict)
        self.detected_icebergs = defaultdict(list)
        self.support_resistance_levels = defaultdict(list)
        self.market_maker_activity = defaultdict(dict)
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate DOM-based trading signals
        
        Args:
            market_data: Dict containing order book data, volume, etc.
            
        Returns:
            List[Signal]: Generated trading signals
        """
        signals = []
        
        try:
            # Process order book data
            if 'order_books' not in market_data:
                return signals
            
            for symbol, order_book_data in market_data['order_books'].items():
                if not order_book_data:
                    continue
                
                # Store order book history
                self.order_book_history[symbol].append({
                    'data': order_book_data,
                    'timestamp': datetime.now()
                })
                
                # Analyze current order book
                analysis = self._analyze_order_book(symbol, order_book_data)
                
                # Detect order flow patterns
                flow_analysis = self._analyze_order_flow(symbol)
                
                # Detect iceberg orders
                iceberg_signals = self._detect_iceberg_orders(symbol, order_book_data)
                
                # Calculate volume profile
                volume_profile = self._calculate_volume_profile(symbol)
                
                # Generate signals based on DOM analysis
                symbol_signals = self._generate_dom_signals(
                    symbol, analysis, flow_analysis, iceberg_signals, volume_profile
                )
                
                signals.extend(symbol_signals)
                
        except Exception as e:
            logger.error(f"Error generating DOM signals: {e}")
        
        return signals
    
    def _analyze_order_book(self, symbol: str, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current order book for imbalances and patterns"""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return {}
            
            analysis = {
                'timestamp': datetime.now(),
                'spread': asks[0]['price'] - bids[0]['price'] if bids and asks else 0,
                'mid_price': (bids[0]['price'] + asks[0]['price']) / 2 if bids and asks else 0
            }
            
            # Calculate bid/ask volume imbalance
            total_bid_volume = sum(entry['amount'] for entry in bids[:self.config['order_book_levels']])
            total_ask_volume = sum(entry['amount'] for entry in asks[:self.config['order_book_levels']])
            
            if total_ask_volume > 0:
                analysis['volume_imbalance'] = total_bid_volume / total_ask_volume
                analysis['imbalance_strength'] = abs(analysis['volume_imbalance'] - 1.0)
            else:
                analysis['volume_imbalance'] = float('inf')
                analysis['imbalance_strength'] = 1.0
            
            # Analyze order sizes
            bid_sizes = [entry['amount'] for entry in bids]
            ask_sizes = [entry['amount'] for entry in asks]
            
            analysis['avg_bid_size'] = np.mean(bid_sizes) if bid_sizes else 0
            analysis['avg_ask_size'] = np.mean(ask_sizes) if ask_sizes else 0
            analysis['max_bid_size'] = max(bid_sizes) if bid_sizes else 0
            analysis['max_ask_size'] = max(ask_sizes) if ask_sizes else 0
            
            # Calculate liquidity metrics
            analysis['bid_liquidity'] = total_bid_volume
            analysis['ask_liquidity'] = total_ask_volume
            analysis['total_liquidity'] = total_bid_volume + total_ask_volume
            analysis['liquidity_ratio'] = analysis['total_liquidity'] / analysis['mid_price'] if analysis['mid_price'] > 0 else 0
            
            # Detect large orders (potential institutional activity)
            large_bids = [entry for entry in bids if entry['amount'] * entry['price'] > self.config['large_order_threshold']]
            large_asks = [entry for entry in asks if entry['amount'] * entry['price'] > self.config['large_order_threshold']]
            
            analysis['large_bid_count'] = len(large_bids)
            analysis['large_ask_count'] = len(large_asks)
            analysis['institutional_bias'] = len(large_bids) - len(large_asks)
            
            # Calculate order book slope (price impact)
            analysis['bid_slope'] = self._calculate_order_book_slope(bids, 'bid')
            analysis['ask_slope'] = self._calculate_order_book_slope(asks, 'ask')
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing order book for {symbol}: {e}")
            return {}
    
    def _calculate_order_book_slope(self, orders: List[Dict], side: str) -> float:
        """Calculate order book slope (measure of liquidity)"""
        try:
            if len(orders) < 5:
                return 0.0
            
            # Calculate cumulative volume and price levels
            cumulative_volume = 0
            price_levels = []
            volumes = []
            
            for order in orders[:10]:  # Use first 10 levels
                cumulative_volume += order['amount']
                price_levels.append(order['price'])
                volumes.append(cumulative_volume)
            
            # Calculate slope using linear regression
            if len(price_levels) >= 2:
                price_array = np.array(price_levels)
                volume_array = np.array(volumes)
                
                # Normalize prices relative to first level
                if side == 'bid':
                    price_diff = price_array[0] - price_array
                else:  # ask
                    price_diff = price_array - price_array[0]
                
                # Calculate slope (volume per price unit)
                if np.std(price_diff) > 0:
                    slope = np.corrcoef(price_diff, volume_array)[0, 1]
                    return slope if not np.isnan(slope) else 0.0
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_order_flow(self, symbol: str) -> Dict[str, Any]:
        """Analyze order flow patterns over time"""
        try:
            if len(self.order_book_history[symbol]) < 2:
                return {}
            
            # Get recent order book snapshots
            recent_books = list(self.order_book_history[symbol])[-self.config['flow_analysis_window']:]
            
            flow_metrics = {
                'net_flow': 0.0,
                'buy_pressure': 0.0,
                'sell_pressure': 0.0,
                'flow_momentum': 0.0,
                'flow_acceleration': 0.0
            }
            
            # Calculate order flow changes
            buy_flow_changes = []
            sell_flow_changes = []
            
            for i in range(1, len(recent_books)):
                prev_book = recent_books[i-1]['data']
                curr_book = recent_books[i]['data']
                
                # Calculate volume changes at best bid/ask
                if (prev_book.get('bids') and curr_book.get('bids') and
                    prev_book.get('asks') and curr_book.get('asks')):
                    
                    # Best bid volume change
                    prev_best_bid_vol = prev_book['bids'][0]['amount'] if prev_book['bids'] else 0
                    curr_best_bid_vol = curr_book['bids'][0]['amount'] if curr_book['bids'] else 0
                    bid_change = curr_best_bid_vol - prev_best_bid_vol
                    buy_flow_changes.append(bid_change)
                    
                    # Best ask volume change  
                    prev_best_ask_vol = prev_book['asks'][0]['amount'] if prev_book['asks'] else 0
                    curr_best_ask_vol = curr_book['asks'][0]['amount'] if curr_book['asks'] else 0
                    ask_change = curr_best_ask_vol - prev_best_ask_vol
                    sell_flow_changes.append(ask_change)
            
            if buy_flow_changes and sell_flow_changes:
                flow_metrics['buy_pressure'] = np.mean(buy_flow_changes)
                flow_metrics['sell_pressure'] = np.mean(sell_flow_changes)
                flow_metrics['net_flow'] = flow_metrics['buy_pressure'] - flow_metrics['sell_pressure']
                
                # Calculate momentum (rate of change)
                if len(buy_flow_changes) >= 3:
                    flow_metrics['flow_momentum'] = (buy_flow_changes[-1] - buy_flow_changes[-3]) / 2
                    
                # Calculate acceleration (change in momentum)
                if len(buy_flow_changes) >= 5:
                    recent_momentum = np.mean(buy_flow_changes[-3:])
                    earlier_momentum = np.mean(buy_flow_changes[-5:-2])
                    flow_metrics['flow_acceleration'] = recent_momentum - earlier_momentum
            
            return flow_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing order flow for {symbol}: {e}")
            return {}
    
    def _detect_iceberg_orders(self, symbol: str, order_book: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potential iceberg orders"""
        iceberg_signals = []
        
        try:
            if not self.config['iceberg_detection']:
                return iceberg_signals
            
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            # Look for consistent order sizes at specific price levels
            for side, orders in [('bid', bids), ('ask', asks)]:
                if len(orders) < 5:
                    continue
                
                # Group orders by price level (with aggregation)
                price_levels = defaultdict(float)
                for order in orders:
                    # Aggregate orders at similar price levels
                    price_key = round(order['price'] / self.config['price_level_aggregation']) * self.config['price_level_aggregation']
                    price_levels[price_key] += order['amount']
                
                # Look for unusually large orders that might be icebergs
                volumes = list(price_levels.values())
                if volumes:
                    avg_volume = np.mean(volumes)
                    std_volume = np.std(volumes)
                    
                    for price, volume in price_levels.items():
                        # Detect potential iceberg (unusually large order)
                        if volume > avg_volume + (2 * std_volume) and volume > self.config['large_order_threshold']:
                            
                            # Check historical behavior at this price level
                            historical_strength = self._check_price_level_history(symbol, price)
                            
                            if historical_strength > self.config['support_resistance_strength']:
                                iceberg_signals.append({
                                    'type': f'iceberg_{side}',
                                    'price': price,
                                    'volume': volume,
                                    'strength': historical_strength,
                                    'significance': volume / avg_volume if avg_volume > 0 else 1,
                                    'timestamp': datetime.now()
                                })
            
            return iceberg_signals
            
        except Exception as e:
            logger.error(f"Error detecting iceberg orders for {symbol}: {e}")
            return []
    
    def _check_price_level_history(self, symbol: str, price: float, tolerance: float = 0.001) -> float:
        """Check historical significance of a price level"""
        try:
            touches = 0
            total_volume = 0
            
            # Check recent order book history
            for book_entry in list(self.order_book_history[symbol])[-50:]:  # Last 50 snapshots
                book_data = book_entry['data']
                
                # Check bids and asks for touches near this price
                all_orders = book_data.get('bids', []) + book_data.get('asks', [])
                
                for order in all_orders:
                    if abs(order['price'] - price) / price <= tolerance:
                        touches += 1
                        total_volume += order['amount']
            
            # Return strength score based on touches and volume
            strength = touches + (total_volume / self.config['large_order_threshold'])
            return min(strength, 10.0)  # Cap at 10
            
        except Exception:
            return 0.0
    
    def _calculate_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """Calculate volume profile from recent data"""
        try:
            if len(self.order_book_history[symbol]) < self.config['volume_profile_periods']:
                return {}
            
            # Get recent order book data
            recent_data = list(self.order_book_history[symbol])[-self.config['volume_profile_periods']:]
            
            # Aggregate volume by price levels
            volume_by_price = defaultdict(float)
            price_range = {'min': float('inf'), 'max': 0}
            
            for book_entry in recent_data:
                book_data = book_entry['data']
                
                # Process bids and asks
                for orders, weight in [(book_data.get('bids', []), 1.0), (book_data.get('asks', []), 1.0)]:
                    for order in orders:
                        price = order['price']
                        volume = order['amount'] * weight
                        
                        # Aggregate by price level
                        price_key = round(price / self.config['price_level_aggregation']) * self.config['price_level_aggregation']
                        volume_by_price[price_key] += volume
                        
                        # Update price range
                        price_range['min'] = min(price_range['min'], price)
                        price_range['max'] = max(price_range['max'], price)
            
            if not volume_by_price:
                return {}
            
            # Calculate volume profile metrics
            volumes = list(volume_by_price.values())
            prices = list(volume_by_price.keys())
            
            # Find Point of Control (POC) - price with highest volume
            max_volume_idx = np.argmax(volumes)
            poc_price = prices[max_volume_idx]
            poc_volume = volumes[max_volume_idx]
            
            # Find Value Area (70% of volume)
            total_volume = sum(volumes)
            target_volume = total_volume * 0.7
            
            # Sort by volume descending
            sorted_levels = sorted(zip(prices, volumes), key=lambda x: x[1], reverse=True)
            
            value_area_volume = 0
            value_area_prices = []
            
            for price, volume in sorted_levels:
                value_area_prices.append(price)
                value_area_volume += volume
                if value_area_volume >= target_volume:
                    break
            
            value_area_high = max(value_area_prices) if value_area_prices else poc_price
            value_area_low = min(value_area_prices) if value_area_prices else poc_price
            
            profile = {
                'poc_price': poc_price,
                'poc_volume': poc_volume,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'total_volume': total_volume,
                'price_range': price_range,
                'volume_distribution': dict(zip(prices, volumes)),
                'calculated_at': datetime.now()
            }
            
            self.volume_profile[symbol] = profile
            return profile
            
        except Exception as e:
            logger.error(f"Error calculating volume profile for {symbol}: {e}")
            return {}
    
    def _generate_dom_signals(self, symbol: str, order_book_analysis: Dict[str, Any],
                            flow_analysis: Dict[str, Any], iceberg_signals: List[Dict],
                            volume_profile: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on DOM analysis"""
        signals = []
        
        try:
            current_time = datetime.now()
            
            # Get current market data for signal pricing
            if not order_book_analysis or 'mid_price' not in order_book_analysis:
                return signals
            
            current_price = order_book_analysis['mid_price']
            
            # Signal 1: Volume Imbalance Trading
            if 'volume_imbalance' in order_book_analysis:
                imbalance = order_book_analysis['volume_imbalance']
                imbalance_strength = order_book_analysis.get('imbalance_strength', 0)
                
                if imbalance > self.config['imbalance_threshold'] and imbalance_strength > 0.3:
                    # Bullish signal (more bids than asks)
                    confidence = min(0.9, 0.5 + (imbalance_strength * 0.4))
                    
                    if confidence >= self.config['confidence_threshold']:
                        signals.append(Signal(
                            symbol=symbol,
                            action='buy',
                            confidence=confidence,
                            price=current_price,
                            stop_loss=current_price * 0.995,  # 0.5% stop loss
                            take_profit=current_price * 1.01,   # 1% take profit
                            timestamp=current_time,
                            strategy_name=self.name,
                            metadata={
                                'signal_type': 'volume_imbalance',
                                'imbalance_ratio': imbalance,
                                'imbalance_strength': imbalance_strength
                            }
                        ))
                
                elif imbalance < (1.0 / self.config['imbalance_threshold']) and imbalance_strength > 0.3:
                    # Bearish signal (more asks than bids)
                    confidence = min(0.9, 0.5 + (imbalance_strength * 0.4))
                    
                    if confidence >= self.config['confidence_threshold']:
                        signals.append(Signal(
                            symbol=symbol,
                            action='sell',
                            confidence=confidence,
                            price=current_price,
                            stop_loss=current_price * 1.005,  # 0.5% stop loss
                            take_profit=current_price * 0.99,   # 1% take profit
                            timestamp=current_time,
                            strategy_name=self.name,
                            metadata={
                                'signal_type': 'volume_imbalance',
                                'imbalance_ratio': imbalance,
                                'imbalance_strength': imbalance_strength
                            }
                        ))
            
            # Signal 2: Iceberg Order Detection
            for iceberg in iceberg_signals:
                if iceberg['significance'] > 2.0:  # Significantly large order
                    action = 'buy' if iceberg['type'] == 'iceberg_bid' else 'sell'
                    confidence = min(0.8, 0.4 + (iceberg['significance'] * 0.1))
                    
                    if confidence >= self.config['confidence_threshold']:
                        signals.append(Signal(
                            symbol=symbol,
                            action=action,
                            confidence=confidence,
                            price=current_price,
                            stop_loss=current_price * (0.995 if action == 'buy' else 1.005),
                            take_profit=current_price * (1.015 if action == 'buy' else 0.985),
                            timestamp=current_time,
                            strategy_name=self.name,
                            metadata={
                                'signal_type': 'iceberg_order',
                                'iceberg_price': iceberg['price'],
                                'iceberg_volume': iceberg['volume'],
                                'significance': iceberg['significance']
                            }
                        ))
            
            # Signal 3: Order Flow Momentum
            if flow_analysis and 'flow_momentum' in flow_analysis:
                momentum = flow_analysis['flow_momentum']
                net_flow = flow_analysis.get('net_flow', 0)
                
                if abs(momentum) > self.config['min_order_size'] * 0.1:
                    action = 'buy' if momentum > 0 else 'sell'
                    confidence = min(0.75, 0.5 + (abs(momentum) / self.config['large_order_threshold']))
                    
                    if confidence >= self.config['confidence_threshold']:
                        signals.append(Signal(
                            symbol=symbol,
                            action=action,
                            confidence=confidence,
                            price=current_price,
                            stop_loss=current_price * (0.997 if action == 'buy' else 1.003),
                            take_profit=current_price * (1.008 if action == 'buy' else 0.992),
                            timestamp=current_time,
                            strategy_name=self.name,
                            metadata={
                                'signal_type': 'flow_momentum',
                                'momentum': momentum,
                                'net_flow': net_flow
                            }
                        ))
            
            # Signal 4: Volume Profile Support/Resistance
            if volume_profile and 'poc_price' in volume_profile:
                poc_price = volume_profile['poc_price']
                current_distance = abs(current_price - poc_price) / current_price
                
                # Signal when price approaches POC
                if current_distance < 0.005:  # Within 0.5% of POC
                    # Determine likely direction based on where price is coming from
                    if current_price > poc_price:
                        action = 'sell'  # Resistance at POC
                    else:
                        action = 'buy'   # Support at POC
                    
                    confidence = 0.65 - (current_distance * 10)  # Closer = higher confidence
                    
                    if confidence >= self.config['confidence_threshold']:
                        signals.append(Signal(
                            symbol=symbol,
                            action=action,
                            confidence=confidence,
                            price=current_price,
                            stop_loss=current_price * (0.998 if action == 'buy' else 1.002),
                            take_profit=current_price * (1.006 if action == 'buy' else 0.994),
                            timestamp=current_time,
                            strategy_name=self.name,
                            metadata={
                                'signal_type': 'poc_interaction',
                                'poc_price': poc_price,
                                'distance_to_poc': current_distance
                            }
                        ))
            
        except Exception as e:
            logger.error(f"Error generating DOM signals for {symbol}: {e}")
        
        return signals
    
    def update_parameters(self, new_params: Dict[str, Any]) -> bool:
        """Update strategy parameters
        
        Args:
            new_params: New parameter values
            
        Returns:
            bool: True if update successful
        """
        try:
            for key, value in new_params.items():
                if key in self.config:
                    self.config[key] = value
                    
            logger.info(f"DOM analysis parameters updated: {list(new_params.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating DOM parameters: {e}")
            return False
    
    def get_dom_analysis_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of DOM analysis for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DOM analysis summary
        """
        try:
            summary = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'order_book_snapshots': len(self.order_book_history[symbol]),
                'volume_profile': self.volume_profile.get(symbol, {}),
                'detected_icebergs': len(self.detected_icebergs[symbol]),
                'support_resistance_levels': len(self.support_resistance_levels[symbol])
            }
            
            # Latest order book analysis
            if self.order_book_history[symbol]:
                latest_book = self.order_book_history[symbol][-1]['data']
                latest_analysis = self._analyze_order_book(symbol, latest_book)
                summary['latest_analysis'] = latest_analysis
            
            # Recent iceberg detections
            recent_icebergs = [ib for ib in self.detected_icebergs[symbol] 
                             if (datetime.now() - ib['timestamp']).seconds < 3600]  # Last hour
            summary['recent_icebergs'] = recent_icebergs
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting DOM analysis summary for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time DOM metrics across all symbols
        
        Returns:
            Real-time metrics summary
        """
        try:
            metrics = {
                'active_symbols': len(self.order_book_history),
                'total_snapshots': sum(len(history) for history in self.order_book_history.values()),
                'average_imbalance': 0.0,
                'total_detected_icebergs': sum(len(icebergs) for icebergs in self.detected_icebergs.values()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate average imbalance across all symbols
            imbalances = []
            for symbol in self.order_book_history.keys():
                if self.order_book_history[symbol]:
                    latest_book = self.order_book_history[symbol][-1]['data']
                    analysis = self._analyze_order_book(symbol, latest_book)
                    if 'volume_imbalance' in analysis:
                        imbalances.append(analysis['volume_imbalance'])
            
            if imbalances:
                metrics['average_imbalance'] = np.mean(imbalances)
                metrics['imbalance_std'] = np.std(imbalances)
                metrics['extreme_imbalances'] = len([i for i in imbalances if abs(i - 1.0) > 1.0])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time DOM metrics: {e}")
            return {'error': str(e)}


__all__ = [
    'DOMAnalysis'
]