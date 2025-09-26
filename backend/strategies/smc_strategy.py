#!/usr/bin/env python3
"""
Smart Money Concept (SMC) Strategy
Implements institutional trading concepts including:
- Break of Structure (BOS)
- Change of Character (ChoCh) 
- Order Blocks
- Fair Value Gaps (FVG)
- Supply and Demand Zones
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from . import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class SMCStrategy(BaseStrategy):
    """Smart Money Concept trading strategy implementation"""
    
    def __init__(self, name: str = "SMC_Strategy", config: Dict[str, Any] = None):
        default_config = {
            'lookback_period': 20,
            'min_structure_points': 3,
            'bos_confirmation_candles': 2,
            'order_block_strength': 3,
            'fvg_min_size': 0.0005,  # Minimum FVG size as % of price
            'risk_reward_ratio': 2.0,
            'timeframes': ['1h', '4h', '1d'],  # Multi-timeframe analysis
            'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
            'volume_threshold': 1.5,  # Volume multiplier for significance
            'atr_period': 14,
            'confidence_threshold': 0.7
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name, default_config)
        
        # Internal state
        self.market_structure = {}
        self.order_blocks = {}
        self.supply_demand_zones = {}
        self.fair_value_gaps = {}
        self.last_swing_highs = {}
        self.last_swing_lows = {}
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate SMC-based trading signals
        
        Args:
            market_data: Dict containing OHLCV data, volume, etc.
            
        Returns:
            List[Signal]: Generated trading signals
        """
        signals = []
        
        try:
            # Extract price data
            if 'ohlcv' not in market_data:
                return signals
                
            for symbol, ohlcv_data in market_data['ohlcv'].items():
                if len(ohlcv_data) < self.config['lookback_period']:
                    continue
                    
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                # Analyze market structure
                structure_analysis = self._analyze_market_structure(df)
                
                # Identify order blocks
                order_blocks = self._identify_order_blocks(df)
                
                # Detect Fair Value Gaps
                fvgs = self._detect_fair_value_gaps(df)
                
                # Find supply and demand zones
                supply_demand = self._find_supply_demand_zones(df)
                
                # Generate signals based on SMC analysis
                symbol_signals = self._generate_smc_signals(
                    symbol, df, structure_analysis, order_blocks, fvgs, supply_demand
                )
                
                signals.extend(symbol_signals)
                
                # Store analysis for future reference
                self.market_structure[symbol] = structure_analysis
                self.order_blocks[symbol] = order_blocks
                self.fair_value_gaps[symbol] = fvgs
                self.supply_demand_zones[symbol] = supply_demand
                
        except Exception as e:
            logger.error(f"Error generating SMC signals: {e}")
            
        return signals
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure for BOS and ChoCh"""
        try:
            # Calculate swing highs and lows
            swing_highs = self._find_swing_points(df['high'], mode='high')
            swing_lows = self._find_swing_points(df['low'], mode='low')
            
            # Determine current trend
            trend = self._determine_trend(swing_highs, swing_lows)
            
            # Detect Break of Structure (BOS)
            bos_signals = self._detect_break_of_structure(df, swing_highs, swing_lows, trend)
            
            # Detect Change of Character (ChoCh)
            choch_signals = self._detect_change_of_character(df, swing_highs, swing_lows, trend)
            
            return {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'trend': trend,
                'bos_signals': bos_signals,
                'choch_signals': choch_signals,
                'last_analysis': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return {}
    
    def _find_swing_points(self, series: pd.Series, mode: str = 'high', window: int = 5) -> List[Dict]:
        """Find swing highs or lows"""
        swing_points = []
        
        try:
            if mode == 'high':
                for i in range(window, len(series) - window):
                    if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                        swing_points.append({
                            'index': series.index[i],
                            'price': series.iloc[i],
                            'type': 'swing_high'
                        })
            else:  # mode == 'low'
                for i in range(window, len(series) - window):
                    if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                        swing_points.append({
                            'index': series.index[i],
                            'price': series.iloc[i],
                            'type': 'swing_low'
                        })
                        
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            
        return swing_points
    
    def _determine_trend(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> str:
        """Determine current market trend"""
        try:
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return 'sideways'
            
            # Get recent swing points
            recent_highs = swing_highs[-2:]
            recent_lows = swing_lows[-2:]
            
            # Check for higher highs and higher lows (uptrend)
            if (recent_highs[1]['price'] > recent_highs[0]['price'] and 
                recent_lows[1]['price'] > recent_lows[0]['price']):
                return 'uptrend'
            
            # Check for lower highs and lower lows (downtrend)
            if (recent_highs[1]['price'] < recent_highs[0]['price'] and 
                recent_lows[1]['price'] < recent_lows[0]['price']):
                return 'downtrend'
            
            return 'sideways'
            
        except Exception as e:
            logger.error(f"Error determining trend: {e}")
            return 'sideways'
    
    def _detect_break_of_structure(self, df: pd.DataFrame, swing_highs: List[Dict], 
                                 swing_lows: List[Dict], trend: str) -> List[Dict]:
        """Detect Break of Structure (BOS) events"""
        bos_events = []
        
        try:
            current_price = df['close'].iloc[-1]
            
            if trend == 'uptrend' and swing_highs:
                # Look for break above recent swing high
                last_high = swing_highs[-1]['price']
                if current_price > last_high:
                    bos_events.append({
                        'type': 'bullish_bos',
                        'price': current_price,
                        'broken_level': last_high,
                        'timestamp': df.index[-1],
                        'confidence': self._calculate_bos_confidence(df, last_high, 'bullish')
                    })
            
            elif trend == 'downtrend' and swing_lows:
                # Look for break below recent swing low
                last_low = swing_lows[-1]['price']
                if current_price < last_low:
                    bos_events.append({
                        'type': 'bearish_bos',
                        'price': current_price,
                        'broken_level': last_low,
                        'timestamp': df.index[-1],
                        'confidence': self._calculate_bos_confidence(df, last_low, 'bearish')
                    })
                    
        except Exception as e:
            logger.error(f"Error detecting BOS: {e}")
            
        return bos_events
    
    def _detect_change_of_character(self, df: pd.DataFrame, swing_highs: List[Dict], 
                                  swing_lows: List[Dict], trend: str) -> List[Dict]:
        """Detect Change of Character (ChoCh) events"""
        choch_events = []
        
        try:
            current_price = df['close'].iloc[-1]
            
            if trend == 'uptrend' and swing_lows:
                # Look for break below recent swing low (trend change)
                last_low = swing_lows[-1]['price']
                if current_price < last_low:
                    choch_events.append({
                        'type': 'bearish_choch',
                        'price': current_price,
                        'broken_level': last_low,
                        'timestamp': df.index[-1],
                        'previous_trend': 'uptrend',
                        'new_trend': 'downtrend'
                    })
            
            elif trend == 'downtrend' and swing_highs:
                # Look for break above recent swing high (trend change)
                last_high = swing_highs[-1]['price']
                if current_price > last_high:
                    choch_events.append({
                        'type': 'bullish_choch',
                        'price': current_price,
                        'broken_level': last_high,
                        'timestamp': df.index[-1],
                        'previous_trend': 'downtrend',
                        'new_trend': 'uptrend'
                    })
                    
        except Exception as e:
            logger.error(f"Error detecting ChoCh: {e}")
            
        return choch_events
    
    def _identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Identify order blocks (institutional buying/selling zones)"""
        order_blocks = []
        
        try:
            # Calculate volume-weighted average price
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            # Calculate average volume
            avg_volume = df['volume'].rolling(window=20).mean()
            
            # Look for high-volume candles with significant price movement
            for i in range(len(df) - 1):
                if df['volume'].iloc[i] > avg_volume.iloc[i] * self.config['volume_threshold']:
                    # Check for bullish order block
                    if (df['close'].iloc[i] > df['open'].iloc[i] and  # Bullish candle
                        df['close'].iloc[i+1] > df['high'].iloc[i]):  # Next candle breaks above
                        
                        order_blocks.append({
                            'type': 'bullish_order_block',
                            'timestamp': df.index[i],
                            'high': df['high'].iloc[i],
                            'low': df['low'].iloc[i],
                            'volume': df['volume'].iloc[i],
                            'strength': self._calculate_order_block_strength(df, i)
                        })
                    
                    # Check for bearish order block
                    elif (df['close'].iloc[i] < df['open'].iloc[i] and  # Bearish candle
                          df['close'].iloc[i+1] < df['low'].iloc[i]):  # Next candle breaks below
                        
                        order_blocks.append({
                            'type': 'bearish_order_block',
                            'timestamp': df.index[i],
                            'high': df['high'].iloc[i],
                            'low': df['low'].iloc[i],
                            'volume': df['volume'].iloc[i],
                            'strength': self._calculate_order_block_strength(df, i)
                        })
                        
        except Exception as e:
            logger.error(f"Error identifying order blocks: {e}")
            
        return order_blocks
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Fair Value Gaps (FVG)"""
        fvgs = []
        
        try:
            for i in range(1, len(df) - 1):
                # Bullish FVG: gap between candle[i-1] high and candle[i+1] low
                if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
                    gap_size = df['low'].iloc[i+1] - df['high'].iloc[i-1]
                    gap_percentage = gap_size / df['close'].iloc[i]
                    
                    if gap_percentage >= self.config['fvg_min_size']:
                        fvgs.append({
                            'type': 'bullish_fvg',
                            'timestamp': df.index[i],
                            'top': df['low'].iloc[i+1],
                            'bottom': df['high'].iloc[i-1],
                            'size': gap_size,
                            'size_percentage': gap_percentage
                        })
                
                # Bearish FVG: gap between candle[i-1] low and candle[i+1] high
                elif df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                    gap_size = df['low'].iloc[i-1] - df['high'].iloc[i+1]
                    gap_percentage = gap_size / df['close'].iloc[i]
                    
                    if gap_percentage >= self.config['fvg_min_size']:
                        fvgs.append({
                            'type': 'bearish_fvg',
                            'timestamp': df.index[i],
                            'top': df['low'].iloc[i-1],
                            'bottom': df['high'].iloc[i+1],
                            'size': gap_size,
                            'size_percentage': gap_percentage
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting FVGs: {e}")
            
        return fvgs
    
    def _find_supply_demand_zones(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Identify supply and demand zones"""
        zones = {'supply_zones': [], 'demand_zones': []}
        
        try:
            # Use swing points to identify potential zones
            swing_highs = self._find_swing_points(df['high'], mode='high')
            swing_lows = self._find_swing_points(df['low'], mode='low')
            
            # Supply zones (resistance areas)
            for swing_high in swing_highs[-5:]:  # Recent swing highs
                zone_top = swing_high['price'] * 1.002  # 0.2% above
                zone_bottom = swing_high['price'] * 0.998  # 0.2% below
                
                zones['supply_zones'].append({
                    'top': zone_top,
                    'bottom': zone_bottom,
                    'center': swing_high['price'],
                    'timestamp': swing_high['index'],
                    'strength': self._calculate_zone_strength(df, swing_high['price'], 'supply')
                })
            
            # Demand zones (support areas)
            for swing_low in swing_lows[-5:]:  # Recent swing lows
                zone_top = swing_low['price'] * 1.002  # 0.2% above
                zone_bottom = swing_low['price'] * 0.998  # 0.2% below
                
                zones['demand_zones'].append({
                    'top': zone_top,
                    'bottom': zone_bottom,
                    'center': swing_low['price'],
                    'timestamp': swing_low['index'],
                    'strength': self._calculate_zone_strength(df, swing_low['price'], 'demand')
                })
                
        except Exception as e:
            logger.error(f"Error finding supply/demand zones: {e}")
            
        return zones
    
    def _generate_smc_signals(self, symbol: str, df: pd.DataFrame, 
                            structure_analysis: Dict, order_blocks: List[Dict],
                            fvgs: List[Dict], supply_demand: Dict) -> List[Signal]:
        """Generate trading signals based on SMC analysis"""
        signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            current_time = df.index[-1]
            
            # Signal generation based on BOS
            for bos in structure_analysis.get('bos_signals', []):
                if bos['confidence'] >= self.config['confidence_threshold']:
                    action = 'buy' if bos['type'] == 'bullish_bos' else 'sell'
                    
                    # Calculate stop loss and take profit
                    sl, tp = self._calculate_sl_tp(df, action, bos['broken_level'])
                    
                    signals.append(Signal(
                        symbol=symbol,
                        action=action,
                        confidence=bos['confidence'],
                        price=current_price,
                        stop_loss=sl,
                        take_profit=tp,
                        timestamp=current_time,
                        strategy_name=self.name,
                        metadata={
                            'signal_type': 'bos',
                            'bos_type': bos['type'],
                            'broken_level': bos['broken_level']
                        }
                    ))
            
            # Signal generation based on ChoCh
            for choch in structure_analysis.get('choch_signals', []):
                action = 'buy' if choch['type'] == 'bullish_choch' else 'sell'
                
                sl, tp = self._calculate_sl_tp(df, action, choch['broken_level'])
                
                signals.append(Signal(
                    symbol=symbol,
                    action=action,
                    confidence=0.8,  # High confidence for trend changes
                    price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=current_time,
                    strategy_name=self.name,
                    metadata={
                        'signal_type': 'choch',
                        'choch_type': choch['type'],
                        'trend_change': f"{choch['previous_trend']} -> {choch['new_trend']}"
                    }
                ))
            
            # Signal generation based on order blocks
            for ob in order_blocks:
                if ob['strength'] >= self.config['order_block_strength']:
                    # Check if price is interacting with order block
                    if ob['low'] <= current_price <= ob['high']:
                        action = 'buy' if ob['type'] == 'bullish_order_block' else 'sell'
                        
                        sl, tp = self._calculate_sl_tp(df, action, 
                                                     ob['low'] if action == 'buy' else ob['high'])
                        
                        signals.append(Signal(
                            symbol=symbol,
                            action=action,
                            confidence=0.7,
                            price=current_price,
                            stop_loss=sl,
                            take_profit=tp,
                            timestamp=current_time,
                            strategy_name=self.name,
                            metadata={
                                'signal_type': 'order_block',
                                'ob_type': ob['type'],
                                'ob_strength': ob['strength']
                            }
                        ))
                        
        except Exception as e:
            logger.error(f"Error generating SMC signals: {e}")
            
        return signals
    
    def _calculate_bos_confidence(self, df: pd.DataFrame, broken_level: float, direction: str) -> float:
        """Calculate confidence for Break of Structure"""
        try:
            # Calculate volume confirmation
            recent_volume = df['volume'].iloc[-3:].mean()
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Calculate price momentum
            price_change = abs(df['close'].iloc[-1] - broken_level) / broken_level
            
            # Base confidence
            confidence = 0.5
            
            # Add volume confirmation
            if volume_ratio > 1.5:
                confidence += 0.2
            elif volume_ratio > 1.2:
                confidence += 0.1
            
            # Add momentum confirmation
            if price_change > 0.01:  # 1% move
                confidence += 0.2
            elif price_change > 0.005:  # 0.5% move
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_order_block_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate order block strength"""
        try:
            volume_ratio = df['volume'].iloc[index] / df['volume'].rolling(window=20).mean().iloc[index]
            candle_size = abs(df['high'].iloc[index] - df['low'].iloc[index])
            atr = self._calculate_atr(df, index)
            size_ratio = candle_size / atr if atr > 0 else 1
            
            strength = (volume_ratio + size_ratio) / 2
            return min(strength, 10.0)  # Cap at 10
            
        except Exception:
            return 1.0
    
    def _calculate_zone_strength(self, df: pd.DataFrame, price: float, zone_type: str) -> float:
        """Calculate supply/demand zone strength"""
        try:
            # Count touches/rejections from this level
            touches = 0
            tolerance = price * 0.002  # 0.2% tolerance
            
            for i in range(len(df)):
                if zone_type == 'supply':
                    if abs(df['high'].iloc[i] - price) <= tolerance:
                        touches += 1
                else:  # demand
                    if abs(df['low'].iloc[i] - price) <= tolerance:
                        touches += 1
            
            return min(touches / 5.0, 1.0)  # Normalize to 0-1
            
        except Exception:
            return 0.5
    
    def _calculate_sl_tp(self, df: pd.DataFrame, action: str, reference_level: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            atr = self._calculate_atr(df)
            current_price = df['close'].iloc[-1]
            
            if action == 'buy':
                stop_loss = reference_level - atr
                take_profit = current_price + (current_price - stop_loss) * self.config['risk_reward_ratio']
            else:  # sell
                stop_loss = reference_level + atr
                take_profit = current_price - (stop_loss - current_price) * self.config['risk_reward_ratio']
            
            return stop_loss, take_profit
            
        except Exception:
            current_price = df['close'].iloc[-1]
            if action == 'buy':
                return current_price * 0.99, current_price * 1.02
            else:
                return current_price * 1.01, current_price * 0.98
    
    def _calculate_atr(self, df: pd.DataFrame, index: int = None) -> float:
        """Calculate Average True Range"""
        try:
            if index is None:
                index = len(df) - 1
            
            period = min(self.config['atr_period'], index + 1)
            start_idx = max(0, index - period + 1)
            
            high_low = df['high'].iloc[start_idx:index+1] - df['low'].iloc[start_idx:index+1]
            high_close = abs(df['high'].iloc[start_idx:index+1] - df['close'].shift(1).iloc[start_idx:index+1])
            low_close = abs(df['low'].iloc[start_idx:index+1] - df['close'].shift(1).iloc[start_idx:index+1])
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.mean()
            
        except Exception:
            return df['close'].iloc[-1] * 0.01  # 1% fallback
    
    def update_parameters(self, new_params: Dict[str, Any]) -> bool:
        """Update strategy parameters"""
        try:
            for key, value in new_params.items():
                if key in self.config:
                    self.config[key] = value
            return True
        except Exception as e:
            logger.error(f"Error updating SMC parameters: {e}")
            return False
    
    def get_analysis_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of SMC analysis for a symbol"""
        return {
            'market_structure': self.market_structure.get(symbol, {}),
            'order_blocks': self.order_blocks.get(symbol, []),
            'fair_value_gaps': self.fair_value_gaps.get(symbol, []),
            'supply_demand_zones': self.supply_demand_zones.get(symbol, {})
        }