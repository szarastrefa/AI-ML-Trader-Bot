#!/usr/bin/env python3
"""
Data Pipeline Module
Real-time data processing and aggregation from multiple sources
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class DataPipeline:
    """Real-time data pipeline for market data processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.data_sources = {}  # broker_name -> connector
        self.market_data = defaultdict(dict)  # symbol -> timeframe -> data
        self.real_time_data = defaultdict(dict)  # symbol -> latest ticker
        self.subscribers = defaultdict(list)  # data_type -> [callbacks]
        self.is_running = False
        self.update_threads = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.data_history = defaultdict(lambda: deque(maxlen=10000))  # Limited history
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'update_interval': 60,  # seconds
            'real_time_interval': 5,  # seconds for real-time updates
            'max_history_length': 10000,
            'supported_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
            'batch_size': 100,
            'retry_attempts': 3,
            'timeout': 30
        }
    
    def add_data_source(self, broker_name: str, connector) -> bool:
        """Add a data source (broker connector)
        
        Args:
            broker_name: Name of the broker
            connector: Broker connector instance
            
        Returns:
            bool: True if added successfully
        """
        try:
            if not connector.is_connected:
                logger.warning(f"Connector {broker_name} is not connected")
                return False
            
            self.data_sources[broker_name] = connector
            logger.info(f"Added data source: {broker_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding data source {broker_name}: {e}")
            return False
    
    def remove_data_source(self, broker_name: str) -> bool:
        """Remove a data source
        
        Args:
            broker_name: Name of the broker to remove
            
        Returns:
            bool: True if removed successfully
        """
        try:
            if broker_name in self.data_sources:
                del self.data_sources[broker_name]
                # Stop update thread for this source
                if broker_name in self.update_threads:
                    self.update_threads[broker_name] = False
                logger.info(f"Removed data source: {broker_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing data source {broker_name}: {e}")
            return False
    
    def start_pipeline(self) -> bool:
        """Start the data pipeline
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.is_running:
                logger.warning("Data pipeline is already running")
                return True
            
            self.is_running = True
            
            # Start real-time data updates for each source
            for broker_name, connector in self.data_sources.items():
                self._start_real_time_updates(broker_name, connector)
            
            # Start periodic data updates
            self._start_periodic_updates()
            
            logger.info("Data pipeline started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting data pipeline: {e}")
            return False
    
    def stop_pipeline(self):
        """Stop the data pipeline"""
        try:
            self.is_running = False
            
            # Stop all update threads
            for broker_name in self.update_threads:
                self.update_threads[broker_name] = False
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Data pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping data pipeline: {e}")
    
    def _start_real_time_updates(self, broker_name: str, connector):
        """Start real-time data updates for a broker"""
        def real_time_worker():
            self.update_threads[broker_name] = True
            
            while self.is_running and self.update_threads.get(broker_name, False):
                try:
                    # Get available symbols (limit to avoid overload)
                    symbols = connector.get_available_symbols()[:50]  # Limit to 50 symbols
                    
                    # Update ticker data
                    for symbol in symbols[:10]:  # Further limit for real-time
                        try:
                            ticker = connector.get_ticker(symbol)
                            self.real_time_data[symbol] = {
                                'broker': broker_name,
                                'ticker': ticker,
                                'timestamp': datetime.now()
                            }
                            
                            # Store in history
                            self.data_history[f"{symbol}_tickers"].append({
                                'broker': broker_name,
                                'data': ticker,
                                'timestamp': datetime.now()
                            })
                            
                            # Notify subscribers
                            self._notify_subscribers('ticker_update', {
                                'symbol': symbol,
                                'broker': broker_name,
                                'ticker': ticker
                            })
                            
                        except Exception as e:
                            logger.error(f"Error updating ticker for {symbol}: {e}")
                    
                    time.sleep(self.config['real_time_interval'])
                    
                except Exception as e:
                    logger.error(f"Error in real-time worker for {broker_name}: {e}")
                    time.sleep(10)  # Wait before retry
        
        # Start worker thread
        thread = threading.Thread(target=real_time_worker, daemon=True)
        thread.start()
    
    def _start_periodic_updates(self):
        """Start periodic historical data updates"""
        def periodic_worker():
            while self.is_running:
                try:
                    self.update_all_data()
                    time.sleep(self.config['update_interval'])
                except Exception as e:
                    logger.error(f"Error in periodic worker: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=periodic_worker, daemon=True)
        thread.start()
    
    def update_all_data(self):
        """Update all historical data from all sources"""
        try:
            update_tasks = []
            
            for broker_name, connector in self.data_sources.items():
                # Submit update tasks to thread pool
                task = self.executor.submit(self._update_broker_data, broker_name, connector)
                update_tasks.append(task)
            
            # Wait for all updates to complete (with timeout)
            for task in update_tasks:
                try:
                    task.result(timeout=self.config['timeout'])
                except Exception as e:
                    logger.error(f"Error in data update task: {e}")
            
        except Exception as e:
            logger.error(f"Error updating all data: {e}")
    
    def _update_broker_data(self, broker_name: str, connector):
        """Update historical data for a specific broker"""
        try:
            if not connector.is_connected:
                logger.warning(f"Connector {broker_name} is not connected, skipping update")
                return
            
            # Get available symbols (limit to avoid overload)
            symbols = connector.get_available_symbols()[:20]  # Limit to 20 symbols
            
            # Update market data for each timeframe
            for timeframe in self.config['supported_timeframes']:
                for symbol in symbols:
                    try:
                        # Calculate time range for historical data
                        end_time = datetime.now()
                        
                        # Different lookback periods for different timeframes
                        lookback_hours = {
                            '1m': 24,    # 1 day
                            '5m': 120,   # 5 days  
                            '15m': 360,  # 15 days
                            '30m': 720,  # 30 days
                            '1h': 1440,  # 60 days
                            '4h': 2880,  # 120 days
                            '1d': 8760   # 365 days
                        }
                        
                        hours_back = lookback_hours.get(timeframe, 720)
                        start_time = end_time - timedelta(hours=hours_back)
                        
                        # Get historical data
                        bars = connector.get_historical_data(symbol, timeframe, start_time, end_time)
                        
                        if bars:
                            # Convert to DataFrame for easier processing
                            df = pd.DataFrame([
                                {
                                    'timestamp': bar.timestamp,
                                    'open': bar.open,
                                    'high': bar.high,
                                    'low': bar.low,
                                    'close': bar.close,
                                    'volume': bar.volume
                                } for bar in bars
                            ])
                            
                            # Store data
                            if symbol not in self.market_data:
                                self.market_data[symbol] = {}
                            
                            self.market_data[symbol][timeframe] = {
                                'broker': broker_name,
                                'data': df,
                                'last_updated': datetime.now(),
                                'bars_count': len(bars)
                            }
                            
                            # Store in history
                            self.data_history[f"{symbol}_{timeframe}"].append({
                                'broker': broker_name,
                                'data': df.tail(10),  # Last 10 bars only
                                'timestamp': datetime.now()
                            })
                            
                            # Notify subscribers
                            self._notify_subscribers('historical_update', {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'broker': broker_name,
                                'bars_count': len(bars)
                            })
                            
                    except Exception as e:
                        logger.error(f"Error updating {symbol} {timeframe} from {broker_name}: {e}")
                        continue
            
            logger.debug(f"Updated data for {broker_name}")
            
        except Exception as e:
            logger.error(f"Error updating broker data for {broker_name}: {e}")
    
    def get_latest_data(self, symbols: List[str], timeframe: str = '1h') -> Dict[str, Any]:
        """Get latest data for specified symbols
        
        Args:
            symbols: List of symbols to get data for
            timeframe: Timeframe for data
            
        Returns:
            Dict: Latest market data
        """
        try:
            result = {
                'ohlcv': {},
                'tickers': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for symbol in symbols:
                # Get OHLCV data
                if symbol in self.market_data and timeframe in self.market_data[symbol]:
                    market_info = self.market_data[symbol][timeframe]
                    df = market_info['data']
                    
                    # Convert DataFrame to list of lists for easier processing
                    if not df.empty:
                        ohlcv_data = []
                        for _, row in df.iterrows():
                            ohlcv_data.append([
                                row['timestamp'],
                                row['open'],
                                row['high'], 
                                row['low'],
                                row['close'],
                                row['volume']
                            ])
                        
                        result['ohlcv'][symbol] = ohlcv_data
                
                # Get real-time ticker
                if symbol in self.real_time_data:
                    ticker_info = self.real_time_data[symbol]
                    result['tickers'][symbol] = {
                        'broker': ticker_info['broker'],
                        'bid': ticker_info['ticker'].bid,
                        'ask': ticker_info['ticker'].ask,
                        'last': ticker_info['ticker'].last,
                        'volume': ticker_info['ticker'].volume,
                        'timestamp': ticker_info['ticker'].timestamp.isoformat()
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return {'ohlcv': {}, 'tickers': {}, 'error': str(e)}
    
    def get_symbol_data(self, symbol: str, timeframe: str = '1h', 
                       bars_count: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data for specific symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            bars_count: Number of bars to return
            
        Returns:
            DataFrame with historical data or None
        """
        try:
            if symbol in self.market_data and timeframe in self.market_data[symbol]:
                df = self.market_data[symbol][timeframe]['data']
                return df.tail(bars_count) if not df.empty else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting symbol data for {symbol}: {e}")
            return None
    
    def subscribe_to_updates(self, data_type: str, callback: Callable):
        """Subscribe to data updates
        
        Args:
            data_type: Type of data to subscribe to ('ticker_update', 'historical_update')
            callback: Callback function to call on updates
        """
        self.subscribers[data_type].append(callback)
        logger.info(f"Added subscriber for {data_type}")
    
    def _notify_subscribers(self, data_type: str, data: Dict[str, Any]):
        """Notify subscribers of data updates"""
        try:
            for callback in self.subscribers.get(data_type, []):
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
        except Exception as e:
            logger.error(f"Error notifying subscribers: {e}")
    
    def get_aggregated_data(self, symbols: List[str], timeframe: str = '1h') -> Dict[str, Any]:
        """Get aggregated data from all sources
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe for data
            
        Returns:
            Aggregated market data
        """
        try:
            aggregated = {
                'symbols': {},
                'summary': {
                    'total_symbols': len(symbols),
                    'available_symbols': 0,
                    'data_sources': list(self.data_sources.keys()),
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            for symbol in symbols:
                symbol_data = {
                    'ohlcv': None,
                    'ticker': None,
                    'sources': []
                }
                
                # Get OHLCV data
                df = self.get_symbol_data(symbol, timeframe)
                if df is not None and not df.empty:
                    symbol_data['ohlcv'] = df.to_dict('records')
                    symbol_data['sources'].append('historical')
                    aggregated['summary']['available_symbols'] += 1
                
                # Get real-time data
                if symbol in self.real_time_data:
                    ticker_info = self.real_time_data[symbol]
                    symbol_data['ticker'] = {
                        'bid': ticker_info['ticker'].bid,
                        'ask': ticker_info['ticker'].ask,
                        'last': ticker_info['ticker'].last,
                        'volume': ticker_info['ticker'].volume,
                        'spread': ticker_info['ticker'].spread,
                        'timestamp': ticker_info['ticker'].timestamp.isoformat(),
                        'broker': ticker_info['broker']
                    }
                    symbol_data['sources'].append('realtime')
                
                aggregated['symbols'][symbol] = symbol_data
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error getting aggregated data: {e}")
            return {'error': str(e)}
    
    def calculate_technical_indicators(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Calculate technical indicators for symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary with calculated indicators
        """
        try:
            df = self.get_symbol_data(symbol, timeframe, 200)  # Get more data for indicators
            
            if df is None or df.empty:
                return {}
            
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else None
            indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
            indicators['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1] if len(df) >= 12 else None
            indicators['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1] if len(df) >= 26 else None
            
            # RSI
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators['rsi_14'] = rsi.iloc[-1]
            
            # MACD
            if 'ema_12' in indicators and 'ema_26' in indicators and indicators['ema_12'] and indicators['ema_26']:
                indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
                
                # MACD Signal line
                if len(df) >= 26:
                    macd_line = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
                    indicators['macd_signal'] = macd_line.ewm(span=9).mean().iloc[-1]
                    indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            if len(df) >= 20:
                bb_middle = df['close'].rolling(20).mean()
                bb_std = df['close'].rolling(20).std()
                indicators['bb_upper'] = (bb_middle + bb_std * 2).iloc[-1]
                indicators['bb_lower'] = (bb_middle - bb_std * 2).iloc[-1]
                indicators['bb_middle'] = bb_middle.iloc[-1]
            
            # Volume indicators
            if 'volume' in df.columns and len(df) >= 20:
                vol_sma = df['volume'].rolling(20).mean()
                indicators['volume_sma'] = vol_sma.iloc[-1]
                indicators['volume_ratio'] = df['volume'].iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1
            
            # Current price info
            indicators['current_price'] = df['close'].iloc[-1]
            indicators['price_change'] = df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) >= 2 else 0
            indicators['price_change_pct'] = (indicators['price_change'] / df['close'].iloc[-2] * 100) if len(df) >= 2 and df['close'].iloc[-2] > 0 else 0
            
            indicators['calculation_timestamp'] = datetime.now().isoformat()
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return {}
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality and pipeline metrics
        
        Returns:
            Data quality metrics
        """
        try:
            metrics = {
                'data_sources': len(self.data_sources),
                'active_symbols': len(self.market_data),
                'real_time_symbols': len(self.real_time_data),
                'pipeline_status': 'running' if self.is_running else 'stopped',
                'last_update': datetime.now().isoformat()
            }
            
            # Data freshness metrics
            freshness = {}
            for symbol, timeframe_data in self.market_data.items():
                symbol_freshness = {}
                for timeframe, data_info in timeframe_data.items():
                    last_updated = data_info['last_updated']
                    age_minutes = (datetime.now() - last_updated).total_seconds() / 60
                    symbol_freshness[timeframe] = {
                        'last_updated': last_updated.isoformat(),
                        'age_minutes': age_minutes,
                        'bars_count': data_info['bars_count'],
                        'is_fresh': age_minutes < (self.config['update_interval'] / 60) * 2  # 2x update interval
                    }
                freshness[symbol] = symbol_freshness
            
            metrics['data_freshness'] = freshness
            
            # Source-specific metrics
            source_metrics = {}
            for source_name in self.data_sources.keys():
                source_symbols = []
                for symbol, timeframe_data in self.market_data.items():
                    for timeframe, data_info in timeframe_data.items():
                        if data_info['broker'] == source_name:
                            source_symbols.append(f"{symbol}_{timeframe}")
                
                source_metrics[source_name] = {
                    'active_datasets': len(source_symbols),
                    'is_connected': self.data_sources[source_name].is_connected
                }
            
            metrics['sources'] = source_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting data quality metrics: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, retention_days: int = 7):
        """Cleanup old data to manage memory usage
        
        Args:
            retention_days: Days of data to retain
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean history
            for key, history in self.data_history.items():
                # Remove old entries
                while history and history[0]['timestamp'] < cutoff_date:
                    history.popleft()
            
            # Clean market data (keep only recent bars)
            for symbol in list(self.market_data.keys()):
                for timeframe in list(self.market_data[symbol].keys()):
                    data_info = self.market_data[symbol][timeframe]
                    if data_info['last_updated'] < cutoff_date:
                        del self.market_data[symbol][timeframe]
                
                # Remove symbol if no timeframes left
                if not self.market_data[symbol]:
                    del self.market_data[symbol]
            
            logger.info(f"Cleaned up data older than {retention_days} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def export_data(self, symbol: str, timeframe: str = '1h', output_file: str = None) -> bool:
        """Export symbol data to CSV file
        
        Args:
            symbol: Symbol to export
            timeframe: Data timeframe
            output_file: Output file path (optional)
            
        Returns:
            bool: True if export successful
        """
        try:
            df = self.get_symbol_data(symbol, timeframe, bars_count=None)  # Get all data
            
            if df is None or df.empty:
                logger.error(f"No data available for {symbol} {timeframe}")
                return False
            
            if output_file is None:
                output_file = f"data/exports/{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Export to CSV
            df.to_csv(output_file, index=False)
            
            logger.info(f"Exported {len(df)} bars for {symbol} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data for {symbol}: {e}")
            return False


__all__ = [
    'DataPipeline'
]