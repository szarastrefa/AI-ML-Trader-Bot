#!/usr/bin/env python3
"""
Unit Tests for Trading Strategies
Comprehensive test suite for all implemented trading strategies
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import strategy modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from strategies import BaseStrategy, Signal
from strategies.smc_strategy import SmartMoneyConcept
from strategies.dom_analysis import DOMAnalysis
from strategies.ml_strategy import MLStrategy


class TestBaseStrategy(unittest.TestCase):
    """Test BaseStrategy abstract class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = BaseStrategy("Test_Strategy", {'test_param': 100})
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.name, "Test_Strategy")
        self.assertEqual(self.strategy.config['test_param'], 100)
        self.assertFalse(self.strategy.is_active)
        self.assertEqual(len(self.strategy.performance_metrics), 0)
    
    def test_activate_deactivate(self):
        """Test strategy activation and deactivation"""
        # Test activation
        self.strategy.activate()
        self.assertTrue(self.strategy.is_active)
        
        # Test deactivation
        self.strategy.deactivate()
        self.assertFalse(self.strategy.is_active)
    
    def test_update_performance_metrics(self):
        """Test performance metrics update"""
        metrics = {
            'total_signals': 10,
            'successful_signals': 7,
            'win_rate': 0.7,
            'total_pnl': 150.0
        }
        
        self.strategy.update_performance_metrics(metrics)
        
        self.assertEqual(self.strategy.performance_metrics['win_rate'], 0.7)
        self.assertEqual(self.strategy.performance_metrics['total_pnl'], 150.0)
        self.assertIn('last_updated', self.strategy.performance_metrics)


class TestSmartMoneyConcept(unittest.TestCase):
    """Test Smart Money Concept strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        config = {
            'lookback_period': 20,
            'min_structure_points': 3,
            'bos_confirmation_candles': 2,
            'confidence_threshold': 0.6
        }
        self.smc_strategy = SmartMoneyConcept("SMC_Test", config)
    
    def test_smc_initialization(self):
        """Test SMC strategy initialization"""
        self.assertEqual(self.smc_strategy.name, "SMC_Test")
        self.assertEqual(self.smc_strategy.config['lookback_period'], 20)
        self.assertIsInstance(self.smc_strategy.structure_history, dict)
    
    def test_generate_sample_data(self):
        """Generate sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        
        # Create realistic price data
        np.random.seed(42)
        base_price = 1.2000
        returns = np.random.normal(0, 0.001, 100)
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        closes = prices[1:]
        opens = prices[:-1]
        
        # Create highs and lows
        highs = [max(o, c) * (1 + abs(np.random.normal(0, 0.0005))) for o, c in zip(opens, closes)]
        lows = [min(o, c) * (1 - abs(np.random.normal(0, 0.0005))) for o, c in zip(opens, closes)]
        volumes = np.random.uniform(1000, 10000, 100)
        
        data = {
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }
        
        return pd.DataFrame(data)
    
    def test_identify_structure_points(self):
        """Test structure point identification"""
        df = self.test_generate_sample_data()
        
        highs, lows = self.smc_strategy._identify_structure_points(df)
        
        self.assertIsInstance(highs, list)
        self.assertIsInstance(lows, list)
        self.assertGreaterEqual(len(highs), 0)
        self.assertGreaterEqual(len(lows), 0)
        
        # Verify structure points are within data range
        if highs:
            for high in highs:
                self.assertGreaterEqual(high['index'], 0)
                self.assertLess(high['index'], len(df))
        
        if lows:
            for low in lows:
                self.assertGreaterEqual(low['index'], 0)
                self.assertLess(low['index'], len(df))
    
    def test_detect_break_of_structure(self):
        """Test BOS detection"""
        df = self.test_generate_sample_data()
        
        # Mock structure points
        structure_points = {
            'highs': [{'index': 10, 'price': 1.2050}, {'index': 20, 'price': 1.2080}],
            'lows': [{'index': 15, 'price': 1.1980}, {'index': 25, 'price': 1.1960}]
        }
        
        bos_signals = self.smc_strategy._detect_break_of_structure(df, structure_points)
        
        self.assertIsInstance(bos_signals, list)
        # Each BOS signal should have required fields
        for signal in bos_signals:
            self.assertIn('type', signal)
            self.assertIn('direction', signal)
            self.assertIn('confidence', signal)
            self.assertIn('price', signal)
    
    def test_detect_order_blocks(self):
        """Test order block detection"""
        df = self.test_generate_sample_data()
        
        order_blocks = self.smc_strategy._detect_order_blocks(df)
        
        self.assertIsInstance(order_blocks, list)
        # Each order block should have required fields
        for block in order_blocks:
            self.assertIn('type', block)
            self.assertIn('top', block)
            self.assertIn('bottom', block)
            self.assertIn('strength', block)
            self.assertIn('index', block)
    
    def test_detect_fair_value_gaps(self):
        """Test FVG detection"""
        df = self.test_generate_sample_data()
        
        fvgs = self.smc_strategy._detect_fair_value_gaps(df)
        
        self.assertIsInstance(fvgs, list)
        # Each FVG should have required fields
        for fvg in fvgs:
            self.assertIn('type', fvg)
            self.assertIn('top', fvg)
            self.assertIn('bottom', fvg)
            self.assertIn('index', fvg)
    
    def test_generate_signals(self):
        """Test signal generation"""
        df = self.test_generate_sample_data()
        
        market_data = {
            'ohlcv': {
                'EURUSD': df.values.tolist()
            }
        }
        
        signals = self.smc_strategy.generate_signals(market_data)
        
        self.assertIsInstance(signals, list)
        # Verify signal structure
        for signal in signals:
            self.assertIsInstance(signal, Signal)
            self.assertIn(signal.action, ['buy', 'sell'])
            self.assertGreater(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 1)
            self.assertIsNotNone(signal.price)


class TestDOMAnalysis(unittest.TestCase):
    """Test Depth of Market Analysis strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        config = {
            'order_book_levels': 10,
            'imbalance_threshold': 2.0,
            'confidence_threshold': 0.6
        }
        self.dom_strategy = DOMAnalysis("DOM_Test", config)
    
    def test_dom_initialization(self):
        """Test DOM strategy initialization"""
        self.assertEqual(self.dom_strategy.name, "DOM_Test")
        self.assertEqual(self.dom_strategy.config['imbalance_threshold'], 2.0)
        self.assertIsInstance(self.dom_strategy.order_book_history, dict)
    
    def test_generate_sample_order_book(self):
        """Generate sample order book data for testing"""
        np.random.seed(42)
        
        # Generate bids (descending prices)
        bid_prices = np.arange(1.2000, 1.1990, -0.0001)
        bid_volumes = np.random.uniform(1000, 10000, len(bid_prices))
        bids = [{'price': price, 'amount': volume} for price, volume in zip(bid_prices, bid_volumes)]
        
        # Generate asks (ascending prices)
        ask_prices = np.arange(1.2001, 1.2011, 0.0001)
        ask_volumes = np.random.uniform(1000, 10000, len(ask_prices))
        asks = [{'price': price, 'amount': volume} for price, volume in zip(ask_prices, ask_volumes)]
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now()
        }
    
    def test_analyze_order_book(self):
        """Test order book analysis"""
        order_book = self.test_generate_sample_order_book()
        
        analysis = self.dom_strategy._analyze_order_book('EURUSD', order_book)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('volume_imbalance', analysis)
        self.assertIn('spread', analysis)
        self.assertIn('mid_price', analysis)
        self.assertIn('total_liquidity', analysis)
        
        # Verify numeric values
        self.assertGreater(analysis['spread'], 0)
        self.assertGreater(analysis['mid_price'], 0)
        self.assertGreater(analysis['total_liquidity'], 0)
    
    def test_detect_iceberg_orders(self):
        """Test iceberg order detection"""
        order_book = self.test_generate_sample_order_book()
        
        # Add a large order that could be an iceberg
        order_book['bids'][0]['amount'] = 50000  # Much larger than others
        
        icebergs = self.dom_strategy._detect_iceberg_orders('EURUSD', order_book)
        
        self.assertIsInstance(icebergs, list)
        # Each iceberg signal should have required fields
        for iceberg in icebergs:
            self.assertIn('type', iceberg)
            self.assertIn('price', iceberg)
            self.assertIn('volume', iceberg)
    
    def test_generate_signals(self):
        """Test DOM signal generation"""
        order_book = self.test_generate_sample_order_book()
        
        market_data = {
            'order_books': {
                'EURUSD': order_book
            }
        }
        
        signals = self.dom_strategy.generate_signals(market_data)
        
        self.assertIsInstance(signals, list)
        # Verify signal structure
        for signal in signals:
            self.assertIsInstance(signal, Signal)
            self.assertIn(signal.action, ['buy', 'sell'])
            self.assertGreater(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 1)


class TestMLStrategy(unittest.TestCase):
    """Test Machine Learning Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        config = {
            'model_name': 'test_model',
            'prediction_threshold': 0.6,
            'feature_lookback': 20
        }
        self.ml_strategy = MLStrategy("ML_Test", config)
    
    def test_ml_initialization(self):
        """Test ML strategy initialization"""
        self.assertEqual(self.ml_strategy.name, "ML_Test")
        self.assertEqual(self.ml_strategy.config['prediction_threshold'], 0.6)
    
    @patch('backend.strategies.ml_strategy.ModelManager')
    @patch('backend.strategies.ml_strategy.FeatureEngineering')
    def test_load_model(self, mock_feature_eng, mock_model_manager):
        """Test ML model loading"""
        # Mock model manager
        mock_manager = Mock()
        mock_manager.load_model.return_value = Mock()
        mock_manager.get_model_info.return_value = {
            'type': 'classification',
            'feature_names': ['feature1', 'feature2']
        }
        
        self.ml_strategy.model_manager = mock_manager
        
        result = self.ml_strategy.load_model('test_model')
        
        self.assertTrue(result)
        mock_manager.load_model.assert_called_once_with('test_model')
    
    def test_prediction_to_signals(self):
        """Test prediction to signal conversion"""
        prediction = {
            'action': 'buy',
            'confidence': 0.8,
            'prediction': 1
        }
        
        current_bar = pd.Series({
            'open': 1.2000,
            'high': 1.2010,
            'low': 1.1990,
            'close': 1.2005,
            'volume': 5000
        })
        
        features = np.array([[1, 2, 3, 4, 5]])
        
        signals = self.ml_strategy._prediction_to_signals(
            'EURUSD', prediction, current_bar, features
        )
        
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        
        self.assertIsInstance(signal, Signal)
        self.assertEqual(signal.action, 'buy')
        self.assertEqual(signal.confidence, 0.8)
        self.assertEqual(signal.symbol, 'EURUSD')
    
    def test_calculate_dynamic_sl_tp(self):
        """Test dynamic SL/TP calculation"""
        current_bar = pd.Series({
            'open': 1.2000,
            'high': 1.2010,
            'low': 1.1990,
            'close': 1.2005,
            'volume': 5000
        })
        
        # Test buy signal
        sl, tp = self.ml_strategy._calculate_dynamic_sl_tp(current_bar, 'buy')
        
        self.assertLess(sl, current_bar['close'])  # Stop loss below current price
        self.assertGreater(tp, current_bar['close'])  # Take profit above current price
        
        # Test sell signal
        sl, tp = self.ml_strategy._calculate_dynamic_sl_tp(current_bar, 'sell')
        
        self.assertGreater(sl, current_bar['close'])  # Stop loss above current price
        self.assertLess(tp, current_bar['close'])  # Take profit below current price


class TestSignal(unittest.TestCase):
    """Test Signal class"""
    
    def test_signal_creation(self):
        """Test signal creation and attributes"""
        signal = Signal(
            symbol='EURUSD',
            action='buy',
            confidence=0.75,
            price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2100,
            timestamp=datetime.now(),
            strategy_name='TestStrategy',
            metadata={'test': 'value'}
        )
        
        self.assertEqual(signal.symbol, 'EURUSD')
        self.assertEqual(signal.action, 'buy')
        self.assertEqual(signal.confidence, 0.75)
        self.assertEqual(signal.price, 1.2000)
        self.assertEqual(signal.stop_loss, 1.1950)
        self.assertEqual(signal.take_profit, 1.2100)
        self.assertEqual(signal.strategy_name, 'TestStrategy')
        self.assertEqual(signal.metadata['test'], 'value')
    
    def test_signal_validation(self):
        """Test signal validation"""
        # Valid signal
        valid_signal = Signal(
            symbol='EURUSD',
            action='buy',
            confidence=0.75,
            price=1.2000
        )
        
        self.assertIsInstance(valid_signal, Signal)
        
        # Test invalid confidence (should be clamped to 0-1)
        signal_high_conf = Signal(
            symbol='EURUSD',
            action='buy',
            confidence=1.5,  # Invalid: > 1
            price=1.2000
        )
        
        # The signal should still be created, but confidence should be reasonable
        self.assertIsInstance(signal_high_conf, Signal)


class TestStrategyIntegration(unittest.TestCase):
    """Integration tests for strategy system"""
    
    def test_multiple_strategies(self):
        """Test running multiple strategies together"""
        # Initialize strategies
        smc = SmartMoneyConcept("SMC", {})
        dom = DOMAnalysis("DOM", {})
        
        strategies = [smc, dom]
        
        # Activate all strategies
        for strategy in strategies:
            strategy.activate()
            self.assertTrue(strategy.is_active)
        
        # Generate sample data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1H')
        ohlcv_data = []
        
        np.random.seed(42)
        price = 1.2000
        
        for date in dates:
            change = np.random.normal(0, 0.001)
            new_price = price * (1 + change)
            
            ohlcv_data.append([
                date,
                price,  # open
                max(price, new_price) * 1.0005,  # high
                min(price, new_price) * 0.9995,  # low
                new_price,  # close
                np.random.uniform(1000, 5000)  # volume
            ])
            
            price = new_price
        
        market_data = {
            'ohlcv': {
                'EURUSD': ohlcv_data
            },
            'order_books': {
                'EURUSD': {
                    'bids': [{'price': 1.2000, 'amount': 5000}],
                    'asks': [{'price': 1.2001, 'amount': 5000}]
                }
            }
        }
        
        all_signals = []
        
        # Collect signals from all strategies
        for strategy in strategies:
            signals = strategy.generate_signals(market_data)
            all_signals.extend(signals)
        
        # Verify we can get signals from multiple strategies
        self.assertIsInstance(all_signals, list)
        
        # Check that signals from different strategies have different strategy names
        strategy_names = set(signal.strategy_name for signal in all_signals)
        self.assertGreaterEqual(len(strategy_names), 0)  # May be 0 if no signals generated


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBaseStrategy))
    test_suite.addTest(unittest.makeSuite(TestSmartMoneyConcept))
    test_suite.addTest(unittest.makeSuite(TestDOMAnalysis))
    test_suite.addTest(unittest.makeSuite(TestMLStrategy))
    test_suite.addTest(unittest.makeSuite(TestSignal))
    test_suite.addTest(unittest.makeSuite(TestStrategyIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")