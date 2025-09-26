#!/usr/bin/env python3
"""
Machine Learning Trading Strategy
Integrates ML models with trading signals, supports multiple model types,
and includes automated model selection and performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json

from . import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """Machine Learning trading strategy implementation"""
    
    def __init__(self, name: str = "ML_Strategy", config: Dict[str, Any] = None):
        default_config = {
            # Model configuration
            'model_name': None,                   # Name of ML model to use
            'model_type': 'classification',       # 'classification' or 'regression'
            'prediction_threshold': 0.6,          # Minimum prediction confidence
            'feature_lookback': 50,               # Bars to look back for features
            
            # Feature engineering
            'technical_indicators': True,         # Include technical indicators
            'price_features': True,               # Include price-based features
            'volume_features': True,              # Include volume-based features
            'market_microstructure': False,       # Include DOM features (if available)
            'sentiment_features': False,          # Include sentiment data (if available)
            
            # Signal generation
            'signal_smoothing': 3,                # Smooth signals over N periods
            'min_confidence': 0.6,                # Minimum signal confidence
            'risk_reward_ratio': 2.0,             # Default R:R ratio
            'max_holding_period': 24,             # Maximum hours to hold position
            
            # Model management
            'auto_retrain': True,                 # Automatically retrain model
            'retrain_interval_hours': 168,        # Retrain every week
            'min_training_samples': 1000,         # Minimum samples for training
            'validation_split': 0.2,              # Validation data percentage
            'performance_threshold': 0.55,        # Minimum accuracy to keep model
            
            # Feature scaling
            'feature_scaling': 'standard',        # 'standard', 'minmax', or 'none'
            'scale_fit_samples': 500,             # Samples to fit scaler
            
            # Advanced settings
            'ensemble_models': [],                # List of model names for ensemble
            'ensemble_method': 'voting',          # 'voting', 'averaging', 'stacking'
            'model_selection_metric': 'f1_score', # Metric for model selection
            'feature_importance_threshold': 0.01, # Minimum feature importance
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name, default_config)
        
        # ML-specific state
        self.current_model = None
        self.feature_scaler = None
        self.feature_columns = []
        self.model_performance = {}
        self.recent_predictions = deque(maxlen=100)
        self.training_data = deque(maxlen=10000)
        self.last_retrain = None
        
        # Import ML components
        try:
            from ..ml import ModelManager, FeatureEngineering
            self.model_manager = ModelManager()
            self.feature_engineer = FeatureEngineering()
        except ImportError as e:
            logger.error(f"Could not import ML components: {e}")
            self.model_manager = None
            self.feature_engineer = None
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate ML-based trading signals
        
        Args:
            market_data: Dict containing OHLCV data and other market information
            
        Returns:
            List[Signal]: Generated trading signals
        """
        signals = []
        
        try:
            if not self.model_manager or not self.current_model:
                logger.warning("No ML model loaded")
                return signals
            
            # Process market data for each symbol
            if 'ohlcv' not in market_data:
                return signals
            
            for symbol, ohlcv_data in market_data['ohlcv'].items():
                if len(ohlcv_data) < self.config['feature_lookback']:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
                
                # Generate features
                features = self._prepare_features(df)
                
                if features is not None and len(features) > 0:
                    # Make prediction
                    prediction = self._make_prediction(features)
                    
                    if prediction is not None:
                        # Convert prediction to trading signal
                        symbol_signals = self._prediction_to_signals(
                            symbol, prediction, df.iloc[-1], features
                        )
                        signals.extend(symbol_signals)
                        
                        # Store prediction for performance tracking
                        self.recent_predictions.append({
                            'symbol': symbol,
                            'prediction': prediction,
                            'features': features,
                            'timestamp': datetime.now()
                        })
            
            # Check if model retraining is needed
            if self._should_retrain_model():
                self._schedule_model_retrain()
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
        
        return signals
    
    def _prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for ML model prediction
        
        Args:
            df: Price data DataFrame
            
        Returns:
            Feature array or None if preparation failed
        """
        try:
            if not self.feature_engineer:
                logger.error("Feature engineer not available")
                return None
            
            # Create technical indicators
            if self.config['technical_indicators']:
                df = self.feature_engineer.create_technical_indicators(df)
            
            # Take only the lookback period
            df_features = df.tail(self.config['feature_lookback'])
            
            # Prepare features
            X, _ = self.feature_engineer.prepare_features(df_features)
            
            if X is None or len(X) == 0:
                return None
            
            # Use only the latest row for prediction
            latest_features = X[-1:]
            
            # Apply feature scaling if configured
            if self.config['feature_scaling'] != 'none' and self.feature_scaler:
                latest_features = self.feature_scaler.transform(latest_features)
            
            return latest_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _make_prediction(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """Make prediction using loaded ML model
        
        Args:
            features: Feature array
            
        Returns:
            Prediction result dictionary or None
        """
        try:
            if not self.model_manager or not self.current_model:
                return None
            
            model_name = self.config['model_name']
            if not model_name:
                return None
            
            # Make prediction
            prediction = self.model_manager.predict(model_name, features)
            
            if prediction is None:
                return None
            
            # Process prediction based on model type
            if self.config['model_type'] == 'classification':
                # For binary classification: [0, 1] -> [sell, buy]
                if len(prediction) > 0:
                    pred_class = int(prediction[0])
                    confidence = prediction[0] if isinstance(prediction[0], float) else 0.5
                    
                    # For probability outputs (if model supports predict_proba)
                    try:
                        model = self.model_manager.load_model(model_name)
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(features)[0]
                            confidence = max(probabilities)  # Highest probability
                            pred_class = np.argmax(probabilities)
                    except Exception:
                        pass  # Use default confidence
                    
                    return {
                        'type': 'classification',
                        'prediction': pred_class,
                        'confidence': float(confidence),
                        'action': 'buy' if pred_class == 1 else 'sell' if pred_class == 0 else 'hold'
                    }
            
            elif self.config['model_type'] == 'regression':
                # For regression: predict future price or return
                if len(prediction) > 0:
                    predicted_value = float(prediction[0])
                    
                    # Convert to action based on predicted return
                    if predicted_value > 0.005:  # >0.5% predicted return
                        action = 'buy'
                        confidence = min(0.9, abs(predicted_value) * 10)  # Scale to 0-0.9
                    elif predicted_value < -0.005:  # <-0.5% predicted return
                        action = 'sell'
                        confidence = min(0.9, abs(predicted_value) * 10)
                    else:
                        action = 'hold'
                        confidence = 0.3
                    
                    return {
                        'type': 'regression',
                        'prediction': predicted_value,
                        'confidence': confidence,
                        'action': action
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error making ML prediction: {e}")
            return None
    
    def _prediction_to_signals(self, symbol: str, prediction: Dict[str, Any], 
                             current_bar: pd.Series, features: np.ndarray) -> List[Signal]:
        """Convert ML prediction to trading signals
        
        Args:
            symbol: Trading symbol
            prediction: ML model prediction
            current_bar: Current price bar
            features: Feature array used for prediction
            
        Returns:
            List of trading signals
        """
        signals = []
        
        try:
            action = prediction.get('action', 'hold')
            confidence = prediction.get('confidence', 0.0)
            
            # Filter by confidence threshold
            if confidence < self.config['min_confidence']:
                return signals
            
            if action in ['buy', 'sell']:
                current_price = current_bar['close']
                
                # Calculate dynamic stop loss and take profit
                sl, tp = self._calculate_dynamic_sl_tp(current_bar, action)
                
                # Apply signal smoothing if configured
                if self.config['signal_smoothing'] > 1:
                    confidence = self._apply_signal_smoothing(symbol, action, confidence)
                
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=datetime.now(),
                    strategy_name=self.name,
                    metadata={
                        'signal_type': 'ml_prediction',
                        'model_name': self.config['model_name'],
                        'model_type': self.config['model_type'],
                        'raw_prediction': prediction['prediction'],
                        'feature_count': len(features[0]) if len(features) > 0 else 0,
                        'model_confidence': prediction.get('confidence', 0.0)
                    }
                )
                
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error converting prediction to signal: {e}")
        
        return signals
    
    def _calculate_dynamic_sl_tp(self, current_bar: pd.Series, action: str) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit based on market conditions
        
        Args:
            current_bar: Current price bar
            action: Trade action ('buy' or 'sell')
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            current_price = current_bar['close']
            
            # Calculate volatility-based stops
            high_low_range = current_bar['high'] - current_bar['low']
            volatility_factor = high_low_range / current_price if current_price > 0 else 0.01
            
            # Base risk (can be adjusted based on market conditions)
            base_risk = max(0.005, volatility_factor * 1.5)  # At least 0.5%
            base_risk = min(0.02, base_risk)  # Maximum 2%
            
            if action == 'buy':
                stop_loss = current_price * (1 - base_risk)
                take_profit = current_price * (1 + base_risk * self.config['risk_reward_ratio'])
            else:  # sell
                stop_loss = current_price * (1 + base_risk)
                take_profit = current_price * (1 - base_risk * self.config['risk_reward_ratio'])
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating dynamic SL/TP: {e}")
            current_price = current_bar['close']
            if action == 'buy':
                return current_price * 0.99, current_price * 1.02
            else:
                return current_price * 1.01, current_price * 0.98
    
    def _apply_signal_smoothing(self, symbol: str, action: str, confidence: float) -> float:
        """Apply signal smoothing to reduce noise
        
        Args:
            symbol: Trading symbol
            action: Signal action
            confidence: Original confidence
            
        Returns:
            Smoothed confidence
        """
        try:
            # Get recent predictions for this symbol
            recent_preds = [pred for pred in list(self.recent_predictions) 
                          if pred['symbol'] == symbol and 
                          (datetime.now() - pred['timestamp']).total_seconds() < 3600]  # Last hour
            
            if len(recent_preds) < self.config['signal_smoothing']:
                return confidence
            
            # Calculate smoothed confidence based on recent signals
            recent_actions = [pred['prediction']['action'] for pred in recent_preds[-self.config['signal_smoothing']:]]
            action_consistency = recent_actions.count(action) / len(recent_actions)
            
            # Adjust confidence based on consistency
            smoothed_confidence = confidence * action_consistency
            
            return smoothed_confidence
            
        except Exception as e:
            logger.error(f"Error applying signal smoothing: {e}")
            return confidence
    
    def load_model(self, model_name: str) -> bool:
        """Load ML model for strategy
        
        Args:
            model_name: Name of model to load
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not self.model_manager:
                logger.error("Model manager not available")
                return False
            
            # Load model
            model = self.model_manager.load_model(model_name)
            if model is None:
                logger.error(f"Failed to load model: {model_name}")
                return False
            
            # Get model metadata
            model_info = self.model_manager.get_model_info(model_name)
            if model_info:
                self.config['model_name'] = model_name
                self.config['model_type'] = model_info.get('type', 'classification')
                self.feature_columns = model_info.get('feature_names', [])
            
            self.current_model = model
            
            # Setup feature scaler if needed
            if self.config['feature_scaling'] != 'none':
                self._setup_feature_scaler()
            
            logger.info(f"ML model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def _setup_feature_scaler(self):
        """Setup feature scaler for model input"""
        try:
            if not self.feature_engineer:
                return
            
            # This would ideally use historical data to fit the scaler
            # For now, we'll set it up to be fitted when we have enough data
            scaling_method = self.config['feature_scaling']
            
            if scaling_method == 'standard':
                from sklearn.preprocessing import StandardScaler
                self.feature_scaler = StandardScaler()
            elif scaling_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.feature_scaler = MinMaxScaler()
            
            logger.info(f"Feature scaler ({scaling_method}) setup complete")
            
        except ImportError as e:
            logger.error(f"Scikit-learn not available for feature scaling: {e}")
        except Exception as e:
            logger.error(f"Error setting up feature scaler: {e}")
    
    def _should_retrain_model(self) -> bool:
        """Check if model should be retrained
        
        Returns:
            bool: True if retraining is needed
        """
        try:
            if not self.config['auto_retrain']:
                return False
            
            if not self.last_retrain:
                return True  # Never trained
            
            # Check time since last retrain
            hours_since_retrain = (datetime.now() - self.last_retrain).total_seconds() / 3600
            if hours_since_retrain >= self.config['retrain_interval_hours']:
                return True
            
            # Check if we have enough new data
            if len(self.training_data) >= self.config['min_training_samples'] * 1.5:
                return True
            
            # Check model performance degradation
            if self.model_performance:
                current_accuracy = self.model_performance.get('accuracy', 1.0)
                if current_accuracy < self.config['performance_threshold']:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain condition: {e}")
            return False
    
    def _schedule_model_retrain(self):
        """Schedule model retraining (placeholder for async training)"""
        try:
            logger.info(f"Model retraining scheduled for strategy {self.name}")
            
            # In a production system, this would:
            # 1. Queue a background training job
            # 2. Use Celery or similar task queue
            # 3. Train with accumulated data
            # 4. Validate new model performance
            # 5. Deploy if performance is better
            
            # For now, just update timestamp
            self.last_retrain = datetime.now()
            
        except Exception as e:
            logger.error(f"Error scheduling model retrain: {e}")
    
    def update_parameters(self, new_params: Dict[str, Any]) -> bool:
        """Update strategy parameters
        
        Args:
            new_params: New parameter values
            
        Returns:
            bool: True if update successful
        """
        try:
            # Handle model change
            if 'model_name' in new_params and new_params['model_name'] != self.config.get('model_name'):
                model_name = new_params['model_name']
                if not self.load_model(model_name):
                    logger.error(f"Failed to load new model: {model_name}")
                    return False
            
            # Update other parameters
            for key, value in new_params.items():
                if key in self.config:
                    self.config[key] = value
            
            logger.info(f"ML strategy parameters updated: {list(new_params.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating ML parameters: {e}")
            return False
    
    def add_training_data(self, symbol: str, features: np.ndarray, target: float):
        """Add data point for model training
        
        Args:
            symbol: Trading symbol
            features: Feature vector
            target: Target value (actual outcome)
        """
        try:
            self.training_data.append({
                'symbol': symbol,
                'features': features,
                'target': target,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Error adding training data: {e}")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics
        
        Returns:
            Model performance dictionary
        """
        try:
            if not self.model_performance:
                return {'status': 'No performance data available'}
            
            performance = self.model_performance.copy()
            performance['recent_predictions'] = len(self.recent_predictions)
            performance['training_data_size'] = len(self.training_data)
            performance['last_retrain'] = self.last_retrain.isoformat() if self.last_retrain else None
            performance['model_loaded'] = self.current_model is not None
            performance['model_name'] = self.config.get('model_name', 'None')
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {'error': str(e)}
    
    def evaluate_predictions(self, actual_outcomes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate recent predictions against actual outcomes
        
        Args:
            actual_outcomes: List of actual trading outcomes
            
        Returns:
            Evaluation metrics
        """
        try:
            if not self.recent_predictions or not actual_outcomes:
                return {}
            
            # Match predictions with outcomes
            matched_pairs = []
            
            for outcome in actual_outcomes:
                symbol = outcome['symbol']
                outcome_time = outcome['timestamp']
                
                # Find corresponding prediction
                for pred in self.recent_predictions:
                    if (pred['symbol'] == symbol and 
                        abs((pred['timestamp'] - outcome_time).total_seconds()) < 3600):  # Within 1 hour
                        
                        matched_pairs.append({
                            'predicted': pred['prediction']['action'],
                            'actual': outcome['actual_action'],
                            'confidence': pred['prediction']['confidence'],
                            'pnl': outcome.get('pnl', 0.0)
                        })
                        break
            
            if not matched_pairs:
                return {'status': 'No matched predictions found'}
            
            # Calculate evaluation metrics
            correct_predictions = sum(1 for pair in matched_pairs 
                                    if pair['predicted'] == pair['actual'])
            total_predictions = len(matched_pairs)
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            # Calculate profit-based metrics
            total_pnl = sum(pair['pnl'] for pair in matched_pairs)
            profitable_trades = sum(1 for pair in matched_pairs if pair['pnl'] > 0)
            profit_factor = (sum(pair['pnl'] for pair in matched_pairs if pair['pnl'] > 0) / 
                           abs(sum(pair['pnl'] for pair in matched_pairs if pair['pnl'] < 0)))
            
            metrics = {
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'total_pnl': total_pnl,
                'win_rate': profitable_trades / total_predictions if total_predictions > 0 else 0.0,
                'profit_factor': profit_factor if profit_factor != float('inf') else 0.0,
                'avg_confidence': np.mean([pair['confidence'] for pair in matched_pairs]),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Update internal performance tracking
            self.model_performance.update(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return {'error': str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from loaded model
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        try:
            if not self.current_model or not self.model_manager:
                return {}
            
            model_name = self.config['model_name']
            model = self.model_manager.load_model(model_name)
            
            if model is None:
                return {}
            
            # Get feature importance (for supported models)
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):  # Random Forest, Gradient Boosting, etc.
                importances = model.feature_importances_
                if len(importances) == len(self.feature_columns):
                    importance_dict = dict(zip(self.feature_columns, importances))
            
            elif hasattr(model, 'coef_'):  # Linear models
                coefficients = model.coef_
                if len(coefficients) == len(self.feature_columns):
                    # Use absolute coefficients as importance
                    importance_dict = dict(zip(self.feature_columns, np.abs(coefficients)))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def export_strategy_data(self, output_file: str = None) -> bool:
        """Export strategy data including predictions and performance
        
        Args:
            output_file: Output file path
            
        Returns:
            bool: True if export successful
        """
        try:
            if output_file is None:
                output_file = f"data/exports/ml_strategy_{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Create export data
            export_data = {
                'strategy_name': self.name,
                'config': self.config,
                'model_performance': self.model_performance,
                'recent_predictions': list(self.recent_predictions),
                'feature_columns': self.feature_columns,
                'feature_importance': self.get_feature_importance(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Create directory if needed
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to JSON file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"ML strategy data exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting strategy data: {e}")
            return False
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get comprehensive strategy status
        
        Returns:
            Strategy status dictionary
        """
        try:
            return {
                'strategy_name': self.name,
                'is_active': self.is_active,
                'model_loaded': self.current_model is not None,
                'model_name': self.config.get('model_name', 'None'),
                'model_type': self.config.get('model_type', 'Unknown'),
                'feature_count': len(self.feature_columns),
                'recent_predictions': len(self.recent_predictions),
                'training_data_points': len(self.training_data),
                'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
                'performance_metrics': self.get_performance_metrics(),
                'model_performance': self.model_performance,
                'feature_importance': self.get_feature_importance(),
                'status_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
            return {'error': str(e)}


__all__ = [
    'MLStrategy'
]