#!/usr/bin/env python3
"""
Machine Learning Framework
Model management, feature engineering, and ML strategy implementation
"""

import os
import joblib
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import numpy as np
import pandas as pd

# ML Libraries
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for ML models with import/export capabilities"""
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Load existing models
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load models from disk"""
        try:
            for file_name in os.listdir(self.models_dir):
                if file_name.endswith(('.pkl', '.joblib', '.onnx')):
                    model_name = os.path.splitext(file_name)[0]
                    self._load_model_metadata(model_name)
        except Exception as e:
            logger.error(f"Error loading existing models: {e}")
    
    def _load_model_metadata(self, model_name: str):
        """Load model metadata"""
        metadata_file = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata for {model_name}: {e}")
    
    def import_model(self, file_path: str, model_name: str, 
                    model_type: str = None, metadata: Dict = None) -> bool:
        """Import model from file
        
        Args:
            file_path: Path to model file
            model_name: Name for the model
            model_type: Type of model ('sklearn', 'pytorch', 'tensorflow', 'onnx')
            metadata: Additional model metadata
            
        Returns:
            bool: True if import successful
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Model file not found: {file_path}")
                return False
            
            # Determine model type from file extension
            if model_type is None:
                if file_path.endswith(('.pkl', '.joblib')):
                    model_type = 'sklearn'
                elif file_path.endswith('.onnx'):
                    model_type = 'onnx'
                elif file_path.endswith('.pth'):
                    model_type = 'pytorch'
                elif file_path.endswith('.h5'):
                    model_type = 'tensorflow'
                else:
                    model_type = 'unknown'
            
            # Copy model to models directory
            file_extension = os.path.splitext(file_path)[1]
            target_path = os.path.join(self.models_dir, f"{model_name}{file_extension}")
            
            import shutil
            shutil.copy2(file_path, target_path)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(target_path)
            
            # Create metadata
            model_metadata = {
                'name': model_name,
                'type': model_type,
                'file_path': target_path,
                'file_size': os.path.getsize(target_path),
                'file_hash': file_hash,
                'imported_at': datetime.now().isoformat(),
                'version': '1.0',
                'is_deployed': False
            }
            
            if metadata:
                model_metadata.update(metadata)
            
            # Save metadata
            metadata_file = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            self.model_metadata[model_name] = model_metadata
            
            logger.info(f"Model {model_name} imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error importing model {model_name}: {e}")
            return False
    
    def export_model(self, model_name: str, output_path: str) -> bool:
        """Export model to file
        
        Args:
            model_name: Name of model to export
            output_path: Output file path
            
        Returns:
            bool: True if export successful
        """
        try:
            if model_name not in self.model_metadata:
                logger.error(f"Model {model_name} not found")
                return False
            
            metadata = self.model_metadata[model_name]
            source_path = metadata['file_path']
            
            if not os.path.exists(source_path):
                logger.error(f"Model file not found: {source_path}")
                return False
            
            import shutil
            shutil.copy2(source_path, output_path)
            
            # Export metadata as well
            metadata_output = output_path.replace('.pkl', '_metadata.json').replace('.onnx', '_metadata.json')
            with open(metadata_output, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model {model_name} exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load model into memory
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Model object or None if failed
        """
        try:
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            
            if model_name not in self.model_metadata:
                logger.error(f"Model {model_name} not found")
                return None
            
            metadata = self.model_metadata[model_name]
            file_path = metadata['file_path']
            model_type = metadata['type']
            
            # Load based on model type
            if model_type == 'sklearn':
                if file_path.endswith('.joblib'):
                    model = joblib.load(file_path)
                else:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                        
            elif model_type == 'onnx' and ONNX_AVAILABLE:
                model = ort.InferenceSession(file_path)
                
            elif model_type == 'pytorch' and PYTORCH_AVAILABLE:
                model = torch.load(file_path, map_location='cpu')
                
            elif model_type == 'tensorflow' and TENSORFLOW_AVAILABLE:
                model = tf.keras.models.load_model(file_path)
                
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
            
            self.loaded_models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def save_model(self, model: Any, model_name: str, model_type: str, 
                  metadata: Dict = None) -> bool:
        """Save model to disk
        
        Args:
            model: Model object to save
            model_name: Name for the model
            model_type: Type of model
            metadata: Additional metadata
            
        Returns:
            bool: True if save successful
        """
        try:
            # Determine file extension and save method
            if model_type == 'sklearn':
                file_path = os.path.join(self.models_dir, f"{model_name}.joblib")
                joblib.dump(model, file_path)
                
            elif model_type == 'pytorch' and PYTORCH_AVAILABLE:
                file_path = os.path.join(self.models_dir, f"{model_name}.pth")
                torch.save(model, file_path)
                
            elif model_type == 'tensorflow' and TENSORFLOW_AVAILABLE:
                file_path = os.path.join(self.models_dir, f"{model_name}.h5")
                model.save(file_path)
                
            else:
                logger.error(f"Unsupported model type for saving: {model_type}")
                return False
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Create metadata
            model_metadata = {
                'name': model_name,
                'type': model_type,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'file_hash': file_hash,
                'saved_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            if metadata:
                model_metadata.update(metadata)
            
            # Save metadata
            metadata_file = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            self.model_metadata[model_name] = model_metadata
            self.loaded_models[model_name] = model
            
            logger.info(f"Model {model_name} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete model from disk and memory
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            bool: True if delete successful
        """
        try:
            if model_name not in self.model_metadata:
                logger.error(f"Model {model_name} not found")
                return False
            
            metadata = self.model_metadata[model_name]
            file_path = metadata['file_path']
            
            # Delete model file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete metadata file
            metadata_file = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            # Remove from memory
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            
            del self.model_metadata[model_name]
            
            logger.info(f"Model {model_name} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models
        
        Returns:
            List of model information dictionaries
        """
        models = []
        for name, metadata in self.model_metadata.items():
            model_info = metadata.copy()
            model_info['is_loaded'] = name in self.loaded_models
            models.append(model_info)
        
        return models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None
        """
        if model_name in self.model_metadata:
            info = self.model_metadata[model_name].copy()
            info['is_loaded'] = model_name in self.loaded_models
            return info
        return None
    
    def predict(self, model_name: str, features: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions using loaded model
        
        Args:
            model_name: Name of the model
            features: Feature array for prediction
            
        Returns:
            Predictions array or None if failed
        """
        try:
            model = self.load_model(model_name)
            if model is None:
                return None
            
            metadata = self.model_metadata[model_name]
            model_type = metadata['type']
            
            # Make prediction based on model type
            if model_type == 'sklearn':
                predictions = model.predict(features)
                
            elif model_type == 'onnx' and ONNX_AVAILABLE:
                input_name = model.get_inputs()[0].name
                predictions = model.run(None, {input_name: features.astype(np.float32)})[0]
                
            elif model_type == 'pytorch' and PYTORCH_AVAILABLE:
                model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features)
                    predictions = model(features_tensor).numpy()
                    
            elif model_type == 'tensorflow' and TENSORFLOW_AVAILABLE:
                predictions = model.predict(features)
                
            else:
                logger.error(f"Prediction not supported for model type: {model_type}")
                return None
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making prediction with model {model_name}: {e}")
            return None
    
    def validate_model(self, model_name: str, validation_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Validate model performance
        
        Args:
            model_name: Name of the model
            validation_data: Tuple of (X_val, y_val)
            
        Returns:
            Dictionary of validation metrics
        """
        try:
            X_val, y_val = validation_data
            predictions = self.predict(model_name, X_val)
            
            if predictions is None:
                return {}
            
            # Calculate metrics (assuming classification)
            if SKLEARN_AVAILABLE:
                metrics = {
                    'accuracy': accuracy_score(y_val, predictions),
                    'precision': precision_score(y_val, predictions, average='weighted'),
                    'recall': recall_score(y_val, predictions, average='weighted'),
                    'f1_score': f1_score(y_val, predictions, average='weighted'),
                    'validation_timestamp': datetime.now().isoformat()
                }
                
                # Update model metadata with validation results
                if model_name in self.model_metadata:
                    self.model_metadata[model_name]['last_validation'] = metrics
                
                return metrics
            
            return {'error': 'Scikit-learn not available for validation'}
            
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            return {'error': str(e)}
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file
        
        Args:
            file_path: Path to file
            
        Returns:
            str: SHA-256 hash
        """
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""


class FeatureEngineering:
    """Feature engineering for financial data"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from OHLCV data
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        try:
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Exponential Moving Averages
            for period in [12, 26, 50]:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price patterns
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Returns and volatility
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Trend indicators
            df['trend_5'] = np.where(df['close'] > df['sma_5'], 1, -1)
            df['trend_20'] = np.where(df['close'] > df['sma_20'], 1, -1)
            df['trend_50'] = np.where(df['close'] > df['sma_50'], 1, -1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception:
            return pd.Series(50, index=prices.index)  # Neutral RSI fallback
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model
        
        Args:
            df: DataFrame with market data and indicators
            target_column: Name of target column (for supervised learning)
            
        Returns:
            Tuple of (features, targets) or (features, None)
        """
        try:
            # Remove non-feature columns
            feature_df = df.copy()
            
            # Remove timestamp and target columns
            columns_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if target_column and target_column in feature_df.columns:
                columns_to_drop.append(target_column)
            
            for col in columns_to_drop:
                if col in feature_df.columns:
                    feature_df = feature_df.drop(columns=[col])
            
            # Handle missing values
            feature_df = feature_df.fillna(method='ffill').fillna(0)
            
            # Store feature column names
            self.feature_columns = list(feature_df.columns)
            
            # Convert to numpy arrays
            X = feature_df.values
            y = df[target_column].values if target_column and target_column in df.columns else None
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([]), None
    
    def scale_features(self, X: np.ndarray, scaler_name: str = 'standard', 
                     fit: bool = True) -> np.ndarray:
        """Scale features using specified scaler
        
        Args:
            X: Feature array
            scaler_name: Name of scaler ('standard', 'minmax')
            fit: Whether to fit the scaler
            
        Returns:
            Scaled feature array
        """
        try:
            if scaler_name not in self.scalers or fit:
                if scaler_name == 'standard':
                    scaler = StandardScaler()
                elif scaler_name == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    logger.error(f"Unknown scaler: {scaler_name}")
                    return X
                
                if fit:
                    scaler.fit(X)
                    self.scalers[scaler_name] = scaler
            
            scaler = self.scalers[scaler_name]
            return scaler.transform(X)
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return X


__all__ = [
    'ModelManager',
    'FeatureEngineering',
    'SKLEARN_AVAILABLE',
    'PYTORCH_AVAILABLE', 
    'TENSORFLOW_AVAILABLE',
    'ONNX_AVAILABLE'
]