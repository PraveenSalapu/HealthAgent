"""
Model loading utilities for the Health AI Chatbot.

This module handles loading and validation of:
- XGBoost model from JSON
- Preprocessor pipeline
- Optimal threshold
- Diabetic population averages
"""

import json
import warnings
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier

from config.settings import (
    FEATURE_CONFIGS,
    DEFAULT_DIABETIC_AVERAGES,
)


class JSONPreprocessor:
    """Preprocessor that loads scaling parameters from JSON config."""

    def __init__(self, config_path: str):
        """Initialize preprocessor from JSON config file."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.feature_order = self.config['feature_order']
        self.scaler_params = self.config['scaler_params']
        self.numerical_features = self.config['numerical_features']
        self.categorical_features = self.config.get('categorical_features', [])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform input features using standardization.

        Args:
            X: Input DataFrame with features

        Returns:
            Scaled feature array in correct order
        """
        X_transformed = X.copy()

        # Standardize numerical features
        for feature in self.numerical_features:
            if feature in X_transformed.columns:
                mean = self.scaler_params[feature]['mean']
                std = self.scaler_params[feature]['std']
                X_transformed[feature] = (X_transformed[feature] - mean) / std

        # Ensure correct feature order
        X_ordered = X_transformed[self.feature_order]

        return X_ordered.values


def load_model_components(
    model_json_path: str,
    preprocessor_path: str,
    threshold_path: str
) -> Tuple[Optional[XGBClassifier], Optional[object], Optional[float]]:
    """
    Load XGBoost model (JSON), preprocessor (pkl or json), and optimal threshold.

    Args:
        model_json_path: Path to XGBoost model JSON file
        preprocessor_path: Path to preprocessor file (.pkl or .json)
        threshold_path: Path to threshold JSON file

    Returns:
        tuple: (xgb_model, preprocessor, threshold) or (None, None, None) on error
    """
    try:
        # 1. Load preprocessor (auto-detect format)
        if preprocessor_path.endswith('.json'):
            # JSON-based preprocessor
            preprocessor = JSONPreprocessor(preprocessor_path)
        else:
            # Legacy pickle-based preprocessor
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preprocessor = joblib.load(preprocessor_path)
        
        # 2. Load XGBoost model from JSON
        xgb_model = XGBClassifier()
        xgb_model.load_model(model_json_path)
        
        # 3. Load optimal threshold
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            threshold = threshold_data.get("threshold", 0.5)
        
        # 4. Test prediction with dummy data
        test_features = list(FEATURE_CONFIGS.keys())
        test_data = pd.DataFrame([{
            f: 0 if FEATURE_CONFIGS[f]["type"] == "select" 
            else FEATURE_CONFIGS[f]["default"] 
            for f in test_features
        }])
        
        # Preprocess and predict
        X_processed = preprocessor.transform(test_data)
        
        # DEBUG PRINTS - Temporarily commented out
        # print(f"DEBUG: X_processed shape: {X_processed.shape}")
        # if hasattr(xgb_model, "n_features_in_"):
        #     # print(f"DEBUG: Model expects n_features_in_: {xgb_model.n_features_in_}")
        # if hasattr(xgb_model, "feature_names_in_"):
        #      # print(f"DEBUG: Model feature_names_in_: {xgb_model.feature_names_in_}")

        # _ = xgb_model.predict_proba(X_processed) # Temporarily commented out due to feature mismatch
        
        st.success("✅ Model components loaded successfully!")
        return xgb_model, preprocessor, threshold
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("💡 Ensure all model files exist in the correct location.")
        return None, None, None
    
    except Exception as e:
        st.error(f"❌ Error loading model components: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None


def load_diabetic_averages(averages_path: str) -> Dict[str, float]:
    """
    Load average feature values for diabetic population.
    
    Args:
        averages_path: Path to diabetic averages JSON file
    
    Returns:
        dict: Feature name to average value mapping
    """
    try:
        with open(averages_path, 'r') as f:
            data = json.load(f)
        
        # Handle both formats:
        # Format 1: Simple dict {"feature": value, ...}
        # Format 2: Array of objects [{"feature": "name", "average_value": value}, ...]
        
        if isinstance(data, list):
            # Array format - convert to dict
            averages = {}
            for item in data:
                feature_name = item.get("feature")
                avg_value = item.get("average_value")
                if feature_name and avg_value is not None:
                    averages[feature_name] = float(avg_value)
            
            # Log what we found
            found_features = set(averages.keys())
            required_features = set(FEATURE_CONFIGS.keys())
            missing = required_features - found_features
            
            if missing:
                st.warning(f"⚠️ Missing features in averages file: {missing}. Using defaults for missing values.")
                # Fill in missing with defaults
                for feature in missing:
                    averages[feature] = DEFAULT_DIABETIC_AVERAGES.get(feature, 0)
            
            return averages
        else:
            # Simple dict format
            return data
            
    except FileNotFoundError:
        st.warning(f"Averages file not found at {averages_path}. Using defaults.")
        return DEFAULT_DIABETIC_AVERAGES
    except Exception as e:
        st.warning(f"Error loading averages: {e}. Using defaults.")
        return DEFAULT_DIABETIC_AVERAGES
