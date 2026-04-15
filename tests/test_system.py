"""Tests for personalized learning system."""

import pytest
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_generator import PersonalizedLearningDataGenerator
from models.models import ModelFactory, PersonalizedLearningNN, get_device
from eval.evaluator import PersonalizedLearningEvaluator


class TestDataGenerator:
    """Test cases for data generator."""
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        generator = PersonalizedLearningDataGenerator()
        assert generator.config is not None
        assert 'data' in generator.config
    
    def test_generate_student_data(self):
        """Test student data generation."""
        generator = PersonalizedLearningDataGenerator()
        X, y = generator.generate_student_data()
        
        assert X.shape[1] == 4  # 4 features
        assert len(y) == len(X)
        assert all(label in [0, 1, 2] for label in y)  # Valid resource types
    
    def test_create_dataframe(self):
        """Test DataFrame creation."""
        generator = PersonalizedLearningDataGenerator()
        X, y = generator.generate_student_data()
        df = generator.create_dataframe(X, y)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(X)
        assert 'resource_recommendation' in df.columns
    
    def test_split_data(self):
        """Test data splitting."""
        generator = PersonalizedLearningDataGenerator()
        X, y = generator.generate_student_data()
        X_train, X_test, y_train, y_test = generator.split_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
    
    def test_scale_features(self):
        """Test feature scaling."""
        generator = PersonalizedLearningDataGenerator()
        X, y = generator.generate_student_data()
        X_train, X_test, y_train, y_test = generator.split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = generator.scale_features(X_train, X_test)
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert scaler is not None


class TestModels:
    """Test cases for models."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert device in ['cpu', 'cuda', 'mps']
    
    def test_neural_network_initialization(self):
        """Test neural network initialization."""
        model = PersonalizedLearningNN()
        assert model.input_size == 4
        assert model.num_classes == 3
    
    def test_neural_network_forward(self):
        """Test neural network forward pass."""
        model = PersonalizedLearningNN()
        x = torch.randn(1, 4)
        output = model(x)
        
        assert output.shape == (1, 3)
    
    def test_model_factory(self):
        """Test model factory methods."""
        # Test logistic regression
        lr_model = ModelFactory.create_logistic_regression()
        assert lr_model is not None
        
        # Test random forest
        rf_model = ModelFactory.create_random_forest()
        assert rf_model is not None
        
        # Test XGBoost
        xgb_model = ModelFactory.create_xgboost()
        assert xgb_model is not None
        
        # Test LightGBM
        lgb_model = ModelFactory.create_lightgbm()
        assert lgb_model is not None


class TestEvaluator:
    """Test cases for evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = PersonalizedLearningEvaluator()
        assert evaluator.resource_names is not None
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        evaluator = PersonalizedLearningEvaluator()
        
        # Create dummy data
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        y_proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], 
                           [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        
        results = evaluator.evaluate_model(y_true, y_pred, y_proba, "Test Model")
        
        assert 'accuracy' in results
        assert 'precision_macro' in results
        assert 'recall_macro' in results
        assert 'f1_macro' in results
        assert results['accuracy'] == 1.0  # Perfect predictions
    
    def test_create_leaderboard(self):
        """Test leaderboard creation."""
        evaluator = PersonalizedLearningEvaluator()
        
        # Create dummy results
        results = [
            {'model_name': 'Model A', 'accuracy': 0.8, 'f1_macro': 0.75},
            {'model_name': 'Model B', 'accuracy': 0.9, 'f1_macro': 0.85}
        ]
        
        leaderboard = evaluator.create_leaderboard(results)
        
        assert isinstance(leaderboard, pd.DataFrame)
        assert 'rank' in leaderboard.columns
        assert leaderboard.iloc[0]['model_name'] == 'Model B'  # Best model first


def test_integration():
    """Integration test for the complete pipeline."""
    # Generate data
    generator = PersonalizedLearningDataGenerator()
    X, y = generator.generate_student_data()
    df = generator.create_dataframe(X, y)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = generator.split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = generator.scale_features(X_train, X_test)
    
    # Train a model
    model = ModelFactory.create_logistic_regression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    evaluator = PersonalizedLearningEvaluator()
    results = evaluator.evaluate_model(y_test, y_pred, model.predict_proba(X_test_scaled), "Integration Test")
    
    # Assertions
    assert results['accuracy'] > 0.5  # Should have reasonable accuracy
    assert len(y_pred) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__])
