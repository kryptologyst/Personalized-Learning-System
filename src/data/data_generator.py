"""Data generation and processing module for personalized learning system."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import os


class PersonalizedLearningDataGenerator:
    """Generate synthetic student learning data for personalized learning system."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        """Initialize data generator with configuration.
        
        Args:
            config_path: Path to data configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed for reproducibility
        np.random.seed(self.config['data']['generation']['random_seed'])
        
    def generate_student_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic student learning data.
        
        Returns:
            Tuple of (features, labels) where features is (n_samples, n_features)
            and labels is (n_samples,) with resource recommendations
        """
        n_samples = self.config['data']['generation']['n_samples']
        feature_configs = self.config['data']['generation']['features']
        
        # Generate features
        study_time = np.random.normal(
            feature_configs['study_time']['params']['mean'],
            feature_configs['study_time']['params']['std'],
            n_samples
        )
        
        learning_style_score = np.random.uniform(
            feature_configs['learning_style_score']['params']['min'],
            feature_configs['learning_style_score']['params']['max'],
            n_samples
        )
        
        previous_performance = np.random.normal(
            feature_configs['previous_performance']['params']['mean'],
            feature_configs['previous_performance']['params']['std'],
            n_samples
        )
        
        engagement_score = np.random.uniform(
            feature_configs['engagement_score']['params']['min'],
            feature_configs['engagement_score']['params']['max'],
            n_samples
        )
        
        resource_preference = np.random.randint(
            feature_configs['resource_preference']['params']['min'],
            feature_configs['resource_preference']['params']['max'] + 1,
            n_samples
        )
        
        # Create feature matrix
        X = np.stack([
            study_time,
            learning_style_score, 
            previous_performance,
            engagement_score
        ], axis=1)
        
        # Generate recommendations based on business logic
        y = self._generate_recommendations(
            previous_performance, study_time, learning_style_score, 
            engagement_score, resource_preference
        )
        
        return X, y
    
    def _generate_recommendations(
        self, 
        performance: np.ndarray,
        study_time: np.ndarray, 
        learning_style: np.ndarray,
        engagement: np.ndarray,
        preference: np.ndarray
    ) -> np.ndarray:
        """Generate resource recommendations based on student profiles.
        
        Args:
            performance: Previous performance scores
            study_time: Study time per week
            learning_style: Learning style preference
            engagement: Engagement level
            preference: Resource preference
            
        Returns:
            Array of recommended resource types (0=video, 1=quiz, 2=reading)
        """
        recommendations = np.zeros(len(performance), dtype=int)
        
        # Business logic for recommendations
        for i in range(len(performance)):
            if performance[i] < 70 and study_time[i] < 4:
                # Low performance + low study time -> Quiz for reinforcement
                recommendations[i] = 1
            elif learning_style[i] > 0.5:
                # Visual learners -> Video content
                recommendations[i] = 0
            elif engagement[i] > 0.5:
                # High engagement -> Reading for deeper learning
                recommendations[i] = 2
            else:
                # Default to quiz for general reinforcement
                recommendations[i] = 1
                
        return recommendations
    
    def create_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Create a pandas DataFrame from features and labels.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            DataFrame with features and labels
        """
        feature_names = [
            'study_time',
            'learning_style_score', 
            'previous_performance',
            'engagement_score'
        ]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['resource_recommendation'] = y
        
        return df
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = self.config['data']['splits']['test_size']
        random_state = self.config['data']['splits']['random_state']
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
    
    def scale_features(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (scaled_X_train, scaled_X_test, scaler)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler


def load_data(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    """Load and prepare data for the personalized learning system.
    
    Args:
        config_path: Path to data configuration file
        
    Returns:
        Dictionary containing data splits and metadata
    """
    generator = PersonalizedLearningDataGenerator(config_path)
    
    # Generate data
    X, y = generator.generate_student_data()
    
    # Create DataFrame for analysis
    df = generator.create_dataframe(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = generator.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = generator.scale_features(X_train, X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'dataframe': df,
        'feature_names': ['study_time', 'learning_style_score', 'previous_performance', 'engagement_score'],
        'resource_names': {0: 'Video', 1: 'Quiz', 2: 'Reading Article'}
    }
