"""Evaluation metrics and model comparison for personalized learning system."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


class PersonalizedLearningEvaluator:
    """Evaluator for personalized learning models."""
    
    def __init__(self, resource_names: Dict[int, str] = None):
        """Initialize evaluator.
        
        Args:
            resource_names: Mapping of resource IDs to names
        """
        self.resource_names = resource_names or {0: 'Video', 1: 'Quiz', 2: 'Reading Article'}
        
    def evaluate_model(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Evaluate a single model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, resource_name in self.resource_names.items():
            metrics[f'precision_{resource_name.lower()}'] = precision_per_class[i]
            metrics[f'recall_{resource_name.lower()}'] = recall_per_class[i]
            metrics[f'f1_{resource_name.lower()}'] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # ROC AUC (if probabilities available)
        if y_proba is not None:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
            except ValueError:
                # Handle case where some classes are missing
                metrics['roc_auc_ovr'] = np.nan
                metrics['roc_auc_ovo'] = np.nan
        
        return metrics
    
    def create_leaderboard(
        self, 
        results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create a model leaderboard.
        
        Args:
            results: List of evaluation results
            
        Returns:
            DataFrame with model rankings
        """
        df = pd.DataFrame(results)
        
        # Sort by accuracy (primary) and f1_macro (secondary)
        df = df.sort_values(['accuracy', 'f1_macro'], ascending=False)
        
        # Add ranking
        df['rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        main_metrics = ['rank', 'model_name', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        other_metrics = [col for col in df.columns if col not in main_metrics]
        df = df[main_metrics + other_metrics]
        
        return df
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        model_name: str,
        save_path: str = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f',
            cmap='Blues',
            xticklabels=list(self.resource_names.values()),
            yticklabels=list(self.resource_names.values())
        )
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Resource Type')
        plt.ylabel('Actual Resource Type')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(
        self, 
        results: List[Dict[str, Any]],
        save_path: str = None
    ) -> None:
        """Plot model comparison charts.
        
        Args:
            results: List of evaluation results
            model_name: Name of the model
            save_path: Path to save the plot
        """
        df = pd.DataFrame(results)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'F1 Score (Macro)', 'Precision (Macro)', 'Recall (Macro)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy
        fig.add_trace(
            go.Bar(x=df['model_name'], y=df['accuracy'], name='Accuracy'),
            row=1, col=1
        )
        
        # F1 Score
        fig.add_trace(
            go.Bar(x=df['model_name'], y=df['f1_macro'], name='F1 Score'),
            row=1, col=2
        )
        
        # Precision
        fig.add_trace(
            go.Bar(x=df['model_name'], y=df['precision_macro'], name='Precision'),
            row=2, col=1
        )
        
        # Recall
        fig.add_trace(
            go.Bar(x=df['model_name'], y=df['recall_macro'], name='Recall'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Model Performance Comparison",
            showlegend=False
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_per_class_metrics(
        self, 
        results: List[Dict[str, Any]],
        save_path: str = None
    ) -> None:
        """Plot per-class metrics comparison.
        
        Args:
            results: List of evaluation results
            save_path: Path to save the plot
        """
        df = pd.DataFrame(results)
        
        # Extract per-class metrics
        precision_cols = [col for col in df.columns if col.startswith('precision_') and col != 'precision_macro' and col != 'precision_weighted']
        recall_cols = [col for col in df.columns if col.startswith('recall_') and col != 'recall_macro' and col != 'recall_weighted']
        f1_cols = [col for col in df.columns if col.startswith('f1_') and col != 'f1_macro' and col != 'f1_weighted']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Precision by Resource Type', 'Recall by Resource Type', 'F1 Score by Resource Type')
        )
        
        # Precision
        for col in precision_cols:
            resource_type = col.replace('precision_', '').title()
            fig.add_trace(
                go.Bar(x=df['model_name'], y=df[col], name=f'{resource_type} Precision'),
                row=1, col=1
            )
        
        # Recall
        for col in recall_cols:
            resource_type = col.replace('recall_', '').title()
            fig.add_trace(
                go.Bar(x=df['model_name'], y=df[col], name=f'{resource_type} Recall'),
                row=1, col=2
            )
        
        # F1 Score
        for col in f1_cols:
            resource_type = col.replace('f1_', '').title()
            fig.add_trace(
                go.Bar(x=df['model_name'], y=df[col], name=f'{resource_type} F1'),
                row=1, col=3
            )
        
        fig.update_layout(
            height=500,
            title_text="Per-Class Performance Comparison",
            showlegend=True
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
        
        fig.show()
    
    def generate_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> str:
        """Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Classification report string
        """
        target_names = list(self.resource_names.values())
        return classification_report(y_true, y_pred, target_names=target_names)


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> Dict[str, float]:
    """Perform cross-validation on a model.
    
    Args:
        model: Model to validate
        X: Features
        y: Labels
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with cross-validation results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'cv_mean': scores.mean(),
        'cv_std': scores.std(),
        'cv_scores': scores
    }


def analyze_prediction_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    resource_names: Dict[int, str]
) -> pd.DataFrame:
    """Analyze prediction errors.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        X: Features
        feature_names: Names of features
        resource_names: Mapping of resource IDs to names
        
    Returns:
        DataFrame with error analysis
    """
    errors = y_true != y_pred
    
    error_df = pd.DataFrame(X[errors], columns=feature_names)
    error_df['true_label'] = y_true[errors]
    error_df['predicted_label'] = y_pred[errors]
    error_df['true_resource'] = error_df['true_label'].map(resource_names)
    error_df['predicted_resource'] = error_df['predicted_label'].map(resource_names)
    
    return error_df
