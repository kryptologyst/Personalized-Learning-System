"""Visualization utilities for personalized learning system."""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import os


class PersonalizedLearningVisualizer:
    """Visualization utilities for personalized learning analysis."""
    
    def __init__(self, resource_names: Dict[int, str] = None):
        """Initialize visualizer.
        
        Args:
            resource_names: Mapping of resource IDs to names
        """
        self.resource_names = resource_names or {0: 'Video', 1: 'Quiz', 2: 'Reading Article'}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_feature_distributions(
        self, 
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot distributions of student features.
        
        Args:
            df: DataFrame with student data
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        features = ['study_time', 'learning_style_score', 'previous_performance', 'engagement_score']
        titles = ['Study Time (hours/week)', 'Learning Style Score', 'Previous Performance', 'Engagement Score']
        
        for i, (feature, title) in enumerate(zip(features, titles)):
            row, col = i // 2, i % 2
            
            axes[row, col].hist(df[feature], bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel(feature.replace('_', ' ').title())
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_resource_distribution(
        self, 
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot distribution of recommended resources.
        
        Args:
            df: DataFrame with student data
            save_path: Path to save the plot
        """
        resource_counts = df['resource_recommendation'].value_counts()
        resource_labels = [self.resource_names[i] for i in resource_counts.index]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(resource_labels, resource_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Resource Recommendation Distribution')
        ax1.set_ylabel('Number of Students')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(resource_counts.values, labels=resource_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Resource Recommendation Proportions')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_correlations(
        self, 
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot correlation matrix of features.
        
        Args:
            df: DataFrame with student data
            save_path: Path to save the plot
        """
        feature_cols = ['study_time', 'learning_style_score', 'previous_performance', 'engagement_score']
        corr_matrix = df[feature_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_performance_comparison(
        self, 
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot model performance comparison.
        
        Args:
            results: List of model evaluation results
            save_path: Path to save the plot
        """
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        titles = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2
            
            bars = axes[row, col].bar(df['model_name'], df[metric], color='skyblue', edgecolor='navy')
            axes[row, col].set_title(title)
            axes[row, col].set_ylabel('Score')
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                  f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(
        self, 
        df: pd.DataFrame,
        results: List[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Create interactive dashboard with Plotly.
        
        Args:
            df: DataFrame with student data
            results: Model evaluation results
            save_path: Path to save the HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Study Time Distribution', 'Learning Style Distribution',
                'Performance Distribution', 'Engagement Distribution',
                'Resource Recommendations', 'Feature Correlations'
            ),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "heatmap"}]]
        )
        
        # Study time distribution
        fig.add_trace(
            go.Histogram(x=df['study_time'], name='Study Time'),
            row=1, col=1
        )
        
        # Learning style distribution
        fig.add_trace(
            go.Histogram(x=df['learning_style_score'], name='Learning Style'),
            row=1, col=2
        )
        
        # Performance distribution
        fig.add_trace(
            go.Histogram(x=df['previous_performance'], name='Performance'),
            row=2, col=1
        )
        
        # Engagement distribution
        fig.add_trace(
            go.Histogram(x=df['engagement_score'], name='Engagement'),
            row=2, col=2
        )
        
        # Resource recommendations
        resource_counts = df['resource_recommendation'].value_counts()
        resource_labels = [self.resource_names[i] for i in resource_counts.index]
        
        fig.add_trace(
            go.Pie(labels=resource_labels, values=resource_counts.values, name="Resources"),
            row=3, col=1
        )
        
        # Feature correlations
        feature_cols = ['study_time', 'learning_style_score', 'previous_performance', 'engagement_score']
        corr_matrix = df[feature_cols].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=feature_cols,
                y=feature_cols,
                colorscale='RdBu',
                zmid=0
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200,
            title_text="Personalized Learning System Dashboard",
            showlegend=False
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
        
        fig.show()


def create_summary_report(
    df: pd.DataFrame,
    results: List[Dict[str, Any]],
    save_path: str = "assets/summary_report.txt"
) -> None:
    """Create a text summary report.
    
    Args:
        df: DataFrame with student data
        results: Model evaluation results
        save_path: Path to save the report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Personalized Learning System - Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Dataset summary
        f.write("Dataset Summary:\n")
        f.write(f"  Total students: {len(df)}\n")
        f.write(f"  Features: {df.shape[1] - 1}\n")
        f.write(f"  Resource types: {df['resource_recommendation'].nunique()}\n\n")
        
        # Feature statistics
        f.write("Feature Statistics:\n")
        for col in ['study_time', 'learning_style_score', 'previous_performance', 'engagement_score']:
            f.write(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}\n")
        f.write("\n")
        
        # Resource distribution
        f.write("Resource Distribution:\n")
        resource_counts = df['resource_recommendation'].value_counts()
        resource_names = {0: 'Video', 1: 'Quiz', 2: 'Reading Article'}
        for resource_id, count in resource_counts.items():
            f.write(f"  {resource_names[resource_id]}: {count} ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        # Model performance
        if results:
            f.write("Model Performance:\n")
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('accuracy', ascending=False)
            
            for _, row in results_df.iterrows():
                f.write(f"  {row['model_name']}:\n")
                f.write(f"    Accuracy: {row['accuracy']:.3f}\n")
                f.write(f"    F1 Score: {row['f1_macro']:.3f}\n")
                f.write(f"    Precision: {row['precision_macro']:.3f}\n")
                f.write(f"    Recall: {row['recall_macro']:.3f}\n")
        
        f.write(f"\nReport generated on: {pd.Timestamp.now()}\n")
    
    print(f"Summary report saved to: {save_path}")
