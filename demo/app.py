"""Streamlit demo for personalized learning system."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import yaml
import joblib
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_generator import PersonalizedLearningDataGenerator
from models.models import ModelFactory, PersonalizedLearningTrainer, get_device


# Page configuration
st.set_page_config(
    page_title="Personalized Learning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .resource-card {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .recommendation-highlight {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample data for the demo."""
    generator = PersonalizedLearningDataGenerator()
    X, y = generator.generate_student_data()
    df = generator.create_dataframe(X, y)
    return df


@st.cache_resource
def load_trained_models():
    """Load pre-trained models (simulated for demo)."""
    # In a real scenario, these would be loaded from saved model files
    models = {}
    
    # Create and train a simple model for demo
    generator = PersonalizedLearningDataGenerator()
    X, y = generator.generate_student_data()
    X_train, X_test, y_train, y_test = generator.split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = generator.scale_features(X_train, X_test)
    
    # Train XGBoost model
    xgb_model = ModelFactory.create_xgboost()
    xgb_model.fit(X_train_scaled, y_train)
    
    models['xgboost'] = xgb_model
    models['scaler'] = scaler
    
    return models


def predict_resource_recommendation(student_profile, models):
    """Predict resource recommendation for a student profile."""
    # Scale the input
    student_scaled = models['scaler'].transform([student_profile])
    
    # Get prediction
    prediction = models['xgboost'].predict(student_scaled)[0]
    probabilities = models['xgboost'].predict_proba(student_scaled)[0]
    
    return prediction, probabilities


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎓 Personalized Learning System</h1>', unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        df = load_sample_data()
        models = load_trained_models()
    
    # Sidebar
    st.sidebar.title("Student Profile")
    st.sidebar.markdown("---")
    
    # Student profile inputs
    study_time = st.sidebar.slider(
        "Study Time (hours/week)", 
        min_value=1.0, max_value=15.0, value=5.0, step=0.5
    )
    
    learning_style = st.sidebar.slider(
        "Learning Style Score", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help="0 = Visual, 1 = Auditory/Kinesthetic"
    )
    
    previous_performance = st.sidebar.slider(
        "Previous Performance", 
        min_value=0.0, max_value=100.0, value=80.0, step=1.0
    )
    
    engagement_score = st.sidebar.slider(
        "Engagement Score", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.1
    )
    
    # Create student profile
    student_profile = [study_time, learning_style, previous_performance, engagement_score]
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Dataset Overview")
        
        # Dataset statistics
        col1_1, col1_2, col1_3, col1_4 = st.columns(4)
        
        with col1_1:
            st.metric("Total Students", len(df))
        
        with col1_2:
            st.metric("Avg Study Time", f"{df['study_time'].mean():.1f}h")
        
        with col1_3:
            st.metric("Avg Performance", f"{df['previous_performance'].mean():.1f}")
        
        with col1_4:
            st.metric("Avg Engagement", f"{df['engagement_score'].mean():.2f}")
        
        # Resource distribution
        st.subheader("📚 Resource Type Distribution")
        resource_counts = df['resource_recommendation'].value_counts()
        resource_names = {0: 'Video', 1: 'Quiz', 2: 'Reading Article'}
        
        fig_pie = px.pie(
            values=resource_counts.values,
            names=[resource_names[i] for i in resource_counts.index],
            title="Distribution of Recommended Resources"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Feature distributions
        st.subheader("📈 Feature Distributions")
        
        fig_dist = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Study Time', 'Learning Style Score', 
                          'Previous Performance', 'Engagement Score')
        )
        
        features = ['study_time', 'learning_style_score', 'previous_performance', 'engagement_score']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for feature, (row, col) in zip(features, positions):
            fig_dist.add_trace(
                go.Histogram(x=df[feature], name=feature.replace('_', ' ').title()),
                row=row, col=col
            )
        
        fig_dist.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Resource Recommendation")
        
        # Get prediction
        prediction, probabilities = predict_resource_recommendation(student_profile, models)
        
        # Display recommendation
        resource_names = {0: 'Video', 1: 'Quiz', 2: 'Reading Article'}
        recommended_resource = resource_names[prediction]
        
        st.markdown(f"""
        <div class="recommendation-highlight">
            <h3>🎯 Recommended Resource</h3>
            <h2>{recommended_resource}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability breakdown
        st.subheader("📊 Prediction Confidence")
        
        prob_df = pd.DataFrame({
            'Resource': [resource_names[i] for i in range(3)],
            'Probability': probabilities
        })
        
        fig_prob = px.bar(
            prob_df, 
            x='Resource', 
            y='Probability',
            title="Prediction Probabilities",
            color='Probability',
            color_continuous_scale='Blues'
        )
        fig_prob.update_layout(yaxis_title="Probability", xaxis_title="Resource Type")
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Student profile summary
        st.subheader("👤 Student Profile Summary")
        
        profile_data = {
            'Metric': ['Study Time', 'Learning Style', 'Performance', 'Engagement'],
            'Value': [f"{study_time:.1f}h", f"{learning_style:.1f}", f"{previous_performance:.0f}", f"{engagement_score:.1f}"]
        }
        
        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True)
    
    # Additional analysis
    st.markdown("---")
    st.subheader("🔍 Detailed Analysis")
    
    # Feature importance (simulated)
    st.subheader("📈 Feature Importance")
    
    # Simulate feature importance
    feature_names = ['Study Time', 'Learning Style', 'Previous Performance', 'Engagement Score']
    importance_values = [0.3, 0.25, 0.35, 0.1]  # Simulated values
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    
    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance for Resource Recommendation",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model performance metrics (simulated)
    st.subheader("📊 Model Performance")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [0.87, 0.85, 0.86, 0.85]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.dataframe(metrics_df, use_container_width=True)
    
    with col4:
        fig_metrics = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            title="Model Performance Metrics",
            color='Value',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Personalized Learning System Demo | 
        <a href='https://github.com/kryptologyst' target='_blank'>kryptologyst</a></p>
        <p><em>This is a research demonstration. Not for operational use.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
