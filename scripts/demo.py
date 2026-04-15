#!/usr/bin/env python3
"""Quick demo script for personalized learning system."""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_generator import PersonalizedLearningDataGenerator
from models.models import ModelFactory


def main():
    """Run a quick demonstration of the personalized learning system."""
    
    print("🎓 Personalized Learning System - Quick Demo")
    print("=" * 50)
    
    # Generate sample data
    print("📊 Generating sample student data...")
    generator = PersonalizedLearningDataGenerator()
    X, y = generator.generate_student_data()
    df = generator.create_dataframe(X, y)
    
    print(f"✅ Generated {len(df)} student profiles")
    print(f"📚 Resource distribution: {df['resource_recommendation'].value_counts().to_dict()}")
    
    # Split and scale data
    print("\n🔄 Preparing data for training...")
    X_train, X_test, y_train, y_test = generator.split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = generator.scale_features(X_train, X_test)
    
    # Train a quick model
    print("\n🤖 Training XGBoost model...")
    model = ModelFactory.create_xgboost()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    accuracy = (predictions == y_test).mean()
    
    print(f"✅ Model trained successfully!")
    print(f"📈 Test accuracy: {accuracy:.3f}")
    
    # Show some example predictions
    print("\n🎯 Example Predictions:")
    resource_names = {0: 'Video', 1: 'Quiz', 2: 'Reading Article'}
    
    for i in range(5):
        student_features = X_test[i]
        predicted_resource = predictions[i]
        actual_resource = y_test[i]
        
        print(f"\nStudent {i+1}:")
        print(f"  Study Time: {student_features[0]:.1f} hours/week")
        print(f"  Learning Style: {student_features[1]:.2f}")
        print(f"  Performance: {student_features[2]:.1f}")
        print(f"  Engagement: {student_features[3]:.2f}")
        print(f"  Recommended: {resource_names[predicted_resource]}")
        print(f"  Actual: {resource_names[actual_resource]}")
        print(f"  Correct: {'✅' if predicted_resource == actual_resource else '❌'}")
    
    print("\n🚀 Demo completed! Run 'streamlit run demo/app.py' for the full interactive demo.")
    print("📖 See README.md for complete usage instructions.")


if __name__ == "__main__":
    main()
