#!/usr/bin/env python3
"""Main training script for personalized learning system."""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_generator import load_data
from models.models import (
    ModelFactory, PersonalizedLearningTrainer, get_device
)
from eval.evaluator import (
    PersonalizedLearningEvaluator, cross_validate_model, 
    analyze_prediction_errors
)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_all_models(
    data: Dict[str, Any], 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Train all models and return results."""
    
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    resource_names = data['resource_names']
    
    evaluator = PersonalizedLearningEvaluator(resource_names)
    results = []
    
    # 1. Logistic Regression (Baseline)
    print("Training Logistic Regression...")
    lr_model = ModelFactory.create_logistic_regression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)
    
    lr_results = evaluator.evaluate_model(y_test, lr_pred, lr_proba, "Logistic Regression")
    results.append(lr_results)
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf_model = ModelFactory.create_random_forest()
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)
    
    rf_results = evaluator.evaluate_model(y_test, rf_pred, rf_proba, "Random Forest")
    results.append(rf_results)
    
    # 3. XGBoost
    print("Training XGBoost...")
    xgb_model = ModelFactory.create_xgboost(config.get('models', {}).get('xgboost', {}))
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_proba = xgb_model.predict_proba(X_test_scaled)
    
    xgb_results = evaluator.evaluate_model(y_test, xgb_pred, xgb_proba, "XGBoost")
    results.append(xgb_results)
    
    # 4. LightGBM
    print("Training LightGBM...")
    lgb_model = ModelFactory.create_lightgbm(config.get('models', {}).get('lightgbm', {}))
    lgb_model.fit(X_train_scaled, y_train)
    lgb_pred = lgb_model.predict(X_test_scaled)
    lgb_proba = lgb_model.predict_proba(X_test_scaled)
    
    lgb_results = evaluator.evaluate_model(y_test, lgb_pred, lgb_proba, "LightGBM")
    results.append(lgb_results)
    
    # 5. Neural Network
    print("Training Neural Network...")
    device = get_device()
    print(f"Using device: {device}")
    
    nn_model = ModelFactory.create_neural_network(
        config=config.get('models', {}).get('neural_network', {})
    )
    
    trainer = PersonalizedLearningTrainer(nn_model, device)
    
    # Train the model
    history = trainer.train(
        X_train_scaled, y_train,
        epochs=config.get('models', {}).get('neural_network', {}).get('epochs', 15),
        batch_size=config.get('models', {}).get('neural_network', {}).get('batch_size', 32),
        verbose=True
    )
    
    # Evaluate
    nn_acc, nn_pred = trainer.evaluate(X_test_scaled, y_test)
    
    # Get probabilities (approximate)
    nn_model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        nn_logits = nn_model(X_tensor)
        nn_proba = torch.softmax(nn_logits, dim=1).cpu().numpy()
    
    nn_results = evaluator.evaluate_model(y_test, nn_pred, nn_proba, "Neural Network")
    nn_results['training_history'] = history
    results.append(nn_results)
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train personalized learning models")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Data config file path")
    parser.add_argument("--output-dir", default="assets", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Load configurations
    config = load_config(args.config)
    
    # Load data
    print("Loading data...")
    data = load_data(args.data_config)
    
    print(f"Data shape: {data['X_train'].shape}")
    print(f"Resource distribution: {np.bincount(data['y_train'])}")
    
    # Train models
    print("Training models...")
    results = train_all_models(data, config)
    
    # Create evaluator
    evaluator = PersonalizedLearningEvaluator(data['resource_names'])
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(results)
    print("\nModel Leaderboard:")
    print(leaderboard[['rank', 'model_name', 'accuracy', 'f1_macro']].to_string(index=False))
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save leaderboard
    leaderboard.to_csv(f"{args.output_dir}/model_leaderboard.csv", index=False)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{args.output_dir}/detailed_results.csv", index=False)
    
    # Generate plots
    print("Generating visualizations...")
    
    # Model comparison plot
    evaluator.plot_model_comparison(results, f"{args.output_dir}/model_comparison.html")
    
    # Per-class metrics plot
    evaluator.plot_per_class_metrics(results, f"{args.output_dir}/per_class_metrics.html")
    
    # Confusion matrices for top models
    top_models = leaderboard.head(3)
    for _, row in top_models.iterrows():
        model_name = row['model_name']
        cm = row['confusion_matrix']
        evaluator.plot_confusion_matrix(
            cm, model_name, 
            f"{args.output_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        )
    
    # Error analysis for best model
    best_model_idx = leaderboard.iloc[0]['rank'] - 1
    best_model_results = results[best_model_idx]
    
    # Get predictions for error analysis (we need to retrain or save predictions)
    print(f"\nBest model: {leaderboard.iloc[0]['model_name']}")
    print(f"Accuracy: {leaderboard.iloc[0]['accuracy']:.4f}")
    print(f"F1 Score: {leaderboard.iloc[0]['f1_macro']:.4f}")
    
    print(f"\nResults saved to {args.output_dir}/")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
