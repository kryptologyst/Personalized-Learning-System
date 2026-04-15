# Personalized Learning System

A research-focused personalized learning system that adapts to individual student learning patterns and recommends optimal educational resources using machine learning.

## Overview

This system analyzes student profiles including study time, learning style preferences, previous performance, and engagement levels to recommend the most suitable learning resources (videos, quizzes, or reading materials). The project demonstrates advanced ML techniques for educational technology applications.

## Features

- **Multi-Model Approach**: Implements logistic regression, random forest, XGBoost, LightGBM, and neural networks
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, and per-class analysis
- **Interactive Demo**: Streamlit-based web application for real-time recommendations
- **Modern Tech Stack**: PyTorch, scikit-learn, XGBoost, LightGBM with proper device fallback
- **Reproducible Research**: Deterministic seeding, configuration management, and structured logging
- **Production Ready**: Clean code structure, type hints, comprehensive testing

## Project Structure

```
personalized-learning-system/
├── src/                    # Source code modules
│   ├── data/              # Data generation and processing
│   ├── models/            # Model implementations
│   ├── eval/              # Evaluation metrics and analysis
│   └── viz/               # Visualization utilities
├── configs/               # Configuration files
├── data/                  # Data storage (raw/processed/external)
├── scripts/               # Training and utility scripts
├── tests/                 # Test suite
├── demo/                  # Streamlit demo application
├── assets/                # Generated outputs and visualizations
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Personalized-Learning-System.git
   cd Personalized-Learning-System
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Training Models

Train all models and generate evaluation results:

```bash
python scripts/train.py --config configs/config.yaml --data-config configs/data.yaml
```

This will:
- Generate synthetic student data
- Train 5 different models (Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network)
- Evaluate performance with comprehensive metrics
- Generate visualizations and save results to `assets/`

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo provides:
- Real-time resource recommendations based on student profiles
- Interactive parameter adjustment
- Dataset visualization and analysis
- Model performance metrics

## Data Schema

### Student Profile Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `study_time` | float | 1-15 hours/week | Weekly study time |
| `learning_style_score` | float | 0-1 | Learning preference (0=visual, 1=auditory/kinesthetic) |
| `previous_performance` | float | 0-100 | Previous academic performance score |
| `engagement_score` | float | 0-1 | Student engagement level |

### Resource Types

| ID | Resource Type | Description |
|----|---------------|-------------|
| 0 | Video | Visual learning content |
| 1 | Quiz | Interactive assessment |
| 2 | Reading Article | Text-based learning material |

## Model Performance

The system implements multiple models with comprehensive evaluation:

### Baseline Models
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble method with feature importance

### Advanced Models
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Efficient gradient boosting
- **Neural Network**: Deep learning with PyTorch

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (macro and weighted)
- Per-class metrics for each resource type
- Cross-validation with 5-fold CV
- Confusion matrices and classification reports

## Configuration

### Data Configuration (`configs/data.yaml`)
- Data generation parameters
- Feature distributions
- Train/test split ratios
- Feature scaling options

### Model Configuration (`configs/config.yaml`)
- Model hyperparameters
- Training parameters
- Evaluation settings
- Resource type mappings

## API Usage

### Basic Usage

```python
from src.data.data_generator import load_data
from src.models.models import ModelFactory
from src.eval.evaluator import PersonalizedLearningEvaluator

# Load data
data = load_data()

# Train model
model = ModelFactory.create_xgboost()
model.fit(data['X_train_scaled'], data['y_train'])

# Make predictions
predictions = model.predict(data['X_test_scaled'])

# Evaluate
evaluator = PersonalizedLearningEvaluator(data['resource_names'])
results = evaluator.evaluate_model(data['y_test'], predictions)
```

### Custom Student Profile

```python
import numpy as np

# Create custom student profile
student_profile = np.array([
    6.5,    # study_time (hours/week)
    0.3,    # learning_style_score (visual learner)
    85.0,   # previous_performance
    0.7     # engagement_score
]).reshape(1, -1)

# Scale features
student_scaled = data['scaler'].transform(student_profile)

# Get recommendation
prediction = model.predict(student_scaled)[0]
resource_name = data['resource_names'][prediction]
print(f"Recommended resource: {resource_name}")
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/ scripts/ demo/
ruff check src/ tests/ scripts/ demo/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Research Applications

This system demonstrates several important concepts in educational technology:

1. **Personalized Learning**: Adaptive content recommendation based on individual profiles
2. **Multi-Modal Learning**: Support for different learning resource types
3. **Educational Data Mining**: Analysis of student behavior patterns
4. **Learning Analytics**: Performance prediction and intervention strategies

### Potential Extensions

- **Real-time Adaptation**: Dynamic model updates based on student feedback
- **Multi-Objective Optimization**: Balance learning effectiveness with engagement
- **Causal Inference**: Analyze intervention effects on learning outcomes
- **Federated Learning**: Privacy-preserving model training across institutions

## Limitations and Disclaimers

### Research Demo Only
This system is designed for research and educational purposes. It is not intended for operational use in production educational environments without proper validation and testing.

### Data Privacy
- All data used in this demo is synthetic
- No real student data is collected or stored
- Privacy considerations should be addressed before deployment

### Model Limitations
- Models are trained on synthetic data and may not generalize to real-world scenarios
- Performance metrics are based on simulated student behavior
- Real-world validation is required for production deployment

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**kryptologyst**  
GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)

## Citation

If you use this system in your research, please cite:

```bibtex
@software{personalized_learning_system,
  title={Personalized Learning System: ML-based Educational Resource Recommendation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Personalized-Learning-System}
}
```

## Acknowledgments

- Educational technology research community
- Open source ML libraries (PyTorch, scikit-learn, XGBoost, LightGBM)
- Streamlit for interactive demos
- Contributors and testers

---

*This project is part of the Environmental & Social Applications research initiative, focusing on technology solutions for educational equity and personalized learning.*
# Personalized-Learning-System
