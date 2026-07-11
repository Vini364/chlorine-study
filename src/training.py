import json
import numpy as np
import pandas as pd
import csv
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def load_features(features_json):
    """Load features from JSON file"""
    print(f"Loading features from {features_json}...")
    with open(features_json, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns.")
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for training.

    Args:
        df: DataFrame with features and concentration
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Separate features and target
    if 'concentration' not in df.columns:
        raise ValueError("DataFrame must contain 'concentration' column")

    # Remove non-feature columns
    feature_columns = [col for col in df.columns
                      if col not in ['concentration', 'is_augmented']]

    X = df[feature_columns].values
    y = df['concentration'].values

    print(f"\nFeatures used for training: {len(feature_columns)}")
    print(f"Feature names: {feature_columns}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Scale features (important for regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

def log_results(results):
    """Save experiment results to CSV"""
    file_exists = os.path.isfile('experiment_log.csv')

    with open('experiment_log.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow([
                'model',
                'train_r2',
                'test_r2',
                'train_rmse',
                'test_rmse',
                'cv_r2_mean',
                'cv_r2_std'
            ])

        writer.writerow([
            results['model_name'],
            results['train_r2'],
            results['test_r2'],
            results['train_rmse'],
            results['test_rmse'],
            results['cv_r2_mean'],
            results['cv_r2_std']
        ])


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, cv_folds=5):
    """
    Train a model and evaluate its performance.

    Args:
        model: Scikit-learn model instance
        model_name: Name of the model for display
        X_train, X_test, y_train, y_test: Train/test splits
        cv_folds: Number of cross-validation folds

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    results = {
        'model_name': model_name,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
    }

    # Cross-validation
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv_folds, scoring='r2')
    results['cv_r2_mean'] = cv_scores.mean()
    results['cv_r2_std'] = cv_scores.std()

    # Print results
    print(f"\nTraining Results:")
    print(f"  R² Score: {results['train_r2']:.4f}")
    print(f"  RMSE: {results['train_rmse']:.4f}")
    print(f"  MAE: {results['train_mae']:.4f}")

    print(f"\nTesting Results:")
    print(f"  R² Score: {results['test_r2']:.4f}")
    print(f"  RMSE: {results['test_rmse']:.4f}")
    print(f"  MAE: {results['test_mae']:.4f}")

    print(f"\nCross-Validation Results:")
    print(f"  Mean R² Score: {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")

    # Store predictions for visualization
    results['y_train'] = y_train
    results['y_train_pred'] = y_train_pred
    results['y_test'] = y_test
    results['y_test_pred'] = y_test_pred
    log_results(results)

    return results

def compare_models(lasso_results, ridge_results):
    """
    Compare Lasso and Ridge regression results.

    Args:
        lasso_results: Results dictionary from Lasso
        ridge_results: Results dictionary from Ridge
    """
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")

    comparison_df = pd.DataFrame([
        {
            'Model': 'Lasso',
            'Train R²': lasso_results['train_r2'],
            'Test R²': lasso_results['test_r2'],
            'Train RMSE': lasso_results['train_rmse'],
            'Test RMSE': lasso_results['test_rmse'],
            'Train MAE': lasso_results['train_mae'],
            'Test MAE': lasso_results['test_mae'],
            'CV R² (mean)': lasso_results['cv_r2_mean'],
            'CV R² (std)': lasso_results['cv_r2_std'],
        },
        {
            'Model': 'Ridge',
            'Train R²': ridge_results['train_r2'],
            'Test R²': ridge_results['test_r2'],
            'Train RMSE': ridge_results['train_rmse'],
            'Test RMSE': ridge_results['test_rmse'],
            'Train MAE': ridge_results['train_mae'],
            'Test MAE': ridge_results['test_mae'],
            'CV R² (mean)': ridge_results['cv_r2_mean'],
            'CV R² (std)': ridge_results['cv_r2_std'],
        }
    ])

    print("\n" + comparison_df.to_string(index=False))

    # Determine which model performs better
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    if lasso_results['test_r2'] > ridge_results['test_r2']:
        better_model = "Lasso"
        improvement = (lasso_results['test_r2'] - ridge_results['test_r2']) * 100
    else:
        better_model = "Ridge"
        improvement = (ridge_results['test_r2'] - lasso_results['test_r2']) * 100

    print(f"\nBetter model (Test R²): {better_model}")
    print(f"Improvement: {improvement:.2f}%")

    # Check for overfitting
    print(f"\nOverfitting Analysis:")
    lasso_overfit = lasso_results['train_r2'] - lasso_results['test_r2']
    ridge_overfit = ridge_results['train_r2'] - ridge_results['test_r2']

    print(f"  Lasso (Train R² - Test R²): {lasso_overfit:.4f}")
    print(f"  Ridge (Train R² - Test R²): {ridge_overfit:.4f}")

    if lasso_overfit < ridge_overfit:
        print(f"  → Lasso shows less overfitting")
    else:
        print(f"  → Ridge shows less overfitting")

    return comparison_df

def plot_predictions(lasso_results, ridge_results, save_path='model_comparison.png'):
    """
    Create visualization comparing actual vs predicted values for both models.

    Args:
        lasso_results: Results dictionary from Lasso
        ridge_results: Results dictionary from Ridge
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lasso plot
    axes[0].scatter(lasso_results['y_test'], lasso_results['y_test_pred'],
                   alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0].plot([lasso_results['y_test'].min(), lasso_results['y_test'].max()],
                [lasso_results['y_test'].min(), lasso_results['y_test'].max()],
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Concentration', fontsize=12)
    axes[0].set_ylabel('Predicted Concentration', fontsize=12)
    axes[0].set_title(f"Lasso Regression (R² = {lasso_results['test_r2']:.4f})", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ridge plot
    axes[1].scatter(ridge_results['y_test'], ridge_results['y_test_pred'],
                   alpha=0.6, edgecolors='k', linewidth=0.5, color='green')
    axes[1].plot([ridge_results['y_test'].min(), ridge_results['y_test'].max()],
                [ridge_results['y_test'].min(), ridge_results['y_test'].max()],
                'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Concentration', fontsize=12)
    axes[1].set_ylabel('Predicted Concentration', fontsize=12)
    axes[1].set_title(f"Ridge Regression (R² = {ridge_results['test_r2']:.4f})", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPrediction plot saved to {save_path}")

    return fig

if __name__ == "__main__":
    # Configuration
    features_file = 'features_data.json'
    alpha = 0.1  # Regularization parameter

    # Load and prepare data
    df = load_features(features_file)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)

    # Initialize models
    lasso_model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    ridge_model = Ridge(alpha=alpha, random_state=42)

    # Train and evaluate Lasso
    lasso_results = train_and_evaluate_model(
        lasso_model, "Lasso Regression", X_train, X_test, y_train, y_test
    )

    # Train and evaluate Ridge
    ridge_results = train_and_evaluate_model(
        ridge_model, "Ridge Regression", X_train, X_test, y_train, y_test
    )

    # Compare models
    comparison_df = compare_models(lasso_results, ridge_results)

    # Save comparison to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    print(f"\nComparison saved to model_comparison.csv")

    # Create visualization
    try:
        plot_predictions(lasso_results, ridge_results)
    except Exception as e:
        print(f"\nNote: Could not create plot visualization: {e}")
        print("This is not critical - results are still valid.")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
