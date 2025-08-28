#!/usr/bin/env python3
# Bot Detection Model Training Script

import os
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, RFECV
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Path definitions
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FEATURES_PATH = BASE_DIR / "data" / "processed" / "bots" / "cresci2017_features.csv"
MODEL_DIR = BASE_DIR / "models" / "bots"

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def load_features(file_path=None):
    """
    Load the pre-processed feature dataset

    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the features CSV file

    Returns:
    --------
    pd.DataFrame
        The loaded features dataset
    """
    if file_path is None:
        file_path = FEATURES_PATH

    logging.info(f"Loading features from {file_path}...")

    try:
        features_df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded features with shape: {features_df.shape}")
        logging.info(f"Feature columns: {features_df.columns.tolist()}")

        # Check for missing values
        missing_values = features_df.isna().sum().sum()
        if missing_values > 0:
            logging.warning(f"Dataset contains {missing_values} missing values that will need to be handled")

        return features_df
    except Exception as e:
        logging.error(f"Error loading features: {str(e)}")
        return None

def prepare_data_for_training(features_df, test_size=0.2, random_state=42):
    """
    Prepare data for model training - including handling missing values

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame containing the extracted features
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, feature_names)
    """
    logging.info("Preparing data for model training...")

    # Check if 'bot' or 'is_bot' column exists
    target_col = 'is_bot'

    # Get available feature columns - we'll use all except the ID and target columns
    exclude_columns = ['user_id', target_col, 'dataset', 'screen_name']
    # Add any other non-numeric columns or columns with too many missing values
    for col in features_df.columns:
        if col not in exclude_columns:
            if features_df[col].dtype == 'object':
                exclude_columns.append(col)
            elif features_df[col].isna().sum() / len(features_df) > 0.5:  # Exclude columns with >50% missing
                logging.warning(f"Excluding column '{col}' due to >50% missing values")
                exclude_columns.append(col)

    feature_columns = [col for col in features_df.columns if col not in exclude_columns]
    logging.info(f"Using features: {feature_columns}")

    if not feature_columns:
        logging.error("No valid features found!")
        return None

    # Check remaining missing values
    missing_counts = features_df[feature_columns].isna().sum()
    missing_features = missing_counts[missing_counts > 0]
    if len(missing_features) > 0:
        logging.info(f"Features with missing values that will be imputed: {missing_features.to_dict()}")

    # Shuffle the dataset
    features_df = features_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    logging.info(f"Dataset shuffled with random state {random_state}")

    # Split features and target
    X = features_df[feature_columns]
    y = features_df[target_col]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logging.info(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    logging.info(f"Class distribution in training: {np.bincount(y_train.astype(int))}")
    logging.info(f"Class distribution in test: {np.bincount(y_test.astype(int))}")

    return X_train, X_test, y_train, y_test, feature_columns

def create_feature_interactions(X_train, X_test, top_features=None, degree=2):
    """
    Create polynomial feature interactions to enhance feature importance

    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    top_features : list, optional
        List of column names to use for interactions (if None, use all)
    degree : int
        Degree of polynomial features

    Returns:
    --------
    tuple
        (X_train_poly, X_test_poly, poly_feature_names)
    """
    logging.info(f"Creating polynomial feature interactions of degree {degree}...")

    if top_features is not None:
        # Only create interactions for selected features
        X_train_select = X_train[top_features]
        X_test_select = X_test[top_features]
    else:
        X_train_select = X_train
        X_test_select = X_test

    # Check for NaN values in the input data
    if isinstance(X_train_select, pd.DataFrame):
        nan_count = X_train_select.isna().sum().sum()
    else:
        nan_count = np.isnan(X_train_select).sum()

    logging.info(f"Input data for feature interactions has {nan_count} missing values that will be imputed")

    # Create and fit a simple imputer
    imputer = SimpleImputer(strategy='median')
    X_train_select_imputed = imputer.fit_transform(X_train_select)
    X_test_select_imputed = imputer.transform(X_test_select)

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_select_imputed)
    X_test_poly = poly.transform(X_test_select_imputed)

    # Generate feature names for the polynomial features
    if hasattr(poly, 'get_feature_names_out'):
        poly_feature_names = poly.get_feature_names_out(X_train_select.columns)
    else:
        # Fallback for older scikit-learn versions
        poly_feature_names = poly.get_feature_names(X_train_select.columns)

    # Combine with original features that weren't included in the interactions
    remaining_features = [col for col in X_train.columns if col not in top_features] if top_features else []
    if remaining_features:
        # We need to handle missing values in the remaining features as well
        remaining_train = X_train[remaining_features]
        remaining_test = X_test[remaining_features]

        # Impute missing values in remaining features
        remaining_imputer = SimpleImputer(strategy='median')
        remaining_train_imputed = remaining_imputer.fit_transform(remaining_train)
        remaining_test_imputed = remaining_imputer.transform(remaining_test)

        # Combine with polynomial features
        X_train_poly = np.hstack((X_train_poly, remaining_train_imputed))
        X_test_poly = np.hstack((X_test_poly, remaining_test_imputed))
        poly_feature_names = np.concatenate((poly_feature_names, np.array(remaining_features)))

    logging.info(f"Created {X_train_poly.shape[1]} features with polynomial interactions")
    return X_train_poly, X_test_poly, poly_feature_names

def select_features_with_rf(X_train, y_train, X_test, feature_names, threshold='mean'):
    """
    Use Random Forest to select the most important features

    Parameters:
    -----------
    X_train : array
        Training features
    y_train : array
        Training target values
    X_test : array
        Test features
    feature_names : list
        Names of features
    threshold : str or float
        Threshold for feature importance ('mean', 'median', or float value)

    Returns:
    --------
    tuple
        (X_train_selected, X_test_selected, selected_feature_names, importance_values)
    """
    logging.info(f"Selecting features using Random Forest with threshold: {threshold}")

    # Check for NaN values in the input data
    if isinstance(X_train, pd.DataFrame):
        nan_count = X_train.isna().sum().sum()
    else:
        nan_count = np.isnan(X_train).sum()

    logging.info(f"Input data for feature selection has {nan_count} missing values that will be imputed")

    # Create and fit a simple imputer
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Initial RF for feature selection
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    # Fit model for feature selection
    rf.fit(X_train_imputed, y_train)

    # Get feature importances
    importances = rf.feature_importances_

    # Create selector with threshold
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)

    # Transform data
    X_train_selected = selector.transform(X_train_imputed)
    X_test_selected = selector.transform(X_test_imputed)

    # Get the selected feature indices
    selected_indices = selector.get_support()

    # Get the selected feature names
    selected_feature_names = np.array(feature_names)[selected_indices]

    # Get importance values for selected features
    selected_importances = importances[selected_indices]

    logging.info(f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]} original features")
    logging.info(f"Top selected features: {', '.join(selected_feature_names[:5])}")

    # Create a mapping of feature names to importance values
    importance_dict = {name: imp for name, imp in zip(selected_feature_names, selected_importances)}

    return X_train_selected, X_test_selected, selected_feature_names, importance_dict

def train_random_forest(X_train, y_train, feature_names=None, enhance_importance=True):
    """
    Train a Random Forest model with parameters optimized for better feature importance

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    feature_names : list, optional
        Names of features for importance analysis
    enhance_importance : bool
        Whether to enhance feature importance through model tuning

    Returns:
    --------
    dict
        Dictionary containing model, selected features, and importance values
    """
    logging.info("Training Random Forest model optimized for feature importance...")

    result = {}

    # Check for missing values if X_train is a DataFrame
    if hasattr(X_train, 'isna'):
        missing_values = X_train.isna().sum().sum()
        logging.info(f"Training data contains {missing_values} missing values that will be imputed")

    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # If X_train is already a numpy array from feature selection
    if isinstance(X_train, np.ndarray):
        # Create simple pipeline
        pipeline = Pipeline(steps=[
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,  # Increased from 2
                min_samples_leaf=2,   # Increased from 1
                max_features='sqrt',  # 'sqrt' usually gives better feature importance distribution
                bootstrap=True,
                oob_score=True,       # Out-of-bag scoring provides better estimates
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            ))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)
        rf_model = pipeline.named_steps['classifier']

    else:
        # Create the preprocessing pipeline for DataFrame input
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, X_train.columns)
            ]
        )

        # Create the full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,  # Increased from 2
                min_samples_leaf=2,   # Increased from 1
                max_features='sqrt',  # 'sqrt' usually gives better feature importance distribution
                bootstrap=True,
                oob_score=True,       # Out-of-bag scoring provides better estimates
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            ))
        ])

        # Train the pipeline
        pipeline.fit(X_train, y_train)
        rf_model = pipeline.named_steps['classifier']

    # Store the model in the result dictionary
    result['model'] = pipeline
    result['rf_model'] = rf_model

    # Feature importance analysis
    if feature_names is not None:
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        logging.info("Feature ranking:")
        for f in range(min(10, len(feature_names))):
            logging.info(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

        result['feature_names'] = feature_names
        result['feature_importances'] = importances
        result['feature_indices'] = indices

    return result

def evaluate_model(model_result, X_test, y_test):
    """
    Evaluate the trained model and generate performance metrics

    Parameters:
    -----------
    model_result : dict
        Dictionary containing model and feature information
    X_test : array-like
        Test features
    y_test : array-like
        Test target values

    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    logging.info("Evaluating model performance...")

    model = model_result['model']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # Print classification report
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))

    # Combine metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    # If the model is a Random Forest, get OOB score
    if 'rf_model' in model_result and hasattr(model_result['rf_model'], 'oob_score_'):
        logging.info(f"Out-of-Bag Score: {model_result['rf_model'].oob_score_:.4f}")
        metrics['oob_score'] = model_result['rf_model'].oob_score_

    return metrics

def visualize_feature_importance(model_result, vis_dir):
    """
    Create visualizations of feature importance

    Parameters:
    -----------
    model_result : dict
        Dictionary containing model and feature information
    vis_dir : str or Path
        Directory to save visualizations

    Returns:
    --------
    list
        Paths to created visualization files
    """
    vis_files = []

    # Create visualizations directory if needed
    os.makedirs(vis_dir, exist_ok=True)

    # Check if we have feature information
    if ('feature_names' not in model_result or
            'feature_importances' not in model_result or
            'feature_indices' not in model_result):
        logging.warning("Missing feature information for visualization")
        return vis_files

    feature_names = model_result['feature_names']
    importances = model_result['feature_importances']
    indices = model_result['feature_indices']

    # Plot top 20 features or all if less than 20
    num_features = min(20, len(indices))

    # 1. Standard horizontal bar plot with larger figure
    plt.figure(figsize=(14, 10))
    plt.title('Top 20 Feature Importances for Bot Detection', fontsize=18)
    plt.barh(range(num_features), importances[indices[:num_features]], align='center', color='#1f77b4')
    plt.yticks(range(num_features), [feature_names[i] for i in indices[:num_features]], fontsize=14)
    plt.xlabel('Relative Importance', fontsize=16)
    plt.tight_layout()

    # Add values to the bars
    for i, v in enumerate(importances[indices[:num_features]]):
        plt.text(v, i, f' {v:.4f}', fontsize=12, va='center')

    std_plot_path = os.path.join(vis_dir, 'feature_importance.png')
    plt.savefig(std_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    vis_files.append(std_plot_path)
    logging.info(f"Standard feature importance plot saved to {std_plot_path}")

    # 2. Horizontal bar plot with logarithmic scale for better visibility of small values
    plt.figure(figsize=(14, 10))
    plt.title('Feature Importances (Log Scale)', fontsize=18)
    plt.barh(range(num_features), importances[indices[:num_features]], align='center', color='#2ca02c')
    plt.yticks(range(num_features), [feature_names[i] for i in indices[:num_features]], fontsize=14)
    plt.xlabel('Log Relative Importance', fontsize=16)
    plt.xscale('log')  # Use logarithmic scale for importance values
    plt.tight_layout()

    log_plot_path = os.path.join(vis_dir, 'feature_importance_log.png')
    plt.savefig(log_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    vis_files.append(log_plot_path)
    logging.info(f"Logarithmic feature importance plot saved to {log_plot_path}")

    # 3. Create a detailed heatmap for top 30 features with their relative importance values
    if len(indices) >= 30:
        plt.figure(figsize=(15, 12))
        top_30_indices = indices[:30]
        top_30_features = [feature_names[i] for i in top_30_indices]
        top_30_importances = importances[top_30_indices]

        # Create a DataFrame for the heatmap
        importance_df = pd.DataFrame({
            'Feature': top_30_features,
            'Importance': top_30_importances
        })
        importance_df = importance_df.set_index('Feature')

        # Create heatmap
        sns.heatmap(importance_df.T, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5, cbar_kws={'label': 'Importance Score'})
        plt.title('Top 30 Feature Importance Heatmap', fontsize=18)
        plt.tight_layout()

        heatmap_path = os.path.join(vis_dir, 'feature_importance_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        vis_files.append(heatmap_path)
        logging.info(f"Feature importance heatmap saved to {heatmap_path}")

    return vis_files

def save_model(model_result, metrics=None):
    """
    Save the trained model and related artifacts

    Parameters:
    -----------
    model_result : dict
        Dictionary containing the model and feature information
    metrics : dict, optional
        Performance metrics

    Returns:
    --------
    str
        Path to the saved model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"random_forest_bot_detector_{timestamp}.joblib"

    # Save the model
    joblib.dump(model_result['model'], model_path)
    logging.info(f"Model saved to {model_path}")

    # Save feature importances if available
    if 'feature_names' in model_result and 'feature_importances' in model_result:
        importances_df = pd.DataFrame({
            'feature': model_result['feature_names'],
            'importance': model_result['feature_importances']
        })
        importances_df = importances_df.sort_values('importance', ascending=False)

        importance_path = MODEL_DIR / f"feature_importances_{timestamp}.csv"
        importances_df.to_csv(importance_path, index=False)
        logging.info(f"Feature importances saved to {importance_path}")

    # Save metrics if provided
    if metrics is not None:
        metrics_path = MODEL_DIR / f"metrics_{timestamp}.txt"
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        logging.info(f"Metrics saved to {metrics_path}")

    return str(model_path)

def enhance_feature_importance(X_train, X_test, y_train, y_test, feature_names):
    """
    Apply techniques to enhance feature importance

    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target values
    y_test : Series
        Test target values
    feature_names : list
        Names of features

    Returns:
    --------
    tuple
        (model_result, metrics)
    """
    logging.info("Enhancing feature importance...")

    # First select the most important features using an initial Random Forest
    X_train_selected, X_test_selected, selected_features, importance_dict = select_features_with_rf(
        X_train, y_train, X_test, feature_names, threshold='mean'
    )

    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in sorted_features[:15]]  # Top 15 features

    logging.info(f"Top 15 features for interaction creation: {top_features}")

    # Create interaction features using the top features
    X_train_poly, X_test_poly, poly_feature_names = create_feature_interactions(
        X_train, X_test, top_features=top_features, degree=2
    )

    # Train the final model with the enhanced feature set
    model_result = train_random_forest(X_train_poly, y_train, poly_feature_names)

    # Evaluate the model
    metrics = evaluate_model(model_result, X_test_poly, y_test)

    # Create visualizations
    vis_dir = MODEL_DIR / "visualizations"
    visualize_feature_importance(model_result, vis_dir)

    return model_result, metrics

def main():
    """Main function to run the model training pipeline"""
    logging.info("Starting bot detection model training with enhanced feature importance")

    # Load pre-processed feature data
    features_df = load_features()

    if features_df is None:
        logging.error("Could not load features data. Exiting.")
        return

    # Prepare data for training
    data = prepare_data_for_training(features_df)

    if data is None:
        logging.error("Data preparation failed. Exiting.")
        return

    X_train, X_test, y_train, y_test, feature_names = data

    # Apply feature importance enhancement techniques
    model_result, metrics = enhance_feature_importance(X_train, X_test, y_train, y_test, feature_names)

    # Save model and artifacts
    save_model(model_result, metrics)

    logging.info("Bot detection model training with enhanced feature importance completed")

if __name__ == "__main__":
    main()
