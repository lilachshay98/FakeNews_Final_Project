import logging
import os
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

def extract_text_features(texts):
    """
    Extract additional features from text data:
    - Word count
    - Word entropy (information density)
    - Average word length
    - Punctuation ratio
    """
    import collections
    from scipy.stats import entropy
    import re
    import numpy as np

    features = []
    logging.info("Extracting additional text features...")

    for text in texts:
        if not isinstance(text, str):
            text = str(text)

        # Word count
        words = text.split()
        word_count = len(words)

        # Word entropy
        if word_count > 0:
            word_counts = collections.Counter(words)
            probs = [count / word_count for count in word_counts.values()]
            word_entropy = entropy(probs)
        else:
            word_entropy = 0

        # Average word length
        if word_count > 0:
            avg_word_length = sum(len(word) for word in words) / word_count
        else:
            avg_word_length = 0

        # Punctuation ratio
        total_chars = len(text)
        punct_count = sum(1 for char in text if char in '.,;:!?\'"-()[]{}/\\')
        punct_ratio = punct_count / total_chars if total_chars > 0 else 0

        features.append([word_count, word_entropy, avg_word_length, punct_ratio])

    return np.array(features)


def get_models_dir():
    """Get the path to the models directory"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')

    # Create the directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logging.info(f"Created models directory at {models_dir}")

    return models_dir


def load_processed_data():
    """
    Load the processed data from the cleaned_combined.csv file
    and extract the features (text) and labels.
    """
    # Get the path to the processed data file
    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_data_path = os.path.join(base_dir, 'data', 'stats', 'cleaned_combined.csv')

    # Log the data loading process
    logging.info(f"Loading data from {processed_data_path}...")

    # Load the data
    if not os.path.exists(processed_data_path):
        logging.error(f"Processed data file not found at {processed_data_path}")
        raise FileNotFoundError(f"Processed data file not found at {processed_data_path}")

    # Read the CSV file
    data = pd.read_csv(processed_data_path)

    # Extract text features and labels
    if 'cleaned_text' in data.columns:
        cleaned_text = data['cleaned_text']
    else:
        logging.error("No text column found in the data")
        raise ValueError("No text column found in the data")

    if 'label' in data.columns:
        y = data['label']
    else:
        logging.error("No label column found in the data")
        raise ValueError("No label column found in the data")

    logging.info(f"Loaded {len(data)} samples with {y.value_counts().to_dict()} class distribution")

    return cleaned_text, y


def get_split_data():
    """
    Load the processed data and split it into training and testing sets.
    Returns:
        X_train, X_test, y_train, y_test: Split data for training and testing.
    """
    # Load the cleaned text and labels
    cleaned_text, y = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_text, y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y  # This ensures class distribution is preserved
    )
    return X_train, X_test, y_train, y_test


def tokenize_text(X_train, X_test):
    """
    Tokenize the text data using TF-IDF vectorization and extract additional features.
    """
    logging.info("Vectorizing text data using TF-IDF...")

    # Check if vectorizer already exists
    vectorizer_path = os.path.join(get_models_dir(), 'tfidf_vectorizer.joblib')

    # Handle NaN values
    logging.info(f"Checking for NaN values: {X_train.isna().sum()} in training, {X_test.isna().sum()} in testing")
    X_train = X_train.fillna("")
    X_test = X_test.fillna("")

    if os.path.exists(vectorizer_path):
        logging.info("Loading existing TF-IDF vectorizer...")
        tfidf_vectorizer = joblib.load(vectorizer_path)
        X_train_tfidf = tfidf_vectorizer.transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
    else:
        logging.info("Creating and fitting new TF-IDF vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Save the vectorizer
        joblib.dump(tfidf_vectorizer, vectorizer_path)
        logging.info(f"Saved TF-IDF vectorizer to {vectorizer_path}")

    logging.info(f"Vectorized training data shape: {X_train_tfidf.shape}")

    # Extract additional text features
    logging.info("Extracting additional text features...")
    X_train_features = extract_text_features(X_train)
    X_test_features = extract_text_features(X_test)

    logging.info(f"Additional features shape: {X_train_features.shape}")

    # Scale the additional features

    scaler = StandardScaler()
    X_train_features_scaled = scaler.fit_transform(X_train_features)
    X_test_features_scaled = scaler.transform(X_test_features)

    # Save the scaler
    scaler_path = os.path.join(get_models_dir(), 'feature_scaler.joblib')
    joblib.dump(scaler, scaler_path)

    # Combine TF-IDF features with additional text features
    X_train_combined = hstack([X_train_tfidf, X_train_features_scaled])
    X_test_combined = hstack([X_test_tfidf, X_test_features_scaled])

    logging.info(f"Combined features shape: {X_train_combined.shape}")

    return X_train_combined, X_test_combined


def save_model(model, name):
    """
    Save a trained model to disk.

    Args:
        model: The trained model to save
        name: Name of the model (used for the filename)
    """
    model_path = os.path.join(get_models_dir(), f"{name.lower().replace(' ', '_')}_model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"Saved {name} model to {model_path}")


def load_model(name):
    """
    Load a saved model from disk.

    Args:
        name: Name of the model to load

    Returns:
        The loaded model or None if not found
    """
    model_path = os.path.join(get_models_dir(), f"{name.lower().replace(' ', '_')}_model.joblib")

    if os.path.exists(model_path):
        logging.info(f"Loading existing {name} model...")
        return joblib.load(model_path)
    else:
        logging.info(f"No saved {name} model found.")
        return None