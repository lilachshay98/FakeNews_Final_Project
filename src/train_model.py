from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import logging

from common_utils import load_model, save_model, get_models_dir, get_split_data, tokenize_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fit_models(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Train and evaluate multiple classification models for fake news detection.

    Implements an ensemble approach by training multiple diverse classification
    algorithms on the same dataset, handling different model requirements for
    non-negative features, and evaluating individual model performance.

    Parameters
    ----------
    X_train_tfidf : scipy.sparse matrix, shape (n_train_samples, n_features)
        Training feature matrix combining TF-IDF vectors and additional
        text features from tokenize_text().
    X_test_tfidf : scipy.sparse matrix, shape (n_test_samples, n_features)
        Testing feature matrix with same feature structure as training data.
    y_train : array-like, shape (n_train_samples,)
        Training labels (0=fake, 1=real).
    y_test : array-like, shape (n_test_samples,)
        Testing labels for model evaluation.

    Returns
    -------
    fitted_models : dict[str, sklearn.base.BaseEstimator]
        Dictionary mapping model names to fitted scikit-learn model instances:
        - 'Naive Bayes': MultinomialNB with Laplace smoothing
        - 'Logistic Regression': Balanced class weights, high iteration limit
        - 'Gradient Boosting': Ensemble with 200 trees and controlled learning rate
    """
    X_train_tfidf = X_train_tfidf.tocsr()
    X_test_tfidf = X_test_tfidf.tocsr()

    n_tfidf_features = 10000
    X_train_tfidf_only = X_train_tfidf[:, :n_tfidf_features]
    X_test_tfidf_only = X_test_tfidf[:, :n_tfidf_features]

    models_config = {
        'Naive Bayes': {
            'model': MultinomialNB(alpha=0.1),
            'requires_non_negative': True
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=10000, C=1.0, class_weight='balanced'),
            'requires_non_negative': False
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7),
            'requires_non_negative': False
        }
    }

    fitted_models = {}
    accuracy_rates = {}

    # Fit individual models
    for name, config in models_config.items():
        model_instance = config['model']
        requires_non_negative = config['requires_non_negative']

        logging.info(f"Training new {name} model...")

        if requires_non_negative:
            logging.info(f"{name} requires non-negative values, using TF-IDF features only")
            model_instance.fit(X_train_tfidf_only, y_train)
            y_pred = model_instance.predict(X_test_tfidf_only)
        else:
            model_instance.fit(X_train_tfidf, y_train)
            y_pred = model_instance.predict(X_test_tfidf)

        save_model(model_instance, name)
        accuracy = accuracy_score(y_test, y_pred)
        fitted_models[name] = {
            'model': model_instance,
            'requires_non_negative': requires_non_negative
        }
        accuracy_rates[name] = accuracy
        logging.info(f"{name} model accuracy: {accuracy * 100:.2f}%")

    # Skip ensemble creation and just print individual model performances
    print("-" * 30)
    for name, accuracy in accuracy_rates.items():
        print(f"{name}: {accuracy * 100:.2f}%")
    print("-" * 30)

    return {name: config['model'] for name, config in fitted_models.items()}


if __name__ == '__main__':
    # Check if models directory exists
    models_dir = get_models_dir()
    logging.info(f"Using models directory: {models_dir}")

    # Load and split the data
    X_train, X_test, y_train, y_test = get_split_data()

    # Tokenize the text data
    X_train_tfidf, X_test_tfidf = tokenize_text(X_train, X_test)

    # Fit multiple models and evaluate their performance
    fitted_models = fit_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

    logging.info("Model processing completed successfully.")
