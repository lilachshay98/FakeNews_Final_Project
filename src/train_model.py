from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
import logging
from scipy.sparse import hstack, csr_matrix
import numpy as np

from ClassificationModel.src.common_utils import load_model, save_model, get_models_dir, get_split_data, tokenize_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fit_models(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Fit multiple classification models and return them.
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
        # SVM removed as requested
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
