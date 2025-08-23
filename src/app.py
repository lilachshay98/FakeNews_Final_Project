#!/usr/bin/env python3
# News Classifier App
# This app uses multiple trained models to classify news as real or fake

import os
import sys
import logging
from joblib import load
import string
import colorama
from colorama import Fore, Style

# Setup logging - redirect to file to keep console clean
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'classifier_app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE)  # Only log to file, not console
    ]
)
colorama.init()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DOMAIN_STATS_DIR = os.path.join(BASE_DIR, 'data/raw')

class NewsClassifier:
    """Class to classify news articles as real or fake using multiple models"""

    def __init__(self):
        """Initialize the classifier by loading all models and vectorizer"""
        print(f"{Fore.CYAN}Starting news classifier application...{Style.RESET_ALL}")
        logging.info("Starting news classifier application...")

        try:
            # Load vectorizer
            self.vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
            print(f"{Fore.CYAN}Loading vectorizer...{Style.RESET_ALL}")
            logging.info(f"Loading vectorizer from {self.vectorizer_path}")
            self.vectorizer = load(self.vectorizer_path)

            # Load all models
            self.models = {}
            model_files = {
                'naive_bayes': 'naive_bayes_model.joblib',
                'logistic_regression': 'logistic_regression_model.joblib',
                'decision_tree': 'decision_tree_model.joblib',
                'random_forest': 'random_forest_model.joblib'
            }

            print(f"{Fore.CYAN}Loading classification models...{Style.RESET_ALL}")
            for name, filename in model_files.items():
                model_path = os.path.join(MODELS_DIR, filename)
                logging.info(f"Loading {name} model from {model_path}")
                self.models[name] = load(model_path)

            print(f"{Fore.GREEN}All models loaded successfully!{Style.RESET_ALL}")
            logging.info("All models loaded successfully")

        except Exception as e:
            print(f"{Fore.RED}Error loading models: {str(e)}{Style.RESET_ALL}")
            logging.error(f"Error loading models: {str(e)}")
            sys.exit(1)

    @staticmethod
    def get_domain_stats():
        """Load domain statistics from domains.txt"""
        domain_stats = {}
        domains_path = os.path.join(DOMAIN_STATS_DIR, 'domains_summary.csv')
        if os.path.exists(domains_path):
            with open(domains_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    domain = parts[0].strip().lower()
                    if domain == '﻿domain':
                        continue  # Skip header
                    fake_ratio = float(parts[4].strip())
                    domain_stats[domain] = fake_ratio
        else:
            logging.warning(f"domains.txt file not found at {domains_path}")
        return domain_stats

    def clean_text(self, text):
        """Clean input text with the same preprocessing as training data"""
        logging.info("Cleaning input text...")

        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def predict(self, text, domain):
        """Make predictions using all models"""
        domain_stats = self.get_domain_stats()
        try:
            print(f"\n{Fore.CYAN}Analyzing text...{Style.RESET_ALL}")

            # Clean the text
            cleaned_text = self.clean_text(text)
            logging.info("Cleaning input text...")

            # Vectorize
            logging.info("Vectorizing text...")
            X = self.vectorizer.transform([cleaned_text])

            # Make predictions with each model
            results = {}
            probabilities = {}

            for name, model in self.models.items():
                logging.info(f"Getting prediction from {name}...")

                # Get prediction
                prediction = model.predict(X)[0]
                results[name] = prediction

                # Get probability if the model supports it
                try:
                    proba = model.predict_proba(X)[0]
                    probabilities[name] = proba
                except:
                    # Some models might not have predict_proba
                    probabilities[name] = [0.5, 0.5] if prediction == 1 else [0.5, 0.5]

            # Add score according to domains list:
            domain_proba = [0.5, 0.5]
            if domain and domain in domain_stats:
                logging.info(f"Adjusting probabilities based on domain: {domain}")
                domain_proba = [domain_stats[domain], 1 - domain_stats[domain]]
                probabilities['domain'] = domain_proba

            # Calculate voting result
            votes = list(results.values())
            final_prediction = 1 if sum(votes) > len(votes)/2 else 0

            # Calculate average probabilities
            avg_proba = [0, 0]
            for name in probabilities:
                avg_proba[0] += probabilities[name][0]
                avg_proba[1] += probabilities[name][1]

            avg_proba[0] /= len(probabilities)
            avg_proba[1] /= len(probabilities)

            # Calculate entropy score (measures uncertainty of prediction)
            import math
            entropy = 0
            for p in avg_proba:
                if p > 0:  # Avoid log(0) which is undefined
                    entropy -= p * math.log2(p)
            # Normalize entropy to percentage (0-100), where 0% means complete certainty
            # and 100% means maximum uncertainty (50-50 probability)
            max_entropy = 1.0  # Binary entropy is max 1.0 when p=[0.5, 0.5]
            entropy_score = (entropy / max_entropy) * 100

            return {
                'prediction': final_prediction,
                'label': 'REAL' if final_prediction == 1 else 'FAKE',
                'confidence': avg_proba[final_prediction] * 100,
                'real_probability': avg_proba[1] * 100,
                'fake_probability': avg_proba[0] * 100,
                'entropy_score': entropy_score,
                'model_votes': results
            }

        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return None

def print_header():
    """Print the application header"""
    print("\n" + "="*50)
    print(f"{Fore.CYAN}=== Fake News Classification System ==={Style.RESET_ALL}")
    print("="*50)
    print("Enter news text to classify (type 'exit' to quit):\n")

def print_result(result):
    """Print the classification result with nice formatting"""
    print("\n" + "="*50)
    print(f"{Fore.YELLOW}=== Classification Result ==={Style.RESET_ALL}")
    print("="*50)

    # Print prediction with color
    if result['label'] == 'REAL':
        print(f"Prediction: {Fore.GREEN}{result['label']}{Style.RESET_ALL}")
    else:
        print(f"Prediction: {Fore.RED}{result['label']}{Style.RESET_ALL}")

    # Print confidence and probabilities
    print(f"Confidence: {Fore.YELLOW}{result['confidence']:.2f}%{Style.RESET_ALL}")
    print(f"Probability of REAL: {Fore.CYAN}{result['real_probability']:.2f}%{Style.RESET_ALL}")
    print(f"Probability of FAKE: {Fore.CYAN}{result['fake_probability']:.2f}%{Style.RESET_ALL}")

    # Print entropy score (lower is better - indicates more certainty)
    entropy_color = Fore.GREEN if result['entropy_score'] < 30 else Fore.YELLOW if result['entropy_score'] < 70 else Fore.RED
    print(f"Entropy Score: {entropy_color}{result['entropy_score']:.2f}%{Style.RESET_ALL} (lower means more certainty)")

    print("\n" + "="*50)

def main():
    """Main function to run the classifier application"""
    classifier = NewsClassifier()

    while True:
        print_header()

        # Get user text input
        news_text = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

        # Strip whitespace to properly check for empty input
        news_text = news_text.strip()

        # Check if user wants to exit
        if news_text.lower() in ['exit', 'quit', 'q']:
            print(f"\n{Fore.GREEN}Thank you for using the Fake News Classifier!{Style.RESET_ALL}")
            break

        # Validate input - skip if empty
        if not news_text:
            print(f"{Fore.RED}Please enter some text to classify. Empty input cannot be processed.{Style.RESET_ALL}")
            continue

        print("Enter the domain hte news came from if you have it (just press enter if you don't have it):\n")
        # Get user domain input
        domain_text = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

        # Make prediction
        result = classifier.predict(news_text, domain_text)

        # Display result if valid
        if result:
            print_result(result)
        else:
            print(f"{Fore.RED}Error: Could not classify the text. Please try again.{Style.RESET_ALL}")

        # Instead of clearing the console which may cause "TERM environment variable not set" error,
        # just print some blank lines to create visual separation
        print("\n\n")

if __name__ == "__main__":
    main()
