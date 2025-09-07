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
import pandas as pd
from page_rank import extract_domain, scrape_outlinks_one, build_graph_from_edges, STATS
import csv
import warnings
from datetime import datetime
import math

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
STATS_DIR = os.path.join(BASE_DIR, 'data/stats')

# Trusted platforms with predefined PageRank scores
TRUSTED_PLATFORMS = {
    'bbc.com': 0.9,
    'cnn.com': 0.95,
    'reuters.com': 0.92,
    'nytimes.com': 0.93,
    'theguardian.com': 0.91
}


class NewsClassifier:
    """Class to classify news articles and tweets"""

    def __init__(self):
        """Initialize the classifier by loading all models and vectorizer"""
        print(f"{Fore.CYAN}Starting classification application...{Style.RESET_ALL}")
        logging.info("Starting classification application...")

        try:
            # Load vectorizer for news classification
            self.vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
            print(f"{Fore.CYAN}Loading vectorizer...{Style.RESET_ALL}")
            logging.info(f"Loading vectorizer from {self.vectorizer_path}")
            self.vectorizer = load(self.vectorizer_path)

            # Load news classification models
            self.news_models = {}
            model_files = {
                'naive_bayes': 'naive_bayes_model.joblib',
                'logistic_regression': 'logistic_regression_model.joblib',
                'decision_tree': 'decision_tree_model.joblib',
                'random_forest': 'random_forest_model.joblib'
            }

            print(f"{Fore.CYAN}Loading news classification models...{Style.RESET_ALL}")
            for name, filename in model_files.items():
                model_path = os.path.join(MODELS_DIR, filename)
                logging.info(f"Loading {name} model from {model_path}")
                self.news_models[name] = load(model_path)

            # Load bot detection model
            print(f"{Fore.CYAN}Loading bot detection model...{Style.RESET_ALL}")
            bot_model_path = os.path.join(MODELS_DIR, 'bots', 'random_forest_bot_detector_latest.joblib')
            if os.path.exists(bot_model_path):
                self.bot_model = load(bot_model_path)
                logging.info(f"Loaded bot detection model from {bot_model_path}")
            else:
                bot_models = [f for f in os.listdir(os.path.join(MODELS_DIR, 'bots')) if f.endswith('.joblib')]
                if bot_models:
                    latest_model = max(bot_models)
                    bot_model_path = os.path.join(MODELS_DIR, 'bots', latest_model)
                    self.bot_model = load(bot_model_path)
                    logging.info(f"Loaded bot detection model from {bot_model_path}")
                else:
                    self.bot_model = None
                    logging.warning("No bot detection model found")

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
        domains_path = os.path.join(STATS_DIR, 'domains_summary.csv')
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

    @staticmethod
    def get_date_stats():
        dates_stats = {}
        dates_path = os.path.join(STATS_DIR, 'monthly_bot_data.csv')
        if os.path.exists(dates_path):
            with open(dates_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    year = parts[5].strip()
                    if year == 'year':
                        continue  # Skip header
                    if not dates_stats.get(year, None):
                        dates_stats[year] = {}
                    month = parts[6].strip()
                    fake_ratio = float(parts[3].strip())
                    dates_stats[year][month] = fake_ratio
        return dates_stats

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

    def get_year_and_month_from_date_input(self, date):
        # Validate and parse date input
        if date and '-' in date:
            parts = date.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                year, month = parts
                if len(month) == 2 and month.startswith('0'):
                    month = month[1:]
                return year, month
        return None, None

    def get_url_pagerank_score(self, user_url, graph=None, alpha=0.85):
        """
        Get PageRank score for a user-provided URL by:
        1. Extracting its domain
        2. Checking if it's a trusted platform
        3. If domain exists in graph, return its score
        4. If not, scrape its outlinks and calculate a temporary score
        """
        # Extract domain from URL
        domain = extract_domain(user_url)
        if not domain:
            logging.warning(f"Could not extract a valid domain from the URL: {user_url}")
            return 0.5, "Could not extract a valid domain from the URL."

        # Check if this is a trusted platform with predefined score
        if domain in TRUSTED_PLATFORMS:
            trusted_score = TRUSTED_PLATFORMS[domain]
            logging.info(f"Domain {domain} is a trusted platform with predefined score: {trusted_score}")
            return trusted_score, f"Domain {domain} is a trusted platform with predefined score: {trusted_score}"

        # Load existing graph if not provided
        if graph is None:
            try:
                # Try to load existing edges
                edges = []
                with open(os.path.join(STATS, "domain_edges.csv"), "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for src, dst, label in reader:
                        edges.append((src, dst, label))

                # Import networkx here to avoid dependency issues if not needed
                import networkx as nx
                graph = build_graph_from_edges(edges)
            except Exception as e:
                logging.error(f"Error loading existing graph: {str(e)}")
                return 0.5, f"Error loading existing graph: {str(e)}"

        try:
            # Import networkx here to avoid dependency issues if not needed
            import networkx as nx

            # If domain already in graph, return its PageRank score
            pr = nx.pagerank(graph, alpha=alpha)
            if domain in pr:
                rank_position = sorted(pr.values(), reverse=True).index(pr[domain]) + 1
                logging.info(f"Domain {domain} exists in graph (rank {rank_position}/{len(pr)})")
                return pr[domain], f"Domain {domain} exists in our database (rank {rank_position}/{len(pr)})"

            # If domain not in graph, fetch its outlinks and calculate temporary score
            logging.info(f"Domain {domain} not in existing graph. Fetching outlinks...")

            # Get outlinks for this domain
            outlinks = scrape_outlinks_one(user_url)
            if not outlinks:
                logging.warning(f"Could not fetch any outlinks for {domain}")
                return 0.5, f"Could not fetch any outlinks for {domain}"

            # Create temporary graph with new domain and its connections
            temp_graph = graph.copy()
            for src, dst in outlinks:
                temp_graph.add_edge(src, dst)

            # Calculate new PageRank scores
            new_pr = nx.pagerank(temp_graph, alpha=alpha)

            # Return the score for our domain
            if domain in new_pr:
                rank_position = sorted(new_pr.values(), reverse=True).index(new_pr[domain]) + 1
                logging.info(f"Calculated temporary score for {domain} (rank {rank_position}/{len(new_pr)})")
                return new_pr[domain], f"Temporary score for {domain} (rank {rank_position}/{len(new_pr)})"
            else:
                logging.warning(f"Domain {domain} has no connections in the graph")
                return 0.5, f"Domain {domain} has no connections in the graph"

        except Exception as e:
            logging.error(f"Error calculating PageRank: {str(e)}")
            return 0.5, f"Error calculating PageRank: {str(e)}"

    def predict(self, text, domain, date, url=None):
        """Make predictions using all models and calculate a cumulative score from multiple factors"""
        domain_stats = self.get_domain_stats()
        date_stats = self.get_date_stats()
        year, month = self.get_year_and_month_from_date_input(date)
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

            for name, model in self.news_models.items():
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

            # Calculate voting result
            votes = list(results.values())
            model_score = sum(votes) / len(votes)  # Score between 0 and 1

            # Calculate average probabilities from models
            avg_proba = [0, 0]
            for name in probabilities:
                avg_proba[0] += probabilities[name][0]
                avg_proba[1] += probabilities[name][1]

            avg_proba[0] /= len(probabilities)
            avg_proba[1] /= len(probabilities)

            # Initialize cumulative scores (higher means more likely to be REAL)
            # Start with model probability score
            cumulative_real_score = avg_proba[1]
            cumulative_fake_score = avg_proba[0]

            # Track contribution from each source for reporting
            score_contributions = {
                'model_probability': avg_proba[1] - 0.5  # Center around 0 for logging
            }

            # Extract domain from URL if provided but domain is not
            if url and not domain:
                domain = extract_domain(url)
                if domain:
                    print(f"{Fore.CYAN}Extracted domain from URL: {domain}{Style.RESET_ALL}")
                    logging.info(f"Domain extracted from URL: {domain}")

            # Apply domain reputation adjustment if available
            domain_info = None
            if domain and domain in domain_stats:
                logging.info(f"Adjusting scores based on domain: {domain}")
                fake_ratio = domain_stats[domain]

                # Apply domain adjustment to scores
                # Higher fake ratio increases fake probability
                domain_adjustment = min(0.3, fake_ratio * 0.5)  # Cap the adjustment at 30%

                # Add to cumulative scores
                cumulative_fake_score += domain_adjustment
                cumulative_real_score -= domain_adjustment

                score_contributions[
                    'domain_reputation'] = -domain_adjustment  # Negative because it reduces "real" score

                # Get page rank and use it as a factor (higher page rank = more likely real)
                if url:
                    page_rank_score, page_rank_message = self.get_url_pagerank_score(url)
                    page_rank_adjustment = (page_rank_score - 0.5) * 0.4  # Scale the effect (max ±0.2)

                    cumulative_real_score += page_rank_adjustment
                    cumulative_fake_score -= page_rank_adjustment

                    score_contributions['page_rank'] = page_rank_adjustment

                    logging.info(f"Applied page rank adjustment: {page_rank_score}, adjustment={page_rank_adjustment}")

            # Apply date-based adjustment if available
            if year and month and year in date_stats and month in date_stats[year]:
                logging.info(f"Adjusting scores based on date: {date}")
                fake_ratio = date_stats[year][month]

                # Apply date adjustment to scores
                date_adjustment = min(0.2, fake_ratio * 0.3)  # Cap the adjustment at 20%

                # Add to cumulative scores
                cumulative_fake_score += date_adjustment
                cumulative_real_score -= date_adjustment

                score_contributions['date_factor'] = -date_adjustment

                print(f"{Fore.YELLOW}Applied date-based adjustment: {date_adjustment:.4f}{Style.RESET_ALL}")
                logging.info(f"Applied date adjustment for {year}-{month}: fake_ratio={fake_ratio}, " +
                             f"adjustment={date_adjustment}")

            # Calculate final probability based on cumulative scores
            # Normalize to ensure they sum to 1
            total_score = cumulative_real_score + cumulative_fake_score
            avg_proba[1] = cumulative_real_score / total_score
            avg_proba[0] = cumulative_fake_score / total_score

            # Make final prediction based on normalized scores
            final_prediction = 1 if avg_proba[1] > avg_proba[0] else 0

            logging.info(f"Final scores - Real: {cumulative_real_score:.4f}, Fake: {cumulative_fake_score:.4f}")
            logging.info(f"Score contributions: {score_contributions}")

            return {
                'prediction': final_prediction,
                'label': 'REAL' if final_prediction == 1 else 'FAKE',
                'confidence': avg_proba[final_prediction] * 100,
                'real_probability': avg_proba[1] * 100,
                'fake_probability': avg_proba[0] * 100,
                'model_votes': results,
                'score_contributions': score_contributions
            }

        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return None

    def predict_bot(self, tweet_text, user_data):
        """
        Analyze if a tweet is from a bot based on user data

        Parameters:
        -----------
        tweet_text : str
            The text content of the tweet
        user_data : dict
            Dictionary containing Twitter user metrics

        Returns:
        --------
        dict
            Result dictionary with bot prediction and confidence
        """
        try:
            # Import warnings to suppress the sklearn feature names warning
            import warnings

            if self.bot_model is None:
                logging.error("Bot detection model not available")
                return {
                    'prediction': None,
                    'label': 'UNKNOWN',
                    'message': 'Bot detection model not available',
                    'confidence': 0
                }

            print(f"\n{Fore.CYAN}Analyzing account for bot characteristics...{Style.RESET_ALL}")

            # Apply the 5 strong human likelihood indicators based on profile_completeness_auc.csv
            # These indicators have high AUC values indicating strong discriminatory power
            human_score = 0.0
            human_indicators = {}

            # 1. Verified status (binary) - AUC: 0.7828
            verified_weight = 0.7828
            if 'verified' in user_data and user_data['verified']:
                human_indicators['verified'] = verified_weight
                human_score += verified_weight

            # 2. Followers to friends ratio - AUC: 0.7501
            followers_to_friends_ratio_weight = 0.7501
            if 'followers_count' in user_data and 'friends_count' in user_data:
                followers_to_friends_ratio = user_data['followers_count'] / (user_data['friends_count'] + 1)
                # Normalize the ratio: most human accounts have ratios between 0.1 and 10
                normalized_ratio = min(1.0, followers_to_friends_ratio / 10.0)
                human_indicators['followers_to_friends_ratio'] = normalized_ratio * followers_to_friends_ratio_weight
                human_score += human_indicators['followers_to_friends_ratio']

            # 3. Number of followers - AUC: 0.7357
            followers_weight = 0.7357
            if 'followers_count' in user_data:
                # Normalize followers count: logarithmic scale (1000 followers is a significant threshold)
                followers_normalized = min(1.0, math.log10(user_data['followers_count'] + 1) / 3.0)
                human_indicators['followers'] = followers_normalized * followers_weight
                human_score += human_indicators['followers']

            # 4. Listed count - AUC: 0.7338
            listed_count_weight = 0.7338
            if 'listed_count' in user_data:
                # Normalize listed count: being in 10+ lists is significant
                listed_normalized = min(1.0, user_data['listed_count'] / 10.0)
                human_indicators['listed_count'] = listed_normalized * listed_count_weight
                human_score += human_indicators['listed_count']

            # 5. Account age in days - AUC: 0.6287
            account_age_weight = 0.6287
            if 'created_at' in user_data:
                try:
                    # Parse Twitter date format
                    created_date = datetime.strptime(user_data['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                    current_date = datetime.now()
                    account_age_days = (current_date - created_date).days
                    # Normalize account age: accounts older than 365 days get full score
                    age_normalized = min(1.0, account_age_days / 365.0)
                    human_indicators['account_age_days'] = age_normalized * account_age_weight
                    human_score += human_indicators['account_age_days']
                except Exception as e:
                    logging.error(f"Error calculating account age: {str(e)}")

            # Calculate human probability based on indicators
            # Maximum possible score if all indicators are at their max
            max_possible_score = verified_weight + followers_to_friends_ratio_weight + followers_weight + listed_count_weight + account_age_weight
            # Normalize to a probability
            human_probability = min(0.95, human_score / max_possible_score)

            # Prepare features for bot detection using the existing model
            features = {}

            # Required features based on the bot model
            required_features = [
                'followers_count', 'friends_count', 'statuses_count',
                'favourites_count', 'listed_count', 'screen_name_length',
                'retweets', 'replies', 'favoriteC', 'hashtag',
                'mentions', 'intertime', 'ffratio', 'favorites',
                'uniqueHashtags', 'uniqueMentions', 'uniqueURL'
            ]

            # Fill available features from user_data
            for feature in required_features:
                if feature in user_data:
                    features[feature] = user_data[feature]
                else:
                    # Use sensible defaults for missing features
                    if feature == 'ffratio' and 'followers_count' in user_data and 'friends_count' in user_data:
                        if user_data['followers_count'] > 0:
                            features[feature] = user_data['friends_count'] / user_data['followers_count']
                        else:
                            features[feature] = user_data['friends_count'] if user_data['friends_count'] > 0 else 1.0
                    elif feature == 'screen_name_length' and 'screen_name' in user_data:
                        features[feature] = len(user_data['screen_name'])
                    else:
                        # Default values for other features
                        features[feature] = 0.0

            # Convert to DataFrame with appropriate columns that the model expects
            X = pd.DataFrame([features])

            # Add URL field which was in the original dataset
            if 'url' not in X.columns:
                X['url'] = 0.0

            # Add 'listed' field which appears in the feature importance file
            if 'listed' not in X.columns:
                X['listed'] = X['listed_count'] if 'listed_count' in X.columns else 0.0

            # Create polynomial features (interactions between features)
            # Based on the feature importance file, we know the model was trained with these interactions
            core_features = ['screen_name_length', 'statuses_count', 'followers_count', 'friends_count',
                             'favourites_count']

            # Add polynomial feature interactions - EXPLICITLY listing all needed combinations
            # to ensure we have exactly what the model expects
            interaction_pairs = [
                ('screen_name_length', 'statuses_count'),
                ('followers_count', 'friends_count'),
                ('screen_name_length', 'friends_count'),
                ('screen_name_length', 'followers_count'),
                ('followers_count', 'favourites_count'),
                ('friends_count', 'favourites_count'),
                ('screen_name_length', 'favourites_count'),
                ('statuses_count', 'followers_count'),
                ('statuses_count', 'friends_count'),
                ('statuses_count', 'favourites_count')
            ]

            for feat1, feat2 in interaction_pairs:
                interaction_name = f"{feat1} {feat2}"
                X[interaction_name] = X[feat1] * X[feat2]

            # Verify we have all 29 features
            expected_features = set([
                'screen_name_length', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count',
                'listed_count', 'url', 'retweets', 'replies', 'favoriteC', 'hashtag', 'mentions', 'intertime',
                'ffratio', 'favorites', 'uniqueHashtags', 'uniqueMentions', 'uniqueURL', 'listed',
                'screen_name_length statuses_count', 'followers_count friends_count',
                'screen_name_length friends_count',
                'screen_name_length followers_count', 'followers_count favourites_count',
                'friends_count favourites_count',
                'screen_name_length favourites_count', 'statuses_count followers_count', 'statuses_count friends_count',
                'statuses_count favourites_count'
            ])

            # Check if we're missing any features and add them
            missing_features = expected_features - set(X.columns)
            for feat in missing_features:
                X[feat] = 0.0  # Add any missing features with default value
                logging.info(f"Added missing feature: {feat}")

            # Make prediction
            try:
                # Try to extract feature names from the model
                model_features = []
                try:
                    if hasattr(self.bot_model, 'feature_names_in_'):
                        model_features = self.bot_model.feature_names_in_.tolist()
                except:
                    pass

                # If model has feature names, ensure we have exactly those features
                if model_features:
                    # Add any missing features
                    for feat in model_features:
                        if feat not in X.columns:
                            X[feat] = 0.0
                    # Keep only the features the model knows about and in the same order
                    X = X[model_features]

                # Suppress the specific warning about feature names
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning,
                                            message="X has feature names, but RandomForestClassifier was fitted without feature names")

                    # Now make the prediction
                    prediction = self.bot_model.predict(X)[0]
                    probabilities = self.bot_model.predict_proba(X)[0]

                # Bot is usually labeled as 0, human as 1
                # Blend model prediction with our human indicators
                model_human_prob = probabilities[1]
                # Use a weighted average, giving more weight to our custom indicators for the specified features
                final_human_prob = (human_probability * 0.7) + (model_human_prob * 0.3)
                is_bot = final_human_prob < 0.5

                return {
                    'prediction': int(is_bot),
                    'label': 'BOT' if is_bot else 'HUMAN',
                    'confidence': (1 - final_human_prob if is_bot else final_human_prob) * 100,
                    'bot_probability': (1 - final_human_prob) * 100,
                    'human_probability': final_human_prob * 100,
                    'human_indicators': human_indicators
                }
            except Exception as e:
                logging.error(f"Error using bot prediction model: {str(e)}")
                logging.error(f"Current feature count: {len(X.columns)}, Features: {X.columns.tolist()}")
                return {
                    'prediction': None,
                    'label': 'ERROR',
                    'message': f"Error using model: {str(e)}",
                    'confidence': 0
                }

        except Exception as e:
            logging.error(f"Error in bot prediction: {str(e)}")
            return None


def print_header():
    """Print the application header with content type selection"""
    print("\n" + "=" * 50)
    print(f"{Fore.CYAN}=== Content Classification System ==={Style.RESET_ALL}")
    print("=" * 50)
    print(f"Please select content type to analyze:")
    print(f"1. {Fore.YELLOW}News Article{Style.RESET_ALL}")
    print(f"2. {Fore.YELLOW}Tweet{Style.RESET_ALL}")
    print(f"Type {Fore.YELLOW}exit{Style.RESET_ALL} to quit")
    print("=" * 50)


def print_result(result):
    """Print the classification result with nice formatting"""
    print("\n" + "=" * 50)
    print(f"{Fore.YELLOW}=== Classification Result ==={Style.RESET_ALL}")
    print("=" * 50)

    # Print prediction with color
    if result['label'] == 'REAL':
        print(f"Prediction: {Fore.GREEN}{result['label']}{Style.RESET_ALL}")
    else:
        print(f"Prediction: {Fore.RED}{result['label']}{Style.RESET_ALL}")

    # Print confidence and probabilities
    print(f"Confidence: {Fore.YELLOW}{result['confidence']:.2f}%{Style.RESET_ALL}")
    print(f"Probability of REAL: {Fore.CYAN}{result['real_probability']:.2f}%{Style.RESET_ALL}")
    print(f"Probability of FAKE: {Fore.CYAN}{result['fake_probability']:.2f}%{Style.RESET_ALL}")

    print("\n" + "=" * 50)


def print_bot_result(result):
    """Print the bot detection result with nice formatting"""
    print("\n" + "=" * 50)
    print(f"{Fore.YELLOW}=== Bot Detection Result ==={Style.RESET_ALL}")
    print("=" * 50)

    # Print prediction with color
    if result['label'] == 'HUMAN':
        print(f"Prediction: {Fore.GREEN}{result['label']}{Style.RESET_ALL}")
    elif result['label'] == 'BOT':
        print(f"Prediction: {Fore.RED}{result['label']}{Style.RESET_ALL}")
    else:
        print(f"Prediction: {Fore.YELLOW}{result['label']}{Style.RESET_ALL}")

    # Print message if available (typically for errors)
    if 'message' in result:
        print(f"Message: {Fore.RED}{result['message']}{Style.RESET_ALL}")

    # Print confidence and probabilities if available
    print(f"Confidence: {Fore.YELLOW}{result['confidence']:.2f}%{Style.RESET_ALL}")

    # Print probabilities only if they're available (they won't be for error cases)
    if 'human_probability' in result and 'bot_probability' in result:
        print(f"Probability of HUMAN: {Fore.CYAN}{result['human_probability']:.2f}%{Style.RESET_ALL}")
        print(f"Probability of BOT: {Fore.CYAN}{result['bot_probability']:.2f}%{Style.RESET_ALL}")

    print("\n" + "=" * 50)


def main():
    """Main function to run the classifier application"""
    classifier = NewsClassifier()

    while True:
        print_header()

        # Get content type selection
        choice = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

        # Strip whitespace to properly check input
        choice = choice.strip()

        # Check if user wants to exit
        if choice.lower() in ['exit', 'quit', 'q']:
            print(f"\n{Fore.GREEN}Thank you for using the Content Classification System!{Style.RESET_ALL}")
            break

        # Process article or tweet based on user choice
        if choice == '1':
            process_article(classifier)
        elif choice == '2':
            process_tweet(classifier)
        else:
            print(
                f"{Fore.RED}Invalid choice. Please enter 1 for News Article, 2 for Tweet, or 'exit' to quit.{Style.RESET_ALL}")

        # Visual separation between iterations
        print("\n\n")


def process_article(classifier):
    """Process a news article for classification"""
    print("\n" + "=" * 50)
    print(f"{Fore.CYAN}=== News Article Classification ==={Style.RESET_ALL}")
    print("=" * 50)
    print("Enter the article text to classify:\n")

    # Get article text
    article_text = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

    if not article_text.strip():
        print(f"{Fore.RED}Empty input. Article classification canceled.{Style.RESET_ALL}")
        return

    print("\nEnter the domain the article came from (press enter to skip):")
    domain = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

    print("\nEnter the full URL of the article (for page rank analysis, press enter to skip):")
    url = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

    # Extract domain from URL if URL is provided but domain is not
    if url and not domain:
        extracted_domain = extract_domain(url)
        if extracted_domain:
            domain = extracted_domain
            print(f"{Fore.CYAN}Extracted domain from URL: {domain}{Style.RESET_ALL}")
            logging.info(f"Domain extracted from URL: {domain}")

    # Classify the article with the URL for page rank analysis
    result = classifier.predict(article_text, domain, "", url)

    # Display result
    if result:
        print_result(result)

        # Show score contributions if available
        if 'score_contributions' in result:
            print(f"\n{Fore.YELLOW}=== Score Contributions ==={Style.RESET_ALL}")
            for source, contribution in result['score_contributions'].items():
                direction = "+" if contribution > 0 else ""
                print(f"{source}: {direction}{contribution:.4f}")
            print("=" * 50)
    else:
        print(f"{Fore.RED}Error: Could not classify the article. Please try again.{Style.RESET_ALL}")


def process_tweet(classifier):
    """Process a tweet for bot detection"""
    print("\n" + "=" * 50)
    print(f"{Fore.CYAN}=== Tweet Analysis ==={Style.RESET_ALL}")
    print("=" * 50)
    print("Enter the tweet text:\n")

    # Get tweet text
    tweet_text = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

    if not tweet_text.strip():
        print(f"{Fore.RED}Empty input. Tweet analysis canceled.{Style.RESET_ALL}")
        return

    # Collect user account information for bot detection
    print(f"\n{Fore.CYAN}Now please provide account metrics for bot detection analysis:{Style.RESET_ALL}")

    user_data = {}

    # Basic metrics
    print("\nNumber of followers:")
    try:
        user_data['followers_count'] = float(input(f"{Fore.CYAN}> {Style.RESET_ALL}") or "0")
    except ValueError:
        user_data['followers_count'] = 0

    print("\nNumber of friends/following:")
    try:
        user_data['friends_count'] = float(input(f"{Fore.CYAN}> {Style.RESET_ALL}") or "0")
    except ValueError:
        user_data['friends_count'] = 0

    print("\nTotal number of statuses/tweets:")
    try:
        user_data['statuses_count'] = float(input(f"{Fore.CYAN}> {Style.RESET_ALL}") or "0")
    except ValueError:
        user_data['statuses_count'] = 0

    print("\nNumber of favorites/likes received:")
    try:
        user_data['favourites_count'] = float(input(f"{Fore.CYAN}> {Style.RESET_ALL}") or "0")
    except ValueError:
        user_data['favourites_count'] = 0

    print("\nNumber of public lists the account is included in:")
    try:
        user_data['listed_count'] = float(input(f"{Fore.CYAN}> {Style.RESET_ALL}") or "0")
    except ValueError:
        user_data['listed_count'] = 0

    print("\nIs the account verified? (y/n):")
    verified_input = input(f"{Fore.CYAN}> {Style.RESET_ALL}").strip().lower()
    user_data['verified'] = verified_input == 'y' or verified_input == 'yes'

    print("\nScreen name or username:")
    user_data['screen_name'] = input(f"{Fore.CYAN}> {Style.RESET_ALL}")
    user_data['screen_name_length'] = len(user_data['screen_name'])

    print("\nAccount creation date (YYYY-MM format, press enter to skip):")
    account_date = input(f"{Fore.CYAN}> {Style.RESET_ALL}")

    # Set default values for advanced metrics
    user_data['retweets'] = 0.2
    user_data['replies'] = 0.1
    user_data['favoriteC'] = 0.3
    user_data['hashtag'] = 0.3
    user_data['mentions'] = 0.4
    user_data['intertime'] = 3600
    user_data['favorites'] = 100
    user_data['uniqueHashtags'] = 0.5
    user_data['uniqueMentions'] = 0.6
    user_data['uniqueURL'] = 0.7

    # Calculate friends-to-followers ratio
    if user_data['followers_count'] > 0:
        user_data['ffratio'] = user_data['friends_count'] / user_data['followers_count']
    else:
        user_data['ffratio'] = user_data['friends_count'] if user_data['friends_count'] > 0 else 1.0

    # Perform bot detection
    bot_result = classifier.predict_bot(tweet_text, user_data)

    if bot_result:
        print_bot_result(bot_result)

        # Also analyze the tweet content for real/fake classification
        print(f"\n{Fore.CYAN}Now analyzing the tweet content for real/fake news classification...{Style.RESET_ALL}")
        # Pass the account date to the content analysis
        content_result = classifier.predict(tweet_text, "", account_date)

        if content_result:
            print_result(content_result)
        else:
            print(
                f"{Fore.RED}Error: Could not classify the tweet content. Bot detection analysis still valid.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Error: Could not perform bot detection analysis. Please try again.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
