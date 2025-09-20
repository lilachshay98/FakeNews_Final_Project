import csv
import logging
import math
import os
import string
import sys
from datetime import datetime

import pandas as pd
from joblib import load

from colorama import Fore, Style

from community_detection import load_accounts, build_follow_graph, top_mentions, \
    subgraph_around_anchors, \
    louvain_partition, analyze_communities, predict_account_label
from page_rank import extract_domain, build_graph_from_edges, scrape_outlinks_one

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATS_DIR = os.path.join(BASE_DIR, 'data/stats')

TRUSTED_PLATFORMS = {
    'bbc.com': 0.9,
    'cnn.com': 0.95,
    'reuters.com': 0.92,
    'nytimes.com': 0.93,
    'theguardian.com': 0.91
}


class NewsClassifier:
    """
    Comprehensive multimodal classifier for fake news detection and bot analysis.

    A unified classification system that combines multiple machine learning models,
    domain reputation analysis, PageRank scoring, community detection, and temporal
    patterns to provide robust content authenticity assessment and bot detection
    capabilities.
    """
    def __init__(self):
        """
        Initialize the classifier by loading all models and vectorizers.

        Loads and initializes all required components including TF-IDF vectorizer,
        ensemble of news classification models, bot detection model, and
        associated preprocessing tools. Displays loading progress with colored
        terminal output.

        Returns
        -------
        None
            Initializes classifier with all loaded models or exits if loading fails.

        Raises
        ------
        SystemExit
            If any required model files cannot be loaded, exits with error code 1.

        Notes
        -----
        Loaded models include:
        - TF-IDF vectorizer for text feature extraction
        - Ensemble of news classifiers: Naive Bayes, Logistic Regression,
          Decision Tree, Random Forest
        - Random Forest bot detection model
        """
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

    def get_url_pagerank_score(self, user_url, graph=None, alpha=0.85):
        """
        Calculate PageRank score for a URL's domain with trusted platform handling.

        Computes PageRank-based credibility score by first checking trusted platforms,
        then existing graph data, and finally scraping outlinks for new domains
        to calculate temporary scores.

        Parameters
        ----------
        user_url : str
            URL to analyze for PageRank scoring.
        graph : networkx.DiGraph, optional
            Existing domain graph for PageRank calculation. If None, loads
            from domain_edges.csv file.
        alpha : float, default=0.85
            Damping parameter for PageRank algorithm.

        Returns
        -------
        score : float
            PageRank score between 0.0 and 1.0, where higher values indicate
            more credible/authoritative domains.
        message : str
            Descriptive message explaining the score source and calculation.

        Notes
        -----
        Scoring hierarchy:
        1. Trusted platforms: Predefined high scores for major news outlets
        2. Existing graph: PageRank from historical domain relationship data
        3. Dynamic scraping: Temporary score from newly scraped outlinks
        4. Fallback: 0.5 neutral score for unreachable/invalid domains
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
                with open(os.path.join(STATS_DIR, "domain_edges.csv"), "r", encoding="utf-8") as f:
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
        """
        Make multimodal fake news predictions using ensemble methods and contextual factors.

        Performs comprehensive fake news detection by combining predictions from
        multiple machine learning models with domain reputation analysis, PageRank
        scoring, and temporal pattern recognition for robust classification.

        Parameters
        ----------
        text : str
            News article text content to analyze.
        domain : str
            Source domain name for reputation analysis.
        date : str
            Publication date in YYYY-MM format for temporal analysis.
        url : str, optional
            Full article URL for PageRank analysis and domain extraction.

        Returns
        -------
        result : dict
            Comprehensive prediction result containing:
            - 'prediction': int, binary prediction (0=fake, 1=real)
            - 'label': str, human-readable prediction ('REAL' or 'FAKE')
            - 'confidence': float, confidence percentage (0-100)
            - 'real_probability': float, probability of being real (0-100)
            - 'fake_probability': float, probability of being fake (0-100)
            - 'model_votes': dict, individual model predictions
            - 'score_contributions': dict, contribution from each scoring component
            - 'component_weights': dict, weighting scheme used for final prediction
            Returns None if prediction fails.

        Notes
        -----
        The ensemble includes Naive Bayes, Logistic Regression, Decision Tree,
        and Random Forest models for robust text-based classification.
        """
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

            # Use avg_proba[1] as the model's real probability score
            model_real_prob = avg_proba[1]

            # Initialize domain and page rank scores with neutral values
            domain_score = 0.5  # Neutral value
            page_rank_score = 0.5  # Neutral value

            # Track contribution from each source for reporting
            score_contributions = {
                'model_prediction': (model_real_prob - 0.5) * 0.4  # Centered around 0 for logging
            }

            # Extract domain from URL if provided but domain is not
            if url and not domain:
                domain = extract_domain(url)
                if domain:
                    print(f"{Fore.CYAN}Extracted domain from URL: {domain}{Style.RESET_ALL}")
                    logging.info(f"Domain extracted from URL: {domain}")

            # Apply domain reputation adjustment if available
            if domain and domain in domain_stats:
                logging.info(f"Adjusting scores based on domain: {domain}")
                fake_ratio = domain_stats[domain]
                # Convert fake_ratio to a credibility score (1 - fake_ratio)
                domain_score = 1.0 - fake_ratio
                score_contributions['domain_score'] = (domain_score - 0.5) * 0.3
                logging.info(f"Applied domain score: {domain_score}")

            # Get page rank and use it as a factor (higher page rank = more likely real)
            if url:
                pr_score, pr_message = self.get_url_pagerank_score(url)
                if pr_score is not None:
                    page_rank_score = pr_score
                    score_contributions['page_rank_score'] = (page_rank_score - 0.5) * 0.3
                    logging.info(f"Applied page rank score: {page_rank_score}, {pr_message}")

            # Apply date-based adjustment if available
            if year and month and year in date_stats and month in date_stats[year]:
                logging.info(f"Adjusting scores based on date: {date}")
                fake_ratio = date_stats[year][month]
                # Apply a small date adjustment to the model score
                date_adjustment = min(0.1, fake_ratio * 0.2)  # Cap the adjustment
                model_real_prob -= date_adjustment
                score_contributions['date_factor'] = -date_adjustment
                print(f"{Fore.YELLOW}Applied date-based adjustment: -{date_adjustment:.4f}{Style.RESET_ALL}")

            # Calculate final probability based on weighted components:
            # 40% model prediction, 30% domain score, 30% page rank
            final_real_probability = (
                    (model_real_prob * 0.4) +  # 40% model prediction
                    (domain_score * 0.3) +  # 30% domain score
                    (page_rank_score * 0.3)  # 30% page rank score
            )

            # Ensure probability is within bounds
            final_real_probability = max(0.01, min(0.99, final_real_probability))
            final_fake_probability = 1.0 - final_real_probability

            # Make final prediction
            final_prediction = 1 if final_real_probability > 0.5 else 0

            logging.info(f"Final probability - Real: {final_real_probability:.4f}, Fake: {final_fake_probability:.4f}")
            logging.info(f"Score contributions: {score_contributions}")
            logging.info(f"Component weights - Model: 40%, Domain: 30%, PageRank: 30%")

            return {
                'prediction': final_prediction,
                'label': 'REAL' if final_prediction == 1 else 'FAKE',
                'confidence': max(final_real_probability, final_fake_probability) * 100,
                'real_probability': final_real_probability * 100,
                'fake_probability': final_fake_probability * 100,
                'model_votes': results,
                'score_contributions': score_contributions,
                'component_weights': {
                    'model_prediction': '40%',
                    'domain_score': '30%',
                    'page_rank_score': '30%'
                }
            }

        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return None

    def get_community_prediction_score(self, account_name):
        """
        Get bot/human prediction score based on community detection analysis.

        Uses community detection algorithms on Twitter follow networks to
        predict account authenticity based on community membership patterns
        and relationship structures.

        Parameters
        ----------
        account_name : str
            Twitter account username (with or without @ prefix).

        Returns
        -------
        prediction_score : dict or float
            Community-based prediction result containing human/bot probabilities
            and community assignment information. Returns 1.0 if account name
            is not provided.

        Notes
        -----
        Community analysis workflow:
        1. Load Twitter account relationship data
        2. Build directed follow graph
        3. Find anchor accounts from mentions
        4. Create ego-graph around anchors
        5. Apply Louvain community detection
        6. Analyze community composition (bot vs human ratio)
        7. Predict account type based on community membership
        """
        if not account_name:
            return 1.0  # Neutral score if no account name provided
        # Load and build graph
        accounts = load_accounts()
        Gd, labels, screen = build_follow_graph(accounts)

        # Find anchors and create subgraph
        anchors = top_mentions(accounts)
        H, anchor_ids = subgraph_around_anchors(Gd, screen, anchors, radius=2, max_nodes=4000, mutual_only=False)

        # Detect communities
        partition = louvain_partition(H)

        # Analyze communities
        community_stats = analyze_communities(labels, partition)

        return predict_account_label(account_name, Gd, H, labels, screen, partition, community_stats, radius=2)

    def predict_bot(self, tweet_text, user_data):
        """
        Analyze if a tweet is from a bot using multimodal detection methods.

        Performs comprehensive bot detection by combining machine learning model
        predictions with profile completeness analysis, community detection, and
        behavioral pattern recognition for robust account classification.

        Parameters
        ----------
        tweet_text : str
            The text content of the tweet to analyze.
        user_data : dict
            Dictionary containing Twitter user metrics including:
            - Basic counts: followers_count, friends_count, statuses_count
            - Profile info: screen_name, verified, created_at
            - Engagement: favourites_count, listed_count
            - Behavioral features for model input

        Returns
        -------
        result : dict
            Comprehensive bot detection result containing:
            - 'prediction': int, binary prediction (0=human, 1=bot)
            - 'label': str, human-readable result ('HUMAN', 'BOT', or 'ERROR')
            - 'confidence': float, prediction confidence percentage
            - 'bot_probability': float, probability of being a bot
            - 'human_probability': float, probability of being human
            - 'human_indicators': dict, AUC-weighted profile indicators
            - 'community_score': dict, community-based prediction
            Returns None if analysis fails completely.
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
            expected_features = {'screen_name_length', 'statuses_count', 'followers_count', 'friends_count',
                                 'favourites_count', 'listed_count', 'url', 'retweets', 'replies', 'favoriteC',
                                 'hashtag', 'mentions', 'intertime', 'ffratio', 'favorites', 'uniqueHashtags',
                                 'uniqueMentions', 'uniqueURL', 'listed', 'screen_name_length statuses_count',
                                 'followers_count friends_count', 'screen_name_length friends_count',
                                 'screen_name_length followers_count', 'followers_count favourites_count',
                                 'friends_count favourites_count', 'screen_name_length favourites_count',
                                 'statuses_count followers_count', 'statuses_count friends_count',
                                 'statuses_count favourites_count'}

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

                # Get community prediction if screen name is provided
                community_score = None
                try:
                    if user_data.get('screen_name'):
                        community_score = self.get_community_prediction_score(user_data['screen_name'])
                        # Ensure community_score is a number we can use
                        if isinstance(community_score, (int, float)):
                            # Incorporate community score into final human probability
                            final_human_prob = (final_human_prob * 0.7) + (community_score['human_probability'] * 0.3)
                        else:
                            logging.warning(f"Community score is not a number: {community_score}")
                except Exception as e:
                    logging.warning(f"Error getting community prediction: {str(e)}")

                # Determine if it's a bot based on final probability
                is_bot = final_human_prob < 0.5

                return {
                    'prediction': int(is_bot),
                    'label': 'BOT' if is_bot else 'HUMAN',
                    'confidence': (1 - final_human_prob if is_bot else final_human_prob) * 100,
                    'bot_probability': (1 - final_human_prob) * 100,
                    'human_probability': final_human_prob * 100,
                    'human_indicators': human_indicators,
                    'community_score': community_score
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

    @staticmethod
    def get_domain_stats():
        """
        Load domain reputation statistics from CSV file.

        Reads domain-level statistics including fake news ratios for
        credibility assessment in news classification.

        Returns
        -------
        domain_stats : dict[str, float]
            Mapping from domain names to fake news ratios (0.0-1.0).
            Empty dict if file not found.
        """
        domain_stats = {}
        domains_path = os.path.join(STATS_DIR, 'domains_summary.csv')
        if os.path.exists(domains_path):
            with open(domains_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    parts = line.strip().split(',')
                    domain = parts[0].strip().lower()
                    if domain == 'domain':
                        continue  # Skip header
                    fake_ratio = float(parts[4].strip())
                    domain_stats[domain] = fake_ratio
        else:
            logging.warning(f"domains.txt file not found at {domains_path}")
        return domain_stats

    @staticmethod
    def get_date_stats():
        """
        Load temporal fake news statistics by year and month.

        Reads monthly bot activity statistics for temporal pattern
        analysis in news authenticity assessment.

        Returns
        -------
        date_stats : dict[str, dict[str, float]]
            Nested mapping from year -> month -> fake news ratio.
            Empty dict if file not found.
        """
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
        """
        Clean input text with preprocessing matching training data.

        Applies consistent text preprocessing including lowercasing,
        punctuation removal, and whitespace normalization to match
        the preprocessing used during model training.

        Parameters
        ----------
        text : str or other
            Input text to preprocess. Non-string inputs are converted to string.

        Returns
        -------
        cleaned_text : str
            Preprocessed text ready for vectorization and model input.

        Notes
        -----
        Preprocessing steps:
        1. Convert to string if necessary
        2. Lowercase conversion
        3. Punctuation removal using string.punctuation
        4. Whitespace normalization (collapse multiple spaces)
        """
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
        """
        Parse year and month from date input string.

        Extracts year and month components from date strings in
        YYYY-MM format for temporal analysis.

        Parameters
        ----------
        date : str
            Date string in YYYY-MM format.

        Returns
        -------
        year : str or None
            Extracted year component or None if parsing fails.
        month : str or None
            Extracted month component (leading zero removed) or None if parsing fails.
        """
        # Validate and parse date input
        if date and '-' in date:
            parts = date.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                year, month = parts
                if len(month) == 2 and month.startswith('0'):
                    month = month[1:]
                return year, month
        return None, None
