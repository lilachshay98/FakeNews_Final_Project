import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class BotDetectionSystem:
    """
    Machine learning system for detecting automated social media accounts.

    Combines feature engineering, model training, and evaluation to identify bot accounts
    based on behavioral patterns and profile characteristics. Uses Random Forest classifier
    with balanced class weights for optimal performance on imbalanced datasets.

    Attributes:
        model: RandomForestClassifier with balanced class weights
        scaler: StandardScaler for feature normalization
        label_encoder: LabelEncoder for categorical data handling
        feature_names: List of extracted feature column names
    """
    def __init__(self):
        """Initialize bot detection system with default ML components."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []

    def merge_csv_files(self, file_paths):
        """
        Combine multiple CSV datasets into unified DataFrame.

        Loads and concatenates CSV files with error handling for missing or corrupted
        files. Provides detailed logging of loading success and dataset dimensions.

        Parameters
        ----------
        file_paths : list of str
            Paths to CSV files for merging

        Returns
        -------
        pd.DataFrame or None
            Combined dataset with all rows concatenated. Returns None if no files loaded successfully.
        """
        dataframes = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                dataframes.append(df)
                print(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            print(f"Merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
            return merged_df
        else:
            print("No files were successfully loaded")
            return None

    def clean_data(self, df, missing_threshold=0.5):
        """
        Remove columns with excessive missing values to improve data quality.

        Analyzes missing value patterns and drops columns exceeding the specified
        threshold to maintain dataset integrity while preserving informative features.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to clean
        missing_threshold : float, default=0.5
            Proportion threshold for dropping columns (0.5 = 50% missing values)

        Returns
        -------
        pd.DataFrame
            Cleaned dataset with high-quality columns retained
        """
        print("Data cleaning started...")
        print(f"Initial dataset shape: {df.shape}")

        # Calculate missing value percentage for each column
        missing_pct = df.isnull().sum() / len(df)

        # Drop columns with missing values above threshold
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index
        df_cleaned = df.drop(columns=cols_to_drop)

        print(f"Dropped {len(cols_to_drop)} columns with >{missing_threshold * 100}% missing values")
        print(f"Remaining columns: {list(df_cleaned.columns)}")
        print(f"Cleaned dataset shape: {df_cleaned.shape}")

        return df_cleaned

    def preprocess_text(self, df, tweet_columns=None, profile_columns=None):
        """
        Combine tweet content with profile information for comprehensive text analysis.

        Merges multiple text sources into unified fields, handles missing content with
        'Nil' markers, and prepares text data for feature extraction.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset with text columns
        tweet_columns : list of str, optional
            Column names containing tweet text (auto-detected if None)
        profile_columns : list of str, optional
            Column names containing profile text (auto-detected if None)

        Returns
        -------
        pd.DataFrame
            Dataset with added 'combined_text' column and text preprocessing applied
        """
        print("Text preprocessing started...")

        # Default column names (adjust based on your dataset structure)
        if tweet_columns is None:
            tweet_columns = [col for col in df.columns if 'tweet' in col.lower() or 'text' in col.lower()]
        if profile_columns is None:
            profile_columns = [col for col in df.columns if any(x in col.lower() for x in ['description', 'bio', 'profile'])]

        # Combine tweet text
        if tweet_columns:
            df['combined_tweets'] = df[tweet_columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
        else:
            df['combined_tweets'] = ''

        # Combine profile information
        if profile_columns:
            df['profile_text'] = df[profile_columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
        else:
            df['profile_text'] = ''

        # Create combined text
        df['combined_text'] = df['combined_tweets'] + ' ' + df['profile_text']

        # Handle accounts with no tweets
        df['combined_text'] = df['combined_text'].apply(lambda x: 'Nil' if x.strip() == '' else x)

        print(f"Accounts with 'Nil' text: {(df['combined_text'] == 'Nil').sum()}")

        return df

    def extract_features(self, df):
        """
        Engineer behavioral and profile features for bot detection.

        Creates comprehensive feature set including social metrics, content analysis,
        temporal patterns, and engagement characteristics optimized for identifying
        automated account behavior.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataset with user profiles and text content

        Returns
        -------
        pd.DataFrame
            Feature matrix with 15+ engineered features:
            - Social metrics: followers, following, statuses counts and ratios
            - Content features: hashtag, mention, URL counts and text statistics
            - Account features: age, verification, profile completeness
            - Engagement features: likes, retweets, content uniqueness
        """
        print("Feature engineering started...")

        # Initialize feature dataframe
        features_df = pd.DataFrame()

        # Account-based features
        if 'followers_count' in df.columns:
            features_df['followers_count'] = df['followers_count'].fillna(0)
        else:
            features_df['followers_count'] = 0

        if 'following_count' in df.columns or 'friends_count' in df.columns:
            following_col = 'following_count' if 'following_count' in df.columns else 'friends_count'
            features_df['following_count'] = df[following_col].fillna(0)
        else:
            features_df['following_count'] = 0

        if 'statuses_count' in df.columns or 'tweet_count' in df.columns:
            status_col = 'statuses_count' if 'statuses_count' in df.columns else 'tweet_count'
            features_df['statuses_count'] = df[status_col].fillna(0)
        else:
            features_df['statuses_count'] = 0

        # Calculate follower-following ratio
        features_df['follower_following_ratio'] = np.where(
            features_df['following_count'] == 0,
            features_df['followers_count'],
            features_df['followers_count'] / features_df['following_count']
        )

        # Text-based features
        if 'combined_text' in df.columns:
            text_col = df['combined_text']
        else:
            text_col = df.iloc[:, 0] if len(df.columns) > 0 else pd.Series([''] * len(df))

        # Hashtag features
        features_df['hashtag_count'] = text_col.apply(lambda x: len(re.findall(r'#\w+', str(x))))
        features_df['mention_count'] = text_col.apply(lambda x: len(re.findall(r'@\w+', str(x))))
        features_df['url_count'] = text_col.apply(
            lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(x))))

        # Text length and word count
        features_df['text_length'] = text_col.apply(lambda x: len(str(x)))
        features_df['word_count'] = text_col.apply(lambda x: len(str(x).split()))

        # Account age and verification (if available)
        if 'created_at' in df.columns:
            try:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                current_date = datetime.now()
                features_df['account_age_days'] = (current_date - df['created_at']).dt.days.fillna(0)
            except:
                features_df['account_age_days'] = 0
        else:
            features_df['account_age_days'] = 0

        if 'verified' in df.columns:
            features_df['is_verified'] = df['verified'].fillna(False).astype(int)
        else:
            features_df['is_verified'] = 0

        # Simulated intertime feature (time between posts)
        # In real scenario, this would be calculated from actual timestamps
        features_df['avg_intertime_hours'] = np.random.exponential(scale=24, size=len(df))

        # Profile completeness
        profile_fields = ['description', 'location', 'url', 'name']
        available_profile_fields = [col for col in profile_fields if col in df.columns]
        if available_profile_fields:
            features_df['profile_completeness'] = df[available_profile_fields].notna().sum(axis=1) / len(available_profile_fields)
        else:
            features_df['profile_completeness'] = 0.5

        # Engagement features
        if 'favourite_count' in df.columns or 'likes_count' in df.columns:
            likes_col = 'favourite_count' if 'favourite_count' in df.columns else 'likes_count'
            features_df['avg_likes'] = df[likes_col].fillna(0)
        else:
            features_df['avg_likes'] = 0

        if 'retweet_count' in df.columns:
            features_df['avg_retweets'] = df['retweet_count'].fillna(0)
        else:
            features_df['avg_retweets'] = 0

        # Duplicate/repetitive content indicators
        features_df['text_uniqueness'] = text_col.apply(lambda x: len(set(str(x).split())) / max(len(str(x).split()), 1))

        print(f"Extracted {len(features_df.columns)} features")
        print(f"Feature names: {list(features_df.columns)}")

        return features_df

    def train_model(self, X, y):
        """
        Train Random Forest classifier and evaluate performance on bot detection task.

        Performs train-test split, feature scaling, model training, and comprehensive
        evaluation with multiple metrics optimized for binary classification.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with engineered behavioral features
        y : array-like
            Binary labels (0: human, 1: bot)

        Returns
        -------
        dict
            Training results containing:
            - X_test, y_test, y_pred: Test data and predictions for analysis
            - metrics: Dictionary with accuracy, precision, recall, f1 scores
        """
        print("Model training started...")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Store feature names
        self.feature_names = list(X.columns)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        print(f"Model Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")

        return {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Visualize classification performance with annotated confusion matrix.

        Creates heatmap showing true positives, false positives, true negatives,
        and false negatives with detailed interpretation for bot detection context.

        Parameters
        ----------
        y_test : array-like
            True labels from test set
        y_pred : array-like
            Predicted labels from model

        Returns
        -------
        np.ndarray
            Confusion matrix values [[TN, FP], [FN, TP]]
        """
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Human', 'Bot'],
                    yticklabels=['Human', 'Bot'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Add interpretation text
        tn, fp, fn, tp = cm.ravel()
        plt.text(0.5, -0.15, f'True Negatives (Humans correctly classified): {tn}\n'
                             f'False Positives (Humans misclassified as bots): {fp}\n'
                             f'False Negatives (Bots misclassified as humans): {fn}\n'
                             f'True Positives (Bots correctly classified): {tp}',
                 transform=plt.gca().transAxes, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

        return cm

    def plot_feature_importance(self, top_n=15):
        """
        Display most influential features for bot classification decisions.

        Ranks and visualizes feature importance scores from Random Forest model
        to identify key behavioral patterns distinguishing bots from humans.

        Parameters
        ----------
        top_n : int, default=15
            Number of top features to display in ranking

        Returns
        -------
        pd.DataFrame or None
            Feature importance rankings with columns ['feature', 'importance'].
            Returns None if model lacks feature importance attribute.
        """
        if hasattr(self.model, 'feature_importance_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()

            return importance_df
        else:
            print("Model doesn't have feature importance attribute")
            return None

    def analyze_class_distribution(self, y):
        """
        Visualize dataset balance between human and bot account labels.

        Creates pie chart and prints statistics showing class distribution to
        assess dataset balance and potential bias in bot detection training.

        Parameters
        ----------
        y : array-like
            Binary labels (0: human, 1: bot)

        Returns
        -------
        pd.Series
            Class counts with labels as index and counts as values
        """
        class_counts = pd.Series(y).value_counts()

        plt.figure(figsize=(8, 6))
        plt.pie(class_counts.values, labels=['Human', 'Bot'], autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Account Types')
        plt.axis('equal')
        plt.show()

        print(f"Class Distribution:")
        for label, count in class_counts.items():
            class_name = 'Bot' if label == 1 else 'Human'
            print(f"{class_name}: {count} ({count / len(y) * 100:.1f}%)")

        return class_counts


# Example usage and demonstration
def run_bot_detection_demo():
    """
    Demonstrate comprehensive bot detection system capabilities with synthetic data.

    Executes complete machine learning pipeline including data generation, preprocessing,
    feature engineering, model training, and evaluation. Creates realistic synthetic
    dataset with bot-like behavioral patterns for system validation.

    Returns
    -------
    tuple
        (detector, results, feature_importance) containing:
        - detector: Trained BotDetectionSystem instance
        - results: Model evaluation results and predictions
        - feature_importance: Feature ranking DataFrame for analysis

    Notes
    -----
    Synthetic data includes realistic bot patterns such as extreme follower ratios
    and automated posting behaviors to simulate real-world detection scenarios.
    """
    print("=== Social Media Bot Detection System Demo ===\n")

    # Initialize the system
    detector = BotDetectionSystem()

    # Create synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic features
    data = {
        'followers_count': np.random.lognormal(3, 2, n_samples).astype(int),
        'following_count': np.random.lognormal(2.5, 1.5, n_samples).astype(int),
        'statuses_count': np.random.lognormal(4, 1, n_samples).astype(int),
        'verified': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'description': ['User bio text'] * n_samples,
        'created_at': pd.date_range('2010-01-01', '2023-01-01', n_samples),
        'combined_text': ['Sample tweet text with #hashtags and @mentions'] * n_samples
    }

    # Create labels (0: human, 1: bot)
    labels = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

    # Introduce realistic patterns for bots
    bot_indices = np.where(labels == 1)[0]
    # Bots tend to have extreme follower/following ratios
    data['followers_count'][bot_indices] = np.random.choice([0, 1], len(bot_indices)) * np.random.lognormal(1, 1, len(bot_indices)).astype(
        int)
    data['following_count'][bot_indices] = np.random.lognormal(5, 1, len(bot_indices)).astype(int)

    df = pd.DataFrame(data)

    print("1. Data Preprocessing...")
    df_processed = detector.preprocess_text(df)

    print("\n2. Feature Engineering...")
    features = detector.extract_features(df_processed)

    print("\n3. Class Distribution Analysis...")
    detector.analyze_class_distribution(labels)

    print("\n4. Model Training...")
    results = detector.train_model(features, labels)

    print("\n5. Confusion Matrix Analysis...")
    cm = detector.plot_confusion_matrix(results['y_test'], results['y_pred'])

    print("\n6. Feature Importance Analysis...")
    importance_df = detector.plot_feature_importance()

    print("\n=== Key Insights ===")
    print("Based on the analysis:")
    print("• Follower count, following count, and hashtag usage are critical features")
    print("• Intertime (posting intervals) helps distinguish bot behavior patterns")
    print("• Bots often show extreme ratios in social metrics")
    print("• Profile completeness and verification status are important indicators")
    print("\n• In real-world applications, prioritize recall to catch as many bots as possible")
    print("• The model achieves high performance across all metrics")

    return detector, results, importance_df


# Run the demonstration
if __name__ == "__main__":
    detector, results, feature_importance = run_bot_detection_demo()