#!/usr/bin/env python3
# Cresci-2017 Twitter Bot Dataset Cleaning Script

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

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
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "bots" / "cresci-2017" / "datasets_full.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "stats" / "bots"

ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
DATA = ROOT / "data" / "raw"
FIGS = ROOT / "figures"
STATS = ROOT / "data" / "stats"

# Create output directory if it doesn't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Define bot and genuine account datasets
BOT_DATASETS = [
    'fake_followers.csv',
    'social_spambots_1.csv',
    'social_spambots_2.csv',
    'social_spambots_3.csv',
    'traditional_spambots_1.csv',
    'traditional_spambots_2.csv',
    'traditional_spambots_3.csv',
    'traditional_spambots_4.csv'
]

GENUINE_DATASETS = [
    'genuine_accounts.csv'
]


def clean_user_data(df):
    """
    Clean and preprocess Twitter user profile data.

    Performs comprehensive data cleaning and feature engineering on Twitter
    user profile datasets, including date parsing, type conversion, missing
    value handling, and derived feature calculation for bot detection analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw user data DataFrame containing Twitter profile information
        with potential inconsistencies, missing values, and mixed data types.

    Returns
    -------
    df_cleaned : pandas.DataFrame
        Cleaned DataFrame with:
        - Standardized data types (datetime, numeric, categorical)
        - Removed columns with excessive missing values (>50%)
        - Engineered features for bot detection:
          * screen_name_length: Username length
          * has_description: Binary indicator for profile description
          * description_length: Profile bio character count
          * account_age_days: Account age in days from creation
    """
    logging.info(f"Cleaning user data: {df.shape[0]} records")

    # Convert empty strings to NaN
    df = df.replace('', np.nan)

    # Drop columns with too many missing values (threshold: 50%)
    missing_threshold = 0.5
    before_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=int(df.shape[0] * (1 - missing_threshold)))
    after_cols = df.shape[1]
    logging.info(f"Dropped {before_cols - after_cols} columns with more than {missing_threshold * 100}% missing values")

    # Twitter API uses these common date formats
    date_formats = [
        '%a %b %d %H:%M:%S %z %Y',  # "Wed May 04 23:30:37 +0000 2011"
        '%Y-%m-%d %H:%M:%S',  # "2011-05-05 01:30:37"
        '%Y-%m-%d'  # "2011-05-05"
    ]

    # Convert datetime columns
    date_cols = [col for col in df.columns if
                 'date' in col.lower() or 'time' in col.lower() or 'created_at' in col.lower()]
    for col in date_cols:
        if col in df.columns:
            # Try each format in order
            for date_format in date_formats:
                try:
                    logging.info(f"Converting {col} to datetime with format {date_format}")
                    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    # If successful (no errors), break the loop
                    break
                except Exception as e:
                    # If this format fails, try the next one
                    continue

            # If all formats fail, use the default parser as fallback
            if pd.api.types.is_object_dtype(df[col]):
                logging.warning(f"Could not parse {col} with specified formats, using default parser")
                df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert numeric columns
    numeric_cols = ['followers_count', 'friends_count', 'statuses_count', 'favourites_count',
                    'listed_count', 'default_profile', 'default_profile_image', 'verified']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Extract basic account features
    if 'screen_name' in df.columns:
        # Calculate username length
        df['screen_name_length'] = df['screen_name'].str.len()

    if 'description' in df.columns:
        # Calculate description length
        df['has_description'] = (~df['description'].isna()).astype(int)
        df['description_length'] = df['description'].str.len().fillna(0)

    # Calculate account age in days if created_at is available
    if 'created_at' in df.columns and pd.api.types.is_datetime64_dtype(df['created_at']):
        now = pd.Timestamp.now()
        df['account_age_days'] = (now - df['created_at']).dt.days

    return df


def clean_tweet_data(df):
    """
    Clean and preprocess Twitter tweet data.

    Performs comprehensive cleaning of tweet datasets including text feature
    extraction, date parsing, engagement metric conversion, and content
    analysis features for bot detection and social media analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw tweet data DataFrame containing Twitter posts with potential
        inconsistencies, mixed data types, and missing values.

    Returns
    -------
    df_cleaned : pandas.DataFrame
        Cleaned DataFrame with:
        - Standardized datetime and numeric columns
        - Removed columns with excessive missing values (>50%)
        - Extracted text features:
          * text_length: Character count of tweet content
          * hashtag_count: Number of hashtags (#) in tweet
          * mention_count: Number of user mentions (@) in tweet
          * url_count: Number of URLs in tweet content
    """
    logging.info(f"Cleaning tweet data: {df.shape[0]} records")

    # Convert empty strings to NaN
    df = df.replace('', np.nan)

    # Drop columns with too many missing values (threshold: 50%)
    missing_threshold = 0.5
    before_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=int(df.shape[0] * (1 - missing_threshold)))
    after_cols = df.shape[1]
    logging.info(f"Dropped {before_cols - after_cols} columns with more than {missing_threshold * 100}% missing values")

    # Twitter API uses these common date formats
    date_formats = [
        '%a %b %d %H:%M:%S %z %Y',  # "Wed May 04 23:30:37 +0000 2011"
        '%Y-%m-%d %H:%M:%S',  # "2011-05-05 01:30:37"
        '%Y-%m-%d'  # "2011-05-05"
    ]

    # Convert datetime columns
    date_cols = [col for col in df.columns if
                 'date' in col.lower() or 'time' in col.lower() or 'created_at' in col.lower()]
    for col in date_cols:
        if col in df.columns:
            # Try each format in order
            for date_format in date_formats:
                try:
                    logging.info(f"Converting {col} to datetime with format {date_format}")
                    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    # If successful (no errors), break the loop
                    break
                except Exception as e:
                    # If this format fails, try the next one
                    continue

            # If all formats fail, use the default parser as fallback
            if pd.api.types.is_object_dtype(df[col]):
                logging.warning(f"Could not parse {col} with specified formats, using default parser")
                df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert numeric columns
    numeric_cols = ['favorite_count', 'retweet_count', 'quoted_status_id']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Extract text features if text column exists
    if 'text' in df.columns:
        df['text_length'] = df['text'].str.len()
        df['hashtag_count'] = df['text'].str.count(r'#\w+')
        df['mention_count'] = df['text'].str.count(r'@\w+')
        df['url_count'] = df['text'].str.count(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    return df


def process_dataset(dataset_name, is_bot):
    """
    Process a single dataset folder containing users and tweets.

    Loads, cleans, and processes both user profile and tweet data from a
    specific dataset folder, handling CSV parsing errors and missing files
    gracefully while maintaining data integrity and provenance tracking.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset folder to process
    is_bot : bool
        Label indicating whether this dataset contains bot accounts (True)
        or genuine human accounts (False).

    Returns
    -------
    result : dict
        Dictionary containing processed data with keys:
        - 'users': pandas.DataFrame with cleaned user profile data
        - 'tweets': pandas.DataFrame with cleaned tweet data
        Missing datasets (users or tweets) are omitted from result.
    """
    dataset_path = RAW_DATA_DIR / dataset_name
    label = 1 if is_bot else 0  # 1 for bot, 0 for human

    logging.info(f"Processing dataset: {dataset_name} (bot={is_bot})")

    result = {}

    # Process users.csv
    users_file = dataset_path / 'users.csv'
    if os.path.exists(users_file):
        try:
            # Handle CSV with no headers by providing column names
            user_columns = [
                'id_str', 'screen_name', 'name', 'followers_count', 'friends_count',
                'statuses_count', 'favourites_count', 'listed_count', 'url', 'lang',
                'time_zone', 'location', 'default_profile', 'default_profile_image',
                'geo_enabled', 'profile_image_url', 'profile_banner_url', 'profile_use_background_image',
                'profile_background_image_url_https', 'profile_text_color', 'profile_image_url_https',
                'profile_sidebar_border_color', 'profile_background_tile', 'profile_sidebar_fill_color',
                'profile_background_image_url', 'profile_background_color', 'profile_link_color',
                'utc_offset', 'is_translator', 'follow_request_sent', 'protected',
                'verified', 'notifications', 'description', 'contributors_enabled',
                'following', 'created_at', 'timestamp', 'crawled_at', 'updated'
            ]

            # Read users file
            users_df = pd.read_csv(users_file, header=None, names=user_columns,
                                   quotechar='"', escapechar='\\',
                                   on_bad_lines='warn', low_memory=False)

            # Add dataset information
            users_df['dataset'] = dataset_name
            users_df['is_bot'] = is_bot

            # Clean user data
            users_df = clean_user_data(users_df)

            result['users'] = users_df

        except Exception as e:
            logging.error(f"Error processing users in {dataset_name}: {str(e)}")
    else:
        logging.warning(f"Users file not found in {dataset_name}")

    # Process tweets.csv
    tweets_file = dataset_path / 'tweets.csv'
    if os.path.exists(tweets_file):
        # Define column names based on Twitter API tweet object schema
        tweet_columns = [
            'created_at', 'id', 'id_str', 'text', 'source', 'truncated',
            'in_reply_to_status_id', 'in_reply_to_status_id_str',
            'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name',
            'user_id', 'geo', 'coordinates', 'place', 'contributors', 'is_quote_status',
            'retweet_count', 'favorite_count', 'favorited', 'retweeted', 'lang',
            'timestamp', 'crawled_at'
        ]

        # Read tweets file with the current encoding
        tweets_df = pd.read_csv(tweets_file,
                                header=None,
                                names=tweet_columns,
                                quotechar='"',
                                encoding='utf-8',
                                escapechar='\\',
                                on_bad_lines='warn',
                                low_memory=False, )

        # Add dataset information
        tweets_df['dataset'] = dataset_name
        tweets_df['is_bot'] = is_bot

        # Clean tweet data
        tweets_df = clean_tweet_data(tweets_df)

        result['tweets'] = tweets_df

    if 'tweets' not in result:
        logging.error(f"All encoding attempts failed for tweets in {dataset_name}")
    else:
        logging.warning(f"Tweets file not found in {dataset_name}")

    return result


def main():
    """
    Main processing function for comprehensive Twitbot-20 dataset cleaning.

    Orchestrates the complete data processing pipeline from raw Twitbot-20
    datasets to cleaned, feature-engineered datasets ready for machine learning.
    Processes multiple bot and human account datasets, combines them, and creates
    comprehensive feature sets for bot detection research.

    Returns
    -------
    None
    """
    logging.info(f"Starting Twitbot-20 dataset cleaning process")

    all_users_dfs = []
    all_tweets_dfs = []
    dataset_results = {}

    # Process bot datasets
    for dataset in BOT_DATASETS:
        result = process_dataset(dataset, is_bot=True)
        dataset_results[dataset] = result
        if 'users' in result:
            all_users_dfs.append(result['users'])
        if 'tweets' in result:
            all_tweets_dfs.append(result['tweets'])

    # Process genuine account datasets
    for dataset in GENUINE_DATASETS:
        result = process_dataset(dataset, is_bot=False)
        dataset_results[dataset] = result
        if 'users' in result:
            all_users_dfs.append(result['users'])
        if 'tweets' in result:
            all_tweets_dfs.append(result['tweets'])

    # Combine all user dataframes
    if all_users_dfs:
        combined_users_df = pd.concat(all_users_dfs, ignore_index=True)
        logging.info(f"Combined users dataset shape: {combined_users_df.shape}")

        # Save separate user dataset
        users_output_path = PROCESSED_DATA_DIR / 'cresci2017_users_cleaned.csv'
        combined_users_df.to_csv(users_output_path, index=False)
        logging.info(f"Saved cleaned users data to {users_output_path}")

        # Save statistics
        bot_counts = combined_users_df['is_bot'].value_counts()
        logging.info(f"User class distribution: Bots={bot_counts.get(1, 0)}, Humans={bot_counts.get(0, 0)}")
    else:
        logging.warning("No user data was processed successfully")
        return

    # Combine all tweet dataframes
    combined_tweets_df = None
    if all_tweets_dfs:
        combined_tweets_df = pd.concat(all_tweets_dfs, ignore_index=True)
        logging.info(f"Combined tweets dataset shape: {combined_tweets_df.shape}")

        # Save separate tweet dataset (for reference)
        tweets_output_path = PROCESSED_DATA_DIR / 'cresci2017_tweets_cleaned.csv'
        combined_tweets_df.to_csv(tweets_output_path, index=False)
        logging.info(f"Saved cleaned tweets data to {tweets_output_path}")

        # Save statistics
        bot_counts = combined_tweets_df['is_bot'].value_counts()
        logging.info(f"Tweet class distribution: Bots={bot_counts.get(1, 0)}, Humans={bot_counts.get(0, 0)}")
    else:
        logging.warning("No tweet data was processed successfully")

    # Now create a comprehensive dataset combining user profiles with their tweets
    logging.info("Creating comprehensive dataset combining user profiles with tweet information...")

    # Group tweets by user_id and compute aggregate tweet features
    tweet_features = {}

    if combined_tweets_df is not None:
        logging.info("Computing tweet aggregates per user...")

        # Ensure user_id is string type for consistency
        if 'user_id' in combined_tweets_df.columns:
            combined_tweets_df['user_id'] = combined_tweets_df['user_id'].astype(str)

        # Group by user_id and compute aggregate features
        tweet_groups = combined_tweets_df.groupby('user_id')

        for user_id, group in tweet_groups:
            # Basic count statistics
            tweet_count = len(group)
            avg_text_length = group['text_length'].mean() if 'text_length' in group.columns else 0
            avg_hashtags = group['hashtag_count'].mean() if 'hashtag_count' in group.columns else 0
            avg_mentions = group['mention_count'].mean() if 'mention_count' in group.columns else 0
            avg_urls = group['url_count'].mean() if 'url_count' in group.columns else 0

            # Timing features
            time_diffs = []

            if 'created_at' in group.columns and pd.api.types.is_datetime64_dtype(group['created_at']):
                sorted_times = sorted(group['created_at'].dropna())

                # Calculate time differences between consecutive tweets
                if len(sorted_times) > 1:
                    for i in range(1, len(sorted_times)):
                        diff_seconds = (sorted_times[i] - sorted_times[i - 1]).total_seconds()
                        time_diffs.append(diff_seconds)

            # Calculate time-based features
            avg_seconds_between_tweets = np.mean(time_diffs) if time_diffs else np.nan
            std_seconds_between_tweets = np.std(time_diffs) if len(time_diffs) > 1 else np.nan

            # Check for tweets posted at exact same second (suspicious bot behavior)
            exact_same_times = 0
            if time_diffs:
                exact_same_times = sum(1 for diff in time_diffs if diff == 0)

            # Store features for this user
            tweet_features[user_id] = {
                'tweet_count': tweet_count,
                'avg_text_length': avg_text_length,
                'avg_hashtags': avg_hashtags,
                'avg_mentions': avg_mentions,
                'avg_urls': avg_urls,
                'avg_seconds_between_tweets': avg_seconds_between_tweets,
                'std_seconds_between_tweets': std_seconds_between_tweets,
                'exact_same_time_tweets': exact_same_times,
                'has_tweets': 1  # Flag indicating this user has tweets
            }

    # Now combine user profiles with their tweet features
    logging.info("Merging user profiles with tweet features...")

    # Ensure id_str is string type for consistency
    combined_users_df['id_str'] = combined_users_df['id_str'].astype(str)

    # Create tweet features dataframe
    if tweet_features:
        tweet_features_df = pd.DataFrame.from_dict(tweet_features, orient='index').reset_index()
        tweet_features_df.rename(columns={'index': 'id_str'}, inplace=True)

        # Merge user profiles with tweet features
        combined_df = pd.merge(combined_users_df, tweet_features_df, on='id_str', how='left')

        # Fill NaN values for users with no tweets
        combined_df['has_tweets'].fillna(0, inplace=True)
        combined_df['tweet_count'].fillna(0, inplace=True)

        # Mark accounts with no tweets as "Nil" for text features
        text_feature_cols = ['avg_text_length', 'avg_hashtags', 'avg_mentions', 'avg_urls',
                             'avg_seconds_between_tweets', 'std_seconds_between_tweets',
                             'exact_same_time_tweets']

        for col in text_feature_cols:
            if col in combined_df.columns:
                combined_df[col].fillna('Nil', inplace=True)
    else:
        # If no tweet data at all, add basic columns
        combined_df = combined_users_df.copy()
        combined_df['has_tweets'] = 0
        combined_df['tweet_count'] = 0
        combined_df['tweet_status'] = 'Nil'

    # Save the comprehensive dataset
    comprehensive_output_path = PROCESSED_DATA_DIR / 'cresci2017_comprehensive.csv'
    combined_df.to_csv(comprehensive_output_path, index=False)
    logging.info(f"Saved comprehensive dataset to {comprehensive_output_path}")

    # Report comprehensive statistics
    bot_counts_comprehensive = combined_df['is_bot'].value_counts()
    logging.info("=== COMPREHENSIVE DATASET STATISTICS ===")
    logging.info(f"Total accounts: {len(combined_df)}")
    logging.info(
        f"Bot accounts: {bot_counts_comprehensive.get(1, 0)} ({bot_counts_comprehensive.get(1, 0) / len(combined_df) * 100:.1f}%)")
    logging.info(
        f"Human accounts: {bot_counts_comprehensive.get(0, 0)} ({bot_counts_comprehensive.get(0, 0) / len(combined_df) * 100:.1f}%)")

    accounts_with_tweets = combined_df['has_tweets'].sum()
    accounts_without_tweets = len(combined_df) - accounts_with_tweets
    logging.info(f"Accounts with tweets: {accounts_with_tweets} ({accounts_with_tweets / len(combined_df) * 100:.1f}%)")
    logging.info(
        f"Accounts without tweets (Nil): {accounts_without_tweets} ({accounts_without_tweets / len(combined_df) * 100:.1f}%)")
    logging.info("=====================================")


if __name__ == "__main__":
    main()
