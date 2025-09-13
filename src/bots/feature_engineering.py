#!/usr/bin/env python3
# Cresci-2017 Twitter Bot Dataset Feature Engineering

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
import re

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
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed" / "bots"


def load_cresci_comprehensive_dataset(file_path=None):
    """
    Load the Cresci 2017 comprehensive dataset into a pandas DataFrame

    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the CSV file. If None, uses the default path.

    Returns:
    --------
    pd.DataFrame
        The loaded dataset
    """
    if file_path is None:
        file_path = PROCESSED_DATA_DIR / "cresci2017_comprehensive.csv"

    logging.info(f"Loading Cresci 2017 comprehensive dataset from {file_path}...")

    try:
        # Using chunksize for large files to avoid memory issues
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=10000):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)

        logging.info(f"Successfully loaded dataset with shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        logging.error(f"Error loading Cresci dataset: {str(e)}")
        return None


def load_tweets_and_users_data():
    """
    Load both tweets and users data for feature engineering

    Returns:
    --------
    tuple
        (tweets_df, users_df) DataFrames containing the processed tweet and user data
    """
    tweets_path = PROCESSED_DATA_DIR / "cresci2017_tweets_cleaned.csv"
    users_path = PROCESSED_DATA_DIR / "cresci2017_users_cleaned.csv"

    try:
        # Load tweets data
        logging.info(f"Loading tweets data from {tweets_path}")
        tweets_df = pd.read_csv(tweets_path)

        # Load users data
        logging.info(f"Loading users data from {users_path}")
        users_df = pd.read_csv(users_path)

        logging.info(f"Loaded {len(tweets_df)} tweets and {len(users_df)} user profiles")

        return tweets_df, users_df

    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None


def parse_entities_field(tweets_df):
    """
    Parse Twitter entities field to extract hashtags, urls, and mentions

    Parameters:
    -----------
    tweets_df : pd.DataFrame
        DataFrame containing tweet data with 'entities' field

    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns for hashtags, urls, and mentions
    """
    logging.info("Parsing tweet entities to extract hashtags, URLs, and mentions")

    # Initialize empty columns
    tweets_df['hashtags'] = None
    tweets_df['urls'] = None
    tweets_df['mentions'] = None

    # Process only if entities column exists
    if 'entities' in tweets_df.columns:
        # Function to extract hashtags from entities JSON
        def extract_hashtags(entities):
            try:
                if pd.isna(entities):
                    return []

                if isinstance(entities, str):
                    entities_dict = json.loads(entities)
                else:
                    entities_dict = entities

                if 'hashtags' in entities_dict and entities_dict['hashtags']:
                    return [tag['text'] for tag in entities_dict['hashtags']]
                return []
            except:
                return []

        # Function to extract URLs from entities JSON
        def extract_urls(entities):
            try:
                if pd.isna(entities):
                    return []

                if isinstance(entities, str):
                    entities_dict = json.loads(entities)
                else:
                    entities_dict = entities

                if 'urls' in entities_dict and entities_dict['urls']:
                    return [url['expanded_url'] for url in entities_dict['urls']]
                return []
            except:
                return []

        # Function to extract mentions from entities JSON
        def extract_mentions(entities):
            try:
                if pd.isna(entities):
                    return []

                if isinstance(entities, str):
                    entities_dict = json.loads(entities)
                else:
                    entities_dict = entities

                if 'user_mentions' in entities_dict and entities_dict['user_mentions']:
                    return [mention['screen_name'] for mention in entities_dict['user_mentions']]
                return []
            except:
                return []

        # Apply extraction functions
        tweets_df['hashtags'] = tweets_df['entities'].apply(extract_hashtags)
        tweets_df['urls'] = tweets_df['entities'].apply(extract_urls)
        tweets_df['mentions'] = tweets_df['entities'].apply(extract_mentions)

    return tweets_df


def extract_features(tweets_df, users_df):
    """
    Extract features from tweets and user data as specified

    Parameters:
    -----------
    tweets_df : pd.DataFrame
        DataFrame containing tweet data
    users_df : pd.DataFrame
        DataFrame containing user profile data

    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    logging.info("Extracting features from tweets and user data...")

    # Check and identify user identifier column - could be 'id', 'user_id', or something else
    logging.info(f"Users dataframe columns: {users_df.columns.tolist()}")

    # Identify the primary user identifier column
    user_id_column = None
    possible_id_columns = ['id', 'user_id', 'id_str']
    for col in possible_id_columns:
        if col in users_df.columns:
            user_id_column = col
            logging.info(f"Found user identifier column: '{col}'")
            break

    # If no standard ID column found, try to identify by column name pattern
    if user_id_column is None:
        for col in users_df.columns:
            if 'id' in col.lower():
                user_id_column = col
                logging.info(f"Using column '{col}' as user identifier")
                break

    # If still no ID column, use the first column as a fallback
    if user_id_column is None:
        user_id_column = users_df.columns[0]
        logging.info(f"No clear ID column found, using first column '{user_id_column}' as identifier")

    # Check tweet user identifier
    tweet_user_id_column = 'user_id'
    if 'user_id' not in tweets_df.columns:
        # Try to find an alternative
        for col in tweets_df.columns:
            if 'user' in col.lower() and 'id' in col.lower():
                tweet_user_id_column = col
                logging.info(f"Using column '{col}' as tweet user identifier")
                break

    # Ensure user_id columns are strings for reliable matching
    tweets_df[tweet_user_id_column] = tweets_df[tweet_user_id_column].astype(str)
    users_df[user_id_column] = users_df[user_id_column].astype(str)

    # Parse entities if not already done
    if 'hashtags' not in tweets_df.columns:
        tweets_df = parse_entities_field(tweets_df)

    # Initialize feature dataframe from users_df
    features_df = users_df.copy()

    # Rename the ID column to user_id for consistency
    features_df = features_df.rename(columns={user_id_column: 'user_id'})

    # Initialize feature columns
    feature_columns = [
        'retweets', 'replies', 'favoriteC', 'hashtag', 'url', 'mentions',
        'intertime', 'ffratio', 'favorites', 'listed',
        'uniqueHashtags', 'uniqueMentions', 'uniqueURL'
    ]

    for col in feature_columns:
        features_df[col] = 0.0

    # Group tweets by user_id
    user_groups = tweets_df.groupby(tweet_user_id_column)

    # Mapping from tweet's user_id format to features_df user_id format
    tweet_users = set(tweets_df[tweet_user_id_column].unique())
    feature_users = set(features_df['user_id'].unique())
    logging.info(f"Found {len(tweet_users)} unique users in tweets and {len(feature_users)} in user profiles")

    # Iterate through each user and calculate features
    user_count = 0
    for user_id, user_tweets in user_groups:
        if user_id not in features_df['user_id'].values:
            continue

        user_count += 1
        if user_count % 1000 == 0:
            logging.info(f"Processed {user_count} users...")

        tweet_count = len(user_tweets)
        if tweet_count == 0:
            continue

        # 1. retweets: ratio between retweet count and tweet count
        retweet_count = user_tweets['retweet_count'].sum() if 'retweet_count' in user_tweets.columns else 0
        retweet_ratio = retweet_count / tweet_count

        # 2. replies: ratio between reply count and tweet count
        reply_count = user_tweets[
            'in_reply_to_status_id'].notna().sum() if 'in_reply_to_status_id' in user_tweets.columns else 0
        reply_ratio = reply_count / tweet_count

        # 3. favoriteC: ratio between favorited tweets and tweet count
        favorite_count = user_tweets['favorite_count'].sum() if 'favorite_count' in user_tweets.columns else 0
        favorite_ratio = favorite_count / tweet_count

        # 4. hashtag: ratio between hashtag count and tweet count
        hashtag_count = user_tweets['hashtags'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        hashtag_ratio = hashtag_count / tweet_count

        # 5. url: ratio between url count and tweet count
        url_count = user_tweets['urls'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        url_ratio = url_count / tweet_count

        # 6. mentions: ratio between mention count and tweet count
        mention_count = user_tweets['mentions'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        mention_ratio = mention_count / tweet_count

        # 7. intertime: average seconds between postings
        if 'created_at' in user_tweets.columns and len(user_tweets) > 1:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_dtype(user_tweets['created_at']):
                    user_tweets['created_at'] = pd.to_datetime(user_tweets['created_at'], errors='coerce')

                # Sort by creation time
                sorted_tweets = user_tweets.sort_values('created_at')

                # Calculate time differences in seconds
                time_diffs = sorted_tweets['created_at'].diff().dt.total_seconds()

                # Average time difference (excluding the first which will be NaN)
                avg_time_diff = time_diffs.iloc[1:].mean()

                # Handle NaN (e.g., if all tweets at the same time)
                intertime = avg_time_diff if not pd.isna(avg_time_diff) else 0
            except Exception as e:
                logging.warning(f"Could not calculate intertime for user {user_id}: {str(e)}")
                intertime = 0
        else:
            intertime = 0

        # Safely get user attributes
        def get_user_attr(attr_name, default=0):
            if attr_name in features_df.columns:
                val = features_df.loc[features_df['user_id'] == user_id, attr_name].values
                return val[0] if len(val) > 0 else default
            return default

        # 8. ffratio: friends-to-followers ratio
        friends_count = get_user_attr('friends_count')
        followers_count = get_user_attr('followers_count')

        # Avoid division by zero
        ffratio = friends_count / followers_count if followers_count > 0 else friends_count

        # 9. favorites: total number of tweets favorited by this account
        favorites = get_user_attr('favourites_count')
        if favorites == 0:  # Try alternate column name
            favorites = get_user_attr('favorites_count')

        # 10. listed: number of times the account has been listed
        listed = get_user_attr('listed_count')

        # 11. uniqueHashtags: ratio between unique hashtag count and tweet count
        all_hashtags = []
        for hashtags in user_tweets['hashtags']:
            if isinstance(hashtags, list):
                all_hashtags.extend(hashtags)
        unique_hashtags_count = len(set(all_hashtags))
        unique_hashtags_ratio = unique_hashtags_count / tweet_count

        # 12. uniqueMentions: ratio between unique mention count and tweet count
        all_mentions = []
        for mentions in user_tweets['mentions']:
            if isinstance(mentions, list):
                all_mentions.extend(mentions)
        unique_mentions_count = len(set(all_mentions))
        unique_mentions_ratio = unique_mentions_count / tweet_count

        # 13. uniqueURL: ratio between unique urls count and tweet count
        all_urls = []
        for urls in user_tweets['urls']:
            if isinstance(urls, list):
                all_urls.extend(urls)
        unique_urls_count = len(set(all_urls))
        unique_urls_ratio = unique_urls_count / tweet_count

        # Update features dataframe
        idx = features_df.index[features_df['user_id'] == user_id].tolist()
        if idx:
            features_df.loc[idx[0], 'retweets'] = retweet_ratio
            features_df.loc[idx[0], 'replies'] = reply_ratio
            features_df.loc[idx[0], 'favoriteC'] = favorite_ratio
            features_df.loc[idx[0], 'hashtag'] = hashtag_ratio
            features_df.loc[idx[0], 'url'] = url_ratio
            features_df.loc[idx[0], 'mentions'] = mention_ratio
            features_df.loc[idx[0], 'intertime'] = intertime
            features_df.loc[idx[0], 'ffratio'] = ffratio
            features_df.loc[idx[0], 'favorites'] = favorites
            features_df.loc[idx[0], 'listed'] = listed
            features_df.loc[idx[0], 'uniqueHashtags'] = unique_hashtags_ratio
            features_df.loc[idx[0], 'uniqueMentions'] = unique_mentions_ratio
            features_df.loc[idx[0], 'uniqueURL'] = unique_urls_ratio

    return features_df


def save_features(features_df, output_path=None):
    """
    Save the extracted features to a CSV file

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame containing the extracted features
    output_path : str or Path, optional
        Path to save the features CSV. If None, uses the default path.

    Returns:
    --------
    str
        Path to the saved CSV file
    """
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "cresci2017_features.csv"

    features_df.to_csv(output_path, index=False)
    logging.info(f"Features saved to {output_path}")
    return output_path


def main():
    """Main function to run the feature engineering pipeline"""
    logging.info("Starting feature engineering process for Cresci 2017 dataset")

    # Load tweets and users data
    tweets_df, users_df = load_tweets_and_users_data()

    if tweets_df is None or users_df is None:
        logging.error("Could not load required data files. Exiting.")
        return

    # Extract features
    features_df = extract_features(tweets_df, users_df)

    # Save features
    save_features(features_df)

    logging.info("Feature engineering process completed")


if __name__ == "__main__":
    main()
