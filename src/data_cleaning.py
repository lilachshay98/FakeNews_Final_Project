import pandas as pd
import string
import os
import logging
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import ssl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def clean_text(text):
    """
    Perform basic text cleaning operations.

    Applies fundamental text preprocessing including lowercasing,
    punctuation removal, and basic whitespace normalization.

    Parameters
    ----------
    text : str
        Input text to clean.

    Returns
    -------
    cleaned_text : str
        Preprocessed text with lowercase conversion, punctuation removal,
        and normalized whitespace.

    Notes
    -----
    This is a simplified cleaning function. For more comprehensive
    preprocessing, use process_text() instead.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = text.split()
    return ' '.join(words)


def load_and_clean_data(raw_data_path):
    """
    Load and preprocess fake news datasets.

    Reads the Fake.csv and True.csv files, standardizes column formats,
    removes unnecessary columns, and combines into a single dataset
    with appropriate labels.

    Parameters
    ----------
    raw_data_path : str
        Path to directory containing 'Fake.csv' and 'True.csv' files.

    Returns
    -------
    combined : pandas.DataFrame
        Combined dataset with columns:
        - 'text': article text content
        - 'label': binary label (0=fake, 1=true)
        Duplicate rows and null text entries are removed.

    Notes
    -----
    The function expects CSV files with standard fake news dataset
    structure including 'title', 'text', 'subject', and 'date' columns.
    Only the 'text' column is retained for analysis.
    """
    logging.info(f'Loading datasets from {raw_data_path}...')
    fake = pd.read_csv(os.path.join(raw_data_path, 'Fake.csv'))
    true = pd.read_csv(os.path.join(raw_data_path, 'True.csv'))

    logging.info('Standardizing columns...')
    fake['label'] = 0
    true['label'] = 1

    logging.info('Cleaning text...')
    fake.drop(columns=["title", "date", "subject"], inplace=True)
    true.drop(columns=["title", "date", "subject"], inplace=True)

    # Drop any unnamed columns in all dataframes
    fake = fake.loc[:, ~fake.columns.str.contains('^Unnamed')]
    true = true.loc[:, ~true.columns.str.contains('^Unnamed')]

    fake.head()
    true.head()

    logging.info('Combining datasets...')
    combined = pd.concat([fake, true], ignore_index=True)

    # Remove rows where text is null
    null_count = combined['text'].isna().sum()
    if null_count > 0:
        logging.info(f'Dropping {null_count} rows with null text values')
        combined = combined.dropna(subset=['text'])

    combined.info()
    combined.drop_duplicates(inplace=True)
    return combined


def save_cleaned_data(df, out_path):
    """
    Save cleaned dataset to CSV file.

    Writes the processed DataFrame to a CSV file with appropriate
    formatting for machine learning workflows.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset to save.
    out_path : str
        Output file path for the saved CSV file.

    Returns
    -------
    None
        Saves DataFrame to specified path without row indices.
    """
    logging.info(f'Saving cleaned data to {out_path}...')
    df.to_csv(out_path, index=False)


def download_nltk_resources():
    """
    Download required NLTK data packages for text processing.

    Downloads essential NLTK resources including tokenizers,
    stopwords, lemmatizer dictionaries, and multilingual support.

    Returns
    -------
    success : bool
        True if all NLTK resources were downloaded successfully,
        False if any download failed.

    Notes
    -----
    Downloads the following NLTK packages:
    - punkt_tab: Tokenization models
    - stopwords: Stopword lists for multiple languages
    - wordnet: WordNet lexical database for lemmatization
    - omw-1.4: Open Multilingual WordNet data
    """
    logging.info('Downloading NLTK resources...')

    try:
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        logging.info('NLTK resources downloaded successfully')
        return True
    except Exception as e:
        logging.error(f'Error downloading NLTK resources: {e}')
        return False


def process_text(text):
    """
    Apply comprehensive text preprocessing for NLP analysis.

    Performs advanced text cleaning including normalization, special
    character removal, tokenization, lemmatization, stopword removal,
    and deduplication for optimal NLP model performance.

    Parameters
    ----------
    text : str or other
        Input text to process. Non-string inputs return empty list.

    Returns
    -------
    cleaned_text : list[str]
        List of processed tokens with:
        - Normalized whitespace and special characters removed
        - Lemmatized to base word forms
        - Stopwords filtered out
        - Short words (≤3 characters) removed
        - Duplicates removed while preserving order
        Returns empty list for invalid inputs.

    Notes
    -----
    Processing pipeline:
    1. Whitespace normalization using regex
    2. Special character removal (non-word characters)
    3. Single character removal between whitespace
    4. Non-alphabetical character filtering
    5. Lowercasing
    6. Tokenization using NLTK word_tokenize
    7. Lemmatization using WordNetLemmatizer
    8. English stopword removal
    9. Short word filtering (length > 3)
    10. Duplicate removal with order preservation
    """
    if pd.isna(text) or not isinstance(text, str):
        return []

    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Remove extra white space from text

    text = re.sub(r'\W', ' ', str(text))  # Remove all the special characters from text

    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove all single characters from text

    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove any character that isn't alphabetical

    text = text.lower()

    words = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    stop_words = set(stopwords.words("english"))
    Words = [word for word in words if word not in stop_words]

    Words = [word for word in Words if len(word) > 3]

    indices = np.unique(Words, return_index=True)[1]
    cleaned_text = np.array(Words)[np.sort(indices)].tolist()

    return cleaned_text


def get_cleaned_data(combined_data):
    """
    Process entire dataset through comprehensive text cleaning pipeline.

    Applies advanced text preprocessing to all documents in the dataset,
    filtering out invalid entries and creating a clean dataset ready
    for machine learning model training.

    Parameters
    ----------
    combined_data : pandas.DataFrame
        Dataset with 'text' and 'label' columns containing raw text data.

    Returns
    -------
    result_df : pandas.DataFrame
        Processed dataset with columns:
        - 'cleaned_text': str, space-separated cleaned tokens
        - 'label': int, original labels for successfully processed texts
        Only includes documents that produced non-empty processed results.

    Notes
    -----
    Processing steps:
    1. Separate features (text) and labels
    2. Applies process_text() to each document
    3. Filters out documents that produce empty results
    4. Maintains label alignment with processed texts
    5. Converts token lists back to strings for compatibility

    The function logs the number of successfully processed documents
    versus the original count for quality assessment.
    """
    logging.info('Processing text data...')
    x = combined_data.drop('label', axis=1)
    y = combined_data.label
    texts = list(x['text'])

    logging.info('Applying advanced text cleaning to each document...')

    # Create a list to store both cleaned texts and corresponding valid indices
    cleaned_texts = []
    valid_indices = []

    for i, text in enumerate(texts):
        if text and isinstance(text, str) and not pd.isna(text):
            processed = process_text(text)
            if processed:  # Only include non-empty processed results
                cleaned_texts.append(processed)
                valid_indices.append(i)

    # Use only labels from valid indices
    valid_labels = y.iloc[valid_indices].values

    # Convert list of word lists to strings for saving
    cleaned_text_strings = [' '.join(words) for words in cleaned_texts]

    # Create new DataFrame with cleaned text and matching labels
    result_df = pd.DataFrame({
        'cleaned_text': cleaned_text_strings,
        'label': valid_labels
    })

    logging.info(f'Processed {len(result_df)} documents out of {len(texts)} original documents')
    return result_df


if __name__ == '__main__':
    # Download NLTK resources first
    download_nltk_resources()

    # Define correct paths
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Get ClassificationModel directory
    raw_data_path = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'stats')

    # Create processed directory if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    out_path = os.path.join(processed_dir, 'cleaned_combined.csv')

    logging.info(f'Raw data path: {raw_data_path}')
    logging.info(f'Output path: {out_path}')
    logging.info('Starting data cleaning process...')

    # Load and clean the data
    df = load_and_clean_data(raw_data_path)

    # Process text and get DataFrame with cleaned text and labels
    cleaned_df = get_cleaned_data(df)

    # Save the cleaned data (includes both cleaned text and labels)
    save_cleaned_data(cleaned_df, out_path)

    logging.info(f'Cleaned data saved to {out_path}')
