#!/usr/bin/env python3
# News Classifier App
# This app uses multiple trained models to classify news as real or fake

import os
import logging
import colorama
from colorama import Fore, Style
from ClassificationModel.src.shared_logic import NewsClassifier
from page_rank import extract_domain


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


def print_header():
    """Print the application header with content type selection"""
    print("\n" + "="*50)
    print(f"{Fore.CYAN}=== Content Classification System ==={Style.RESET_ALL}")
    print("="*50)
    print(f"Please select content type to analyze:")
    print(f"1. {Fore.YELLOW}News Article{Style.RESET_ALL}")
    print(f"2. {Fore.YELLOW}Tweet{Style.RESET_ALL}")
    print(f"Type {Fore.YELLOW}exit{Style.RESET_ALL} to quit")
    print("="*50)

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

    print("\n" + "="*50)

def print_bot_result(result):
    """Print the bot detection result with nice formatting"""
    print("\n" + "="*50)
    print(f"{Fore.YELLOW}=== Bot Detection Result ==={Style.RESET_ALL}")
    print("="*50)

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

    print("\n" + "="*50)

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
            print(f"{Fore.RED}Invalid choice. Please enter 1 for News Article, 2 for Tweet, or 'exit' to quit.{Style.RESET_ALL}")

        # Visual separation between iterations
        print("\n\n")

def process_article(classifier):
    """Process a news article for classification"""
    print("\n" + "="*50)
    print(f"{Fore.CYAN}=== News Article Classification ==={Style.RESET_ALL}")
    print("="*50)
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

    else:
        print(f"{Fore.RED}Error: Could not classify the article. Please try again.{Style.RESET_ALL}")

def process_tweet(classifier):
    """Process a tweet for bot detection"""
    print("\n" + "="*50)
    print(f"{Fore.CYAN}=== Tweet Analysis ==={Style.RESET_ALL}")
    print("="*50)
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

    else:
        print(f"{Fore.RED}Error: Could not perform bot detection analysis. Please try again.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()