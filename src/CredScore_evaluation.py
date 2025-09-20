import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from shared_logic import NewsClassifier
from pathlib import Path
import seaborn as sns

# adjusting file paths
ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
DATA = ROOT / "data" / "raw"
FIGS = ROOT / "figures"


class ConfusionMatrixMetrics:
    """
    A class for computing and visualizing classification performance metrics.

    Provides comprehensive evaluation metrics and visualization capabilities
    for binary classification tasks, with automatic normalization of confusion
    matrices and professional-quality plotting functionality.

    Parameters
    ----------
    tp : int
        True positives - correctly predicted positive instances.
    fp : int
        False positives - incorrectly predicted positive instances.
    fn : int
        False negatives - incorrectly predicted negative instances.
    tn : int
        True negatives - correctly predicted negative instances.
    """
    def __init__(self, tp, fp, fn, tn):
        """
        Initialize ConfusionMatrixMetrics with confusion matrix components.

        Parameters
        ----------
        tp : int
            True positives count.
        fp : int
            False positives count.
        fn : int
            False negatives count.
        tn : int
            True negatives count.
        """
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def accuracy(self):
        """
        Calculate classification accuracy.

        Computes the proportion of correct predictions among all predictions.

        Returns
        -------
        accuracy : float
            Accuracy score in range [0, 1], where 1 represents perfect accuracy.
            Formula: (TP + TN) / (TP + FP + FN + TN)
        """
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

    def precision(self):
        """
        Calculate precision (positive predictive value).

        Measures the proportion of positive predictions that were actually correct.
        Indicates how reliable positive predictions are.

        Returns
        -------
        precision : float
            Precision score in range [0, 1], where 1 represents perfect precision.
            Formula: TP / (TP + FP). Returns 0 if no positive predictions made.
        """
        if (self.tp + self.fp) == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        """
        Calculate recall (sensitivity or true positive rate).

        Measures the proportion of actual positives that were correctly identified.
        Indicates how well the model finds positive instances.

        Returns
        -------
        recall : float
            Recall score in range [0, 1], where 1 represents perfect recall.
            Formula: TP / (TP + FN). Returns 0 if no actual positives exist.
        """
        if (self.tp + self.fn) == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def f1_score(self):
        """
        Calculate recall (sensitivity or true positive rate).

        Measures the proportion of actual positives that were correctly identified.
        Indicates how well the model finds positive instances.

        Returns
        -------
        recall : float
            Recall score in range [0, 1], where 1 represents perfect recall.
            Formula: TP / (TP + FN). Returns 0 if no actual positives exist.
        """
        p = self.precision()
        r = self.recall()
        if (p + r) == 0:
            return 0
        return 2 * (p * r) / (p + r)

    def false_positive_rate(self):
        """
        Calculate false positive rate (1 - specificity).

        Measures the proportion of actual negatives that were incorrectly
        classified as positive. Used in ROC curve analysis.

        Returns
        -------
        fpr : float
            False positive rate in range [0, 1], where 0 represents no false positives.
            Formula: FP / (FP + TN). Returns 0 if no actual negatives exist.
        """
        if (self.fp + self.tn) == 0:
            return 0
        return self.fp / (self.fp + self.tn)

    def confusion_matrix(self):
        """
        Return normalized confusion matrix (values between 0-1).

        Returns
        -------
        cm : numpy.ndarray
            2x2 normalized confusion matrix in format:
            [[TN, FP],
             [FN, TP]]
        """
        # Standard format: rows=actual, columns=predicted
        cm = np.array([[self.tn, self.fp],
                       [self.fn, self.tp]], dtype=float)

        # normalize by total number of samples
        total = self.tp + self.fp + self.fn + self.tn
        if total > 0:
            cm = cm / total
        return cm

    def plot_confusion_matrix(self, title, save_path):
        """
        Plot and save confusion matrix as a heatmap.

        Parameters
        ----------
        title : str
            Title for the plot
        save_path : str
            Path to save the figure. If None, only displays.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        class_names = ['Fake', 'Real']

        cm = self.confusion_matrix()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=['Predicted ' + class_names[0], 'Predicted ' + class_names[1]],
                    yticklabels=['Actual ' + class_names[0], 'Actual ' + class_names[1]],
                    cbar_kws={'label': 'Normalized Frequency'}, ax=ax)

        # Add title and labels
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Class', fontsize=12, fontweight='bold')

        # Add text annotations with percentages
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                        ha='center', va='center', fontsize=10,
                        color='white' if cm[i, j] > 0.5 else 'black')

        plt.tight_layout()
        # Save if path provided
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()
        return fig

    def report(self):
        """
        Generate a comprehensive dictionary of classification metrics.

        Computes all standard binary classification metrics in a single call,
        providing a complete performance summary suitable for logging,
        comparison, or further analysis.

        Returns
        -------
        metrics_dict : dict
            Dictionary containing computed metrics with the following keys:
            - 'accuracy': Overall classification accuracy [0, 1]
            - 'precision': Positive predictive value [0, 1]
            - 'recall': Sensitivity/True positive rate [0, 1]
            - 'f1_score': Harmonic mean of precision and recall [0, 1]
            - 'false_positive_rate': False alarm rate [0, 1]
        """
        return {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1_score(),
            'false_positive_rate': self.false_positive_rate()
        }

    def plot_roc_curve(self, y_true, y_scores, title, path):
        """
        Generate and save ROC (Receiver Operating Characteristic) curve visualization.

        Creates a publication-quality ROC curve plot showing the trade-off between
        true positive rate and false positive rate across different classification
        thresholds. Includes AUC score and random classifier baseline.

        Parameters
        ----------
        y_true : array-like, shape (n_samples,)
            True binary labels (0 or 1). Must be binary classification labels.
        y_scores : array-like, shape (n_samples,)
            Predicted probabilities or decision scores for the positive class.
            Higher scores should correspond to higher likelihood of positive class.
        title : str
            Title for the ROC curve plot. Should be descriptive of the task.
        path : str or Path
            File path where the plot will be saved. Recommended to use .png extension.

        Returns
        -------
        None
            Saves plot to specified path and displays it. Prints confirmation message.
        Raises
        ------
        ValueError
            If y_true contains non-binary values or if array lengths don't match.
            """
        # testing the confusion matrix at various thresholds as learned in class
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(FIGS / path)
        plt.show()


if __name__ == "__main__":

    # Initialize classifier
    classifier = NewsClassifier()

    # # News Articles evaluation
    #
    # # load the test dataset
    # fakeddit_df = pd.read_csv(ROOT / DATA / "all_validate.tsv", sep='\t')
    # fakeddit_df = fakeddit_df.dropna(subset=['clean_title', '2_way_label'])
    #
    # # Sample 20% of the data randomly
    # sample_size = int(len(fakeddit_df) * 0.20)
    # fakeddit_df_sample = fakeddit_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    #
    # # Extract text and labels from the sampled data
    # X_test = fakeddit_df_sample['clean_title'].reset_index(drop=True)
    # y_test = fakeddit_df_sample['2_way_label'].reset_index(drop=True)
    #
    # domains = fakeddit_df.get('domain', [None] * len(X_test))
    # urls = fakeddit_df.get('url', [None] * len(X_test))
    # dates = fakeddit_df.get('created_utc', [None] * len(X_test))
    #
    # # Prepare lists for true and predicted labels
    # news_true_labels = []
    # news_predicted_labels = []
    # news_probability_scores = []
    #
    # # Make predictions using NewsClassifier
    # for idx in range(len(X_test)):
    #     text = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
    #     true_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
    #
    #     domain = domains.iloc[idx] if hasattr(domains, 'iloc') and domains.iloc[idx] is not None else None
    #     url = urls.iloc[idx] if hasattr(urls, 'iloc') and urls.iloc[idx] is not None else None
    #
    #     # Use the advanced predict method
    #     prediction_result = classifier.predict(text=text, domain=domain, date="", url=url)
    #
    #     if prediction_result is not None:
    #         predicted_label = prediction_result['prediction']  # 0 or 1
    #         fake_prob_score = prediction_result['fake_probability'] / 100.0
    #
    #         news_true_labels.append(true_label)
    #         news_predicted_labels.append(predicted_label)
    #         news_probability_scores.append(fake_prob_score)
    #
    # # Calculate confusion matrix for news
    # news_cm = confusion_matrix(news_true_labels, news_predicted_labels)
    # news_tn, news_fp, news_fn, news_tp = news_cm.ravel()
    #
    # # Create metrics instance for news
    # news_metrics = ConfusionMatrixMetrics(tp=news_tp, fp=news_fp, fn=news_fn, tn=news_tn)
    #
    # news_report = news_metrics.report()
    # print("=== NEWS ARTICLES EVALUATION ===")
    # print("Confusion Matrix:\n", news_metrics.confusion_matrix())
    # print(f"True Negatives (TN): {news_tn}")
    # print(f"False Positives (FP): {news_fp}")
    # print(f"False Negatives (FN): {news_fn}")
    # print(f"True Positives (TP): {news_tp}")
    # print("Metrics report:")
    # for k, v in news_report.items():
    #     print(f"{k.capitalize()}: {v:.4f}")
    #
    # # Convert to numpy arrays
    # news_true_labels = np.array(news_true_labels)
    # news_probability_scores = np.array(news_probability_scores)
    #
    # # Plot ROC curve and confusion matrix
    # news_metrics.plot_roc_curve(news_true_labels, news_probability_scores, "News Articles Detection ROC Curve",
    #                FIGS / "news_roc_curve.png")
    # news_metrics.plot_confusion_matrix("News Articles - Normalized Confusion Matrix Heatmap", "news_cm_heatmap.png")
    #
    # print("\n" + "="*50 + "\n")

    # Bot Detection evaluation

    # load the test dataset
    # tweets_test_df = pd.read_csv(ROOT / DATA / "test.json")
    # tweets_test_df = tweets_test_df.dropna(subset=['tweet', 'BinaryNumTarget'])
    #
    # # Sample 20% of the data randomly
    # sample_size = int(len(tweets_test_df) * 0.05)
    # tweets_test_df_sample = tweets_test_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    #
    # # Extract text and labels from the sampled data
    # X_test = tweets_test_df_sample['tweet'].reset_index(drop=True)
    # y_test = tweets_test_df_sample['BinaryNumTarget'].reset_index(drop=True)  # 1=human, 0=bot

    # Load JSON file instead of CSV
    with open(ROOT / DATA / "test.json", 'r') as f:
        data = json.load(f)

    # Convert JSON data to DataFrame
    tweets_data = []
    for entry in data:
        # Extract tweets from the 'tweet' array with null checking
        tweets = entry.get('tweet', [])

        # Handle cases where tweets field is None or not a list
        if tweets is None:
            tweets = []
        elif not isinstance(tweets, list):
            tweets = [str(tweets)]

        label = int(entry.get('label', 0))  # Convert string label to int

        if tweets:
            # Create a row for each tweet
            for tweet_text in tweets:
                if tweet_text and tweet_text.strip():
                    tweets_data.append({
                        'tweet': tweet_text,
                        'label': label  # 1=human, 0=bot based on your JSON structure
                    })

    # Create DataFrame
    tweets_test_df = pd.DataFrame(tweets_data)

    # Drop rows with missing tweet text or labels
    tweets_test_df = tweets_test_df.dropna(subset=['tweet', 'label'])

    # Sample 5% of the data randomly
    sample_size = int(len(tweets_test_df) * 0.05)
    tweets_test_df_sample = tweets_test_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Extract text and labels from the sampled data
    X_test = tweets_test_df_sample['tweet'].reset_index(drop=True)
    y_test = tweets_test_df_sample['label'].reset_index(drop=True)  # 1=human, 0=bot

    # Prepare lists for true and predicted labels
    tweets_true_labels = []
    tweets_predicted_labels = []
    tweets_probability_scores = []

    # Make predictions using NewsClassifier bot detection
    for idx in range(len(X_test)):
        text = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
        true_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]

        # Prepare user_data from the dataset features
        user_data = {
            'followers_count': tweets_test_df_sample.iloc[idx].get('followers_count', 0),
            'friends_count': tweets_test_df_sample.iloc[idx].get('friends_count', 0),
            'statuses_count': tweets_test_df_sample.iloc[idx].get('statuses_count', 0),
            'favourites_count': tweets_test_df_sample.iloc[idx].get('favourites_count', 0),
            'listed_count': tweets_test_df_sample.iloc[idx].get('listed_count', 0),
            'screen_name': tweets_test_df_sample.iloc[idx].get('screen_name', ''),
            'verified': tweets_test_df_sample.iloc[idx].get('verified', False),
            'screen_name_length': len(str(tweets_test_df_sample.iloc[idx].get('screen_name', ''))),
            'retweets': tweets_test_df_sample.iloc[idx].get('retweets', 0.2),
            'replies': tweets_test_df_sample.iloc[idx].get('replies', 0.1),
            'favoriteC': tweets_test_df_sample.iloc[idx].get('favoriteC', 0.3),
            'hashtag': tweets_test_df_sample.iloc[idx].get('hashtag', 0.3),
            'mentions': tweets_test_df_sample.iloc[idx].get('mentions', 0.4),
            'intertime': tweets_test_df_sample.iloc[idx].get('intertime', 3600),
            'favorites': tweets_test_df_sample.iloc[idx].get('favourites', 100),
            'uniqueHashtags': tweets_test_df_sample.iloc[idx].get('uniqueHashtags', 0.5),
            'uniqueMentions': tweets_test_df_sample.iloc[idx].get('uniqueMentions', 0.6),
            'uniqueURL': tweets_test_df_sample.iloc[idx].get('uniqueURL', 0.7)
        }

        # Calculate friends-to-followers ratio if not present
        if user_data['followers_count'] > 0:
            user_data['ffratio'] = user_data['friends_count'] / user_data['followers_count']
        else:
            user_data['ffratio'] = user_data['friends_count'] if user_data['friends_count'] > 0 else 1.0

        # Use the bot prediction method
        prediction_result = classifier.predict_bot(tweet_text=text, user_data=user_data)

        if prediction_result is not None and 'prediction' in prediction_result:
            predicted_label = prediction_result['prediction']  # 0=human, 1=bot
            bot_prob_score = prediction_result['bot_probability'] / 100.0

            tweets_true_labels.append(true_label)
            tweets_predicted_labels.append(predicted_label)
            tweets_probability_scores.append(bot_prob_score)

    # Calculate confusion matrix for tweets (FIXED VARIABLES)
    tweets_cm = confusion_matrix(tweets_true_labels, tweets_predicted_labels)
    tweets_tn, tweets_fp, tweets_fn, tweets_tp = tweets_cm.ravel()

    # Create metrics instance for tweets
    tweets_metrics = ConfusionMatrixMetrics(tp=tweets_tp, fp=tweets_fp, fn=tweets_fn, tn=tweets_tn)

    tweets_report = tweets_metrics.report()
    print("=== BOT DETECTION EVALUATION ===")
    print("Bot Detection Confusion Matrix:\n", tweets_metrics.confusion_matrix())
    print(f"True Negatives (TN - Correctly identified humans): {tweets_tn}")
    print(f"False Positives (FP - Humans classified as bots): {tweets_fp}")
    print(f"False Negatives (FN - Bots classified as humans): {tweets_fn}")
    print(f"True Positives (TP - Correctly identified bots): {tweets_tp}")
    print("Bot Detection Metrics report:")
    for k, v in tweets_report.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # Convert to numpy arrays
    tweets_true_labels = np.array(tweets_true_labels)
    tweets_probability_scores = np.array(tweets_probability_scores)

    # Plot ROC curve
    tweets_metrics.plot_roc_curve(tweets_true_labels, tweets_probability_scores, "Bot Detection ROC Curve",
                   FIGS / "tweets_roc_curve.png")
    tweets_metrics.plot_confusion_matrix("Twitter Accounts - Normalized Confusion Matrix Heatmap", "tweets_cm_heatmap"
                                                                                                   ".png")
