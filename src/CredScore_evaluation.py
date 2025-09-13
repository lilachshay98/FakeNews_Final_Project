import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
from train_model import fitted_models, X_train_tfidf, X_test_tfidf, y_train, y_test, X_test
from shared_logic import NewsClassifier
from common_utils import get_split_data, tokenize_text


class ConfusionMatrixMetrics:
    def __init__(self, tp, fp, fn, tn):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

    def precision(self):
        if (self.tp + self.fp) == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if (self.tp + self.fn) == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if (p + r) == 0:
            return 0
        return 2 * (p * r) / (p + r)

    def false_positive_rate(self):
        if (self.fp + self.tn) == 0:
            return 0
        return self.fp / (self.fp + self.tn)

    def confusion_matrix(self):
        return np.array([[self.tp, self.fp],
                         [self.fn, self.tn]])

    def report(self):
        return {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1_score(),
            'false_positive_rate': self.false_positive_rate()
        }


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Initialize classifier
    classifier = NewsClassifier()

    # Prepare lists for true and predicted labels
    true_labels = []
    predicted_labels = []

    # for idx, text,  in enumerate(X_test_tfidf):
    #     true_label = y_test.iloc[idx]  # 0=fake, 1=real
    #     classifier.predict(



    # metrics = ConfusionMatrixMetrics(tp=40, fp=10, fn=5, tn=45)
    # report = metrics.report()
    # print("Confusion Matrix:\n", metrics.confusion_matrix())
    # print("Metrics report:")
    # for k, v in report.items():
    #     print(f"{k.capitalize()}: {v:.4f}")
    #
    # # Example true binary labels and predicted scores (replace with actual)
    # y_true = np.array([1] * 40 + [0] * 10 + [1] * 5 + [0] * 45)  # 1=positive, 0=negative
    # np.random.seed(42)
    # y_scores = np.concatenate([
    #     np.random.uniform(0.7, 1, 40),  # scores for true positives
    #     np.random.uniform(0, 0.4, 10),  # scores for false positives
    #     np.random.uniform(0.6, 1, 5),  # scores for false negatives (simulate partial confidence)
    #     np.random.uniform(0, 0.3, 45)  # scores for true negatives
    # ])
    #
    # # Plot ROC curve
    # plot_roc_curve(y_true, y_scores)
