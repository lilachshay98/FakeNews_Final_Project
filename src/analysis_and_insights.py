import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

# adjusting file paths
ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
DATA = ROOT / "data" / "raw"
FIGS = ROOT / "figures"


def measure_traffic(out_png=FIGS/"engagement.png"):
    """
    Plot stacked bars of engagement components (mentions, replies, retweets, favourites)
    for Real vs. Fake using raw totals.

    Parameters
    ----------
    out_png : str, default="engagement.png"
        Path to save the figure.

    Returns
    -------
    totals : pandas.DataFrame
        Totals per label (index: ["Fake", "Real"]) with columns
        ["mentions", "replies", "retweets", "favourites"].
    """
    # load csv file
    df = pd.read_csv(DATA/"Features_For_Traditional_ML_Techniques.csv")

    # ensure values are numeric + fill missing for the four attention components
    components = ["mentions", "replies", "retweets", "favourites"]
    for col in components:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # map labels (1 = Real, 0 = Fake)
    label_map = {1.0: "Real", 0.0: "Fake", 1: "Real", 0: "Fake"}
    df["label"] = df["BinaryNumTarget"].map(label_map)

    # aggregate totals per label for each component
    totals = df.groupby("label")[components].sum().sort_index()

    # plot stacked bars with absolute totals
    plt.figure(figsize=(8, 5))
    bottoms = np.zeros(len(totals.index), dtype=float)

    for comp in components:
        vals = totals[comp].values.astype(float)
        plt.bar(totals.index, vals, bottom=bottoms, label=comp)
        bottoms += vals

    plt.ylabel("Total Reactions (millions)")
    plt.title("Engagement Traffic in Real vs. Fake News")
    plt.xlabel("Label")
    plt.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    return totals


def identify_trends(out_csv=FIGS/"bots_human_accounts_created.csv",
                    out_png=FIGS/"account_creation_over_time.png"):
    """
        Count monthly bot vs. human account creations and plot the time series.

        Parameters
        ----------
        out_csv : str, default="bots_human_accounts_created.csv"
            Path to save the monthly counts CSV.
        out_png : str, default="account_creation_over_time.png"
            Path to save the line chart.

        Returns
        -------
        None
        """
    # load datasets and merge into one
    files = [DATA/"dev.json", DATA/"test.json", DATA/"train.json"]
    data = []
    for file in files:
        with open(file, "r") as f:
            data.extend(json.load(f))

    # create the count for each label (0=Bot, 1=Human)
    month_counts = defaultdict(lambda: {"bots": 0, "humans": 0})

    # Loop over accounts
    for account in data:
        created_at = account["profile"].get("created_at", None)
        label = account.get("label", None)  # "0"=bot, "1"=human

        if created_at and label is not None:
            try:
                # Parse and truncate to month
                month = pd.to_datetime(created_at).strftime("%Y-%m")
            except Exception:
                continue

            if label == "0":  # bot
                month_counts[month]["bots"] += 1
            elif label == "1":  # human
                month_counts[month]["humans"] += 1

    # Convert to DataFrame
    df = pd.DataFrame([
        {"month": m, "bots": c["bots"], "humans": c["humans"]}
        for m, c in month_counts.items()
    ])

    # Sort by month
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")
    df.to_csv(out_csv)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["month"], df["bots"], label="Bots", color="red", marker="o")
    plt.plot(df["month"], df["humans"], label="Humans", color="blue", marker="o")
    plt.xlabel("Date of Account Created")
    plt.ylabel("Number of Accounts Created")
    plt.title("Bot vs Human Account Creation Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()


def ascii_count_name(name: str) -> np.ndarray:
    """
    Count occurrences of ASCII code points (0..127) in a string.

    Parameters
    ----------
    name : str
        Input string; non-ASCII characters are ignored.

    Returns
    -------
    counts : numpy.ndarray, shape (128,), dtype=int64
        counts[i] is the number of occurrences of chr(i).
    """
    counts = np.zeros(128, dtype=np.int64)
    if not name:
        return counts
    ascii_name = name.encode("ascii", "ignore").decode("ascii")
    for ch in ascii_name:
        o = ord(ch)
        if 0 <= o < 128:
            counts[o] += 1
    return counts


if __name__ == '__main__':
    # 1. Engagement Traffic
    measure_traffic()

    # 2. Fake vs Real Accounts Created Over Time
    identify_trends()
