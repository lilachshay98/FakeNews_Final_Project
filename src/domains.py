import pandas as pd
from urllib.parse import urlparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# adjusting file paths
ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
DATA = ROOT / "data" / "raw"
FIGS = ROOT / "figures"
STATS = ROOT / "data" / "stats"


def extract_domain(url):
    try:
        d = urlparse(url).netloc.lower()
        if d.startswith("www."):
            d = d[4:]
        return d
    except:
        return None


def process_news_dataset(df, label):
    df = df.copy()
    df["domain"] = df["news_url"].apply(extract_domain)
    df["label"] = label
    return df[["domain", "label"]]


def process_general_dataset(df):
    df = df.copy()
    df = df.rename(columns={"2_way_label": "label"})[["domain", "label"]]
    df["label"] = df["label"].map({0: "fake", 1: "real"})
    return df


def build_domain_summary(min_total=20, min_each=5):
    # load
    train = pd.read_csv(DATA/"all_train.tsv", sep="\t")
    test_public = pd.read_csv(DATA/"all_test_public.tsv", sep="\t")
    politifact_real = pd.read_csv(DATA/"politifact_real.csv")
    politifact_fake = pd.read_csv(DATA/"politifact_fake.csv")
    gossipcop_real = pd.read_csv(DATA/"gossipcop_real.csv")
    gossipcop_fake = pd.read_csv(DATA/"gossipcop_fake.csv")

    # normalize
    news = pd.concat([
        process_news_dataset(politifact_real, "real"),
        process_news_dataset(politifact_fake, "fake"),
        process_news_dataset(gossipcop_real, "real"),
        process_news_dataset(gossipcop_fake, "fake"),
    ], ignore_index=True)

    general = pd.concat([
        process_general_dataset(train),
        process_general_dataset(test_public),
    ], ignore_index=True)

    combined = pd.concat([news, general], ignore_index=True)
    combined = combined[combined["domain"].notna() & (combined["domain"] != "")]

    # counts
    summary = combined.groupby(["domain", "label"]).size().unstack(fill_value=0)

    # filters: total >= min_total and each class >= min_each
    summary = summary[(summary.sum(axis=1) >= min_total) &
                      (summary.get("fake", 0) >= min_each) &
                      (summary.get("real", 0) >= min_each)]

    # metrics
    summary["total"] = summary["fake"] + summary["real"]
    summary["fake_ratio"] = summary["fake"] / summary["total"]

    # sort by fake_ratio for convenience (you can change this)
    return summary.sort_values("fake_ratio", ascending=False)


# ---------- plots ----------
def plot_topN_total_stacked(summary: pd.DataFrame, top_n=20, outpath=FIGS/"topN_total_stacked.png"):
    """Linear stacked bars with clear R%/F% labels inside the bars (good for percentages)."""
    top = summary.sort_values("total", ascending=False).head(top_n).copy()
    x = np.arange(len(top))
    real = top["real"].to_numpy(float)
    fake = top["fake"].to_numpy(float)
    total = real + fake
    rpct = real / total * 100
    fpct = fake / total * 100

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x, real, label="real")
    ax.bar(x, fake, bottom=real, label="fake")
    ax.set_ylabel("Articles")
    ax.set_title(f"Top {top_n} Domains by Total Articles (stacked, linear)")
    ax.set_xticks(x); ax.set_xticklabels(list(top.index), rotation=45, ha="right")
    ax.legend()

    for i in range(len(top)):
        ax.text(x[i], real[i]*0.5, f"R {rpct[i]:.1f}%", ha="center", va="center", fontsize=8, color="white")
        if fpct[i] >= 3:
            ax.text(x[i], real[i] + fake[i]*0.5, f"F {fpct[i]:.1f}%", ha="center", va="center", fontsize=8, color="white")
        else:
            ax.text(x[i], (real[i]+fake[i])*1.02, f"F {fpct[i]:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)


def plot_topN_total_stacked_log(summary: pd.DataFrame, top_n=20, outpath=FIGS/"topN_total_stacked_log.png"):
    import numpy as np, matplotlib.pyplot as plt
    top = summary.sort_values("total", ascending=False).head(top_n)
    x = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, top["real"], label="real")
    ax.bar(x, top["fake"], bottom=top["real"], label="fake")
    ax.set_yscale("log")
    ax.set_ylabel("Articles (log scale)")
    ax.set_title(f"Top {top_n} Domains by Total Articles (stacked, log scale)")
    ax.set_xticks(x); ax.set_xticklabels(list(top.index), rotation=45, ha="right")
    ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)


if __name__ == "__main__":
    res = build_domain_summary(min_total=20, min_each=5)
    res.to_csv(STATS/"domains_summary.csv", encoding="utf-8-sig")
    print("Wrote: domains_summary.csv")

    plot_topN_total_stacked(res, top_n=20)
    plot_topN_total_stacked_log(res, top_n=20)

    print("Wrote: topN_total_stacked.png, topN_total_stacked_log.png")
