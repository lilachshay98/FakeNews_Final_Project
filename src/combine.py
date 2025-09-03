import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# adjusting file paths
ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
DATA = ROOT / "data" / "raw"
FIGS = ROOT / "figures"
STATS = ROOT / "data" / "stats"

SUMMARY_CSV = STATS / "domains_summary.csv"  # from domains.py
EDGES_CSV = STATS / "domain_edges.csv"  # from page_rank.py
OUT_CSV = STATS / "domain_summary_with_pr.csv"


def load_summary(path=SUMMARY_CSV):
    df = pd.read_csv(path)
    if "domain" not in df.columns:
        df.rename(columns={df.columns[0]: "domain"}, inplace=True)
    return df


def load_edges(path=EDGES_CSV):
    df = pd.read_csv(path)
    return df[["source_domain", "target_domain"]]


def build_pagerank(edges_df, alpha=0.85):
    G = nx.DiGraph()
    G.add_edges_from(edges_df.itertuples(index=False, name=None))
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["domain", "pagerank", "in_degree", "out_degree"])
    pr = nx.pagerank(G, alpha=alpha)
    deg_in, deg_out = dict(G.in_degree()), dict(G.out_degree())
    return pd.DataFrame({
        "domain": list(pr.keys()),
        "pagerank": list(pr.values()),
        "in_degree": [deg_in.get(d, 0) for d in pr.keys()],
        "out_degree": [deg_out.get(d, 0) for d in pr.keys()],
    })


def merge_tables(summary_df, pr_df):
    m = summary_df.merge(pr_df, on="domain", how="left").fillna({"pagerank": 0, "in_degree": 0, "out_degree": 0})
    m["majority_label"] = np.where(m["fake"] > m["real"], "fake", "real")
    # PageRank percentile (0–100) without scipy
    pr = m["pagerank"].to_numpy()
    order = pr.argsort().argsort()  # ranks 0..N-1
    m["pr_percentile"] = 100 * order / max(len(pr) - 1, 1)
    return m


def violin_by_label(df, outpath=FIGS/"violin_prpct_by_label.png"):
    data = [df.loc[df["majority_label"] == "real", "pr_percentile"],
            df.loc[df["majority_label"] == "fake", "pr_percentile"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.25)
    ax.set_xticks([1, 2]);
    ax.set_xticklabels(["real-majority", "fake-majority"])
    ax.set_ylabel("PageRank percentile")
    ax.set_title("Influence distribution by majority label (violin)")
    fig.tight_layout();
    fig.savefig(outpath, dpi=220);
    plt.close(fig)


def main():
    if not Path(SUMMARY_CSV).exists():
        raise FileNotFoundError(f"Missing {SUMMARY_CSV}. Run domains.py first.")
    if not Path(EDGES_CSV).exists():
        raise FileNotFoundError(f"Missing {EDGES_CSV}. Run page_rank.py first.")

    summary = load_summary()
    pr_df = build_pagerank(load_edges())
    merged = merge_tables(summary, pr_df)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote: {OUT_CSV} (rows={len(merged)})")

    violin_by_label(merged)
    print("Saved: violin_prpct_by_label.png")


if __name__ == "__main__":
    main()
