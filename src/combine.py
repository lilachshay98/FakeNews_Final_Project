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
    """
    Load domain summary data from CSV file.

    Parameters
    ----------
    path : pathlib.Path, optional
        Path to the domain summary CSV file. Default is SUMMARY_CSV.

    Returns
    -------
    summary_df : pandas.DataFrame
        containing domain summary data with 'domain' column
        and associated statistics (fake/real counts).
    """
    df = pd.read_csv(path)
    if "domain" not in df.columns:
        df.rename(columns={df.columns[0]: "domain"}, inplace=True)
    return df


def load_edges(path=EDGES_CSV):
    """
    Load domain edge data for network graph construction.

    Parameters
    ----------
    path : pathlib.Path, optional
        Path to the domain edges CSV file. Default is EDGES_CSV.

    Returns
    -------
    edges_df : pandas.DataFrame
        with columns ['source_domain', 'target_domain']
        representing directed edges between domains.
    """
    df = pd.read_csv(path)
    return df[["source_domain", "target_domain"]]


def build_pagerank(edges_df, alpha=0.85):
    """
    Calculate PageRank scores and degree centrality for domain network.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        with columns ['source_domain', 'target_domain']
        representing directed edges between domains.
    alpha : float, optional
        parameter for PageRank algorithm. Default is 0.85.

    Returns
    -------
    pagerank_df : pandas.DataFrame
        with columns:
        - 'domain': domain names
        - 'pagerank': PageRank scores (0.0 to 1.0)
        - 'in_degree': number of incoming edges
        - 'out_degree': number of outgoing edges
        Returns empty DataFrame if no nodes in graph.
    """
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
    """
    Merge domain summary with PageRank data and compute derived metrics.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Domain summary DataFrame with 'domain', 'fake', 'real' columns.
    pr_df : pandas.DataFrame
        PageRank DataFrame with 'domain', 'pagerank', 'in_degree',
        'out_degree' columns.

    Returns
    -------
    merged_df : pandas.DataFrame
        Combined DataFrame with additional columns:
        - 'majority_label': 'fake' or 'real' based on higher count
        - 'pr_percentile': PageRank percentile (0-100) without scipy
        Missing PageRank values filled with 0.
    """
    m = summary_df.merge(pr_df, on="domain", how="left").fillna({"pagerank": 0, "in_degree": 0, "out_degree": 0})
    m["majority_label"] = np.where(m["fake"] > m["real"], "fake", "real")

    pr = m["pagerank"].to_numpy()
    order = pr.argsort().argsort()  # ranks 0..N-1
    m["pr_percentile"] = 100 * order / max(len(pr) - 1, 1)
    return m


def violin_by_label(df, outpath=FIGS/"violin_prpct_by_label.png"):
    """
    Create violin plot of PageRank percentiles by majority label.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged DataFrame with 'majority_label' and 'pr_percentile' columns.
    outpath : pathlib.Path, optional
        Output path for the violin plot image. Default saves to figures
        directory as 'violin_prpct_by_label.png'.

    Returns
    -------
    None
        Saves violin plot as PNG file showing PageRank percentile
        distributions for real-majority vs fake-majority domains.
    """
    data = [df.loc[df["majority_label"] == "real", "pr_percentile"],
            df.loc[df["majority_label"] == "fake", "pr_percentile"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.25)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["real-majority", "fake-majority"])
    ax.set_ylabel("PageRank percentile")
    ax.set_title("Influence distribution by majority label (violin)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    """
    Main pipeline function that combines domain summary with PageRank analysis.

    Raises
    ------
    FileNotFoundError

    Returns
    -------
    None
        Executes the complete analysis pipeline:
        1. Loads domain summary and edge data
        2. Computes PageRank scores and network metrics
        3. Merges datasets and adds derived columns
        4. Saves combined results to domain_summary_with_pr.csv
        5. Generates violin plot visualization
    """
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
