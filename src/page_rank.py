import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import random
import time
import csv
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

# adjusting file paths
ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
DATA = ROOT / "data" / "raw"
FIGS = ROOT / "figures"
STATS = ROOT / "data" / "stats"
# -------- Config --------
FILES = [
    (DATA/"politifact_real.csv", "Politifact_Real"),
    (DATA/"politifact_fake.csv", "Politifact_Fake"),
    (DATA/"gossipcop_real.csv", "GossipCop_Real"),
    (DATA/"gossipcop_fake.csv", "GossipCop_Fake"),
]
PER_FILE_LIMIT = 500
REQUEST_TIMEOUT = 8
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
SLEEP_EVERY = 25
SLEEP_SECS = 1.0

EXCLUDE_DOMAINS_SUBSTR = [
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "pinterest.com",
    "reddit.com", "t.co", "bit.ly", "goo.gl", "doubleclick.net", "googlesyndication.com",
    "google-analytics.com", "taboola.com", "outbrain.com", "cdn.", "privacy", "terms"
]


def extract_domain(url: str):
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return None
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


def is_excluded(domain: str) -> bool:
    return any(sub in domain for sub in EXCLUDE_DOMAINS_SUBSTR)


def scrape_outlinks_one(url: str):
    try:
        src = extract_domain(url)
        if not src:
            return []
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        out_edges = []
        for a in soup.find_all("a", href=True):
            dst = extract_domain(a["href"])
            if not dst:
                continue
            if dst == src:  # ignoring self-links
                continue
            if is_excluded(dst):
                continue
            out_edges.append((src, dst))
        return list(set(out_edges))
    except Exception:
        return []


def scrape_file_build_edges(csv_path: str, label: str, per_file_limit: int):
    df = pd.read_csv(csv_path)
    if "news_url" not in df.columns:
        print(f"[WARN] {csv_path} has no 'news_url' column. Skipping.")
        return []
    urls = df["news_url"].dropna().astype(str).unique().tolist()
    random.seed(7)
    random.shuffle(urls)
    urls = urls[:min(per_file_limit, len(urls))]
    edges = []
    for i, u in enumerate(urls, 1):
        page_edges = scrape_outlinks_one(u)
        for s, t in page_edges:
            edges.append((s, t, label))
        print(f"[{i}/{len(urls)}] {label}: {u} -> {len(page_edges)} outlinks")
        if i % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECS)
    return edges


def build_graph_from_edges(edges):
    G = nx.DiGraph()
    for s, t, _ in edges:
        G.add_edge(s, t)
    return G


def run_pagerank_and_plot(G, top_k=30, alpha=0.85):
    pr = nx.pagerank(G, alpha=alpha)
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print("\n=== Top PageRank domains ===")
    for dom, score in top:
        print(f"{dom:35s} {score:.6f}")

    # Draw network (only top_k nodes to keep it readable)
    top_nodes = set([dom for dom, _ in top])
    H = G.subgraph(top_nodes).copy()

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(H, seed=7, k=0.5)  # layout
    node_sizes = [5000 * pr[n] for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_edges(H, pos, arrowstyle="->", arrowsize=10, alpha=0.5)
    nx.draw_networkx_labels(H, pos, font_size=8)
    plt.title("PageRank Graph (Top domains)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS/"pagerank_graph.png", dpi=300)
    plt.close()
    print("Saved network graph to pagerank_graph.png")


if __name__ == "__main__":
    all_edges = []
    for path, label in FILES:
        print(f"\n--- Scraping {label} (limit={PER_FILE_LIMIT}) ---")
        edges = scrape_file_build_edges(path, label, PER_FILE_LIMIT)
        all_edges.extend(edges)
    G = build_graph_from_edges(all_edges)
    print(f"\nGraph summary: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    with open(STATS/"domain_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_domain", "target_domain", "dataset_label"])
        w.writerows(all_edges)
    print("Saved edges to domain_edges.csv")
    run_pagerank_and_plot(G, top_k=50)
