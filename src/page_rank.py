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
    (DATA / "politifact_real.csv", "Politifact_Real"),
    (DATA / "politifact_fake.csv", "Politifact_Fake"),
    (DATA / "gossipcop_real.csv", "GossipCop_Real"),
    (DATA / "gossipcop_fake.csv", "GossipCop_Fake"),
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


def extract_domain(url):
    """
    Extract the domain name from a given URL.

    Parses a URL string to extract the domain component, handling
    common URL formats and normalizing the result by removing
    'www.' prefixes.

    Parameters
    ----------
    url : str
        URL string to extract domain from. Must include protocol
        (http:// or https://).

    Returns
    -------
    domain : str or None
        Normalized domain name (lowercase, without 'www.' prefix)
        or None if URL is invalid or domain cannot be extracted.
    """
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return None
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


def is_excluded(domain):
    """
    Check if a domain should be excluded from analysis.

    Determines whether a domain contains substrings that indicate
    it should be filtered out (social media, tracking, CDN, etc.).

    Parameters
    ----------
    domain : str
        Domain name to check against exclusion list.

    Returns
    -------
    excluded : bool
        True if domain contains any excluded substring, False otherwise.
    """
    return any(sub in domain for sub in EXCLUDE_DOMAINS_SUBSTR)


def scrape_outlinks_one(url):
    """
    Scrape outgoing links from a single webpage.

    Downloads and parses a webpage to extract all outgoing links,
    filtering for valid domains and excluding self-links and
    blacklisted domains.

    Parameters
    ----------
    url : str
        URL of the webpage to scrape for outgoing links.

    Returns
    -------
    out_edges : list[tuple[str, str]]
        List of (source_domain, target_domain) tuples representing
        outgoing links. Empty list if scraping fails or no valid
        links found.

    Notes
    -----
    The function:
    1. Extracts source domain from input URL
    2. Makes HTTP GET request with timeout and user agent
    3. Parses HTML using BeautifulSoup
    4. Extracts all <a href="..."> links
    5. Filters out self-links and excluded domains
    6. Returns deduplicated list of domain pairs

    Handles all exceptions gracefully by returning empty list.
    """
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


def scrape_file_build_edges(csv_path, label, per_file_limit):
    """
    Scrape outlinks from URLs in a CSV file to build network edges.

    Processes a dataset file containing news URLs, randomly samples
    URLs up to the specified limit, and scrapes outgoing links from
    each to build a network of domain relationships.

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing URLs in 'news_url' column.
    label : str
        Dataset label to associate with extracted edges (e.g., 'Fake', 'Real').
    per_file_limit : int
        Maximum number of URLs to process from the file.

    Returns
    -------
    edges : list[tuple[str, str, str]]
        List of (source_domain, target_domain, label) tuples representing
        scraped links with dataset labels. Empty list if file has no
        'news_url' column.

    Notes
    -----
    Processing workflow:
    1. Load CSV and extract unique URLs from 'news_url' column
    2. Randomly shuffle URLs (seed=7 for reproducibility)
    3. Limit to specified number of URLs
    4. Scrape each URL for outgoing links
    5. Add periodic sleep delays to be respectful to servers
    6. Return all edges with associated dataset label

    Progress is printed during processing showing current URL
    and number of outlinks found.
    """
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
    """
    Build a directed NetworkX graph from edge list.

    Creates a NetworkX directed graph from a list of domain
    relationships, ignoring edge labels for graph construction.

    Parameters
    ----------
    edges : list[tuple[str, str, str]]
        List of (source_domain, target_domain, label) tuples
        representing directed edges between domains.

    Returns
    -------
    G : networkx.DiGraph
        Directed graph where nodes are domain names and edges
        represent link relationships between domains.
    """
    G = nx.DiGraph()
    for s, t, _ in edges:
        G.add_edge(s, t)
    return G


def run_pagerank_and_plot(G, top_k=30, alpha=0.85):
    """
    Calculate PageRank scores and create network visualization.

    Computes PageRank centrality scores for all nodes in the graph,
    displays the top-ranked domains, and creates a network visualization
    of the most important domains.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph of domain relationships.
    top_k : int, default=30
        Number of top-ranked domains to display and visualize.
    alpha : float, default=0.85
        Damping parameter for PageRank algorithm. Higher values
        give more weight to incoming links vs. random jumps.

    Returns
    -------
    None
        Prints PageRank rankings to stdout and saves network
        visualization to 'pagerank_graph.png' in figures directory.

    Notes
    -----
    The function:
    1. Calculates PageRank scores using NetworkX
    2. Prints top_k domains with their scores
    3. Creates subgraph of only top-ranked domains
    4. Generates spring layout visualization
    5. Sizes nodes proportionally to PageRank scores
    6. Saves high-resolution plot (300 DPI) to file

    The visualization helps identify the most influential domains
    in the scraped network based on incoming link patterns.
    """
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
    plt.savefig(FIGS / "pagerank_graph.png", dpi=300)
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
    with open(STATS / "domain_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_domain", "target_domain", "dataset_label"])
        w.writerows(all_edges)
    print("Saved edges to domain_edges.csv")
    run_pagerank_and_plot(G, top_k=50)
