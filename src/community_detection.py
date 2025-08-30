
import csv
import json
import math
import os
import re
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

# Twitter handle: 1–15 chars, letters/digits/underscore
MENTION_RE = re.compile(r'(?<!\w)@([A-Za-z0-9_]{1,15})')


# 1. load + build the graph
def load_accounts():
    """
    Load records from train.json, dev.json, and test.json (if present) and
    return them as one data frame.
    """
    files = ["datasets/train.json", "datasets/dev.json", "datasets/dev.json"]
    data = []
    for p in files:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data.extend(json.load(f))
    return data


def build_follow_graph(accounts):
    """
        Build a directed follow graph and keep the largest weakly connected component.

        Parameters
        ----------
        accounts : Iterable[dict]
            Records with keys like "ID", "label", "profile.screen_name", "neighbor.following".

        Returns
        -------
        Gd : nx.DiGraph
            Directed graph (nodes: string IDs; edge u->v means u follows v).
        labels : dict[str, {"0","1",None}]
            id -> label ("0" bot, "1" human, or None).
        screen : dict[str, str]
            id -> normalized screen name (lowercase, no "@").
        """
    labels = {}  # "0"=bot, "1"=human, may be None
    screen = {}  # id -> screen_name (account name, normalized)
    following = defaultdict(list)

    for acc in accounts:
        uid = str(acc.get("ID"))
        labels[uid] = acc.get("label")
        prof = acc.get("profile") or {}
        sn = (prof.get("screen_name") or "").strip().lstrip("@").lower()
        screen[uid] = sn
        outs = (acc.get("neighbor") or {}).get("following") or []
        following[uid] = [str(v) for v in outs]

    Gd = nx.DiGraph()
    Gd.add_nodes_from(labels.keys())
    for u, outs in following.items():
        for v in outs:
            if v in labels:  # keep edges inside known node set
                Gd.add_edge(u, v)

    # keep largest weakly-connected component for a single blob
    if Gd.number_of_nodes():
        wcc = max(nx.weakly_connected_components(Gd), key=len)
        Gd = Gd.subgraph(wcc).copy()

    return Gd, labels, screen


# 2. calculate anchors by identifying the most mentioned accounts
def top_mentions(data, top_k=5, exclude_self=True, include_at=False):
    """"
    Return the top-k most-mentioned handles across tweets.

    Parameters
    ----------
    data : Iterable[dict]
        Records with optional "profile.screen_name" and "tweet" (str or Iterable[str]).
    top_k : int, default=5
        Number of handles to return.
    exclude_self : bool, default=True
        Exclude mentions equal to the author's handle.
    include_at : bool, default=False
        Prefix results with "@".

    Returns
    -------
    handles : list[str]
        Lowercased handles (prefixed with "@" if requested).
    """
    counts = Counter()

    for acc in data:
        prof = acc.get("profile") or {}
        author = (prof.get("screen_name") or "").strip().lstrip("@").lower()

        tweets = acc.get("tweet")
        if not tweets:
            continue
        if isinstance(tweets, str):
            tweets = [tweets]

        for t in tweets:
            if not isinstance(t, str):
                continue
            for m in MENTION_RE.findall(t):
                h = m.lower()
                if exclude_self and h == author:
                    continue
                counts[h] += 1

    handles = [('@' + h) if include_at else h for h, _ in counts.most_common(top_k)]
    return handles


# 3. create a graph centered around the anchors
def subgraph_around_anchors(Gd, screen, anchors, radius=2, max_nodes=4000, reciprocal=False):
    """
        Create an undirected subgraph around anchor accounts within a hop radius.

        Parameters
        ----------
        Gd : nx.DiGraph
            Directed follow graph (node IDs as strings).
        screen : dict[str, str]
            id -> screen name (lowercase, no "@").
        anchors : Iterable[str]
            Seed handles (with/without "@", case-insensitive).
        radius : int, default=2
            Ego-graph radius.
        max_nodes : int, default=4000
            Cap on subgraph size; trims by degree if exceeded.
        reciprocal : bool, default=False
            Keep only mutual follows before subgraphing.

        Returns
        -------
        H : nx.Graph
            Undirected subgraph around anchors.
        anchor_ids : list[str]
            Anchor node IDs found in the graph.
        """
    anchors = [a.strip().lstrip("@").lower() for a in anchors]
    id_for = {sn: uid for uid, sn in screen.items()}
    anchor_ids = [id_for[a] for a in anchors if a in id_for]

    Gu = Gd.to_undirected()
    if reciprocal:
        # keep only mutual follows
        Gu = nx.Graph((u, v) for u, v in Gu.edges() if Gd.has_edge(u, v) and Gd.has_edge(v, u))

    if not anchor_ids:
        # fallback to a 2-core if anchors not found
        H = nx.k_core(Gu, k=2) if Gu.number_of_nodes() else Gu.copy()
        return H, []

    H = nx.Graph()
    for aid in anchor_ids:
        if aid in Gu:
            H = nx.compose(H, nx.ego_graph(Gu, aid, radius=radius))
    if H.number_of_nodes() > max_nodes:
        nodes_sorted = sorted(H.nodes(), key=lambda n: Gu.degree(n), reverse=True)[:max_nodes]
        H = H.subgraph(nodes_sorted).copy()
    return H, anchor_ids


# 4. use Louvain algorithm to calculate the best partition
def louvain_partition(Gu):
    """
        Compute Louvain communities for an undirected graph.

        Parameters
        ----------
        Gu : nx.Graph
            Graph to partition.

        Returns
        -------
        partition : dict[Hashable, int]
            Node -> community ID.
        """
    try:
        import community as community_louvain
    except ImportError:
        import community.community_louvain as community_louvain
    return community_louvain.best_partition(Gu, resolution=1.0, random_state=42)


# 5. plot
def helper_plot_top_mentions(data, exclude_self=True):
    """
    Count @mentions across tweets and write the top 100 to CSV.

    Parameters
    ----------
    data : Iterable[dict]
        Records with optional "profile.screen_name" and "tweet".
    exclude_self : bool, default=True
        Exclude mentions equal to the author's handle.

    Returns
    -------
    counts : collections.Counter
        handle (lowercase, no "@") -> count.
    """
    counts = Counter()
    for acc in data:
        prof = acc.get("profile") or {}
        author = (prof.get("screen_name") or "").strip().lstrip("@").lower()

        tweets = acc.get("tweet")
        if not tweets:
            continue
        if isinstance(tweets, str):
            tweets = [tweets]

        for t in tweets:
            if not isinstance(t, str):
                continue
            for m in MENTION_RE.findall(t):
                h = m.lower()
                if exclude_self and h == author:
                    continue
                counts[h] += 1

    # save only the top 100 to a csv file
    top_k = 100
    top = counts.most_common(top_k)
    out_path = Path("../top_100_mentioned_accounts.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "account", "count"])
        for i, (account, cnt) in enumerate(top, 1):
            writer.writerow([i, f"@{account}", cnt])

    return counts


def plot_top_mentions(counts, top_k=5, out_png="top5_mentions.png", color_map=None):
    """
    Plot a horizontal bar chart of top-k mentioned accounts.

    Parameters
    ----------
    counts : collections.Counter
        Handle (with/without "@") -> count.
    top_k : int, default=5
        Number of bars.
    out_png : str, default="top5_mentions.png"
        Output path.
    color_map : dict[str, Any] or None, default=None
        Handle -> color (case-insensitive; "@" ignored).

    Returns
    -------
    None
    """
    color_map = {(k.lstrip('@').lower()): v for k, v in (color_map or {}).items()}

    topk = counts.most_common(top_k)
    if not topk:
        raise ValueError("No mentions found to plot.")

    handles = [h for h, _ in topk][::-1]  # largest at top
    values = [v for _, v in topk][::-1]

    # choose colors per handle (case-insensitive, no '@')
    palette = list(plt.cm.tab10.colors)  # fallback palette
    colors = []
    for i, h in enumerate(handles):
        key = h.lstrip('@').lower()
        colors.append(color_map.get(key, palette[i % len(palette)]))

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

    bars = ax.barh(
        range(len(handles)), values,
        height=0.7,
        color=colors,  # <-- per-account colors
        edgecolor="black",
        linewidth=1.0
    )

    ax.set_yticks(range(len(handles)))
    ax.set_yticklabels([f"@{h}" for h in handles])
    ax.set_xlabel("Number of mentions")
    ax.set_title("Top 5 most-mentioned accounts")

    xmax = max(values)
    for rect, v in zip(bars, values):
        x = rect.get_width()
        y = rect.get_y() + rect.get_height() / 2
        ax.text(x + 0.01 * xmax, y, f"{v:,}", va="center", ha="left", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="-", linewidth=0.5, color="0.9")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved to {out_png}")


def plot_follow_graph(H, labels, screen, anchor_ids, partition, out_png="follow_graph_louvain.png"):
    """
        Draw the follow subgraph with bot/human colors, highlighted anchors, and community count.

        Parameters
        ----------
        H : nx.Graph
            Subgraph to draw.
        labels : dict[str, {"0","1",None}]
            id -> label.
        screen : dict[str, str]
            id -> screen name (lowercase, no "@").
        anchor_ids : list[str]
            Anchor node IDs to emphasize.
        partition : dict[Hashable, int]
            Node -> community ID.
        out_png : str, default="follow_graph_louvain.png"
            Output path.

        Returns
        -------
        None
        """
    n = H.number_of_nodes()
    k = 1.0 / math.sqrt(max(n, 1))
    pos = nx.spring_layout(H, k=k, iterations=300, seed=42)

    # color by label
    bots = [n for n in H if labels.get(n) == "0"]
    humans = [n for n in H if labels.get(n) == "1"]
    other = [n for n in H if labels.get(n) not in ("0", "1")]

    deg = H.degree()

    def size(node):
        return 20 + 4 * math.sqrt(deg[node])

    plt.figure(figsize=(16, 6), dpi=150)

    # draw edges
    nx.draw_networkx_edges(H, pos, edge_color="#7f7f7f", width=0.5, alpha=0.35)

    # draw nodes
    nx.draw_networkx_nodes(H, pos, nodelist=humans,
                           node_color="#7ED957", edgecolors="none",
                           node_size=[size(x) for x in humans], label="human user")
    nx.draw_networkx_nodes(H, pos, nodelist=bots,
                           node_color="#FF5C5C", edgecolors="none",
                           node_size=[size(x) for x in bots], label="bot user")
    if other:
        nx.draw_networkx_nodes(H, pos, nodelist=other,
                               node_color="#C0C0C0", edgecolors="none",
                               node_size=[size(x) for x in other], label="unknown")

    # emphasize anchors
    for aid in anchor_ids:
        if aid in H and aid in pos:
            x, y = pos[aid]
            plt.scatter([x], [y], s=260, facecolors="none", edgecolors="black", linewidths=2, zorder=3)
            plt.text(x, y, "@" + screen.get(aid, ""),
                     fontsize=10, weight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.5, alpha=0.85))

    # show #communities from Louvain
    n_comms = len(set(partition.get(n, -1) for n in H))
    plt.legend(loc="upper right", frameon=False)
    plt.axis("off")
    plt.title(f"Follow graph - Louvain communities: {n_comms}")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_png}")


if __name__ == "__main__":
    accounts = load_accounts()
    Gd, labels, screen = build_follow_graph(accounts)

    anchors = top_mentions(accounts)
    H, anchor_ids = subgraph_around_anchors(Gd, screen, anchors, radius=2, max_nodes=4000, reciprocal=False)

    # Louvain on the undirected subgraph
    partition = louvain_partition(H)

    counts = helper_plot_top_mentions(accounts, exclude_self=True)
    plot_top_mentions(counts, top_k=5, out_png="top5_mentions.png")
    plot_follow_graph(H, labels, screen, anchor_ids, partition, out_png="follow_graph_louvain.png")
