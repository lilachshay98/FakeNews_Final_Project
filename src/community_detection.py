import csv
import json
import math
import os
import re
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import community.community_louvain as community_louvain
import random

ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
DATA = ROOT / "data" / "raw"
FIGS = ROOT / "figures"
STATS = ROOT / "data" / "stats"
# Twitter handle: 1–15 chars, letters/digits/underscore
MENTION_RE = re.compile(r'(?<!\w)@([A-Za-z0-9_]{1,15})')


# 1. load + build the graph
def load_accounts():
    """
    Load records from train.json, dev.json, and test.json (if present) and
    return them as one data frame.
    """
    files = [DATA / "dev.json", DATA / "test.json", DATA / "train.json"]
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
    labels = {}  # "0"=human, "1"=bot, may be None
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
        Lowercase handles (prefixed with "@" if requested).
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
def subgraph_around_anchors(Gd, screen, anchors, radius=2, max_nodes=4000, mutual_only=False):
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
    mutual_only : bool, default=False
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
    if mutual_only:
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


# 4. use the Louvain algorithm to calculate the best partition
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
    return community_louvain.best_partition(Gu, resolution=1.0, random_state=42)


# 5. analyze results
def analyze_communities(labels, partition):
    """
    Analyze each community's bot/human composition.

    Parameters
    ----------
    labels : dict[str, {"0","1",None}]
        Node ID to label mapping ("0"=bot, "1"=human)
    partition : dict[str, int]
        Node ID to community ID mapping

    Returns
    -------
    community_stats : dict
        Community ID -> statistics dictionary
    """
    community_stats = {}

    for community_id in set(partition.values()):
        # Get all nodes in this community
        community_nodes = [node for node, comm in partition.items() if comm == community_id]

        # Count bots and humans
        bots = sum(1 for node in community_nodes if labels.get(node) == "0")
        humans = sum(1 for node in community_nodes if labels.get(node) == "1")
        unknown = sum(1 for node in community_nodes if labels.get(node) not in ("0", "1"))
        total = len(community_nodes)

        # Calculate percentages
        bot_pct = (bots / total) * 100 if total > 0 else 0
        human_pct = (humans / total) * 100 if total > 0 else 0

        # Determine community type and predicted label
        if bot_pct > 70:
            comm_type = "bot-dominant"
            predicted_label = "0"
            confidence = bot_pct / 100
        elif human_pct > 70:
            comm_type = "human-dominant"
            predicted_label = "1"
            confidence = human_pct / 100
        else:
            comm_type = "mixed"
            predicted_label = None  # will be 50/50, random choice
            confidence = 0.5

        community_stats[community_id] = {'total_nodes': total, 'bots': bots, 'humans': humans, 'unknown': unknown,
                                         'bot_percentage': bot_pct, 'human_percentage': human_pct, 'type': comm_type,
                                         'predicted_label': predicted_label, 'confidence': confidence,
                                         'nodes': community_nodes}
    return community_stats


def find_connected_community(account_id, Gd, partition, radius):
    """
    Find if an account is connected to any existing community within radius.

    Searches the network to determine if a given account has connections
    to nodes in existing communities within a specified hop radius.

    Parameters
    ----------
    account_id : str
        The account ID to search connections for.
    Gd : networkx.DiGraph
        The full directed graph.
    partition : dict[str, int]
        Node ID to community ID mapping.
    radius : int
        Search radius in number of hops.

    Returns
    -------
    community_id : int or None
        Community ID if connected community found, None otherwise.
        Returns the community with most connections if multiple found.
    """
    if account_id not in Gd:
        return None

    # Get neighbors within radius
    ego_net = nx.ego_graph(Gd.to_undirected(), account_id, radius=radius)

    # Count connections to each community
    community_connections = Counter()
    for neighbor in ego_net.nodes():
        if neighbor in partition and neighbor != account_id:
            community_connections[partition[neighbor]] += 1

    # Return the most connected community, or None if no connections
    if community_connections:
        return community_connections.most_common(1)[0][0]


def add_edges_to_community_graph(account_id, community_id, Gd, H, partition):
    """
    Add edges between a new account and its community members.

    Updates the community graph by adding edges between the specified
    account and other members of the same community, based on existing
    connections in the original directed graph.

    Parameters
    ----------
    account_id : str
        The account ID to connect to community members.
    community_id : int
        The community ID the account belongs to.
    Gd : networkx.DiGraph
        The original directed graph with all connections.
    H : networkx.Graph
        The community subgraph to update (modified in place).
    partition : dict[str, int]
        Node ID to community ID mapping.

    Returns
    -------
    None
        Modifies H in place by adding appropriate edges.
    """
    # Add edges to accounts in the same community that are connected in the original graph
    community_members = [node for node, comm in partition.items() if comm == community_id and node != account_id]

    for member in community_members:
        if member in H.nodes():
            # Check if there's an edge in the original directed graph (in either direction)
            if Gd.has_edge(account_id, member) or Gd.has_edge(member, account_id):
                H.add_edge(account_id, member)


def create_new_community_stats(node_list, labels):
    """
    Create statistics for a new community.

    Computes comprehensive statistics for a community given its member
    nodes and their labels.

    Parameters
    ----------
    node_list : list[str]
        List of node IDs in the community.
    labels : dict[str, {"0","1",None}]
        Node ID to label mapping.

    Returns
    -------
    stats : dict
        Statistics dictionary with same structure as analyze_communities
        output, containing counts, percentages, type classification,
        and confidence measures.
    """
    bots = sum(1 for node in node_list if labels.get(node) == "0")
    humans = sum(1 for node in node_list if labels.get(node) == "1")
    unknown = sum(1 for node in node_list if labels.get(node) not in ("0", "1"))
    total = len(node_list)

    bot_pct = (bots / total) * 100 if total > 0 else 0
    human_pct = (humans / total) * 100 if total > 0 else 0

    # Determine community type and predicted label
    if bot_pct > 70:
        comm_type = "bot-dominant"
        predicted_label = "0"
        confidence = bot_pct / 100
    elif human_pct > 70:
        comm_type = "human-dominant"
        predicted_label = "1"
        confidence = human_pct / 100
    else:
        comm_type = "mixed"
        predicted_label = None
        confidence = 0.5

    return {
        'total_nodes': total, 'bots': bots, 'humans': humans, 'unknown': unknown, 'bot_percentage': bot_pct,
        'human_percentage': human_pct, 'type': comm_type, 'predicted_label': predicted_label,
        'confidence': confidence, 'nodes': node_list.copy() }


def update_community_stats(community_id, community_stats, labels):
    """
    Update statistics for an existing community after adding/changing a node.

    Recalculates all statistics for a community when its composition
    has changed due to node additions or label updates.

    Parameters
    ----------
    community_id : int
        The community ID to update statistics for.
    community_stats : dict[int, dict]
        Community statistics dictionary (modified in place).
    labels : dict[str, {"0","1",None}]
        Current node ID to label mapping.

    Returns
    -------
    None
        Modifies community_stats in place with updated statistics.
    """
    if community_id not in community_stats:
        return

    # Get current nodes in the community
    current_nodes = community_stats[community_id]['nodes']

    # Recalculate stats
    updated_stats = create_new_community_stats(current_nodes, labels)
    community_stats[community_id].update(updated_stats)


def predict_account_label(account_name, Gd, H, labels, screen, partition, community_stats, radius=2, random_seed=42):
    """
    Predict an account's label based on community membership or create new community and updates communities.

    Parameters
    ----------
    account_name : str
        Account name (with or without @)
    Gd : nx.DiGraph
        Full directed graph
    H : nx.Graph
        Subgraph with communities
    labels : dict
        Node labels (will be updated)
    screen : dict
        Node ID to screen name mapping (will be updated if needed)
    partition : dict
        Node to community mapping (will be updated)
    community_stats : dict
        Community statistics (will be updated)
    radius : int, default=2
        How far to search for connected communities
    random_seed : int, default=42
        Seed for random label assignment

    Returns
    -------
    dict : Prediction information and community assignment
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Normalize account name
    normalized_name = account_name.strip().lstrip("@").lower()

    # Find or create account ID
    account_id = None
    for uid, screen_name in screen.items():
        if screen_name == normalized_name:
            account_id = uid
            break

    # If account not found in screen names, create new ID
    if not account_id:
        # Generate new unique ID
        existing_ids = set(screen.keys())
        account_id = str(max(int(id_) for id_ in existing_ids if id_.isdigit()) + 1) if existing_ids else "1"
        screen[account_id] = normalized_name
        print(f"Created new account ID {account_id} for @{normalized_name}")

    # Check if account is already in a community
    if account_id in partition:
        community_id = partition[account_id]
        community_info = community_stats[community_id]

        # Predict label based on existing community
        if community_info['predicted_label'] is not None:
            predicted_label = community_info['predicted_label']
            bot_probability = community_info['bot_percentage'] / 100
            human_probability = community_info['human_percentage'] / 100
        else:
            # Mixed community - 50/50 chance
            predicted_label = random.choice(["0", "1"])
            bot_probability = 0.5
            human_probability = 0.5

        # Update the account's label
        labels[account_id] = predicted_label

        # Update community stats with new prediction
        update_community_stats(community_id, community_stats, labels)

        print(f"Account @{normalized_name} already in community {community_id}")
        return {
            'account_id': account_id, 'account_name': normalized_name, 'community_id': community_id,
            'predicted_label': predicted_label, 'label_type': 'bot' if predicted_label == "0" else 'human',
            'bot_probability': bot_probability, 'human_probability': human_probability,
            'action': 'updated_existing', 'community_type': community_info['type'],
            'confidence': community_info['confidence'] }

    # Try to find connected community
    connected_community = find_connected_community(account_id, Gd, partition, radius)

    if connected_community is not None:
        # Add to existing connected community
        partition[account_id] = connected_community
        if account_id not in H.nodes():
            H.add_node(account_id)

        # Add edges to the community graph if connections exist
        add_edges_to_community_graph(account_id, connected_community, Gd, H, partition)

        # Predict label based on community dominance
        community_info = community_stats[connected_community]
        if community_info['predicted_label'] is not None:
            predicted_label = community_info['predicted_label']
            bot_probability = community_info['bot_percentage'] / 100
            human_probability = community_info['human_percentage'] / 100
        else:
            # Mixed community - 50/50 chance
            predicted_label = random.choice(["0", "1"])
            bot_probability = 0.5
            human_probability = 0.5

        # Update the account's label
        labels[account_id] = predicted_label

        # Update community stats
        update_community_stats(connected_community, community_stats, labels)

        print(f"Added @{normalized_name} to existing community {connected_community}")
        return {
            'account_id': account_id, 'account_name': normalized_name, 'community_id': connected_community,
            'predicted_label': predicted_label, 'label_type': 'bot' if predicted_label == "0" else 'human',
            'bot_probability': bot_probability, 'human_probability': human_probability,
            'action': 'added_to_existing', 'community_type': community_info['type'],
            'confidence': community_info['confidence'] }
    else:
        # Create new community with random label
        new_community_id = max(partition.values()) + 1 if partition else 0
        partition[account_id] = new_community_id

        if account_id not in H.nodes():
            H.add_node(account_id)

        # Randomly assign label (50/50 chance)
        predicted_label = random.choice(["0", "1"])
        labels[account_id] = predicted_label

        # Create new community stats
        community_stats[new_community_id] = create_new_community_stats([account_id], labels)

        print(f"Created new community {new_community_id} for @{normalized_name}")
        return {
            'account_id': account_id, 'account_name': normalized_name, 'community_id': new_community_id,
            'predicted_label': predicted_label, 'label_type': 'bot' if predicted_label == "0" else 'human',
            'bot_probability': 0.5, 'human_probability': 0.5,
            'action': 'created_new_community', 'community_type': 'new', 'confidence': 0.5 }


# 6. print and plot results
def print_community_summary(community_stats):
    """
    Print a summary of all communities.

    Displays comprehensive statistics about detected communities including
    size, composition, and type distribution.

    Parameters
    ----------
    community_stats : dict[int, dict]
        Community statistics from analyze_communities().

    Returns
    -------
    None
        Prints formatted summary to stdout.
    """
    print(f"\n{'=' * 60}")
    print(f"COMMUNITY ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total communities: {len(community_stats)}")
    print()

    # Sort communities by size
    sorted_communities = sorted(community_stats.items(), key=lambda x: x[1]['total_nodes'], reverse=True)

    print(f"{'ID':<4} {'Size':<6} {'Bots':<6} {'Humans':<7} {'Bot%':<6} {'Human%':<8} {'Type':<15}")
    print("-" * 60)

    for comm_id, stats in sorted_communities:
        print(f"{comm_id:<4} {stats['total_nodes']:<6} {stats['bots']:<6} {stats['humans']:<7} "
              f"{stats['bot_percentage']:<6.1f} {stats['human_percentage']:<8.1f} {stats['type']:<15}")

    # Overall statistics
    total_nodes = sum(stats['total_nodes'] for stats in community_stats.values())
    total_bots = sum(stats['bots'] for stats in community_stats.values())
    total_humans = sum(stats['humans'] for stats in community_stats.values())

    bot_dominant = sum(1 for stats in community_stats.values() if stats['type'] == 'bot-dominant')
    human_dominant = sum(1 for stats in community_stats.values() if stats['type'] == 'human-dominant')
    mixed = sum(1 for stats in community_stats.values() if stats['type'] == 'mixed')

    print(f"\n{'=' * 60}")
    print(f"OVERALL STATISTICS:")
    print(f"Total nodes in communities: {total_nodes}")
    print(f"Total bots: {total_bots} ({total_bots / total_nodes * 100:.1f}%)")
    print(f"Total humans: {total_humans} ({total_humans / total_nodes * 100:.1f}%)")
    print(f"Bot-dominant communities: {bot_dominant}")
    print(f"Human-dominant communities: {human_dominant}")
    print(f"Mixed communities: {mixed}")


def plot_follow_graph(H, labels, screen, anchor_ids, partition, out_png=FIGS / "follow_graph_louvain.png"):
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
    nx.draw_networkx_nodes(H, pos, nodelist=humans, node_color="#7ED957", edgecolors="none",
                           node_size=[size(x) for x in humans], label="human user")
    nx.draw_networkx_nodes(H, pos, nodelist=bots, node_color="#FF5C5C", edgecolors="none",
                           node_size=[size(x) for x in bots], label="bot user")
    if other:
        nx.draw_networkx_nodes(H, pos, nodelist=other, node_color="#C0C0C0", edgecolors="none",
                               node_size=[size(x) for x in other], label="unknown")

    # emphasize anchors
    for aid in anchor_ids:
        if aid in H and aid in pos:
            x, y = pos[aid]
            plt.scatter([x], [y], s=260, facecolors="none", edgecolors="black", linewidths=2, zorder=3)
            plt.text(x, y, "@" + screen.get(aid, ""),
                     fontsize=10, weight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.5, alpha=0.85))

    # show communities from Louvain
    n_comms = len(set(partition.get(n, -1) for n in H))
    plt.legend(loc="upper right", frameon=False)
    plt.axis("off")
    plt.title(f"Follow graph - Louvain communities: {n_comms}")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_png}")


def plot_community_colored_graph(H, partition, community_stats, out_png=FIGS / "community_colored_graph.png"):
    """
    Plot the graph with nodes colored by community dominance type.

    Parameters
    ----------
    H : nx.Graph
        The subgraph to visualize
    partition : dict
        Node ID to community ID mapping
    community_stats : dict
        Community statistics with dominance types
    out_png : str or Path
        Output path for the image

    Returns
    -------
    None
    """
    # Map community ID to color based on dominance type
    community_colors = {}
    for comm_id, stats in community_stats.items():
        if stats['type'] == 'bot-dominant':
            community_colors[comm_id] = '#FF5C5C'  # Red
        elif stats['type'] == 'human-dominant':
            community_colors[comm_id] = '#7ED957'  # Green
        else:  # mixed
            community_colors[comm_id] = '#FFA500'  # Orange

    # Assign colors to nodes based on their community
    node_colors = []
    for node in H.nodes():
        comm_id = partition.get(node)
        if comm_id is not None and comm_id in community_colors:
            node_colors.append(community_colors[comm_id])
        else:
            node_colors.append('#808080')  # Dark grey for nodes without community

    # Calculate node sizes based on degree
    deg = H.degree()

    def size(node):
        return 20 + 4 * math.sqrt(deg[node])

    node_sizes = [size(node) for node in H.nodes()]

    # Create layout
    n = H.number_of_nodes()
    k = 1.0 / math.sqrt(max(n, 1))
    pos = nx.spring_layout(H, k=k, iterations=300, seed=42)

    # Create the plot
    plt.figure(figsize=(16, 8), dpi=150)

    # Draw edges
    nx.draw_networkx_edges(H, pos, edge_color="#7f7f7f", width=0.5, alpha=0.35)

    # Draw nodes with community colors
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=0.5,
                           alpha=0.9)

    # Create legend with orange for mixed communities
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#FF5C5C', label='Bot-dominant community')
    green_patch = mpatches.Patch(color='#7ED957', label='Human-dominant community')
    orange_patch = mpatches.Patch(color='#FFA500', label='Mixed community')

    plt.legend(handles=[red_patch, green_patch, orange_patch],
               loc='upper right', frameon=False)

    # Show community count
    n_comms = len(set(partition.values()))
    plt.title(f"Follow graph - Communities colored by dominance ({n_comms} communities)")
    plt.axis('off')
    plt.tight_layout()

    # Save and show
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Community-colored graph saved to {out_png}")

    # Print community breakdown
    bot_dominant = sum(1 for stats in community_stats.values() if stats['type'] == 'bot-dominant')
    human_dominant = sum(1 for stats in community_stats.values() if stats['type'] == 'human-dominant')
    mixed = sum(1 for stats in community_stats.values() if stats['type'] == 'mixed')

    print(f"\nCommunity breakdown:")
    print(f"Bot-dominant (red): {bot_dominant}")
    print(f"Human-dominant (green): {human_dominant}")
    print(f"Mixed (orange): {mixed}")


def get_community_color_mapping(community_stats):
    """
    Get a mapping of community IDs to colors based on their dominance.

    Parameters
    ----------
    community_stats : dict
        Community statistics with dominance types

    Returns
    -------
    dict : Community ID to color mapping
    """
    community_colors = {}
    for comm_id, stats in community_stats.items():
        if stats['type'] == 'bot-dominant':
            community_colors[comm_id] = '#FF5C5C'  # Red
        elif stats['type'] == 'human-dominant':
            community_colors[comm_id] = '#7ED957'  # Green
        else:  # mixed
            community_colors[comm_id] = '#FFA500'  # Orange
    return community_colors


def print_community_color_legend():
    """
    Print the color coding used for communities.

    Displays the color legend for community visualizations to help
    interpret the plots and understand the color scheme.

    Returns
    -------
    None
        Prints formatted color legend to stdout.
    """
    print("\n" + "=" * 50)
    print("COMMUNITY COLOR LEGEND:")
    print("=" * 50)
    print("Red (#FF5C5C): Bot-dominant communities (>70% bots)")
    print("Green (#7ED957): Human-dominant communities (>70% humans)")
    print("Orange (#FFA500): Mixed communities (neither dominates)")
    print("=" * 50)


if __name__ == "__main__":
    # Load and build graph
    accounts = load_accounts()
    Gd, labels, screen = build_follow_graph(accounts)

    # Find anchors and create subgraph
    anchors = top_mentions(accounts)
    H, anchor_ids = subgraph_around_anchors(Gd, screen, anchors, radius=2, max_nodes=4000, mutual_only=False)

    # Detect communities
    partition = louvain_partition(H)

    # Analyze communities
    community_stats = analyze_communities(labels, partition)

    # Visualize
    plot_follow_graph(H, labels, screen, anchor_ids, partition)
    plot_community_colored_graph(H, partition, community_stats)

    # Print initial analysis
    print_community_summary(community_stats)

