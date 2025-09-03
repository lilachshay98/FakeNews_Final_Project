import pandas as pd
from urllib.parse import urlparse
from pathlib import Path

# adjusting file paths
ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
DATA = ROOT / "data" / "raw"
FIGS = ROOT / "figures"
STATS = ROOT / "data" / "stats"


# -----------------------------
# 30-point component (CSV-based)
# -----------------------------
def weighted_ratio_score(real, total, k=20.0, neutral=0.5):
    """
    0–30 credibility using a weighted (shrunken) real/total ratio:
        effective_ratio = (real + k * neutral) / (total + k)
        score           = round(30 * effective_ratio)
    If total <= 0, falls back to 'neutral'. Clamped to [0, 30].
    """
    if total <= 0:
        eff = neutral
    else:
        eff = (float(real) + k * neutral) / (float(total) + k)
    return int(round(max(0.0, min(1.0, eff)) * 30))


def score_domain_from_table(df, domain, k=20.0, neutral=0.5,
                            domain_col='domain', real_col='real', total_col='total',
                            aggregate_duplicates=True):
    """
    Look up a domain (exact match in df[domain_col]) and return its 0–30 score.
    If not found → returns neutral*30.
    """
    sub = df[df[domain_col].astype(str).str.lower() == str(domain).lower()]
    if sub.empty:
        return int(round(30 * neutral))

    if aggregate_duplicates:
        real = float(sub[real_col].sum())
        total = float(sub[total_col].sum())
    else:
        real = float(sub.iloc[0][real_col])
        total = float(sub.iloc[0][total_col])

    return weighted_ratio_score(real, total, k=k, neutral=neutral)


# -----------------------------------
# 10-point component (TLD credibility)
# -----------------------------------
def tld_bonus_score(domain):
    """
    Return 0–10 TLD bonus.
    If domain string ENDS WITH any credible suffix (case-insensitive), return 10, else 0.
    Example credible_suffixes: ["com", "it", "co.uk", "gov"]
    """
    credible_suffixes = ["com", "it", "co.uk", "gov"]
    d = str(domain).lower()
    for suf in credible_suffixes:
        if d.endswith(suf.lower()):
            return 10
    return 0


# --------------------------
# Final 0–40 combined scorer
# --------------------------
def final_domain_score_0_40(path_to_df, domain,
                            k=20.0, neutral=0.5,
                            domain_col='domain', real_col='real', total_col='total',
                            aggregate_duplicates=True):
    """
    Final score out of 40:
      - 0–30 from CSV-based weighted ratio
      - 0–10 from TLD bonus
    """
    df = pd.read_csv(path_to_df)
    score_0_30 = score_domain_from_table(
        df, domain, k=k, neutral=neutral,
        domain_col=domain_col, real_col=real_col, total_col=total_col,
        aggregate_duplicates=aggregate_duplicates
    )
    score_0_10 = tld_bonus_score(domain)
    return int(score_0_30 + score_0_10)


# todo: might delete if not needed
def normalize_domain(s):
    """Strip scheme/port/'www.' so URLs and bare domains match."""
    if '://' not in s:
        s = 'http://' + s
    host = urlparse(s).netloc.lower()
    if ':' in host:
        host = host.split(':', 1)[0]
    if host.startswith('www.'):
        host = host[4:]
    return host


def CredScore_algorithm(content, source, label):
    if label == "article":

        # 2. domain reputation score (0-40)
        domain_score = final_domain_score_0_40(STATS/"domains_summary.csv", source)
        return domain_score

    if label == "twitter":
        pass


if __name__ == "__main__":
    print(CredScore_algorithm("", "guardian.co.uk", "article"))
