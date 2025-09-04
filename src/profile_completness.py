# profile_completeness_auc.py
import json
import re
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # used only for AUC

# Paths
ROOT = Path(__file__).resolve().parents[1]  # goes from src/ up to project root
DATA = ROOT / "data" / "raw"
STATS = ROOT / "data" / "stats"
STATS.mkdir(parents=True, exist_ok=True)

NONEY = {"", "none", "null", "na", "n/a", "none ", " null", " None", "None "}
TW_FORMAT = "%a %b %d %H:%M:%S %z %Y"  # e.g., Tue Feb 04 12:13:10 +0000 2020


def deNone(x):
    if x is None:
        return None
    s = str(x).strip()
    return None if s.lower() in NONEY else s


def to_int(x, default=0):
    x = deNone(x)
    if x is None:
        return default
    try:
        return int(str(x).strip().replace(",", ""))
    except:
        try:
            return int(float(x))
        except:
            return default


def to_bool(x, default=False):
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return default


def parse_created_at(x):
    s = deNone(x)
    if not s:
        return None
    try:
        return datetime.strptime(s, TW_FORMAT)
    except Exception:
        try:
            return datetime.fromisoformat(s)
        except:
            return None


def account_age_days(dt):
    if not dt:
        return np.nan
    now = datetime.now(timezone.utc)
    diff = (now - dt)
    return max(diff.days, 0)


def extract_features(item):
    prof = item.get("profile") or {}

    # basic string fields
    name = deNone(prof.get("name"))
    screen_name = deNone(prof.get("screen_name"))
    description = deNone(prof.get("description"))
    url = deNone(prof.get("url"))

    entities_raw = prof.get("entities")
    ent_urls_count = 0
    if isinstance(entities_raw, str):
        try:
            urls = re.findall(r"https?://[^\s'\"}]+", entities_raw)
            ent_urls_count = len(urls)
        except:
            ent_urls_count = 0

    has_name = 1 if name else 0
    has_screen_name = 1 if screen_name else 0
    has_description = 1 if description else 0
    desc_len = len(description) if description else 0
    has_url = 1 if url else 0
    has_entities_urls = 1 if ent_urls_count > 0 else 0

    default_profile_image = to_bool(prof.get("default_profile_image"), default=False)
    default_profile = to_bool(prof.get("default_profile"), default=True)
    profile_background_img = deNone(
        prof.get("profile_background_image_url_https")
        or prof.get("profile_background_image_url")
    )
    has_profile_image = 0 if default_profile_image else 1
    has_background_image = 1 if profile_background_img else 0

    verified = to_bool(prof.get("verified"), default=False)
    geo_enabled = to_bool(prof.get("geo_enabled"), default=False)
    has_lang = 1 if deNone(prof.get("lang")) else 0
    has_time_zone = 1 if deNone(prof.get("time_zone")) else 0

    # counts
    followers = to_int(prof.get("followers_count"))
    friends = to_int(prof.get("friends_count"))
    listed_count = to_int(prof.get("listed_count"))
    statuses = to_int(prof.get("statuses_count"))
    favourites = to_int(prof.get("favourites_count"))

    created = parse_created_at(prof.get("created_at"))
    age_days = account_age_days(created)
    age_days_filled = age_days if not math.isnan(age_days) else 0

    ff_ratio = followers / (friends + 1)
    activity_rate = statuses / (age_days_filled + 1)
    fav_rate = favourites / (age_days_filled + 1)

    return {
        "user_id": deNone(item.get("ID")),
        "screen_name": (screen_name or ""),
        "label_raw": str(item.get("label")).strip() if item.get("label") is not None else None,
        "has_name": has_name,
        "has_screen_name": has_screen_name,
        "has_description": has_description,
        "desc_len": desc_len,
        "has_url": has_url,
        "has_entities_urls": has_entities_urls,
        "has_profile_image": has_profile_image,
        "has_background_image": has_background_image,
        "default_profile": 1 if default_profile else 0,
        "verified": 1 if verified else 0,
        "geo_enabled": 1 if geo_enabled else 0,
        "has_lang": has_lang,
        "has_time_zone": has_time_zone,
        "followers": followers,
        "friends": friends,
        "listed_count": listed_count,
        "statuses": statuses,
        "favourites": favourites,
        "account_age_days": age_days if not math.isnan(age_days) else np.nan,
        "followers_to_friends_ratio": ff_ratio,
        "activity_rate": activity_rate,
        "fav_rate": fav_rate,
    }


def iter_items(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        for item in data:
            yield item


def main():
    # Collect splits
    files = [DATA / "dev.json", DATA / "test.json", DATA / "train.json"]

    dfs = []
    for p in files:
        part = pd.DataFrame(extract_features(it) for it in iter_items(p))
        part["split"] = p.stem  # "dev" / "test" / "train"
        dfs.append(part)

    df = pd.concat(dfs, ignore_index=True)

    # Map label to is_human (1) / is_bot (0) – adjust if your encoding is flipped
    HUMAN_VALUE = "0"
    df["is_human"] = (df["label_raw"] == HUMAN_VALUE).astype(int)

    # Features to evaluate (AUC only)
    num_feats = [
        "desc_len", "followers", "friends", "listed_count", "statuses", "favourites",
        "account_age_days", "followers_to_friends_ratio", "activity_rate", "fav_rate",
    ]
    bin_feats = [
        "has_name", "has_screen_name", "has_description", "has_url", "has_entities_urls",
        "has_profile_image", "has_background_image", "verified", "geo_enabled", "has_lang",
        "has_time_zone", "default_profile",
    ]

    rows = []

    # Numeric features: compute AUC using the raw value
    for col in num_feats:
        joined = df[["is_human", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(joined) >= 30 and joined[col].nunique() > 1:
            try:
                auc = roc_auc_score(joined["is_human"], joined[col])
            except Exception:
                auc = np.nan
            rows.append({"feature": col, "type": "numeric", "AUC": auc, "num_valid_samples": int(len(joined))})

    # Binary features: compute AUC treating the 0/1 flag as the score
    for col in bin_feats:
        joined = df[["is_human", col]].dropna()
        if len(joined) >= 30 and joined[col].nunique() > 1:
            try:
                auc = roc_auc_score(joined["is_human"], joined[col])
            except Exception:
                auc = np.nan
            rows.append({"feature": col, "type": "binary", "AUC": auc, "num_valid_samples": int(len(joined))})

    out = pd.DataFrame(rows).sort_values(["AUC"], ascending=False)
    out_path = STATS / "profile_completeness_auc.csv"
    out.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(out.head(15))


if __name__ == "__main__":
    main()
