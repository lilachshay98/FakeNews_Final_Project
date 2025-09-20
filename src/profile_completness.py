# profile_completeness_auc.py
import json
import re
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
STATS = ROOT / "data" / "stats"
STATS.mkdir(parents=True, exist_ok=True)

NONE_VALS = {"", "none", "null", "na", "n/a", "none ", " null", " None", "None "}
TW_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def deNone(x):
    """
    Clean null-like values from input data.

    Converts various representations of null values to Python None
    for consistent handling in downstream processing.

    Parameters
    ----------
    x : any
        Input value to check for null-like representations.

    Returns
    -------
    value : str or None
        Stripped string value if input represents valid data,
        None if input represents null/missing data.

    Notes
    -----
    Recognizes the following as null values (case-insensitive):
    "", "none", "null", "na", "n/a", and variations with whitespace.
    """
    if x is None:
        return None
    s = str(x).strip()
    return None if s.lower() in NONE_VALS else s


def to_int(x, default=0):
    """
    Convert input to integer with fallback handling.

    Safely converts various input types to integer values,
    handling common edge cases like comma-separated numbers
    and floating point representations.

    Parameters
    ----------
    x : any
        Input value to convert to integer.
    default : int, default=0
        Default value to return if conversion fails.

    Returns
    -------
    value : int
        Converted integer value or default if conversion fails.

    Notes
    -----
    Conversion attempts:
    1. Direct integer conversion after comma removal
    2. Float-to-integer conversion as fallback
    3. Returns default value if both methods fail
    """
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
    """
    Convert input to boolean value.

    Interprets various string representations as boolean values
    using common conventions for true/false indicators.

    Parameters
    ----------
    x : any
        Input value to convert to boolean.
    default : bool, default=False
        Default value to return if input doesn't match known patterns.

    Returns
    -------
    value : bool
        Boolean interpretation of input or default value.

    Notes
    -----
    Recognizes as True: "true", "1", "yes" (case-insensitive)
    Recognizes as False: "false", "0", "no" (case-insensitive)
    Returns default for all other inputs.
    """
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return default


def parse_created_at(x):
    """
    Parse Twitter account creation date from various formats.

    Attempts to parse datetime strings using Twitter's standard format
    and ISO format as fallback, handling timezone information appropriately.

    Parameters
    ----------
    x : str or any
        Date string to parse, typically from Twitter API data.

    Returns
    -------
    dt : datetime.datetime or None
        Parsed datetime object with timezone information,
        or None if parsing fails.
    """
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
    """
    Calculate account age in days from creation datetime.

    Computes the difference between account creation time and
    current time, returning the age in whole days.

    Parameters
    ----------
    dt : datetime.datetime or None
        Account creation datetime with timezone information.

    Returns
    -------
    age : float or numpy.nan
        Account age in days (non-negative), or NaN if datetime is None.

    Notes
    -----
    Uses current UTC time as reference point and ensures
    non-negative results by taking maximum of 0 and calculated difference.
    """
    if not dt:
        return np.nan
    now = datetime.now(timezone.utc)
    diff = (now - dt)
    return max(diff.days, 0)


def extract_features(item):
    """
        Extract comprehensive profile features for bot detection analysis.

        Processes a Twitter user profile record to extract and compute
        various features used in bot detection, including profile completeness
        indicators, engagement metrics, and derived ratios.

        Parameters
        ----------
        item : dict
            Twitter user profile record containing user data and metadata.

        Returns
        -------
        features : dict
            Dictionary containing extracted features:

            Identity features:
            - 'user_id': str, user identifier
            - 'screen_name': str, username
            - 'label_raw': str, original classification label

            Profile completeness features (binary):
            - 'has_name': int, whether profile has display name
            - 'has_screen_name': int, whether profile has username
            - 'has_description': int, whether profile has bio
            - 'has_url': int, whether profile has website URL
            - 'has_entities_urls': int, whether profile has entity URLs
            - 'has_profile_image': int, whether uses custom profile image
            - 'has_background_image': int, whether has custom background

            Profile characteristics (binary):
            - 'default_profile': int, whether uses default profile settings
            - 'verified': int, whether account is verified
            - 'geo_enabled': int, whether geolocation is enabled
            - 'has_lang': int, whether language is specified
            - 'has_time_zone': int, whether timezone is set

            Numeric features:
            - 'desc_len': int, description/bio length in characters
            - 'followers': int, number of followers
            - 'friends': int, number of accounts followed
            - 'listed_count': int, number of lists account appears on
            - 'statuses': int, total number of tweets/posts
            - 'favourites': int, number of tweets liked
            - 'account_age_days': float, account age in days

            Derived metrics:
            - 'followers_to_friends_ratio': float, followers/friends ratio
            - 'activity_rate': float, tweets per day since creation
            - 'fav_rate': float, favorites per day since creation

        Notes
        -----
        Features are designed based on research findings that bots often have:
        - Incomplete profiles (missing names, descriptions, custom images)
        - Unusual follower/following ratios
        - High activity rates
        - Default profile settings

        Entity URLs are extracted from profile entities field using regex
        pattern matching for HTTP/HTTPS URLs.
        """
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
    """
    Iterate over items in JSON or JSONL dataset files.

    Provides unified iteration interface for both regular JSON files
    (containing arrays) and JSON Lines format files, handling
    encoding and empty lines appropriately.

    Parameters
    ----------
    path : str or Path
        Path to dataset file (JSON or JSONL format).

    Yields
    ------
    item : dict
        Individual record from the dataset.

    Raises
    ------
    FileNotFoundError
        If the specified dataset file does not exist.

    """
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
