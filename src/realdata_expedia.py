import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


NUMERIC_FEATURE_COLUMNS = [
    "site_id",
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_country_id",
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "promotion_flag",
    "srch_destination_id",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "random_bool",
    "comp1_rate",
    "comp1_inv",
    "comp1_rate_percent_diff",
    "comp2_rate",
    "comp2_inv",
    "comp2_rate_percent_diff",
    "comp3_rate",
    "comp3_inv",
    "comp3_rate_percent_diff",
    "comp4_rate",
    "comp4_inv",
    "comp4_rate_percent_diff",
    "comp5_rate",
    "comp5_inv",
    "comp5_rate_percent_diff",
    "comp6_rate",
    "comp6_inv",
    "comp6_rate_percent_diff",
    "comp7_rate",
    "comp7_inv",
    "comp7_rate_percent_diff",
    "comp8_rate",
    "comp8_inv",
    "comp8_rate_percent_diff",
]


def _build_reward(df: pd.DataFrame, reward_mode: str) -> np.ndarray:
    booking = pd.to_numeric(df["booking_bool"], errors="coerce").fillna(0.0).to_numpy()
    if reward_mode == "booking_bool":
        return booking.astype(np.float32)
    if reward_mode == "price_times_booking":
        price = pd.to_numeric(df["price_usd"], errors="coerce").fillna(0.0).to_numpy()
        return (price * booking).astype(np.float32)
    if reward_mode == "gross_bookings_usd":
        if "gross_bookings_usd" not in df.columns:
            raise ValueError("gross_bookings_usd is not available in input dataframe.")
        return pd.to_numeric(df["gross_bookings_usd"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    raise ValueError(f"Unsupported reward_mode: {reward_mode}")


def load_expedia_train_triplets(
    train_csv_path: str,
    max_rows: Optional[int] = None,
    random_seed: int = 2026,
    reward_mode: str = "booking_bool",
) -> dict:
    required_cols = [
        "prop_starrating",
        "price_usd",
        "booking_bool",
        "gross_bookings_usd",
    ]
    usecols = list(dict.fromkeys(NUMERIC_FEATURE_COLUMNS + required_cols))
    df = pd.read_csv(train_csv_path, usecols=usecols, nrows=max_rows, low_memory=False)

    for col in NUMERIC_FEATURE_COLUMNS + ["price_usd", "prop_starrating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic cleaning: finite values, clipping and float32 consistency.
    X_df = df[NUMERIC_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X_df.to_numpy(dtype=np.float32)
    X = np.clip(X, -20.0, 20.0)

    P = pd.to_numeric(df["price_usd"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    P = np.clip(P, 0.0, 1e4)

    Y = _build_reward(df, reward_mode=reward_mode)
    Y = np.clip(np.nan_to_num(Y, nan=0.0), 0.0, 1e6).astype(np.float32)

    starrating = pd.to_numeric(df["prop_starrating"], errors="coerce").fillna(-1).to_numpy(np.int64)

    rng = np.random.RandomState(random_seed)
    perm = rng.permutation(len(X))
    return {
        "X": X[perm],
        "P": P[perm],
        "Y": Y[perm],
        "starrating": starrating[perm],
        "feature_columns": NUMERIC_FEATURE_COLUMNS,
    }


def get_shifted_split(data: dict, split_mode: str, split_config: Optional[Dict[str, Any]] = None) -> dict:
    """
    Hotel-type split: train on low star ratings, test on high (default 1–3 vs 4–5).
    """
    if split_mode != "hotel_type":
        raise ValueError(f"Only split_mode='hotel_type' is supported, got: {split_mode!r}")

    if split_config is None:
        split_config = {}

    star = data["starrating"]
    train_stars = split_config.get("train_stars", [1, 2, 3])
    test_stars = split_config.get("test_stars", [4, 5])
    train_mask = np.isin(star, np.asarray(train_stars))
    test_mask = np.isin(star, np.asarray(test_stars))

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Empty split: train_stars={train_stars}, test_stars={test_stars}"
        )

    return {
        "train": {
            "X": data["X"][train_mask],
            "P": data["P"][train_mask],
            "Y": data["Y"][train_mask],
            "group": star[train_mask],
        },
        "test": {
            "X": data["X"][test_mask],
            "P": data["P"][test_mask],
            "Y": data["Y"][test_mask],
            "group": star[test_mask],
        },
        "group_name": "prop_starrating",
    }
