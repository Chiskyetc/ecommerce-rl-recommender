"""Turn the raw RetailRocket files into cleaned, indexed sessions with a time-based split."""
import json

import numpy as np
import pandas as pd

from config import (EVENT_CODES, MAX_SESSION_LEN, MAX_VISITOR_EVENTS, MIN_ITEM_COUNT,
                    MIN_SESSION_LEN, PROCESSED, RAW, SESSION_GAP_MS, SPLIT)


def load_events():
    df = pd.read_csv(RAW / "events.csv", usecols=["timestamp", "visitorid", "event", "itemid"])
    df["event"] = df["event"].map(EVENT_CODES).astype("int8")
    return df.drop_duplicates()


def add_sessions(df):
    df = df.sort_values(["visitorid", "timestamp"], kind="mergesort").reset_index(drop=True)
    gap = df.groupby("visitorid")["timestamp"].diff()
    df["sid"] = (gap.isna() | (gap > SESSION_GAP_MS)).cumsum().astype("int64")
    return df


def drop_repeats(df):
    # refreshing the same product page shows up as back-to-back identical events
    same = (
        (df["sid"] == df["sid"].shift())
        & (df["itemid"] == df["itemid"].shift())
        & (df["event"] == df["event"].shift())
    )
    return df[~same]


def filter_sessions(df):
    visitor_counts = df.groupby("visitorid").size()
    df = df[df["visitorid"].isin(visitor_counts[visitor_counts <= MAX_VISITOR_EVENTS].index)]
    # two passes is enough for the counts to settle
    for _ in range(2):
        item_counts = df["itemid"].value_counts()
        df = df[df["itemid"].isin(item_counts[item_counts >= MIN_ITEM_COUNT].index)]
        sess_len = df.groupby("sid")["itemid"].transform("size")
        df = df[(sess_len >= MIN_SESSION_LEN) & (sess_len <= MAX_SESSION_LEN)]
    return df


def time_split(df):
    starts = df.groupby("sid")["timestamp"].min().sort_values()
    n = len(starts)
    cut1, cut2 = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    label = pd.Series("train", index=starts.index)
    label.iloc[cut1:cut2] = "val"
    label.iloc[cut2:] = "test"
    df = df.copy()
    df["split"] = df["sid"].map(label)
    return df, starts.iloc[cut1], starts.iloc[cut2]


def index_items(df):
    train_items = np.sort(df.loc[df["split"] == "train", "itemid"].unique())
    item2idx = {int(it): i + 1 for i, it in enumerate(train_items)}  # 0 is padding
    df = df[df["itemid"].isin(item2idx)].copy()
    df["item"] = df["itemid"].map(item2idx).astype("int32")
    sess_len = df.groupby("sid")["item"].transform("size")
    return df[sess_len >= 2], item2idx


def item_categories(item2idx):
    props = pd.read_csv(RAW / "categoryid_rows.csv", header=None,
                        names=["timestamp", "itemid", "property", "value"])
    props = props.sort_values("timestamp").drop_duplicates("itemid", keep="last")
    tree = pd.read_csv(RAW / "category_tree.csv")
    parent = dict(zip(tree["categoryid"], tree["parentid"]))
    cats = pd.DataFrame({"itemid": list(item2idx), "item": list(item2idx.values())})
    cats = cats.merge(props[["itemid", "value"]], on="itemid", how="left")
    cats["category"] = cats["value"].fillna(-1).astype(int)
    cats["parent"] = cats["category"].map(parent).fillna(-1).astype(int)
    return cats.drop(columns="value").sort_values("item")


def build_categories_raw():
    # grep-equivalent pass over the large property files so we only keep categoryid rows
    out = RAW / "categoryid_rows.csv"
    if out.exists():
        return
    with open(out, "w") as fo:
        for part in ("item_properties_part1.csv", "item_properties_part2.csv"):
            with open(RAW / part) as fi:
                for line in fi:
                    if ",categoryid," in line:
                        fo.write(line)


def main():
    build_categories_raw()
    raw = load_events()
    df = add_sessions(raw)
    df = drop_repeats(df)
    df = filter_sessions(df)
    df, val_start, test_start = time_split(df)
    df, item2idx = index_items(df)
    cats = item_categories(item2idx)

    cols = ["sid", "visitorid", "timestamp", "itemid", "item", "event", "split"]
    df[cols].to_parquet(PROCESSED / "sessions.parquet", index=False)
    cats.to_parquet(PROCESSED / "items.parquet", index=False)

    stats = {
        "raw_events": int(len(raw)),
        "raw_visitors": int(raw["visitorid"].nunique()),
        "raw_items": int(raw["itemid"].nunique()),
        "events": int(len(df)),
        "sessions": int(df["sid"].nunique()),
        "items": len(item2idx),
        "val_start": str(pd.to_datetime(val_start, unit="ms")),
        "test_start": str(pd.to_datetime(test_start, unit="ms")),
    }
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        stats[split] = {
            "sessions": int(part["sid"].nunique()),
            "events": int(len(part)),
            "views": int((part["event"] == 0).sum()),
            "addtocart": int((part["event"] == 1).sum()),
            "transactions": int((part["event"] == 2).sum()),
        }
    (PROCESSED / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
