import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotstyle
from config import FIGURES, PROCESSED, RAW, RESULTS
from plotstyle import INK, INK_2, SERIES

plotstyle.apply()


def funnel(raw):
    counts = raw["event"].value_counts().reindex(["view", "addtocart", "transaction"])
    labels = ["View", "Add to cart", "Purchase"]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh(labels[::-1], counts.values[::-1], color=SERIES[0], height=0.55)
    ax.set_xscale("log")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(counts.values[::-1]):
        share = v / counts.sum() * 100
        ax.text(v * 1.15, i, f"{v:,}  ({share:.1f}%)", va="center", color=INK, fontsize=9)
    ax.set_xlim(1e4, counts.max() * 12)
    ax.set_title("Events by type (raw data, log scale)")
    plotstyle.save(fig, FIGURES / "01_event_funnel.png")


def daily_activity(raw, stats):
    day = pd.to_datetime(raw["timestamp"], unit="ms").dt.floor("D")
    daily = raw.groupby([day, "event"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(daily.index, daily["view"], color=SERIES[0], label="Views")
    ax.set_ylabel("Views per day")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    top = ax.get_ylim()[1]
    for key, label, ha in (("val_start", "validation ", "right"), ("test_start", " test", "left")):
        x = pd.Timestamp(stats[key]).floor("D")
        ax.axvline(x, color=INK_2, linewidth=1, linestyle="--")
        ax.text(x, top * 0.97, label, color=INK_2, fontsize=8, va="top", ha=ha)
    ax.set_title("Daily views over the collection period")
    plotstyle.save(fig, FIGURES / "02_daily_activity.png")


def hourly(raw):
    hour = pd.to_datetime(raw["timestamp"], unit="ms").dt.hour
    share = raw.groupby([hour, "event"]).size().unstack(fill_value=0)
    share = share / share.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, (col, label) in enumerate([("view", "Views"), ("transaction", "Purchases")]):
        ax.plot(share.index, share[col], color=SERIES[i], label=label, marker="o", markersize=3)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Share of events (%)")
    ax.legend(loc="upper left")
    ax.set_title("When shoppers browse and buy")
    plotstyle.save(fig, FIGURES / "03_hourly_pattern.png")


def session_lengths(sessions):
    lens = sessions.groupby("sid").size()
    fig, ax = plt.subplots(figsize=(7, 3))
    bins = np.arange(2, 41)
    ax.hist(lens.clip(upper=40), bins=bins, color=SERIES[0], rwidth=0.85)
    ax.set_xlabel("Events per session (sessions over 40 grouped at 40)")
    ax.set_ylabel("Sessions")
    ax.set_title(f"Session length after cleaning (median {lens.median():.0f}, mean {lens.mean():.1f})")
    plotstyle.save(fig, FIGURES / "04_session_length.png")
    return lens


def long_tail(sessions):
    counts = sessions.loc[sessions["split"] == "train", "item"].value_counts().values
    cum = np.cumsum(counts) / counts.sum() * 100
    top20 = cum[int(len(counts) * 0.2) - 1]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(1, len(counts) + 1) / len(counts) * 100, cum, color=SERIES[0])
    ax.axvline(20, color=INK_2, linewidth=1, linestyle="--")
    ax.text(21, 10, f"top 20% of products = {top20:.0f}% of interactions", color=INK_2, fontsize=9)
    ax.set_xlabel("Products, most popular first (%)")
    ax.set_ylabel("Cumulative interactions (%)")
    ax.set_title("Popularity is concentrated in a small share of products")
    plotstyle.save(fig, FIGURES / "05_long_tail.png")
    return float(top20)


def conversion(sessions):
    per = sessions.groupby("sid")["event"].agg(lambda s: s.max())
    return {
        "sessions_with_cart": float((per >= 1).mean()),
        "sessions_with_purchase": float((per == 2).mean()),
    }


def main():
    raw = pd.read_csv(RAW / "events.csv", usecols=["timestamp", "event", "itemid", "visitorid"])
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    stats = json.loads((PROCESSED / "stats.json").read_text())

    funnel(raw)
    daily_activity(raw, stats)
    hourly(raw)
    lens = session_lengths(sessions)
    top20 = long_tail(sessions)

    summary = {
        "raw_view_to_cart_ratio": float((raw["event"] == "view").sum() / (raw["event"] == "addtocart").sum()),
        "raw_visitors_single_event": float((raw.groupby("visitorid").size() == 1).mean()),
        "clean_session_len_median": float(lens.median()),
        "clean_session_len_mean": float(lens.mean()),
        "train_top20pct_item_share": top20,
        **conversion(sessions),
    }
    (RESULTS / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
