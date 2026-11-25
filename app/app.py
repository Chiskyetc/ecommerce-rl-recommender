"""Demo: replay a real test-period session and compare what each model recommends next.

    streamlit run app/app.py

A specific session can be opened directly, e.g. http://localhost:8501/?session=180765&step=5
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import steps  # noqa: E402
from bandit import BanditRecommender, FeatureBuilder, LinearBandit  # noqa: E402
from baselines import ItemKNN, Popularity, load_train  # noqa: E402
from config import MODELS, PROCESSED, RESULTS, REWARDS, ROOT  # noqa: E402
from models import TorchRecommender  # noqa: E402

FEEDBACK = ROOT / "data" / "feedback.csv"
EVENT_LABEL = {0: "viewed", 1: "added to cart", 2: "purchased"}

st.set_page_config(page_title="Session Recommender Demo", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    from compare import load_net, model_label

    n = steps.n_items()
    train = load_train()
    knn = ItemKNN(n, decay=0.8, include_seen=True, seen_boost=2.0).fit(train)
    models = {"Popularity": Popularity(n).fit(train), "Item-KNN": knn}

    theta_path = MODELS / "linucb_theta.npy"
    if theta_path.exists():
        b = LinearBandit.from_theta(np.load(theta_path))
        models["LinUCB bandit"] = BanditRecommender(b, FeatureBuilder(knn, train, n))

    betas_path = RESULTS / "beta_tuning.json"
    betas = json.loads(betas_path.read_text()) if betas_path.exists() else {}
    for path in sorted(MODELS.glob("*.pt")):
        net, ckpt = load_net(path)
        label = model_label(ckpt)
        if ckpt["q_head"]:
            beta = betas.get(label, {}).get("chosen", 1.0)
            models[label] = TorchRecommender(net, "mix", beta=beta)
        else:
            models[label] = TorchRecommender(net, "ce")

    hybrid_path = MODELS / "linucb_hybrid_theta.npy"
    if hybrid_path.exists() and "GRU (supervised)" in models:
        b = LinearBandit.from_theta(np.load(hybrid_path))
        fb = FeatureBuilder(knn, train, n, scorer=models["GRU (supervised)"])
        models["LinUCB bandit + GRU score"] = BanditRecommender(b, fb)

    items = pd.read_parquet(PROCESSED / "items.parquet").set_index("item")
    pop = np.bincount(train["item"], minlength=n + 1)
    return models, items, pop


@st.cache_data
def load_test():
    return steps.build("test")


def product_label(item, items):
    row = items.loc[item]
    cat = "unknown" if row["category"] < 0 else str(row["category"])
    return f"#{row['itemid']}", cat


def main():
    models, items, pop = load_models()
    test = load_test()
    sids = np.unique(test["sid"])

    st.title("What should this shopper see next?")
    st.caption("Each session below is a real browsing session from the test period, which the "
               "models never saw during training. Product names are anonymised in the source "
               "data, so products are shown by ID and category.")

    with st.sidebar:
        st.header("Session")
        params = st.query_params
        if "sid" not in st.session_state:
            asked = params.get("session")
            if asked is not None and asked.isdigit() and int(asked) in set(sids.tolist()):
                st.session_state.sid = int(asked)
            else:
                st.session_state.sid = int(np.random.choice(sids))
        if st.button("Pick a random session"):
            st.session_state.sid = int(np.random.choice(sids))
        st.caption(f"Session ID: {st.session_state.sid}")
        rows = np.flatnonzero(test["sid"] == st.session_state.sid)
        start = params.get("step", "1")
        start = min(max(int(start), 1), len(rows)) if start.isdigit() else 1
        step = st.slider("Step in session", 1, len(rows), start) if len(rows) > 1 else 1
        preferred = ["Item-KNN", "LinUCB bandit", "LinUCB bandit + GRU score"]
        default = [m for m in preferred if m in models] or [m for m in models if m != "Popularity"][:3]
        model_names = st.multiselect("Models to compare", list(models), default=default)
        reveal = st.checkbox("Reveal what the shopper did next", value=True)

    row = rows[step - 1]
    state, state_ev = test["state"][row], test["state_ev"][row]

    st.subheader("Session so far")
    hist = [
        {"#": i + 1, "product": product_label(it, items)[0],
         "category": product_label(it, items)[1], "action": EVENT_LABEL[ev - 1]}
        for i, (it, ev) in enumerate((it, ev) for it, ev in zip(state, state_ev) if it > 0)
    ]
    st.dataframe(pd.DataFrame(hist), hide_index=True, width="stretch")

    target, t_event = int(test["target"][row]), int(test["event"][row])
    if reveal:
        prod, cat = product_label(target, items)
        st.info(f"Next, the shopper **{EVENT_LABEL[t_event]}** product {prod} (category {cat}).")

    cols = st.columns(max(len(model_names), 1))
    seen = set(int(x) for x in state if x > 0)
    for col, name in zip(cols, model_names):
        rec = models[name].recommend(state[None], state_ev[None], 10)[0]
        table = []
        for rank, it in enumerate(rec, 1):
            prod, cat = product_label(int(it), items)
            table.append({"rank": rank, "product": prod, "category": cat,
                          "seen": "yes" if int(it) in seen else "",
                          "match": "<- next" if reveal and int(it) == target else ""})
        hit = target in rec
        with col:
            st.markdown(f"**{name}**")
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
            if reveal:
                st.caption(f"Reward: {REWARDS[t_event] if hit else 0.0:.1f}"
                           + (f" (hit at rank {list(rec).index(target) + 1})" if hit else " (miss)"))

    st.divider()
    st.subheader("Your feedback")
    st.caption("Prepared for future user testing. No responses were collected or used in this "
               "study's evaluation.")
    with st.form("feedback", clear_on_submit=True):
        rated = st.selectbox("Which model's list looks most useful for this shopper?",
                             model_names or list(models))
        relevance = st.slider("How relevant were its recommendations? (1 = not at all, 5 = very)",
                              1, 5, 3)
        would_click = st.radio("Would you click on at least one of them?", ["Yes", "No"],
                               horizontal=True)
        note = st.text_input("Anything else? (optional)")
        if st.form_submit_button("Submit"):
            entry = pd.DataFrame([{
                "time": datetime.now().isoformat(timespec="seconds"),
                "session": st.session_state.sid, "step": step, "model": rated,
                "relevance": relevance, "would_click": would_click, "note": note,
            }])
            entry.to_csv(FEEDBACK, mode="a", header=not FEEDBACK.exists(), index=False)
            st.success("Thanks, saved.")


main()
