import itertools
import json

import evaluate as ev
import steps
from baselines import ItemKNN, Popularity, load_train
from config import RESULTS


def main():
    n = steps.n_items()
    train = load_train()
    val, test = steps.build("val"), steps.build("test")

    pop = Popularity(n).fit(train)

    grid = []
    for include_seen, decay, boost in itertools.product([False, True], [0.5, 0.8], [0.5, 1.0, 2.0]):
        if not include_seen and boost != 1.0:
            continue
        knn = ItemKNN(n, decay=decay, include_seen=include_seen, seen_boost=boost).fit(train)
        r, _ = ev.evaluate(knn, val, n)
        grid.append({"include_seen": include_seen, "decay": decay, "boost": boost,
                     "val_NDCG@10": r["all"]["NDCG@10"], "val_HR@10": r["all"]["HR@10"]})
        print(grid[-1])
    (RESULTS / "knn_tuning.json").write_text(json.dumps(grid, indent=2))

    best = max((g for g in grid if g["include_seen"]), key=lambda g: g["val_NDCG@10"])
    best_unseen = max((g for g in grid if not g["include_seen"]), key=lambda g: g["val_NDCG@10"])

    results = []
    for name, model in [
        ("Popularity", pop),
        ("Item-KNN (new items only)",
         ItemKNN(n, decay=best_unseen["decay"], include_seen=False).fit(train)),
        ("Item-KNN",
         ItemKNN(n, decay=best["decay"], include_seen=True, seen_boost=best["boost"]).fit(train)),
    ]:
        r, _ = ev.evaluate(model, test, n, name=name)
        results.append(r)
    table = ev.save(results, "baselines_test")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
