from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

for p in (PROCESSED, RESULTS, FIGURES, MODELS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42

SESSION_GAP_MS = 30 * 60 * 1000
MIN_ITEM_COUNT = 5
MIN_SESSION_LEN = 3
MAX_SESSION_LEN = 100
MAX_VISITOR_EVENTS = 1000
SPLIT = (0.8, 0.1, 0.1)

EVENT_CODES = {"view": 0, "addtocart": 1, "transaction": 2}
EVENT_NAMES = {v: k for k, v in EVENT_CODES.items()}
REWARDS = {0: 0.2, 1: 0.6, 2: 1.0}

STATE_LEN = 10
SLATE_SIZE = 10
CUTOFFS = (5, 10, 20)
