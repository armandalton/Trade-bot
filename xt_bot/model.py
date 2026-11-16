"""Gradient boosting classifier for P(TP before SL) + Platt calibration fitted on validation only."""
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import FEATURES
from .config import DATA_DIR

MODEL_PATH = DATA_DIR / "model.joblib"


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def make_xy(df):
    return df[FEATURES].values.astype(float), (df["result"] == "tp").astype(int).values


def train_model(train_df, val_df):
    X, y = make_xy(train_df)
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.05, min_samples_leaf=60,
                                         l2_regularization=1.0, early_stopping=False, random_state=7)
    clf.fit(X, y)
    Xv, yv = make_xy(val_df)
    raw = clf.predict_proba(Xv)[:, 1]
    calibrator = None
    if len(yv) >= 100 and yv.sum() >= 20 and (len(yv) - yv.sum()) >= 20:
        calibrator = LogisticRegression(C=1e6).fit(_logit(raw).reshape(-1, 1), yv)
    return {"clf": clf, "calibrator": calibrator, "features": list(FEATURES)}


def predict_proba(bundle, df):
    if len(df) == 0:
        return np.array([])
    raw = bundle["clf"].predict_proba(df[bundle["features"]].values.astype(float))[:, 1]
    if bundle["calibrator"] is None:
        return raw
    return bundle["calibrator"].predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


def pick_threshold(prob, net_r, min_trades):
    """Choose the probability cut-off on VALIDATION data: best mean net R with >= min_trades trades."""
    best = None
    for thr in np.arange(0.25, 0.86, 0.025):
        m = prob >= thr
        n = int(m.sum())
        if n < min_trades:
            continue
        exp = float(np.mean(net_r[m]))
        if best is None or exp > best[1] + 1e-12:
            best = (float(round(thr, 3)), exp, n)
    return best  # (threshold, expectancy_r, n) or None


def save_bundle(bundle, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def predict_kinds(bundles, df):
    """bundles: {'REVERSAL': bundle|None, 'BREAKOUT': bundle|None}. Each row is scored by the model of its own kind."""
    prob = np.zeros(len(df))
    for kind, b in bundles.items():
        if b is None:
            continue
        m = (df["kind"] == kind).values
        if m.any():
            prob[m] = predict_proba(b, df[m])
    return prob


def select_best(df, thresholds, kinds=None):
    """For every touch (symbol, idx) there are two candidates (REVERSAL / BREAKOUT). Keep the candidate with the
    higher calibrated probability, but only if it passes the threshold of its own kind (thresholds chosen on
    validation data). `df` needs columns: prob, kind, symbol, idx, signal_time."""
    if len(df) == 0:
        return df
    d = df.copy()
    d["thr"] = d["kind"].map(thresholds)
    d = d[d["thr"].notna() & (d["prob"] >= d["thr"])]
    if kinds is not None:
        d = d[d["kind"].isin(kinds)]
    d = d.sort_values(["prob"], ascending=False).drop_duplicates(["symbol", "idx"])
    return d.sort_values(["signal_time", "symbol"]).reset_index(drop=True)
