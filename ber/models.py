"""Optional grouped OOF reranking and pairwise ranking experiments."""
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from .common import digest


def score_context(rows, scores):
    result = np.zeros((len(scores), 5), np.float32)
    for row in rows:
        start, size = row["offset"], row["count"]
        p = scores[start:start+size]
        if not len(p): continue
        order = np.argsort(-p, kind="stable")
        ranks = np.empty(size); ranks[order] = np.arange(size)
        source_rank = np.zeros(size)
        for source in ("S2", "S3"):
            ix = [i for i in order if row["candidates"][i].startswith(source)]
            for rank, i in enumerate(ix): source_rank[i] = rank
        sorted_p = p[order]
        margin = sorted_p[0]-sorted_p[1] if size > 1 else sorted_p[0]
        result[start:start+size] = np.column_stack([p, p.max()-p, 1/(1+ranks), 1/(1+source_rank), np.full(size, margin)])
    return result


def oof_scores(estimator, x, y, weights, rows, folds=3):
    scores = np.full(len(y), np.nan)
    for fold_no in range(folds):
        held = np.zeros(len(y), bool)
        for row in rows:
            if int(digest(row["group"])[:8], 16) % folds == fold_no:
                held[row["offset"]:row["offset"]+row["count"]] = True
        fit = ~held & (weights > 0)
        if not held.any() or len(np.unique(y[fit])) != 2:
            raise ValueError("OOF fold too small")
        model = clone(estimator)
        model.fit(x[fit], y[fit], sample_weight=weights[fit])
        scores[held] = model.predict_proba(x[held])[:, 1]
    if not np.isfinite(scores).all(): raise ValueError("Incomplete OOF predictions")
    return scores


class PairwiseRanker:
    """Within-anchor logistic pair differences, calibrated separately afterward."""
    def __init__(self, seed=2026):
        self.seed = seed
        self.model = LogisticRegression(C=.1, max_iter=500, random_state=seed)

    def fit_groups(self, x, y, rows, max_pairs=100_000):
        rng = np.random.default_rng(self.seed)
        differences = []
        for row in rows:
            indices = np.arange(row["offset"], row["offset"]+row["count"])
            positive, negative = indices[y[indices] == 1], indices[y[indices] == 0]
            if not len(positive) or not len(negative): continue
            for p in positive:
                n = rng.choice(negative)
                differences.append(x[p]-x[n])
                if len(differences) >= max_pairs: break
            if len(differences) >= max_pairs: break
        if not differences: raise ValueError("No within-anchor ranking comparisons")
        d = np.asarray(differences)
        self.model.fit(np.concatenate([d, -d]), np.r_[np.ones(len(d)), np.zeros(len(d))])
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def get_params(self):
        return {"kind": "pairwise_logistic", "seed": self.seed}


class SeedEnsemble:
    def __init__(self, models):
        self.models = models
        self.max_iter = models[0].max_iter

    def predict_proba(self, x):
        return np.mean([m.predict_proba(x) for m in self.models], axis=0)

    def get_params(self):
        return {"kind": "seed_ensemble", "members": [m.get_params() for m in self.models]}
