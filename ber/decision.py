"""Anchor decisions fitted on calibration only, with an unseen-country policy."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def anchor_features(row, probabilities):
    p = sorted(probabilities, reverse=True)
    top = p[0] if p else 0.
    second = p[1] if len(p) > 1 else 0.
    return [top, second, top-second, sum(p), np.log1p(len(p)), row.get("address_missing", False),
            row.get("channel_agreement", 0), sum(x > .5 for x in p)]


def fit_gate(rows, probabilities):
    x = [anchor_features(r, probabilities[r["offset"]:r["offset"]+r["count"]]) for r in rows]
    y = [bool(r["truth"]) for r in rows]
    if len(set(y)) != 2:
        return None
    return LogisticRegression(C=1, max_iter=1000, random_state=2026).fit(x, y)


def utility_set(ids, probabilities, missing_rate=0., p_nonempty=1.):
    """Independent-Bernoulli expected F0.5; Poisson missing links truncated at 16.

    Experimental approximation: pair independence and homogeneous missing rate
    are not guaranteed. Never silently replaces the calibrated threshold policy.
    """
    import math
    order = np.argsort(-np.asarray(probabilities), kind="stable")
    probs = np.asarray(probabilities)[order]
    # Cardinality distribution of all retrieved links, prefix/suffix convolutions.
    prefix = [np.array([1.])]
    for p in probs:
        prefix.append(np.convolve(prefix[-1], [1-p, p]))
    suffix = [None]*(len(probs)+1)
    suffix[-1] = np.array([1.])
    for i in range(len(probs)-1, -1, -1):
        suffix[i] = np.convolve(suffix[i+1], [1-probs[i], probs[i]])
    tail = np.array([math.exp(-missing_rate)*missing_rate**n/math.factorial(n) for n in range(17)])
    tail[-1] += max(0., 1-tail.sum())
    best_score, best = 1-p_nonempty, []
    for k in range(1, len(probs)+1):
        outside = np.convolve(suffix[k], tail)
        score = 0.
        for tp, probability in enumerate(prefix[k]):
            if tp:
                score += probability*np.sum(outside*(1.25*tp/(.25*(tp+np.arange(len(outside)))+k)))
        if score > best_score:
            best_score, best = score, [ids[i] for i in order[:k]]
    return best


def decide(bundle, row, probabilities):
    if not len(probabilities):
        return []
    country = row.get("country", "")
    gate = bundle.get("anchor_gate")
    nonempty = float(gate.predict_proba([anchor_features(row, probabilities)])[0, 1]) if gate is not None else 1.
    policy = bundle.get("policies", {}).get(country, bundle.get("robust_policy", {}))
    threshold = policy.get("threshold", bundle["threshold"])
    empty_gate = policy.get("empty_gate", bundle["empty_gate"])
    if max(probabilities) < empty_gate or nonempty < policy.get("gate_threshold", bundle.get("gate_threshold", 0.)):
        return []
    if bundle.get("decoder") == "utility":
        return utility_set(row["candidates"], probabilities, bundle.get("missing_rate", 0), nonempty)
    source_thresholds = policy.get("sources", {})
    return [i for i, p in zip(row["candidates"], probabilities) if p >= source_thresholds.get(i[:2], threshold)]
