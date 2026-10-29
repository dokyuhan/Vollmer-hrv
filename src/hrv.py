"""Time-domain HRV measures: HRVTool v1.07 (Marcus Vollmer, MIT) ``HRV.m`` -> ``SDNN``, ``RMSSD``, ``pNNx`` / ``pNN50`` (global versions),
plus WINDOWING that is NOT part of HRVTool: the Visuomotor authors compute one value per block; here windows are cut out of the beats of the
whole recording and each window gets the same measures plus quality counts. ``bpm``, ``ibi``, ``ln_rmssd``, ``pnn20`` and the quality fields are ours.
"""
import numpy as np

ECG_METRIC_KEYS = ["bpm", "ibi", "sdnn", "rmssd", "pnn20", "pnn50"]
ECG_METRIC_ATTRS = {
    "bpm":   {"units": "beats/min", "description": "60000 / mean(valid RR intervals in ms). NaN if fewer than min_pairs valid intervals"},
    "ibi":   {"units": "ms", "description": "mean valid RR interval. NaN if fewer than min_pairs valid intervals"},
    "sdnn":  {"units": "ms", "description": "SD of valid RR intervals, divided by n (HRV.SDNN(RR,0)). NaN if fewer than min_pairs valid intervals"},
    "rmssd": {"units": "ms", "description": "sqrt(mean(successive differences^2)) over pairs of consecutive valid RR (HRV.RMSSD(RR,0)). NaN if n_pairs < min_pairs"},
    "pnn20": {"units": "fraction (0-1)", "description": "fraction of successive RR differences > 20 ms (NOT a percentage). NaN if n_pairs < min_pairs"},
    "pnn50": {"units": "fraction (0-1)", "description": "fraction of successive RR differences > 50 ms (HRV.pNN50; NOT a percentage). NaN if n_pairs < min_pairs"},
}
ECG_QUALITY_INT_KEYS = ["n_beats", "n_rr", "n_valid_rr", "n_pairs"]
ECG_QUALITY_KEYS = ECG_QUALITY_INT_KEYS + ["valid_share"]
ECG_QUALITY_ATTRS = {
    "n_beats": {"description": "beats (QRS annotations) whose sample lies inside the window"},
    "n_rr": {"description": "beat-to-beat intervals with BOTH beats inside the window"},
    "n_valid_rr": {"description": "of those, intervals kept by the RR filter (HRV.RRfilter)"},
    "n_pairs": {"description": "pairs of consecutive valid intervals (the successive differences RMSSD/pNN are computed from)"},
    "valid_share": {"description": "n_valid_rr / n_rr (0 when n_rr = 0)"},
}

def sdnn(v):
    """HRV.SDNN(RR, 0) with the default flag: standard deviation of the intervals, divided by n (HRV.m example: repmat([1 .9],1,5) -> 0.0500).
    Unit-agnostic (seconds in -> seconds out)."""
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean((v - v.mean()) ** 2)))


def rmssd(d):
    """HRV.RMSSD(RR, 0) with the default flag, from the successive differences ``d``: sqrt(mean(d^2)) (HRV.m example: repmat([1 .9],1,5) -> 0.1000)."""
    d = np.asarray(d, float)
    return float(np.sqrt(np.mean(d ** 2)))


def pnnx(d, x_ms):
    """Fraction of successive differences ``d`` (seconds) whose magnitude exceeds x_ms (HRV.pNNx / HRV.pNN50 with the default flag, divided by n)."""
    d = np.asarray(d, float)
    return float(np.mean(np.abs(d) > x_ms / 1000.0))


def hrv_windows_from_beats(beats, rr, starts, win, min_pairs):
    """HRV measures and quality fields for every window, from the beats/RR series of the WHOLE recording.
    A window [s, s+win) (samples) contains the intervals whose two beats both lie inside it. ``min_pairs`` is chosen by the caller (HRV.m itself returns NaN below a sample size of 5). Values are NaN below ``min_pairs`` valid intervals
    (ibi, bpm, sdnn) or valid successive pairs (rmssd, pnn20, pnn50). No accept/reject decision is made: valid_share and n_pairs are stored.
    Returns {"metrics": {...}, "derived": {"ln_rmssd": ...}, "quality": {...}}."""
    n = len(starts)
    out = {k: np.full(n, np.nan) for k in ECG_METRIC_KEYS}
    out["ln_rmssd"] = np.full(n, np.nan)
    q = {k: np.zeros(n, dtype=np.int64) for k in ECG_QUALITY_INT_KEYS}
    q["valid_share"] = np.zeros(n)
    for i, s in enumerate(starts):
        lo, hi = np.searchsorted(beats, s, "left"), np.searchsorted(beats, s + win, "left")
        r = rr[lo:hi - 1] if (len(rr) and hi - 1 > lo) else np.array([])
        ok = ~np.isnan(r)
        n_valid = int(ok.sum())
        n_pairs = int(np.sum(ok[1:] & ok[:-1])) if len(r) > 1 else 0
        q["n_beats"][i], q["n_rr"][i], q["n_valid_rr"][i], q["n_pairs"][i] = hi - lo, len(r), n_valid, n_pairs
        q["valid_share"][i] = n_valid / len(r) if len(r) else 0.0
        if n_valid >= min_pairs:
            v_ms = 1000.0 * r[ok]
            out["ibi"][i] = float(v_ms.mean())
            out["bpm"][i] = 60000.0 / out["ibi"][i]
            out["sdnn"][i] = sdnn(v_ms)
        if n_pairs >= min_pairs:
            d = np.diff(r); d = d[~np.isnan(d)]                               
            out["rmssd"][i] = 1000.0 * rmssd(d)
            out["pnn20"][i] = pnnx(d, 20)
            out["pnn50"][i] = pnnx(d, 50)
            out["ln_rmssd"][i] = float(np.log(out["rmssd"][i])) if out["rmssd"][i] > 0 else np.nan
    return {"metrics": {k: out[k] for k in ECG_METRIC_KEYS}, "derived": {"ln_rmssd": out["ln_rmssd"]}, "quality": q}
