"""QRS detector: port of HRVTool v1.07 (Marcus Vollmer, MIT) -- ``singleqrs.m``, ``mvqrs_ann.m``, ``mvqrs_checkbeat.m``,
``tma_filter.m``, ``windowed_extrema.m`` and ``HRV.nanzscore`` (from ``HRV.m``).

Pipeline (singleqrs): trimmed-moving-average baseline removal -> z-score -> windowed range (max - min) -> adaptive threshold between the
running min and max -> beat annotated at the START of the QRS rise (not at the R apex). Because the range is used, the QRS polarity does not matter.
Python indices are 0-based, MATLAB's 1-based. The sampling rate is always a parameter.
"""
import numpy as np
import pandas as pd

from .rr import rr_filter

def _mround(x):
    """MATLAB round(): half away from zero."""
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))

def windowed_extrema(sig, wl):
    """windowed_extrema.m: TRAILING window of wl samples (partial at the start). Returns (min, max)."""
    s = pd.Series(sig)
    return s.rolling(wl, min_periods=1).min().to_numpy(), s.rolling(wl, min_periods=1).max().to_numpy()

def tma_filter(sig, wl, pct, chunk=20000):
    """tma_filter.m: signal minus trimmed moving average (centered window, NaN outside the signal)."""
    if wl % 2 != 1:
        wl += 1
    n, h = len(sig), (wl - 1) // 2
    pad = np.concatenate([np.full(h, np.nan), sig, np.full(h, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, wl)
    lo, hi = 1 + _mround(wl * pct / 2), wl - _mround(wl * (1 - pct) / 2)
    out = np.empty(n)
    for a in range(0, n, chunk):
        w = np.sort(win[a:a + chunk], axis=1)
        out[a:a + chunk] = np.nanmean(w[:, lo - 1:hi], axis=1)
    return sig - out

def nanzscore(x):
    """HRV.m -> ``nanzscore`` with opt = 0 (divide by n-1)."""
    m = np.nanmean(x); n = np.sum(~np.isnan(x))
    return (x - m) / np.sqrt(np.nansum((x - m) ** 2) / max(n - 1, 1))

def _ma_causal(x, w):
    """MATLAB ``filter(ones(1,w)/w, 1, x)``."""
    c = np.cumsum(x); out = c.copy(); out[w:] = c[w:] - c[:-w]
    return out / w

def mvqrs_checkbeat(sig, Fs, wl, Beat_min, Beat_max, threshold):
    """mvqrs_checkbeat.m. Returns 1-based annotations plus the filtered running max/min and the threshold curve."""
    N = len(sig)
    w1, w2, w3 = _mround(Fs * 60 / Beat_min), _mround(Fs * 60 / (.5 * Beat_max)), _mround(Fs * 60 / Beat_max)
    wmax = _ma_causal(windowed_extrema(sig, w1)[1], w1)
    wmin = _ma_causal(windowed_extrema(sig, w2)[0], w3)
    idx = np.arange(N)
    wmax_s = wmax[np.minimum(idx + w1 - 1, N - 1)]; wmin_s = wmin[np.minimum(idx + w3 - 1, N - 1)]
    thr_range = wmax_s - wmin_s
    thr = threshold * thr_range + wmin_s; thr8 = .8 * threshold * thr_range + wmin_s
    thr_g, thr_s = sig > thr, sig <= thr8
    const = (np.diff(sig) == 0); k = wl - 1      
    cs = np.concatenate([[0], np.cumsum(const)])
    starts = np.where(cs[k:] - cs[:-k] == k)[0] if len(const) >= k else np.array([], int)
    cc = starts[thr_g[starts]]                                     
    if len(cc) == 0:
        return np.array([], int), wmax, wmin, thr
    cc_accept = cc[np.diff(np.concatenate([[-1], cc])) > 1] + 1           
    ip = np.concatenate([[1], np.where(np.diff(thr_g.astype(int)) == 1)[0] + 1]) 
    Ann = np.unique(ip[np.searchsorted(ip, cc_accept, side="right") - 1])  
    ip2 = np.concatenate([[0], np.where(np.diff(thr_s.astype(int)) == 1)[0] + 1])
    for i in range(1, len(Ann)):                                           
        if np.searchsorted(ip2, Ann[i], "left") == np.searchsorted(ip2, Ann[i - 1], "left"):
            Ann[i] = Ann[i - 1]
    return np.unique(Ann), wmax, wmin, thr

def mvqrs_ann(sig, Fs, wl_we, Beat_min, Beat_max, threshold, R):
    """mvqrs_ann.m: beat annotations (1-based) of the z-scored, baseline-free signal; low-range zones are declined."""
    n = len(sig)
    f = windowed_extrema(sig, wl_we)[1] - windowed_extrema(sig, wl_we)[0]  
    Ann, wmax, wmin, thr = mvqrs_checkbeat(f, Fs, _mround(Fs / 25), Beat_min, Beat_max, threshold)
    sw, sm = _mround(Fs * 60 / Beat_min), _mround(Fs * 60 / Beat_max)
    rng = np.empty(n); r1 = wmax[sw - 1:] - wmin[sm - 1:n - sw + sm]
    rng[:len(r1)] = r1; rng[len(r1) - 1:] = rng[len(r1) - 2]
    d = np.zeros(n + 1)
    for k in np.where(rng <= R)[0]:                                      
        d[max(0, k - _mround(sm / 2))] += 1; d[min(n - 1, k + _mround(sw / 2)) + 1] -= 1
    valid = np.cumsum(d[:-1]) == 0
    return Ann[valid[Ann - 1]]

def singleqrs(signal_, Fs, Beat_min, Beat_max, threshold=.5, pct=.25, R=.4):
    """singleqrs.m. ``Beat_min`` / ``Beat_max`` (bpm) are chosen by the caller (singleqrs.m itself defaults to 50 / 220).
    threshold, pct and R keep singleqrs.m's own defaults, and the filter windows are its defaults too (wl_tma = ceil(.2 Fs), wl_we = ceil(Fs/3)).
    Returns 0-based beat samples (QRS onset).
    Returns no beats for recordings shorter than 5 s, with non-finite samples, or constant."""
    sig = np.asarray(signal_, float)
    if len(sig) < 5 * Fs or not np.isfinite(sig).all() or sig.std() == 0:
        return np.array([], dtype=np.int64)
    z = nanzscore(tma_filter(sig, int(np.ceil(.2 * Fs)), pct))
    Ann = mvqrs_ann(z, Fs, int(np.ceil(Fs / 3)), Beat_min, Beat_max, threshold, R)
    Ann = Ann[(Ann > 0) & (Ann <= len(z))]
    return (Ann - 1).astype(np.int64)

def detect_beats_and_rr(x, Fs, Beat_min, Beat_max, limit):
    """Whole-segment detection in the call order of the Visuomotor authors' step4_pat2qrs.m ``get_hrv``: beats, then RR = diff(beats)/Fs, then RRfilter.
    All study parameters (Beat_min, Beat_max in bpm, RR-filter limit in percent) come from the caller.
    Returns (beats [0-based samples], rr_seconds [len(beats)-1, NaN where the filter rejects the interval]; rr[i] = beats[i] -> beats[i+1])."""
    beats = singleqrs(x, Fs, Beat_min, Beat_max)
    if len(beats) > 3:
        rr = rr_filter(np.diff(beats) / Fs, limit)
    else:
        rr = np.full(max(len(beats) - 1, 0), np.nan)
    return beats, rr
