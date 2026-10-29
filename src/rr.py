"""RR-interval artifact filter: port of HRVTool v1.07 (Marcus Vollmer, MIT), file ``HRV.m``, methods ``rrx`` and ``RRfilter``.

Python indices are 0-based, MATLAB's are 1-based; the logic is line by line the same.
"""
import numpy as np

def rrx(RR):
    """HRV.m -> ``rrx``: relative successive difference, 2*(RR[i]-RR[i-1])/(RR[i]+RR[i-1]); first element NaN."""
    out = np.full(len(RR), np.nan)
    out[1:] = 2 * (RR[1:] - RR[:-1]) / (RR[1:] + RR[:-1])
    return out

def _consec(w):
    """MATLAB ``w(diff(w)==1)``: entries of ``w`` that are directly followed by the next integer."""
    return w[:-1][np.diff(w) == 1] if len(w) > 1 else w[:0]

def rr_filter(RR, limit):
    """HRV.m -> ``RRfilter``. RR in seconds; ``limit`` = boundary of trusted relative RR intervals in percent (HRV.m's own default is 20; no
    default is set here, the caller chooses). Returns a copy in which implausible intervals are NaN.
    Example from HRV.m (limit 20): [1.0, .98, .9, .1, .4, 1.0, .98, .9] -> [1.0, .98, .9, nan, nan, nan, .98, .9]."""
    RR = np.array(RR, float); n = len(RR); lim50 = max(limit, 50)
    RR[RR > 4] = np.nan                                     
    with np.errstate(invalid="ignore"):
        pct = 100 * rrx(RR)
        c = np.zeros(n, bool); c[:-1] = (pct[:-1] > lim50) & (pct[1:] < -lim50)     
        RR[c] = np.nan; pct = 100 * rrx(RR)
        for wl in [x - limit % 10 for x in range(80, limit - 1, -10)]:
            wbp = _consec(_consec(np.where(np.abs(np.diff(pct)) > wl)[0])) 
            for off in (1, 2):
                idx = wbp + off; RR[idx[idx < n]] = np.nan
            pct = 100 * rrx(RR)
        pm = np.where(np.isnan(RR[:n - 2]))[0]
        RR[pm[np.abs(pct[pm + 2]) > 15] + 1] = np.nan; pct = 100 * rrx(RR)      
        pm = np.where(np.abs(pct) > lim50)[0]
        RR[pm - 1] = np.nan; RR[pm] = np.nan                         
    return RR
