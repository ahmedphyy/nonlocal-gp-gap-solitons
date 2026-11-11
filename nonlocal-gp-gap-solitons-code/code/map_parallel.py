"""Base sweep of the parameter map with failure categories, run in
parallel over (family, b).  Same algorithm as the base sweep of
revision_computations.stage_mapcheck (and of run_computations.stage_map),
with every converged point classified as accepted or converged but not
localized, and the first Newton failure in each direction recorded."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
from multiprocessing import Pool
import numpy as np
from nonlocal_gp import (fourier_grid, newton_local, bloch_hump, gauss_hump,
                         first_gap, odd_part)
from revision_computations import rel_tail, update_summary, C0, TAIL_REL, DATA

L, N = 16 * np.pi, 1024
A_STEP = 0.02


def dip(xx, x0=np.pi / 2, amp=0.5, width=1.5):
    return odd_part(bloch_hump(xx, x0, amp, width)
                    - bloch_hump(xx, -x0, amp, width)).real


def sweep_one(args):
    fam, b = args
    x, _ = fourier_grid(L, N)
    if fam == "A":
        cnl, parity = C0, "even"
        seeds = [bloch_hump(x, 0.0, 1.0, 3.0), gauss_hump(x, 0.0, 1.2, 2.0)]
    else:
        cnl, parity = -C0, "odd"
        seeds = [dip(x), dip(x, x0=np.pi, amp=1.0, width=2.0),
                 dip(x, amp=1.0, width=2.0)]
    lo, hi = first_gap(b)
    u_mid = mid = None
    for frac in (0.5, 0.35, 0.65, 0.25, 0.75):
        aa = lo + frac * (hi - lo)
        for sd in seeds:
            cand = newton_local(aa, b, cnl, L, N, sd, damped=False,
                                parity=parity)
            if cand is not None and np.max(np.abs(cand)) > 1e-3 \
                    and rel_tail(cand, L) < TAIL_REL:
                u_mid, mid = cand, aa
                break
        if u_mid is not None:
            break
    acc, nonloc, fail = [], [], []
    if u_mid is None:
        return fam, b, None, acc, nonloc, fail
    for direction in (+1, -1):
        u = u_mid.copy()
        a = mid
        while lo - 0.1 < a < hi + 0.1:
            un = newton_local(a, b, cnl, L, N, u, damped=True, parity=parity)
            if un is None or np.max(np.abs(un)) < 1e-4:
                fail.append((a, b))
                break
            u = un
            if rel_tail(u, L) < TAIL_REL:
                acc.append((a, b))
            else:
                nonloc.append((a, b))
            a = a + direction * A_STEP
    return fam, b, mid, acc, nonloc, fail


if __name__ == "__main__":
    t0 = time.time()
    b_pos = np.round(np.arange(0.10, 1.5001, 0.05), 4)
    b_vals = np.concatenate([-b_pos[::-1], b_pos])
    tasks = [(fam, float(b)) for fam in ("A", "B") for b in b_vals]
    with Pool(4) as pool:
        res = pool.map(sweep_one, tasks, chunksize=1)
    cat = {f: dict(acc=[], nonloc=[], fail=[], nostart=[]) for f in ("A", "B")}
    for fam, b, mid, acc, nonloc, fail in res:
        if mid is None:
            cat[fam]["nostart"].append(b)
        cat[fam]["acc"] += acc
        cat[fam]["nonloc"] += nonloc
        cat[fam]["fail"] += fail
    # the starting state is recorded once for each continuation direction;
    # keep each parameter point once
    for f in cat:
        for k in ("acc", "nonloc", "fail"):
            if cat[f][k]:
                cat[f][k] = [tuple(p) for p in
                             np.unique(np.round(np.array(cat[f][k]), 10), axis=0)]
    out = {}
    for f in cat:
        for k, v in cat[f].items():
            out["%s_%s" % (f, k)] = np.array(v) if len(v) else np.zeros((0, 2))
    np.savez(os.path.join(DATA, "revision_map.npz"), **out)
    counts = {f: {k: len(v) for k, v in cat[f].items()} for f in cat}
    nostart = {f: cat[f]["nostart"] for f in cat}
    update_summary("mapcheck_base", {"b_step": 0.05, "a_step": A_STEP,
                                     "tail_rel": TAIL_REL, "counts": counts,
                                     "nostart_b": nostart,
                                     "grid": {"L_over_pi": 16, "N": N}})
    for f in counts:
        print("family %s: %s (no start at b = %s)" % (f, counts[f], nostart[f]))
    print("MAPPAR_DONE %.0f s" % (time.time() - t0), flush=True)
