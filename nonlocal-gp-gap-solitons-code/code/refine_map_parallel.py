"""Boundary refinement of the parameter map, run in parallel.  Same
algorithm as the refinement part of revision_computations.stage_mapcheck:
for selected b, a start is found at mid-gap on the base grid
(L = 16 pi, N = 1024) and continued in both directions under four
settings; the accepted interval and the stopping reasons are recorded."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import time
from multiprocessing import Pool
import numpy as np
from nonlocal_gp import (fourier_grid, newton_local, bloch_hump, gauss_hump,
                         first_gap, odd_part)
from revision_computations import (accepted_interval, change_window, resample,
                                   rel_tail, update_summary, C0, TAIL_REL)

L0, N0 = 16 * np.pi, 1024
CONFIGS = [("base step", 16 * np.pi, 1024, 0.02),
           ("half step", 16 * np.pi, 1024, 0.01),
           ("larger window", 24 * np.pi, 1536, 0.01),
           ("finer grid", 16 * np.pi, 2048, 0.01)]


def dip(xx, x0=np.pi / 2, amp=0.5, width=1.5):
    return odd_part(bloch_hump(xx, x0, amp, width)
                    - bloch_hump(xx, -x0, amp, width)).real


def start(fam, b):
    x, _ = fourier_grid(L0, N0)
    if fam == "A":
        cnl, parity = C0, "even"
        seeds = [bloch_hump(x, 0.0, 1.0, 3.0), gauss_hump(x, 0.0, 1.2, 2.0)]
    else:
        cnl, parity = -C0, "odd"
        seeds = [dip(x), dip(x, x0=np.pi, amp=1.0, width=2.0),
                 dip(x, amp=1.0, width=2.0)]
    lo, hi = first_gap(b)
    for frac in (0.5, 0.35, 0.65, 0.25, 0.75):
        aa = lo + frac * (hi - lo)
        for sd in seeds:
            cand = newton_local(aa, b, cnl, L0, N0, sd, damped=False,
                                parity=parity)
            if cand is not None and np.max(np.abs(cand)) > 1e-3 \
                    and rel_tail(cand, L0) < TAIL_REL:
                return aa, cand, cnl, parity
    return None


def task(args):
    fam, b, cfg = args
    t0 = time.time()
    st = start(fam, b)
    if st is None:
        return (fam, b, cfg[0], None)
    mid, u_mid, cnl, parity = st
    name, Lc, Nc, da = cfg
    if Lc != L0:
        seed = np.real(change_window(u_mid, L0, Lc))
    elif Nc != N0:
        seed = np.real(resample(u_mid, Nc))
    else:
        seed = u_mid
    # the rounding level of the residual on the collocation grid grows with
    # the largest wavenumber; on the finer grid it exceeds 1e-12
    tol = 1e-11 if (Nc / Lc) > (N0 / L0) * 1.01 else 1e-12
    us = newton_local(mid, b, cnl, Lc, Nc, seed, damped=True, parity=parity,
                      tol=tol)
    if us is None:
        return (fam, b, name, {"start_failed": True})
    r = accepted_interval(mid, b, cnl, Lc, Nc, us, parity, da, tol=tol)
    out = dict(a_start=float(mid), lower=r[-1][0], lower_stop=r[-1][1],
               upper=r[1][0], upper_stop=r[1][1], seconds=time.time() - t0)
    print("%s b=%+.2f %-13s: [%s, %s] (%s | %s) %.0f s"
          % (fam, b, name, r[-1][0], r[1][0], r[-1][1], r[1][1],
             time.time() - t0), flush=True)
    return (fam, b, name, out)


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    cfgs = [c for c in CONFIGS if not only or c[0] in only]
    tasks = [(fam, b, cfg) for fam in ("A", "B")
             for b in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5) for cfg in cfgs]
    # most expensive first
    tasks.sort(key=lambda t: -t[2][2] / t[2][3])
    with Pool(3) as pool:
        res = pool.map(task, tasks, chunksize=1)
    from revision_computations import load_summary
    prev = load_summary().get("mapcheck_refinement", [])
    rows = {"%s_%+.1f" % (r["fam"], r["b"]): r for r in prev}
    for fam, b, name, out in res:
        key = "%s_%+.1f" % (fam, b)
        rows.setdefault(key, {"fam": fam, "b": b, "start": out is not None})
        if out is not None:
            rows[key]["start"] = True
            rows[key][name] = out
    order = [r for k, r in sorted(rows.items(), key=lambda kv: (kv[1]["fam"], kv[1]["b"]))]
    update_summary("mapcheck_refinement", order)
    print("REFINE_DONE", flush=True)
