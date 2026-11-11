"""Recompute the family C separation states whose refined-grid residual
shows under-resolution, on a grid with 1.5 times the resolution."""
import os
import numpy as np
from nonlocal_gp import newton_complex_pt
from revision_computations import (diag_full, hump_positions, resample, load_summary,
                                   update_summary, A0, B0, C0, DATA)
L, N0, N1 = 36 * np.pi, 2304, 3456
dc = dict(np.load(os.path.join(DATA, "revision_familyC.npz")))
s = load_summary()
rows = s["familyC"]["separations"]
for r in rows:
    if not r.get("converged") or r["res_nonlocal"] < 1e-9:
        continue
    key = "u_m%d" % r["m_seed"]
    u = newton_complex_pt(A0, B0, C0, L, N1, resample(dc[key], N1), damped=False, maxit=20)
    if u is None:
        u = newton_complex_pt(A0, B0, C0, L, N1, resample(dc[key], N1), damped=True, maxit=100)
    if u is None:
        print("m=%d: no convergence at N=%d" % (r["m_seed"], N1)); continue
    dg = diag_full(u, A0, B0, C0, L)
    pf, pg = hump_positions(u, L, N1)
    print("m=%2d N=%d: R_nl=%.2e tail=%.1e |u|^2=%.4f Q=%.4f Re at %.2f pi, Im at %.2f pi (N=%d: R_nl=%.1e |u|^2=%.4f)"
          % (r["m_seed"], N1, dg["res_nonlocal"], dg["tail_rel"], dg["l2sq"], dg["Q"], pf, pg,
             N0, r["res_nonlocal"], r["l2sq"]), flush=True)
    res_coarse = r["res_nonlocal"]
    r.update(dg, pos_re=pf, pos_im=pg, N=N1, res_N2304=res_coarse)
    dc[key + "_N%d" % N1] = u
update_summary("familyC", s["familyC"])
np.savez(os.path.join(DATA, "revision_familyC.npz"), **dc)
