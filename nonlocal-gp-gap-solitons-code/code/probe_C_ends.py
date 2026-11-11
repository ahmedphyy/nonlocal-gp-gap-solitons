"""Probe the ends of the family C continuation with smaller steps."""
import json, os
import numpy as np
from nonlocal_gp import newton_complex_pt, fourier_grid
from revision_computations import diag_full, rel_tail, update_summary, change_window, A0, B0, C0, DATA
L, N = 36 * np.pi, 2304
d = np.load(os.path.join(DATA, "revision_familyC.npz"))
dr = np.load(os.path.join(DATA, "flagship.npz"))
u0 = newton_complex_pt(A0, B0, C0, L, N, change_window(dr["u_C1"], float(dr["L"]), L), damped=True, maxit=100)
import sys
dirs = [int(v) for v in sys.argv[1:]] or [-1, +1]
from revision_computations import load_summary
out = load_summary().get("familyC_ends", {}) if len(dirs) < 2 else {}
for direction, a_end in ((-1, float(d["a"][0])), (+1, float(d["a"][-1]))):
    if direction not in dirs:
        continue
    # march from the reference state to the end with the original step
    u = u0.copy(); a = A0
    while (a_end - a) * direction > 1e-9:
        a = round(a + direction * 0.0025, 6)
        un = newton_complex_pt(a, B0, C0, L, N, u, damped=True, maxit=60)
        if un is None:
            un = newton_complex_pt(a, B0, C0, L, N, u, damped=False, maxit=30)
        u = un
    rows = []
    for da in (0.001, 0.0005, 0.00025):
        while True:
            an = round(a + direction * da, 6)
            un = newton_complex_pt(an, B0, C0, L, N, u, damped=True, maxit=60)
            if un is None:
                un = newton_complex_pt(an, B0, C0, L, N, u, damped=False, maxit=30)
            if un is None:
                break
            dg = diag_full(un, an, B0, C0, L)
            if dg["tail_rel"] >= 5e-3 or min(dg["max_re"], dg["max_im"]) < 1e-2:
                break
            u, a = un, an
            rows.append(dict(a=a, l2sq=dg["l2sq"], Q=dg["Q"], res=dg["res_nonlocal"]))
            print("dir %+d step %.5f: a=%.5f |u|^2=%.4f Q=%.4f res=%.1e" % (direction, da, a, dg["l2sq"], dg["Q"], dg["res_nonlocal"]), flush=True)
            if len(rows) > 40:
                break
    out[str(direction)] = dict(last_a=a, rows=rows)
    print("dir %+d: last converged a = %.5f" % (direction, a), flush=True)
update_summary("familyC_ends", out)
