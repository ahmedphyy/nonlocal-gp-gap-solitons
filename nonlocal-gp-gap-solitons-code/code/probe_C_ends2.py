"""Family C branch ends: follow the lower end without a cap, and
diagnose the failure at the upper end."""
import os
import numpy as np
from nonlocal_gp import newton_complex_pt, diff2_matrix, fourier_grid
from revision_computations import (diag_full, update_summary, change_window,
                                   rel_tail, A0, B0, C0, DATA)
L, N = 36 * np.pi, 2304
x, _ = fourier_grid(L, N)
dr = np.load(os.path.join(DATA, "flagship.npz"))
u0 = newton_complex_pt(A0, B0, C0, L, N, change_window(dr["u_C1"], float(dr["L"]), L),
                       damped=True, maxit=100)
out = {}
# lower end, step 0.001, no cap
u, a, rows = u0.copy(), A0, []
reason = None
while a > 0.72:
    an = round(a - 0.001, 6)
    un = newton_complex_pt(an, B0, C0, L, N, u, damped=True, maxit=60)
    if un is None:
        reason = "Newton failure"; break
    dg = diag_full(un, an, B0, C0, L)
    if dg["tail_rel"] >= 5e-3:
        reason = "localization criterion (tau=%.2e)" % dg["tail_rel"]; break
    if min(dg["max_re"], dg["max_im"]) < 1e-2:
        reason = "lost mixed character"; break
    u, a = un, an
    rows.append(dict(a=a, l2sq=dg["l2sq"], Q=dg["Q"], res=dg["res_nonlocal"], tail=dg["tail_rel"],
                     max_re=dg["max_re"], max_im=dg["max_im"]))
    if len(rows) % 10 == 0:
        print("lower: a=%.4f |u|^2=%.4f Q=%.4f res=%.1e tail=%.1e" % (a, dg["l2sq"], dg["Q"],
              dg["res_nonlocal"], dg["tail_rel"]), flush=True)
print("lower end: last accepted a=%.4f, stop at %.4f: %s" % (a, round(a - 0.001, 6), reason), flush=True)
np.savez(os.path.join(DATA, "revision_familyC_lower.npz"), a=[r["a"] for r in rows],
         l2sq=[r["l2sq"] for r in rows], Q=[r["Q"] for r in rows], u_end=u)
out["lower"] = dict(last_a=a, stop=reason, npts=len(rows),
                    end=rows[-1] if rows else None)
# upper end: march to 1.24125 with the steps of the earlier probe
u, a = u0.copy(), A0
while a < 1.24 - 1e-9:
    a = round(a + 0.0025, 6)
    u = newton_complex_pt(a, B0, C0, L, N, u, damped=True, maxit=60)
for an in (1.241, 1.24125):
    u = newton_complex_pt(an, B0, C0, L, N, u, damped=True, maxit=60); a = an
dg = diag_full(u, a, B0, C0, L)
print("upper: at a=%.5f tau=%.2e |u|^2=%.4f" % (a, dg["tail_rel"], dg["l2sq"]), flush=True)
# diagnose a = 1.2415 with plain Newton, recording residual and tail
D2 = diff2_matrix(L, N)
an = 1.2415
uu = newton_complex_pt(an, B0, C0, L, N, u, damped=False, maxit=40, verbose=True)
if uu is not None:
    dg2 = diag_full(uu, an, B0, C0, L)
    print("upper: undamped Newton converged at a=%.4f: tau=%.2e max_re=%.3f max_im=%.3f |u|^2=%.4f"
          % (an, dg2["tail_rel"], dg2["max_re"], dg2["max_im"], dg2["l2sq"]), flush=True)
    out["upper_try"] = dict(a=an, converged=True, **dg2)
else:
    out["upper_try"] = dict(a=an, converged=False)
    print("upper: undamped Newton did not converge at a=%.4f" % an, flush=True)
out["upper"] = dict(last_a=a, tail_rel=dg["tail_rel"], l2sq=dg["l2sq"], Q=dg["Q"])
update_summary("familyC_ends2", out)
