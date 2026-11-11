"""Compare the truncated parameterizations of degree 21 and 29 (and of
81 and 121 Fourier modes) at coordinates of modulus 0.9, as a numerical
indication of the truncation error of the degree-21 parameterization."""
import json, os
import numpy as np
from nonlocal_gp import manifold_parameterization, manifold_eval
from revision_computations import update_summary, A0, B0, C0
th = np.linspace(0, 2 * np.pi, 241)[:-1]
pars = {(nt, M): manifold_parameterization(A0, B0, C0, N_taylor=nt, M=M, l_scale=0.4)
        for nt, M in ((21, 40), (29, 40), (29, 60))}
out = {}
for lab, sg in (("real", 0.9), ("imaginary", 0.9j), ("complex", 0.9 * np.exp(1j * np.pi / 3))):
    # the bundle is normalized by its maximum over 4M sample points, which
    # depends on M; compare different M at matched normalization by
    # rescaling the coordinate with the ratio of the first coefficients
    r = (pars[(29, 60)]["W"][1, 0][60 + 1] / pars[(29, 40)]["W"][1, 0][40 + 1]).real
    w = {k: manifold_eval(p, th, sg * np.ones_like(th)) for k, p in pars.items()
         if k != (29, 60)}
    w[(29, 60)] = manifold_eval(pars[(29, 60)], th, (sg / r) * np.ones_like(th))
    d_deg = [float(np.max(np.abs(w[(21, 40)][i] - w[(29, 40)][i]))) for i in (0, 1)]
    d_mod = [float(np.max(np.abs(w[(29, 40)][i] - w[(29, 60)][i]))) for i in (0, 1)]
    out["normalization_ratio_M60_M40"] = float(r)
    out[lab] = {"deg21_vs_29_u": d_deg[0], "deg21_vs_29_up": d_deg[1],
                "M40_vs_60_u": d_mod[0], "M40_vs_60_up": d_mod[1]}
    print("%-9s |W21-W29|: u %.2e, u' %.2e   |W(M=40)-W(M=60)| at degree 29: u %.2e, u' %.2e"
          % (lab, d_deg[0], d_deg[1], d_mod[0], d_mod[1]))
update_summary("manifold_degree_comparison", out)
