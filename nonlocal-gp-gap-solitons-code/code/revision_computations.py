"""
Additional computations requested in the QTDS review (second round).

Usage:  python3 revision_computations.py [stage ...]
Stages: gaps convergence localres familyC doubled manifold shooting
        branches oddsearch fourier mapcheck all

Every stage reads the reference solutions written by run_computations.py
(data/flagship.npz, data/continuation.npz) and writes its own results to
data/revision_<stage>.npz and to data/revision_summary.json.
"""

import json
import os
import sys
import time

import numpy as np
from numpy.fft import fft, ifft
from scipy.integrate import solve_ivp

from nonlocal_gp import (
    fourier_grid, newton_local, newton_complex_pt, newton_doubled,
    residual_nonlocal, residual_local, pt_defect, l2_norm_sq, quasipower,
    tail_amplitude, reflect, even_part, odd_part, refine, spectral_deriv,
    bloch_hump, gauss_hump, mathieu_bands, first_gap, floquet_exponent,
    manifold_parameterization, manifold_eval, manifold_orbit_residual,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SUMMARY = os.path.join(DATA, "revision_summary.json")

A0, B0, C0 = 1.1025, 0.55125, -0.826875
TAIL_REL = 5e-3


def load_summary():
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as fh:
            return json.load(fh)
    return {}


def update_summary(key, value):
    s = load_summary()
    s[key] = value
    with open(SUMMARY, "w") as fh:
        json.dump(s, fh, indent=2, default=float)


def rel_tail(u, L):
    return tail_amplitude(u, L) / np.max(np.abs(u))


def shift_lattice(u, m, L, N):
    dx = 2.0 * L / N
    s = m * np.pi / dx
    si = int(round(s))
    assert abs(s - si) < 1e-9
    return np.roll(u, si)


def resample(u, N2):
    """Trigonometric interpolant of u (same L) on N2 points."""
    N1 = u.size
    uh = fft(u) / N1
    out = np.zeros(N2, dtype=complex)
    h = min(N1, N2) // 2
    out[:h] = uh[:h]
    out[-h + 1:] = uh[-h + 1:]
    nyq = 0.5 * uh[N1 // 2] if N1 <= N2 else 0.0
    out[h] += nyq
    out[-h] += nyq
    return ifft(out) * N2


def change_window(u, L1, L2):
    """Same spacing, window [-L1, L1) -> [-L2, L2) by cropping or zero padding."""
    N1 = u.size
    dx = 2.0 * L1 / N1
    N2 = int(round(2.0 * L2 / dx))
    off = int(round((L2 - L1) / dx))
    out = np.zeros(N2, dtype=complex)
    for j2 in range(N2):
        j1 = j2 - off
        if 0 <= j1 < N1:
            out[j2] = u[j1]
    return out


def diag_full(u, a, b, c, L):
    u = np.asarray(u, dtype=complex)
    return {
        "res_nonlocal": float(residual_nonlocal(u, a, b, c, L)),
        "res_local": float(residual_local(u, a, b, c, L)),
        "tail_rel": float(rel_tail(u, L)),
        "l2sq": float(l2_norm_sq(u, L)),
        "Q": float(np.real(quasipower(u, L))),
        "max_re": float(np.max(np.abs(np.real(u)))),
        "max_im": float(np.max(np.abs(np.imag(u)))),
    }


def load_reference():
    d = np.load(os.path.join(DATA, "flagship.npz"))
    return d, float(d["L"]), int(d["N"])


# ----------------------------------------------------------------------
def stage_gaps():
    print("== Mathieu gaps and Bloch discretization ==")
    out = {}
    for b in (0.55125, 1.0, 1.5, -1.5):
        lo, hi = first_gap(b)
        out[str(b)] = [float(lo), float(hi)]
        print("  b=%+.5f first gap (%.8f, %.8f); leading order (%.4f, %.4f)"
              % (b, lo, hi, 1 - abs(b) / 2, 1 + abs(b) / 2))
    # dependence of the edges on the number of retained modes
    conv = {}
    for M in (4, 8, 16, 32, 48):
        lo, hi = first_gap(1.5, M=M)
        conv[str(2 * M + 1)] = [float(lo), float(hi)]
        print("  b=1.5, %3d modes: (%.15f, %.15f)" % (2 * M + 1, lo, hi))
    bs = np.round(np.arange(-1.6, 1.6001, 0.01), 4)
    edges = np.array([[*first_gap(bb)] for bb in bs])
    allbands = np.array([np.sort(np.concatenate([
        mathieu_bands(bb, [0.0], 4)[0], mathieu_bands(bb, [1.0], 4)[0]]))
        for bb in bs])
    np.savez(os.path.join(DATA, "revision_gaps.npz"), bs=bs, gap=edges,
             edges=allbands)
    update_summary("gaps", {"first_gap": out, "mode_convergence_b1.5": conv})


# ----------------------------------------------------------------------
def solve_family(tag, a, b, c, L, N, seed):
    if tag == "A":
        f = newton_local(a, b, c, L, N, np.real(seed), damped=True,
                         parity="even", maxit=80)
        return None if f is None else f.astype(complex)
    if tag == "B":
        g = newton_local(a, b, -c, L, N, np.imag(seed), damped=True,
                         parity="odd", maxit=80)
        return None if g is None else 1j * g
    return newton_complex_pt(a, b, c, L, N, seed, damped=True, maxit=120)


def stage_convergence():
    print("== convergence in N and L at the reference parameters ==")
    d, Lr, Nr = load_reference()
    a, b, c = A0, B0, C0
    ref = {"A": d["u_A1"], "B": d["u_B1"], "C": d["u_C1"]}
    cases = [("L", 24 * np.pi, 1536), ("L", 32 * np.pi, 2048),
             ("L", 48 * np.pi, 3072), ("L", 64 * np.pi, 4096),
             ("N", 48 * np.pi, 2048), ("N", 48 * np.pi, 4096)]
    rows = []
    store = {}
    for fam in ("A", "B", "C"):
        ur = ref[fam]
        x_r, _ = fourier_grid(Lr, Nr)
        mref = np.abs(x_r) < 24 * np.pi - 0.5 * (2 * Lr / Nr)
        # reference values on the reference grid
        base = diag_full(ur, a, b, c, Lr)
        base.update(fam=fam, series="ref", L_over_pi=48, N=Nr, diff=0.0)
        rows.append(base)
        for series, L, N in cases:
            if series == "L" and N == Nr:
                continue
            t0 = time.time()
            if series == "L":
                seed = change_window(ur, Lr, L)
            else:
                seed = resample(ur, N)
            u = solve_family(fam, a, b, c, L, N, seed)
            if u is None:
                print("  %s %s L=%g pi N=%d: no convergence" % (fam, series,
                                                                L / np.pi, N))
                continue
            dg = diag_full(u, a, b, c, L)
            if series == "L":
                x, _ = fourier_grid(L, N)
                m = np.abs(x) < 24 * np.pi - 0.5 * (2 * L / N)
                diff = float(np.max(np.abs(u[m] - ur[mref])))
            else:
                Nf = 12288
                diff = float(np.max(np.abs(refine(u, L, Nf // N)
                                           - refine(ur, Lr, Nf // Nr))))
            dg.update(fam=fam, series=series, L_over_pi=float(L / np.pi), N=N,
                      diff=diff)
            rows.append(dg)
            store["u_%s_%s_%d_%d" % (fam, series, int(L / np.pi), N)] = u
            print("  %s %s L=%2d pi N=%4d: res=%.2e tail=%.2e |u|^2=%.10f "
                  "Q=%.10f diff=%.2e (%.0f s)"
                  % (fam, series, L / np.pi, N, dg["res_nonlocal"],
                     dg["tail_rel"], dg["l2sq"], dg["Q"], diff,
                     time.time() - t0))
    np.savez(os.path.join(DATA, "revision_convergence.npz"), **store)
    update_summary("convergence", rows)


# ----------------------------------------------------------------------
def stage_localres():
    print("== local residual identities ==")
    d, L, N = load_reference()
    c = C0
    out = {}
    for tag in ("A1", "B1", "C1"):
        u = d["u_" + tag]
        uf = refine(u, L, 2)
        f, g = np.real(uf), np.imag(uf)
        pred_c = 2 * abs(c) * np.max(np.abs(g * uf ** 2))
        pred_mc = 2 * abs(c) * np.max(np.abs(f * uf ** 2))
        res_c = residual_local(u, A0, B0, c, L)
        res_mc = residual_local(u, A0, B0, -c, L)
        # literal use of the real local equation u'' + (a - V) u - c u^3
        Nf = uf.size
        x = -L + 2 * L * np.arange(Nf) / Nf
        lit = spectral_deriv(uf, L, 2) + (A0 - B0 * np.cos(2 * x)) * uf \
            - c * uf ** 3
        out[tag] = {"R_loc_c": float(res_c), "pred_2|c||g u^2|": float(pred_c),
                    "R_loc_minus_c": float(res_mc),
                    "pred_2|c||f u^2|": float(pred_mc),
                    "R_literal_u3": float(np.max(np.abs(lit)))}
        print("  %s: R_loc(c)=%.6f (pred %.6f)  R_loc(-c)=%.6f (pred %.6f)"
              "  literal u^3 residual=%.2e"
              % (tag, res_c, pred_c, res_mc, pred_mc, np.max(np.abs(lit))))
    update_summary("local_residuals", out)


# ----------------------------------------------------------------------
def hump_positions(u, L, N):
    """Lattice positions (units of pi, x > 0) of the maxima of |Re u| and
    |Im u|."""
    x, _ = fourier_grid(L, N)
    m = x > 0
    f, g = np.abs(np.real(u[m])), np.abs(np.imag(u[m]))
    return float(x[m][np.argmax(f)] / np.pi), float(x[m][np.argmax(g)] / np.pi)


def stage_familyC():
    print("== family C: separations and continuation in a ==")
    a, b, c = A0, B0, C0
    L, N = 36 * np.pi, 2304
    x, _ = fourier_grid(L, N)
    d, Lr, Nr = load_reference()
    fA = newton_local(a, b, c, L, N, bloch_hump(x, 0.0, 1.0, 3.0),
                      damped=False, parity="even")
    gB = newton_local(a, b, -c, L, N,
                      odd_part(bloch_hump(x, np.pi / 2, 0.5, 1.5)
                               - bloch_hump(x, -np.pi / 2, 0.5, 1.5)).real,
                      damped=False, parity="odd")
    sep_rows = []
    store = {}
    for m in range(3, 15):
        t0 = time.time()
        fseed = even_part(shift_lattice(fA, m, L, N)
                          + shift_lattice(fA, -m, L, N)).real
        u = newton_complex_pt(a, b, c, L, N, fseed + 1j * gB, damped=True,
                              maxit=200)
        row = {"m_seed": m}
        if u is None:
            row["converged"] = False
            print("  seed m=%2d: no convergence (%.0f s)" % (m, time.time() - t0))
        else:
            dg = diag_full(u, a, b, c, L)
            pf, pg = hump_positions(u, L, N)
            mixed = dg["max_re"] > 1e-2 and dg["max_im"] > 1e-2
            row.update(converged=True, mixed=bool(mixed), pos_re=pf,
                       pos_im=pg, **dg)
            store["u_m%d" % m] = u
            print("  seed m=%2d: mixed=%s Re-hump at %.2f pi, Im-hump at "
                  "%.2f pi, res=%.2e tail=%.1e |u|^2=%.4f Q=%.4f (%.0f s)"
                  % (m, mixed, pf, pg, dg["res_nonlocal"], dg["tail_rel"],
                     dg["l2sq"], dg["Q"], time.time() - t0))
        sep_rows.append(row)

    # continuation of the reference mixed state in the frequency
    uC = newton_complex_pt(a, b, c, L, N, change_window(d["u_C1"], Lr, L),
                           damped=True, maxit=100)
    assert uC is not None
    da = 0.0025
    branch = {}
    for direction in (+1, -1):
        u = uC.copy()
        aa = a
        rows = []
        reason = "end of range"
        while 0.70 < aa < 1.30:
            un = newton_complex_pt(aa, b, c, L, N, u, damped=True, maxit=60)
            if un is None:
                reason = "Newton failure"
                break
            dg = diag_full(un, aa, b, c, L)
            if dg["max_re"] < 1e-2 or dg["max_im"] < 1e-2:
                reason = "lost mixed character"
                break
            if dg["tail_rel"] >= TAIL_REL:
                reason = "localization criterion"
                break
            u = un
            pf, pg = hump_positions(u, L, N)
            dg.update(a=float(aa), pos_re=pf, pos_im=pg)
            rows.append(dg)
            aa = round(aa + direction * da, 6)
        branch[str(direction)] = {"rows": rows, "stop": reason,
                                  "stop_a": float(aa)}
        print("  continuation dir %+d: %d points, last accepted a=%.4f, "
              "stopped at a=%.4f (%s)" % (direction, len(rows),
                                          rows[-1]["a"], aa, reason))
    allrows = branch["-1"]["rows"][::-1] + branch["1"]["rows"][1:]
    np.savez(os.path.join(DATA, "revision_familyC.npz"),
             a=np.array([r["a"] for r in allrows]),
             l2sq=np.array([r["l2sq"] for r in allrows]),
             Q=np.array([r["Q"] for r in allrows]),
             res=np.array([r["res_nonlocal"] for r in allrows]),
             max_re=np.array([r["max_re"] for r in allrows]),
             max_im=np.array([r["max_im"] for r in allrows]),
             pos_re=np.array([r["pos_re"] for r in allrows]),
             **store)
    update_summary("familyC", {
        "grid": {"L_over_pi": 36, "N": N},
        "separations": sep_rows,
        "continuation": {k: {"stop": v["stop"], "stop_a": v["stop_a"],
                             "first_a": v["rows"][0]["a"],
                             "last_a": v["rows"][-1]["a"],
                             "npts": len(v["rows"])}
                         for k, v in branch.items()},
        "continuation_range": [allrows[0]["a"], allrows[-1]["a"]],
        "max_res_along_branch": float(max(r["res_nonlocal"]
                                          for r in allrows)),
        "max_tail_along_branch": float(max(r["tail_rel"] for r in allrows)),
    })


# ----------------------------------------------------------------------
def stage_doubled():
    print("== doubled system: complete diagnostics ==")
    a, b, c = A0, B0, C0
    d, Lr, Nr = load_reference()
    Nd = int(os.environ.get("DOUBLED_N", "1536"))
    Ld = float(os.environ.get("DOUBLED_L_OVER_PI", "24")) * np.pi
    tagD = "" if Nd == 1536 else "_%d" % Nd
    xd, _ = fourier_grid(Ld, Nd)
    uPT = newton_complex_pt(a, b, c, Ld, Nd, change_window(d["u_C1"], Lr, Ld),
                            damped=True, maxit=100)
    assert uPT is not None
    results = []
    best = None
    for eps in (3e-3, 1e-3, 3e-4, 1e-2):
        p1 = 1.0 + eps * np.cos(np.pi * xd / Ld) \
            + 0.5 * eps * np.sin(2 * np.pi * xd / Ld)
        p2 = 1.0 - 0.75 * eps * np.cos(2 * np.pi * xd / Ld) \
            + 0.5j * eps * np.sin(np.pi * xd / Ld)
        su = uPT * p1
        sw = np.conj(reflect(uPT)) * p2
        raw_seed_defect = float(np.max(np.abs(sw - np.conj(reflect(su)))))
        t0 = time.time()
        uu, ww = newton_doubled(a, b, c, Ld, Nd, su, sw, maxit=60)
        if uu is None:
            print("  eps=%.0e: no convergence" % eps)
            results.append({"eps": eps, "converged": False})
            continue
        # residual of the doubled system on the refined grid
        uf, wf = refine(uu, Ld, 2), refine(ww, Ld, 2)
        Nf = uf.size
        xf = -Ld + 2 * Ld * np.arange(Nf) / Nf
        Vf = b * np.cos(2 * xf)
        Ru = spectral_deriv(uf, Ld, 2) + (a - Vf) * uf - c * uf ** 2 * wf
        Rw = spectral_deriv(wf, Ld, 2) + (a - Vf) * wf - c * wf ** 2 * uf
        res_dbl = float(max(np.max(np.abs(Ru)), np.max(np.abs(Rw))))
        # Wronskian J = u w' - u' w
        J = uf * spectral_deriv(wf, Ld, 1) - spectral_deriv(uf, Ld, 1) * wf
        Jmax = float(np.max(np.abs(J)))
        Jscale = float(np.max(np.abs(uf)) * np.max(np.abs(
            spectral_deriv(wf, Ld, 1))))
        # proportionality w = kappa u
        kappa = np.vdot(uu, ww) / np.vdot(uu, uu)
        prop_defect = float(np.max(np.abs(ww - kappa * uu)))
        # reversor (constraint) defect before and after the admissible
        # alignment (u, w) -> (alpha u, w / alpha); the phase of alpha drops
        s = np.conj(reflect(uu))
        raw_defect = float(np.max(np.abs(ww - s)))
        zeta = np.vdot(s, ww) / np.vdot(s, s)
        rho = np.sqrt(abs(zeta))
        aligned_defect = float(np.max(np.abs(ww / rho - rho * s)))
        # full alignment alpha = sqrt(kappa): (alpha u, w / alpha) should be
        # (u_PT, u_PT) up to the sign symmetry u -> -u
        alpha = np.sqrt(kappa)
        ua, wa = alpha * uu, ww / alpha
        if np.max(np.abs(ua + uPT)) < np.max(np.abs(ua - uPT)):
            ua, wa = -ua, -wa
        ptdef = float(pt_defect(ua))
        cdef = float(np.max(np.abs(wa - np.conj(reflect(ua)))))
        res_nl = float(residual_nonlocal(ua, a, b, c, Ld))
        diff_pt = float(np.max(np.abs(ua - uPT)))
        row = dict(eps=eps, converged=True, seed_defect=raw_seed_defect,
                   res_doubled=res_dbl, J_max=Jmax, J_scale=Jscale,
                   kappa_re=float(np.real(kappa)),
                   kappa_im=float(np.imag(kappa)), prop_defect=prop_defect,
                   raw_constraint_defect=raw_defect,
                   zeta_abs=float(abs(zeta)), zeta_arg=float(np.angle(zeta)),
                   aligned_constraint_defect=aligned_defect,
                   aligned_pt_defect=ptdef,
                   aligned_constraint_defect_full=cdef,
                   res_nonlocal_aligned=res_nl, diff_to_PT_solver=diff_pt)
        results.append(row)
        print("  eps=%.0e (%.0f s): seed defect %.1e, R_dbl %.1e, |J| %.1e, "
              "kappa=%.6f%+.6fi, raw defect %.1e, aligned defect %.1e, "
              "PT defect %.1e, R_nl %.1e, |u-u_PT| %.1e"
              % (eps, time.time() - t0, raw_seed_defect, res_dbl, Jmax,
                 np.real(kappa), np.imag(kappa), raw_defect, aligned_defect,
                 ptdef, res_nl, diff_pt))
        if best is None or res_dbl < best[0]:
            best = (res_dbl, eps, ua, wa)
    _, eps, ua, wa = best
    np.savez(os.path.join(DATA, "revision_doubled%s.npz" % tagD), x=xd, u=ua,
             w=wa, uPT=uPT, eps=eps, L=Ld, N=Nd)
    update_summary("doubled" + tagD, {"grid": {"L_over_pi": Ld / np.pi,
                                               "N": Nd,
                                               "tail_rel_PT": float(
                                                   rel_tail(uPT, Ld))},
                               "runs": results, "plotted_eps": eps})


# ----------------------------------------------------------------------
def fit_sigma(par, xj, uj, sigma_guess):
    """Complex manifold coordinate s with W_1(x_j, s) = u(x_j) (Newton,
    holomorphic in s)."""
    s = complex(sigma_guess)
    for _ in range(30):
        w, _ = manifold_eval(par, np.mod(xj, 2 * np.pi), s)
        w = complex(np.ravel(w)[0])
        h = 1e-7 * max(1.0, abs(s))
        w2, _ = manifold_eval(par, np.mod(xj, 2 * np.pi), s + h)
        dw = (complex(np.ravel(w2)[0]) - w) / h
        ds = (uj - w) / dw
        s += ds
        if abs(ds) < 1e-15 * max(1.0, abs(s)):
            break
    return s


def stage_manifold():
    print("== manifold: truncation study, complex coordinate, tails ==")
    a, b, c = A0, B0, C0
    rows = []
    for M in (20, 40, 60):
        for nt in (5, 9, 13, 17, 21):
            t0 = time.time()
            par = manifold_parameterization(a, b, c, N_taylor=nt, M=M,
                                            l_scale=0.4)
            r_real = manifold_orbit_residual(par, a, b, c, 0.9, (0.0, 4 * np.pi))
            r_cplx = manifold_orbit_residual(par, a, b, c,
                                             0.9 * np.exp(1j * np.pi / 3),
                                             (0.0, 4 * np.pi))
            r_imag = manifold_orbit_residual(par, a, b, c, 0.9j,
                                             (0.0, 4 * np.pi))
            # last Fourier coefficients of the highest order (tail of the
            # Fourier series)
            W = par["W"]
            ftail = float(np.max(np.abs(W[nt, 0][:3])))
            rows.append(dict(M=M, modes=2 * M + 1, N_taylor=nt,
                             res_real=float(r_real), res_complex=float(r_cplx),
                             res_imag=float(r_imag), fourier_tail=ftail))
            print("  modes=%3d degree<=%2d: orbit residual sigma=0.9: %.2e, "
                  "0.9e^{i pi/3}: %.2e, 0.9i: %.2e  (%.1f s)"
                  % (2 * M + 1, nt, r_real, r_cplx, r_imag, time.time() - t0))
    par = manifold_parameterization(a, b, c, N_taylor=21, M=40, l_scale=0.4)
    W = par["W"]
    lam = par["lambda"]
    norms = [float(np.sum(np.abs(W[n, 0]))) for n in range(W.shape[0])]
    odd = [n for n in range(1, len(norms)) if n % 2 == 1]
    ratios = [norms[n + 2] / norms[n] for n in odd if n + 2 < len(norms)]
    even_max = float(max(norms[n] for n in range(2, len(norms), 2)))
    # symmetry W(theta + pi, -sigma) = W(theta, sigma)
    th = np.linspace(0, 2 * np.pi, 97)
    sg = 0.7 * np.exp(1j * 0.4)
    w0, _ = manifold_eval(par, th, sg * np.ones_like(th))
    w1, _ = manifold_eval(par, th + np.pi, -sg * np.ones_like(th))
    sym = float(np.max(np.abs(w0 - w1)))
    print("  odd-degree ratios:", ["%.3f" % r for r in ratios])
    print("  max even-degree norm %.1e, symmetry defect %.1e" % (even_max, sym))

    # tail comparison for families A, B, C on the reference grid
    d, L, N = load_reference()
    x, _ = fourier_grid(L, N)
    tails = {}
    for tag, xm, win in (("A1", 30.0, (24.0, 44.0)),
                         ("B1", 30.0, (24.0, 44.0)),
                         ("C1", 60.0, (54.0, 74.0))):
        u = d["u_" + tag]
        j = int(np.argmin(np.abs(x - xm)))
        w1x = complex(np.exp(1j * np.arange(-par["M"], par["M"] + 1) * x[j])
                      @ W[1, 0])
        s_guess = u[j] / w1x
        s_j = fit_sigma(par, x[j], u[j], s_guess)
        m = (x > win[0]) & (x < win[1])
        sig = s_j * np.exp(lam * (x[m] - x[j]))
        uu, _ = manifold_eval(par, np.mod(x[m], 2 * np.pi), sig)
        diff = float(np.max(np.abs(uu - u[m])))
        size = [float(np.max(np.abs(u[m]))), float(np.min(np.abs(u[m])))]
        tails[tag] = dict(x_match=float(x[j]), window=list(win),
                          sigma_at_match=[float(np.real(s_j)),
                                          float(np.imag(s_j))],
                          sigma_abs_max=float(np.max(np.abs(sig))),
                          sup_diff=diff, tail_size_max=size[0],
                          tail_size_min=size[1])
        print("  tail %s: x_j=%.3f sigma=%.4e%+.4ei window %s diff=%.2e "
              "|u| in [%.1e, %.1e]" % (tag, x[j], np.real(s_j), np.imag(s_j),
                                       win, diff, size[1], size[0]))
    np.savez(os.path.join(DATA, "revision_manifold.npz"), norms=norms,
             W=W, lam=lam, M=par["M"], N_taylor=par["N_taylor"],
             l_scale=par["l_scale"])
    update_summary("manifold", {"truncation": rows, "norms_deg21": norms,
                                "odd_ratios": ratios,
                                "even_max": even_max,
                                "symmetry_defect": sym, "lambda": lam,
                                "tails": tails})


# ----------------------------------------------------------------------
def shoot_profile(par, a, b, c, Lm, sig, x_eval):
    """Backward integration from the manifold point W(Lm, sig) to x = 0."""
    w, wp = manifold_eval(par, np.mod(Lm, 2 * np.pi), sig)
    u0, up0 = complex(np.ravel(w)[0]), complex(np.ravel(wp)[0])

    def rhs(xx, y):
        u = y[0] + 1j * y[1]
        up = y[2] + 1j * y[3]
        upp = -(a - b * np.cos(2 * xx)) * u + c * u ** 3
        return [up.real, up.imag, upp.real, upp.imag]

    sol = solve_ivp(rhs, [Lm, 0.0], [u0.real, u0.imag, up0.real, up0.imag],
                    method="DOP853", rtol=1e-13, atol=1e-15, t_eval=x_eval)
    y = sol.y
    return (y[0] + 1j * y[1]), (y[2] + 1j * y[3])


def stage_shooting():
    print("== boundary-value problem of Theorem (manifold shooting) ==")
    a, b, c = A0, B0, C0
    par = manifold_parameterization(a, b, c, N_taylor=21, M=40, l_scale=0.4)
    lam = par["lambda"]
    d, L, N = load_reference()
    x, _ = fourier_grid(L, N)
    out = {}
    for tag, Lm in (("A1", 30.0), ("A1", 40.0), ("C1", 60.0), ("C1", 70.0)):
        u = d["u_" + tag]
        j = int(np.argmin(np.abs(x - Lm)))
        Lm = float(x[j])
        w1x = complex(np.exp(1j * np.arange(-par["M"], par["M"] + 1) * Lm)
                      @ par["W"][1, 0])
        s = fit_sigma(par, Lm, u[j], u[j] / w1x)
        s_start = s

        def G(sv):
            uu, up = shoot_profile(par, a, b, c, Lm, sv, [0.0])
            return np.array([np.real(up[-1]), np.imag(uu[-1])])

        hist = []
        for it in range(12):
            g0 = G(s)
            hist.append(float(np.max(np.abs(g0))))
            if np.max(np.abs(g0)) < 1e-13:
                break
            h = 1e-8 * max(1e-3, abs(s))
            J = np.column_stack([(G(s + h) - g0) / h,
                                 (G(s + 1j * h) - g0) / h])
            if tag.startswith("A"):
                # real coordinate: one condition f'(0) = 0, one unknown
                ds = -g0[0] / J[0, 0]
            else:
                dsv = np.linalg.solve(J, -g0)
                ds = dsv[0] + 1j * dsv[1]
            s = s + ds
        xs = x[(x >= 0) & (x <= Lm + 1e-12)][::-1]
        us, _ = shoot_profile(par, a, b, c, Lm, s, xs)
        mcol = (x >= 0) & (x <= Lm + 1e-12)
        diff = float(np.max(np.abs(us[::-1] - u[mcol])))
        # beyond Lm: manifold flow versus collocation tail
        mt = (x > Lm) & (x < Lm + 20)
        ut, _ = manifold_eval(par, np.mod(x[mt], 2 * np.pi),
                              s * np.exp(lam * (x[mt] - Lm)))
        diff_tail = float(np.max(np.abs(ut - u[mt])))
        key = "%s_L%d" % (tag, int(round(Lm)))
        out[key] = dict(L_m=Lm, sigma_L=[float(np.real(s)), float(np.imag(s))],
                        sigma_from_fit=[float(np.real(s_start)),
                                        float(np.imag(s_start))],
                        bc_history=hist, diff_on_0_L=diff,
                        diff_tail_20=diff_tail)
        print("  %s L_m=%.3f: sigma_L=%.6e%+.6ei, |bc| history %s, "
              "sup|u_shoot - u_coll| on [0,L_m]=%.2e, tail diff=%.2e"
              % (tag, Lm, np.real(s), np.imag(s),
                 ["%.1e" % h for h in hist], diff, diff_tail))
    update_summary("shooting", out)


# ----------------------------------------------------------------------
def stage_branches():
    print("== branch endpoints, reference points, quasipower identities ==")
    dd = np.load(os.path.join(DATA, "continuation.npz"))
    lo, hi = dd["gap"]
    out = {"gap": [float(lo), float(hi)]}
    for fam in ("A", "B"):
        a = dd[fam + "_a"]
        n2 = dd[fam + "_n2"]
        Q = dd[fam + "_Q"]
        tail = dd[fam + "_tail"] / dd[fam + "_amp"]
        acc = tail < TAIL_REL
        sgn = 1.0 if fam == "A" else -1.0
        idx = np.where(acc)[0]
        j0 = int(np.argmin(np.abs(a - A0)))
        out[fam] = dict(
            computed_range=[float(a[0]), float(a[-1])],
            accepted_range=[float(a[idx[0]]), float(a[idx[-1]])],
            n_accepted=int(acc.sum()),
            ref_l2sq=float(n2[j0]), ref_Q=float(Q[j0]),
            max_identity_defect=float(np.max(np.abs(Q - sgn * n2))),
            n2_at_ends=[float(n2[idx[0]]), float(n2[idx[-1]])])
        print("  %s: computed a in [%.4f, %.4f], accepted [%.4f, %.4f], "
              "|u|^2 at ends %.3f, %.3f; max |Q -+ |u|^2| = %.1e"
              % (fam, a[0], a[-1], a[idx[0]], a[idx[-1]], n2[idx[0]],
                 n2[idx[-1]], out[fam]["max_identity_defect"]))
    update_summary("branches", out)


# ----------------------------------------------------------------------
def stage_oddsearch():
    print("== search for odd localized solutions of the local equation at c ==")
    a, b, c = A0, B0, C0
    L, N = 24 * np.pi, 1536
    x, _ = fourier_grid(L, N)
    trials = []
    found = []
    for x0 in (np.pi / 2, np.pi, 3 * np.pi / 2):
        for amp in (0.3, 0.6, 1.0, 1.5):
            for width in (1.5, 3.0, 6.0):
                for carrier in ("cos", "gauss"):
                    if carrier == "cos":
                        hump = lambda s: bloch_hump(x, s, amp, width)
                    else:
                        hump = lambda s: gauss_hump(x, s, amp, width)
                    seed = odd_part(hump(x0) - hump(-x0)).real
                    for damped in (False, True):
                        g = newton_local(a, b, c, L, N, seed, damped=damped,
                                         parity="odd", maxit=100)
                        ok = g is not None and np.max(np.abs(g)) > 1e-3 and \
                            rel_tail(g, L) < TAIL_REL
                        trials.append(ok)
                        if g is not None and np.max(np.abs(g)) > 1e-3:
                            found.append(dict(x0=x0, amp=amp, width=width,
                                              carrier=carrier,
                                              damped=damped,
                                              tail=float(rel_tail(g, L)),
                                              amp_out=float(np.max(np.abs(g)))))
    print("  %d Newton runs, %d produced a nonzero odd localized solution, "
          "%d a nonzero odd profile of any kind"
          % (len(trials), sum(trials), len(found)))
    update_summary("oddsearch", {"runs": len(trials),
                                 "localized": int(sum(trials)),
                                 "nonzero_any": found,
                                 "grid": {"L_over_pi": 24, "N": N}})


# ----------------------------------------------------------------------
def stage_fourier():
    print("== Fourier coefficient decay on the reference grid ==")
    d, L, N = load_reference()
    out = {}
    for tag in ("A1", "B1", "C1"):
        u = d["u_" + tag]
        uh = np.abs(fft(u)) / N
        _, k = fourier_grid(L, N)
        rel = uh / uh.max()
        kk = np.abs(k)
        order = np.argsort(kk)
        kk, rel = kk[order], rel[order]
        # envelope: max of relative coefficient for |k| >= K
        env = np.maximum.accumulate(rel[::-1])[::-1]
        below = kk[env < 1e-13]
        out[tag] = {"k_max": float(kk.max()),
                    "rel_at_k20": float(env[np.searchsorted(kk, 20.0)]),
                    "rel_at_kmax": float(env[-1]),
                    "k_below_1e-13": float(below[0]) if below.size else None}
        print("  %s: k_max=%.2f, max rel coefficient for |k|>=20: %.1e, at "
              "k_max: %.1e, envelope below 1e-13 from k=%s"
              % (tag, kk.max(), out[tag]["rel_at_k20"], env[-1],
                 out[tag]["k_below_1e-13"]))
    update_summary("fourier", out)


# ----------------------------------------------------------------------
def accepted_interval(a_start, b, cnl, L, N, seed, parity, da, tol=1e-12):
    """Continue from a solution at a_start in both directions with step da
    and return the accepted interval and the stopping reasons."""
    lo, hi = first_gap(b)
    res = {}
    for direction in (+1, -1):
        u = seed.copy()
        a = a_start
        last = None
        reason = "end of range"
        while lo - 0.1 < a < hi + 0.1:
            un = newton_local(a, b, cnl, L, N, u, damped=True, parity=parity,
                              tol=tol)
            if un is None or np.max(np.abs(un)) < 1e-4:
                reason = "solver failure"
                break
            u = un
            if rel_tail(u, L) >= TAIL_REL:
                reason = "localization criterion"
                break
            last = a
            a = round(a + direction * da, 6)
        res[direction] = (last, reason)
    return res


def stage_mapcheck():
    print("== existence map: categories and boundary refinement ==")
    c = C0
    # base sweep with recorded failure categories, as in run_computations
    L, N = 16 * np.pi, 1024
    x, _ = fourier_grid(L, N)
    b_pos = np.round(np.arange(0.10, 1.5001, 0.05), 4)
    b_vals = np.concatenate([-b_pos[::-1], b_pos])
    a_step = 0.02

    def dip(xx, x0=np.pi / 2, amp=0.5, width=1.5):
        return odd_part(bloch_hump(xx, x0, amp, width)
                        - bloch_hump(xx, -x0, amp, width)).real

    seedsA = [bloch_hump(x, 0.0, 1.0, 3.0), gauss_hump(x, 0.0, 1.2, 2.0)]
    seedsB = [dip(x), dip(x, x0=np.pi, amp=1.0, width=2.0),
              dip(x, amp=1.0, width=2.0)]
    cat = {}
    starts = {}
    for fam, cnl, parity, seeds in (("A", c, "even", seedsA),
                                    ("B", -c, "odd", seedsB)):
        acc, nonloc, fail, nostart = [], [], [], []
        t0 = time.time()
        for b in b_vals:
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
            if u_mid is None:
                nostart.append(b)
                continue
            starts[(fam, float(b))] = (mid, u_mid)
            for direction in (+1, -1):
                u = u_mid.copy()
                a = mid
                while lo - 0.1 < a < hi + 0.1:
                    un = newton_local(a, b, cnl, L, N, u, damped=True,
                                      parity=parity)
                    if un is None or np.max(np.abs(un)) < 1e-4:
                        fail.append((a, b))
                        break
                    u = un
                    if rel_tail(u, L) < TAIL_REL:
                        acc.append((a, b))
                    else:
                        nonloc.append((a, b))
                    a = a + direction * a_step
        cat[fam] = dict(acc=np.array(acc), nonloc=np.array(nonloc),
                        fail=np.array(fail), nostart=np.array(nostart))
        print("  family %s: %d accepted, %d converged but not localized, "
              "%d solver failures, %d values of b without a start (%.0f s)"
              % (fam, len(acc), len(nonloc), len(fail), len(nostart),
                 time.time() - t0))
    np.savez(os.path.join(DATA, "revision_map.npz"),
             **{"%s_%s" % (f, k): v for f in cat for k, v in cat[f].items()})

    # refinement of the boundaries at selected b
    configs = [("base step", 16 * np.pi, 1024, 0.02),
               ("half step", 16 * np.pi, 1024, 0.01),
               ("larger window", 24 * np.pi, 1536, 0.01),
               ("finer grid", 16 * np.pi, 2048, 0.01)]
    rows = []
    for fam, cnl, parity in (("A", c, "even"), ("B", -c, "odd")):
        for b in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5):
            key = (fam, float(np.round(b, 4)))
            if key not in starts:
                rows.append(dict(fam=fam, b=b, start=False))
                print("  %s b=%+.2f: no start in the base sweep" % (fam, b))
                continue
            mid, u_mid = starts[key]
            row = dict(fam=fam, b=b, start=True, a_start=float(mid))
            for name, Lc, Nc, da in configs:
                t0 = time.time()
                if Lc != L:
                    seed = np.real(change_window(u_mid, L, Lc))
                elif Nc != N:
                    seed = np.real(resample(u_mid, Nc))
                else:
                    seed = u_mid
                us = newton_local(mid, b, cnl, Lc, Nc, seed, damped=True,
                                  parity=parity)
                if us is None:
                    row[name] = None
                    continue
                r = accepted_interval(mid, b, cnl, Lc, Nc, us, parity, da)
                row[name] = dict(lower=r[-1][0], lower_stop=r[-1][1],
                                 upper=r[1][0], upper_stop=r[1][1])
                print("  %s b=%+.2f %-13s: accepted [%s, %s] stops (%s | %s)"
                      " (%.0f s)" % (fam, b, name, r[-1][0], r[1][0],
                                     r[-1][1], r[1][1], time.time() - t0))
            rows.append(row)
    update_summary("mapcheck", {"b_step": 0.05, "a_step": a_step,
                                "tail_rel": TAIL_REL,
                                "counts": {f: {k: int(len(v))
                                               for k, v in cat[f].items()}
                                           for f in cat},
                                "refinement": rows})


# ----------------------------------------------------------------------
def stage_branchext():
    print("== branches A and B toward the bifurcation edges, larger window ==")
    b, c = B0, C0
    lo, hi = first_gap(b)
    dd = np.load(os.path.join(DATA, "continuation.npz"))
    L1, N1 = 24 * np.pi, 1536
    L, N = 96 * np.pi, 6144
    out = {}
    store = {}
    for fam, cnl, parity, direction in (("A", c, "even", +1),
                                        ("B", -c, "odd", -1)):
        a = dd[fam + "_a"]
        tail = dd[fam + "_tail"] / dd[fam + "_amp"]
        acc = np.where(tail < TAIL_REL)[0]
        # start well inside the accepted range, 0.05 away from its end
        a_end = a[acc[-1]] if direction > 0 else a[acc[0]]
        a_start = round(a_end - direction * 0.05, 6)
        x1, _ = fourier_grid(L1, N1)
        from nonlocal_gp import bloch_hump as bh
        if fam == "A":
            seed1 = newton_local(1.1025, b, cnl, L1, N1, bh(x1, 0, 1.0, 3.0),
                                 damped=False, parity="even")
        else:
            seed1 = newton_local(1.1025, b, cnl, L1, N1,
                                 odd_part(bh(x1, np.pi / 2, 0.5, 1.5)
                                          - bh(x1, -np.pi / 2, 0.5, 1.5)).real,
                                 damped=False, parity="odd")
        u = seed1
        # march on the small window to a_start
        aa = 1.1025
        while (aa - a_start) * direction < 0:
            aa = round(aa + direction * 0.0025, 6)
            u = newton_local(aa, b, cnl, L1, N1, u, damped=True, parity=parity)
        u = np.real(change_window(u, L1, L))
        rows = []
        reason = "end of range"
        while (hi - aa if direction > 0 else aa - lo) > 1e-4:
            un = newton_local(aa, b, cnl, L, N, u, damped=True, parity=parity)
            if un is None or np.max(np.abs(un)) < 1e-4:
                reason = "solver failure"
                break
            t = rel_tail(un, L)
            if t >= TAIL_REL:
                reason = "localization criterion"
                break
            u = un
            rows.append(dict(a=float(aa), l2sq=float(l2_norm_sq(u, L)),
                             Q=float(np.real(quasipower(
                                 u.astype(complex) * (1 if fam == "A" else 1j),
                                 L))),
                             amp=float(np.max(np.abs(u))), tail_rel=float(t)))
            print("  %s a=%.4f |u|^2=%.4f amp=%.4f tail=%.1e"
                  % (fam, aa, rows[-1]["l2sq"], rows[-1]["amp"], t))
            aa = round(aa + direction * 0.0025, 6)
        out[fam] = dict(rows=rows, stop=reason, stop_a=float(aa),
                        last_a=rows[-1]["a"] if rows else None,
                        edge=float(hi if direction > 0 else lo))
        store[fam + "_a"] = np.array([r["a"] for r in rows])
        store[fam + "_n2"] = np.array([r["l2sq"] for r in rows])
        store[fam + "_Q"] = np.array([r["Q"] for r in rows])
        print("  %s: last accepted a=%.4f (edge %.4f), stop: %s"
              % (fam, out[fam]["last_a"], out[fam]["edge"], reason))
    np.savez(os.path.join(DATA, "revision_branchext.npz"), **store)
    update_summary("branchext", {"grid": {"L_over_pi": 96, "N": N},
                                 "A": {k: v for k, v in out["A"].items()
                                       if k != "rows"},
                                 "B": {k: v for k, v in out["B"].items()
                                       if k != "rows"}})


# ----------------------------------------------------------------------
def stage_convC4096():
    """Family C at L = 48 pi, N = 4096 with the undamped iteration: the
    damped iteration stops when the residual reaches its rounding floor
    on the finer grid before the tolerance 1e-12."""
    print("== family C, L = 48 pi, N = 4096, undamped Newton ==")
    d, Lr, Nr = load_reference()
    a, b, c = A0, B0, C0
    ur = d["u_C1"]
    N, L = 4096, Lr
    from nonlocal_gp import diff2_matrix
    x, _ = fourier_grid(L, N)
    u = newton_complex_pt(a, b, c, L, N, resample(ur, N), damped=False,
                          maxit=12, verbose=True)
    if u is None:
        print("  no convergence")
        return
    D2 = diff2_matrix(L, N)
    V = b * np.cos(2 * x)
    f, g = np.real(u), np.imag(u)
    gridres = max(np.max(np.abs(D2 @ f + (a - V) * f - c * (f ** 3 - 3 * f * g ** 2))),
                  np.max(np.abs(D2 @ g + (a - V) * g - c * (3 * f ** 2 * g - g ** 3))))
    dg = diag_full(u, a, b, c, L)
    Nf = 12288
    diff = float(np.max(np.abs(refine(u, L, Nf // N) - refine(ur, Lr, Nf // Nr))))
    dg.update(fam="C", series="N", L_over_pi=48.0, N=N, diff=diff,
              grid_residual=float(gridres), solver="undamped")
    print("  res=%.2e grid res=%.2e tail=%.2e |u|^2=%.10f Q=%.10f diff=%.2e"
          % (dg["res_nonlocal"], gridres, dg["tail_rel"], dg["l2sq"], dg["Q"], diff))
    s = load_summary()
    rows = [r for r in s["convergence"] if not (r["fam"] == "C" and r["series"] == "N" and r["N"] == 4096)]
    rows.append(dg)
    update_summary("convergence", rows)
    pc = os.path.join(DATA, "revision_convergence.npz")
    store = dict(np.load(pc)) if os.path.exists(pc) else {}
    store["u_C_N_48_4096"] = u
    np.savez(pc, **store)


# ----------------------------------------------------------------------
def stage_polishsep():
    """Undamped polishing of the family C separation states, which the
    damped iteration may leave at its rounding floor."""
    print("== family C separations: undamped polishing ==")
    a, b, c = A0, B0, C0
    L, N = 36 * np.pi, 2304
    dc = dict(np.load(os.path.join(DATA, "revision_familyC.npz")))
    s = load_summary()
    rows = s["familyC"]["separations"]
    for r in rows:
        key = "u_m%d" % r["m_seed"]
        if key not in dc:
            continue
        u = newton_complex_pt(a, b, c, L, N, dc[key], damped=False, maxit=15)
        if u is None:
            r["polished"] = False
            print("  m=%d: polishing failed, kept R_nl=%.2e" % (r["m_seed"], r["res_nonlocal"]))
            continue
        dg = diag_full(u, a, b, c, L)
        pf, pg = hump_positions(u, L, N)
        mixed = dg["max_re"] > 1e-2 and dg["max_im"] > 1e-2
        r.update(dg, pos_re=pf, pos_im=pg, mixed=bool(mixed), polished=True)
        dc[key] = u
        print("  m=%2d: R_nl=%.2e tail=%.1e |u|^2=%.4f Q=%.4f Re at %.2f pi, Im at %.2f pi"
              % (r["m_seed"], dg["res_nonlocal"], dg["tail_rel"], dg["l2sq"], dg["Q"], pf, pg))
    np.savez(os.path.join(DATA, "revision_familyC.npz"), **dc)
    update_summary("familyC", s["familyC"])


STAGES = {
    "gaps": stage_gaps, "convergence": stage_convergence,
    "localres": stage_localres, "familyC": stage_familyC,
    "doubled": stage_doubled, "manifold": stage_manifold,
    "shooting": stage_shooting, "branches": stage_branches,
    "oddsearch": stage_oddsearch, "fourier": stage_fourier,
    "mapcheck": stage_mapcheck, "branchext": stage_branchext, "convC4096": stage_convC4096, "polishsep": stage_polishsep,
}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if "all" in args:
        args = list(STAGES.keys())
    for st in args:
        t0 = time.time()
        STAGES[st]()
        print("[%s] finished in %.1f s\n" % (st, time.time() - t0))
