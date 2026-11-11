"""
Production computations for the nonlocal PT-symmetric gap soliton study.

Usage:  python3 run_computations.py [stage ...]
Stages: bands flagship alfimov continuation map manifold all

Results are stored as .npz files in ../data together with a JSON summary
of the diagnostics (residuals by direct substitution, norms, quasipower,
Floquet exponents, spectral gaps).
"""

import json
import os
import sys
import time

import numpy as np

from nonlocal_gp import (
    fourier_grid, newton_local, newton_complex_pt, newton_doubled,
    residual_nonlocal, residual_local, pt_defect, l2_norm_sq, quasipower,
    tail_amplitude, reflect, even_part, odd_part,
    bloch_hump, gauss_hump, mathieu_bands, first_gap, floquet_exponent,
    manifold_parameterization, manifold_eval, manifold_orbit_residual,
)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA, exist_ok=True)

SUMMARY_PATH = os.path.join(DATA, "summary.json")


def load_summary():
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as fh:
            return json.load(fh)
    return {}


def save_summary(s):
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(s, fh, indent=2, default=float)


# flagship parameter set and the two Alfimov-Konotop-Salerno sets
P1 = dict(a=1.1025, b=0.55125, c=-0.826875)
P2 = dict(a=1.0, b=-1.0, c=1.0)
P3 = dict(a=1.0, b=1.0, c=1.0)


def shift_lattice(u, m, L, N):
    """Shift a grid function by m*pi (dx must divide pi)."""
    dx = 2.0 * L / N
    s = m * np.pi / dx
    si = int(round(s))
    assert abs(s - si) < 1e-9, "pi must be a grid multiple"
    return np.roll(u, si)


def diagnostics(u, a, b, c, L):
    return {
        "max_re": float(np.max(np.abs(np.real(u)))),
        "max_im": float(np.max(np.abs(np.imag(u)))),
        "res_nonlocal": float(residual_nonlocal(u, a, b, c, L)),
        "res_local": float(residual_local(u, a, b, c, L)),
        "pt_defect": float(pt_defect(np.asarray(u, dtype=complex))),
        "l2sq": float(l2_norm_sq(u, L)),
        "Q_re": float(np.real(quasipower(u, L))),
        "Q_im": float(np.imag(quasipower(u, L))),
        "tail": float(tail_amplitude(u, L)),
    }


def dipole_seed(x, x0=np.pi / 2, amp=0.5, width=1.5):
    return odd_part(bloch_hump(x, x0, amp, width)
                    - bloch_hump(x, -x0, amp, width)).real


# ----------------------------------------------------------------------
def stage_bands():
    print("== band structure ==")
    ks = np.linspace(0, 1, 201)
    out = {}
    for tag, b in [("flagship", P1["b"]), ("b1", 1.0)]:
        bands = mathieu_bands(b, ks, n_bands=5)
        out[f"bands_{tag}"] = bands
        print(f"  b={b}: first gap {first_gap(b)}")
    # gap edges as functions of b for the existence map
    bs = np.linspace(-1.6, 1.6, 161)
    edges = np.array([np.sort(np.concatenate([
        mathieu_bands(bb, [0.0], 3)[0], mathieu_bands(bb, [1.0], 3)[0]
    ])) for bb in bs])
    np.savez(os.path.join(DATA, "bands.npz"), k=ks, bs=bs, edges=edges, **out)
    s = load_summary()
    s["gap_flagship"] = first_gap(P1["b"])
    s["gap_b1"] = first_gap(1.0)
    lam, _, mu = floquet_exponent(P1["a"], P1["b"])
    s["floquet_flagship"] = {"lambda": lam, "mu": mu}
    lam2, _, mu2 = floquet_exponent(1.0, 1.0)
    s["floquet_b1"] = {"lambda": lam2, "mu": mu2}
    save_summary(s)


# ----------------------------------------------------------------------
def stage_flagship():
    print("== flagship families (a, b, c) =", P1)
    a, b, c = P1["a"], P1["b"], P1["c"]
    L, N = 48 * np.pi, 3072
    x, _ = fourier_grid(L, N)
    sols = {}
    diag = {}

    t0 = time.time()
    # family A: real even one-hump (identical to the local gap soliton)
    fA = newton_local(a, b, c, L, N, bloch_hump(x, 0.0, 1.0, 3.0),
                      damped=False, parity="even")
    sols["A1"] = fA.astype(complex)
    # family B: imaginary odd dipole (solves the flipped-sign local eq.)
    gB = newton_local(a, b, -c, L, N, dipole_seed(x),
                      damped=False, parity="odd")
    sols["B1"] = 1j * gB
    # family B: double dipole (imaginary odd)
    seed = shift_lattice(gB, 8, L, N) + shift_lattice(gB, -8, L, N)
    gB2 = newton_local(a, b, -c, L, N, odd_part(seed).real,
                       damped=True, parity="odd", maxit=150)
    if gB2 is not None and np.max(np.abs(gB2)) > 1e-3:
        sols["B2"] = 1j * gB2
    print("  real families done (%.1f s)" % (time.time() - t0))

    # family B via the complex solver (independent cross-computation for
    # the comparison figure): perturbed complex seed, no reality imposed
    t0 = time.time()
    pert = gauss_hump(x, 0.0, 0.15, 6.0)
    uB1c = newton_complex_pt(a, b, c, L, N, pert + 1j * gB, damped=True,
                             maxit=150)
    if uB1c is not None:
        sols["B1c"] = uB1c
    # family C: mixed complex PT state; central imaginary dipole with a
    # pair of real staggered humps
    fseed = even_part(shift_lattice(fA, 8, L, N)
                      + shift_lattice(fA, -8, L, N)).real
    uC1 = newton_complex_pt(a, b, c, L, N, fseed + 1j * gB, damped=True,
                            maxit=200)
    if uC1 is not None and np.max(np.abs(np.imag(uC1))) > 1e-2 \
            and np.max(np.abs(np.real(uC1))) > 1e-2:
        sols["C1"] = uC1
    print("  complex families done (%.1f s)" % (time.time() - t0))

    # cross-validation of C1 in the full doubled system: perturbed,
    # asymmetric initial data, no symmetry imposed; gauge-aligned defect
    xd = ud = None
    if "C1" in sols:
        t0 = time.time()
        Nd, Ld = 1536, 24 * np.pi
        xd, _ = fourier_grid(Ld, Nd)
        idx = [int(round((xx + L) / (2 * L / N))) % N for xx in xd]
        seedc = sols["C1"][idx]
        uDpt = newton_complex_pt(a, b, c, Ld, Nd, seedc, damped=True,
                                 maxit=100)
        if uDpt is not None:
            rng = np.random.default_rng(7)
            p1 = 1.0 + 0.02 * np.cos(np.pi * xd / Ld) \
                + 0.01 * np.sin(2 * np.pi * xd / Ld)
            p2 = 1.0 - 0.015 * np.cos(2 * np.pi * xd / Ld) \
                + 0.01j * np.sin(np.pi * xd / Ld)
            su = uDpt * p1
            sw = np.conj(reflect(uDpt)) * p2
            uu, ww = newton_doubled(a, b, c, Ld, Nd, su, sw, maxit=80)
            if uu is not None:
                sref = np.conj(reflect(uu))
                z = np.vdot(sref, ww)
                phase = z / abs(z)
                defect = float(np.max(np.abs(ww - phase * sref)))
                # gauge-align u so that it is PT-symmetric itself
                th = 0.5 * np.angle(phase)
                ua = uu * np.exp(-1j * th)
                s = load_summary()
                s["doubled_check_C1"] = {
                    "constraint_defect": defect,
                    "res_u_nonlocal": float(residual_nonlocal(
                        ua, a, b, c, Ld)),
                    "pt_defect_aligned": float(pt_defect(ua)),
                }
                save_summary(s)
                ud = ua
                print("  doubled-system check: gauge-aligned defect=%.2e, "
                      "res=%.2e (%.1f s)"
                      % (defect, s["doubled_check_C1"]["res_u_nonlocal"],
                         time.time() - t0))

    for tag, u in sols.items():
        diag[tag] = diagnostics(u, a, b, c, L)
        print("  %s: %s" % (tag,
                            {k: ('%.3e' % v) for k, v in diag[tag].items()}))

    extra = {}
    if ud is not None:
        extra["x_D"] = xd
        extra["u_D"] = ud
    np.savez(os.path.join(DATA, "flagship.npz"), x=x, L=L, N=N,
             **{f"u_{k}": v for k, v in sols.items()}, **extra)
    s = load_summary()
    s["flagship"] = {"params": P1, "L": L, "N": N, "diag": diag}
    save_summary(s)


# ----------------------------------------------------------------------
def stage_alfimov():
    print("== Alfimov-Konotop-Salerno parameter sets ==")
    diag_all = {}
    for tag, P in [("P2", P2), ("P3", P3)]:
        a, b, c = P["a"], P["b"], P["c"]
        L, N = 24 * np.pi, 1536
        x, _ = fourier_grid(L, N)
        sols = {}
        # even one-hump (b < 0: single hump at 0; b > 0: well pair)
        f1 = newton_local(a, b, c, L, N, bloch_hump(x, 0.0, 1.0, 3.0),
                          damped=False, parity="even")
        if f1 is None or np.max(np.abs(f1)) < 1e-3:
            f1 = newton_local(a, b, c, L, N, gauss_hump(x, 0.0, 1.2, 2.0),
                              damped=False, parity="even")
        sols["A1"] = f1.astype(complex)
        # even two-hump states at several separations
        for m in (1, 2, 3):
            s2 = shift_lattice(f1, m, L, N) + shift_lattice(f1, -m, L, N)
            f2 = newton_local(a, b, c, L, N, even_part(s2).real,
                              damped=True, parity="even", maxit=150)
            if f2 is not None and np.max(np.abs(f2)) > 1e-3 and \
                    l2_norm_sq(f2, L) > 1.2 * l2_norm_sq(f1, L):
                sols[f"A2m{m}"] = f2.astype(complex)
        # family B: odd solution of the flipped-sign equation
        for x0, amp, w in [(np.pi / 2, 0.5, 1.5), (np.pi / 2, 1.0, 2.0),
                           (np.pi, 1.0, 2.0), (np.pi, 0.6, 3.0)]:
            sd = odd_part(bloch_hump(x, x0, amp, w)
                          - bloch_hump(x, -x0, amp, w)).real
            gB = newton_local(a, b, -c, L, N, sd, damped=False, parity="odd")
            if gB is not None and np.max(np.abs(gB)) > 1e-3:
                sols["B1"] = 1j * gB
                break
        d = {}
        for k, u in sols.items():
            d[k] = diagnostics(u, a, b, c, L)
            print("  %s/%s: %s" % (tag, k,
                                   {kk: ('%.3e' % vv)
                                    for kk, vv in d[k].items()}))
        if "B1" not in sols:
            print("  %s: no family B soliton found (odd, flipped sign)"
                  % tag)
        diag_all[tag] = d
        np.savez(os.path.join(DATA, f"alfimov_{tag}.npz"), x=x, L=L, N=N,
                 **{f"u_{k}": v for k, v in sols.items()})
    s = load_summary()
    s["alfimov"] = {"P2": {"params": P2, "diag": diag_all.get("P2")},
                    "P3": {"params": P3, "diag": diag_all.get("P3")}}
    save_summary(s)


# ----------------------------------------------------------------------
def continue_in_a(b, cnl, L, N, seed, a_vals, parity):
    """March a family in a with warm starts.  Returns dict of arrays."""
    res_a, res_n2, res_Q, res_amp, res_tail = [], [], [], [], []
    u = seed.copy()
    for a in a_vals:
        un = newton_local(a, b, cnl, L, N, u, damped=True, parity=parity)
        if un is None or np.max(np.abs(un)) < 1e-4:
            break
        u = un
        res_a.append(a)
        res_n2.append(l2_norm_sq(u, L))
        res_amp.append(np.max(np.abs(u)))
        res_tail.append(tail_amplitude(u, L))
        if parity == "even":
            res_Q.append(np.real(quasipower(u.astype(complex), L)))
        else:
            res_Q.append(np.real(quasipower(1j * u, L)))
    return dict(a=np.array(res_a), n2=np.array(res_n2),
                Q=np.array(res_Q), amp=np.array(res_amp),
                tail=np.array(res_tail))


def stage_continuation():
    print("== continuation in a at fixed b, c ==")
    b, c = P1["b"], P1["c"]
    lo, hi = first_gap(b)
    L, N = 24 * np.pi, 1536
    x, _ = fourier_grid(L, N)
    a0 = P1["a"]
    da = 0.0025

    fA = newton_local(a0, b, c, L, N, bloch_hump(x, 0.0, 1.0, 3.0),
                      damped=False, parity="even")
    gB = newton_local(a0, b, -c, L, N, dipole_seed(x),
                      damped=False, parity="odd")

    up_A = continue_in_a(b, c, L, N, fA, np.arange(a0, hi + 0.02, da),
                         "even")
    dn_A = continue_in_a(b, c, L, N, fA, np.arange(a0, lo - 0.02, -da),
                         "even")
    up_B = continue_in_a(b, -c, L, N, gB, np.arange(a0, hi + 0.02, da),
                         "odd")
    dn_B = continue_in_a(b, -c, L, N, gB, np.arange(a0, lo - 0.02, -da),
                         "odd")

    def merge(dn, up):
        return {k: np.concatenate([dn[k][::-1], up[k][1:]])
                for k in up.keys()}

    A = merge(dn_A, up_A)
    B = merge(dn_B, up_B)
    print("  family A: a in [%.4f, %.4f], %d points"
          % (A["a"][0], A["a"][-1], len(A["a"])))
    print("  family B: a in [%.4f, %.4f], %d points"
          % (B["a"][0], B["a"][-1], len(B["a"])))
    np.savez(os.path.join(DATA, "continuation.npz"),
             gap=[lo, hi], b=b, c=c,
             **{f"A_{k}": v for k, v in A.items()},
             **{f"B_{k}": v for k, v in B.items()})
    s = load_summary()
    s["continuation"] = {
        "b": b, "c": c, "gap": [lo, hi],
        "A_range": [float(A["a"][0]), float(A["a"][-1])],
        "B_range": [float(B["a"][0]), float(B["a"][-1])]}
    save_summary(s)


# ----------------------------------------------------------------------
def stage_map():
    print("== existence map in the (a, b) plane ==")
    c = P1["c"]
    L, N = 16 * np.pi, 1024
    x, _ = fourier_grid(L, N)
    b_pos = np.round(np.arange(0.10, 1.5001, 0.05), 4)
    b_vals = np.concatenate([-b_pos[::-1], b_pos])
    a_step = 0.02
    tail_rel = 5e-3     # localization criterion within the fixed window

    def sweep(cnl, parity, seed_list):
        pts = []
        for b in b_vals:
            lo, hi = first_gap(b)
            u_mid = None
            mid = None
            for frac in (0.5, 0.35, 0.65, 0.25, 0.75):
                aa = lo + frac * (hi - lo)
                for sd in seed_list:
                    cand = newton_local(aa, b, cnl, L, N, sd, damped=False,
                                        parity=parity)
                    if cand is not None and np.max(np.abs(cand)) > 1e-3 \
                            and tail_amplitude(cand, L) < \
                            tail_rel * np.max(np.abs(cand)):
                        u_mid, mid = cand, aa
                        break
                if u_mid is not None:
                    break
            if u_mid is None:
                continue
            for direction in (+1, -1):
                u = u_mid.copy()
                a = mid
                while lo - 0.1 < a < hi + 0.1:
                    un = newton_local(a, b, cnl, L, N, u, damped=True,
                                      parity=parity)
                    if un is None or np.max(np.abs(un)) < 1e-4:
                        break
                    u = un
                    if tail_amplitude(u, L) < tail_rel * np.max(np.abs(u)):
                        pts.append((a, b))
                    a = a + direction * a_step
        return np.array(pts) if pts else np.zeros((0, 2))

    seedsA = [bloch_hump(x, 0.0, 1.0, 3.0), gauss_hump(x, 0.0, 1.2, 2.0)]
    seedsB = [dipole_seed(x),
              dipole_seed(x, x0=np.pi, amp=1.0, width=2.0),
              dipole_seed(x, amp=1.0, width=2.0)]

    t0 = time.time()
    ptsA = sweep(c, "even", seedsA)
    print("  family A sweep: %d points (%.0f s)"
          % (len(ptsA), time.time() - t0))
    t0 = time.time()
    ptsB = sweep(-c, "odd", seedsB)
    print("  family B sweep: %d points (%.0f s)"
          % (len(ptsB), time.time() - t0))
    np.savez(os.path.join(DATA, "existence_map.npz"),
             ptsA=ptsA, ptsB=ptsB, c=c)


# ----------------------------------------------------------------------
def stage_manifold():
    print("== stable manifold parameterization ==")
    a, b, c = P1["a"], P1["b"], P1["c"]
    par = manifold_parameterization(a, b, c, N_taylor=18, M=40, l_scale=0.4)
    lam = par["lambda"]
    W = par["W"]
    taylor_norms = np.array([np.sum(np.abs(W[n, 0]))
                             for n in range(W.shape[0])])
    print("  lambda = %.8f" % lam)
    res_orbit = manifold_orbit_residual(par, a, b, c, sigma0=0.9,
                                        x_span=(0.0, 4 * np.pi))
    print("  orbit residual (direct substitution): %.3e" % res_orbit)

    # match the flagship one-hump tail against the manifold flow
    dat = np.load(os.path.join(DATA, "flagship.npz"))
    x = dat["x"]
    u = dat["u_A1"]
    xr = 30.0
    j = int(np.argmin(np.abs(x - xr)))
    modes = np.arange(-par["M"], par["M"] + 1)
    w1x = np.real(np.exp(1j * modes * x[j]) @ W[1, 0])
    sigma0 = np.real(u[j]) / w1x * np.exp(-lam * x[j])
    # refine the matching coordinate with the full Taylor series
    for _ in range(4):
        um, _ = manifold_eval(par, np.mod(x[j], 2 * np.pi),
                              sigma0 * np.exp(lam * x[j]))
        sigma0 = sigma0 * float(np.real(u[j]) / np.real(np.ravel(um)[0]))
    m = (x > 24) & (x < 44)
    uu, _ = manifold_eval(par, np.mod(x[m], 2 * np.pi),
                          sigma0 * np.exp(lam * x[m]))
    mm = float(np.max(np.abs(np.real(uu) - np.real(u[m]))))
    print("  tail match sup difference: %.3e (sigma0=%.4f)" % (mm, sigma0))

    np.savez(os.path.join(DATA, "manifold.npz"),
             lam=lam, W=W, M=par["M"], N_taylor=par["N_taylor"],
             taylor_norms=taylor_norms, sigma0=sigma0)
    s = load_summary()
    s["manifold"] = {"lambda": lam, "orbit_residual": float(res_orbit),
                     "tail_match": mm, "sigma0": float(sigma0),
                     "taylor_norms": taylor_norms.tolist()}
    save_summary(s)


# ----------------------------------------------------------------------
STAGES = {
    "bands": stage_bands,
    "flagship": stage_flagship,
    "alfimov": stage_alfimov,
    "continuation": stage_continuation,
    "map": stage_map,
    "manifold": stage_manifold,
}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if "all" in args:
        args = ["bands", "flagship", "alfimov", "continuation", "map",
                "manifold"]
    for st in args:
        t0 = time.time()
        STAGES[st]()
        print("[%s] finished in %.1f s\n" % (st, time.time() - t0))
