"""
Figures of the article and its Supplementary Material.  Reads the data
written by run_computations.py, revision_computations.py and the
auxiliary scripts listed in the README. In the manuscript repository
(when ../Revised_draft exists) each figure goes to the figures folder of
the document that uses it, Revised_draft/Main_manuscript/figures or
Revised_draft/Supplementary_material/figures; otherwise all figures go
to ../figures.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from nonlocal_gp import (spectral_deriv, reflect, fourier_grid,
                         manifold_parameterization)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
_DRAFT = os.path.join(HERE, "..", "Revised_draft")
# figures of the Supplementary Material; all others belong to the article
SUPPLEMENT_FIGS = {"fig02_local_vs_nonlocal", "fig08_existence"}


def figdir(name):
    if not os.path.isdir(_DRAFT):
        d = os.path.join(HERE, "..", "figures")
    elif name in SUPPLEMENT_FIGS:
        d = os.path.join(_DRAFT, "Supplementary_material", "figures")
    else:
        d = os.path.join(_DRAFT, "Main_manuscript", "figures")
    os.makedirs(d, exist_ok=True)
    return d

C_A = "#0072B2"
C_B = "#D55E00"
C_C = "#009E73"
C_K = "#000000"
C_G = "#777777"
C_O = "#E69F00"

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

A0, B0, C0 = 1.1025, 0.55125, -0.826875


def save2d(fig, name):
    path = os.path.join(figdir(name), name + ".eps")
    fig.savefig(path, bbox_inches="tight", format="eps")
    plt.close(fig)
    print("saved", path)


def savepng(fig, name, tight=False):
    path = os.path.join(figdir(name), name + ".png")
    if tight:
        fig.savefig(path, format="png", bbox_inches="tight")
    else:
        fig.savefig(path, format="png")
    plt.close(fig)
    print("saved", path)


def rs():
    with open(os.path.join(DATA, "revision_summary.json")) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------
def fig_humps():
    """Figure of one-hump and multihump profiles, with legends placed
    outside the data."""
    d1 = np.load(os.path.join(DATA, "flagship.npz"))
    d2 = np.load(os.path.join(DATA, "alfimov_P2.npz"))
    d3 = np.load(os.path.join(DATA, "alfimov_P3.npz"))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0))
    ax = axs[0, 0]
    x = d1["x"]
    ax.plot(x / np.pi, np.real(d1["u_A1"]), color=C_A, label="one-hump")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.85, 1.05)
    ax.set_title("(a)", loc="left")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylabel(r"$\mathrm{Re}\, u$")
    ax = axs[0, 1]
    ax.plot(x / np.pi, np.imag(d1["u_B1"]), color=C_B, label="dipole")
    ax.plot(x / np.pi, np.imag(d1["u_B2"]), color=C_K, ls="--", lw=1.0,
            label="double dipole")
    ax.set_xlim(-14, 14)
    ax.set_ylim(-1.25, 1.75)
    ax.set_title("(b)", loc="left")
    ax.legend(loc="upper center", ncol=2)
    ax.set_ylabel(r"$\mathrm{Im}\, u$")
    for ax, dd, key2, lab in [(axs[1, 0], d2, "u_A2m3", "(c)"),
                              (axs[1, 1], d3, "u_A2m2", "(d)")]:
        xx = dd["x"]
        u1 = np.real(dd["u_A1"])
        s1 = -1.0 if abs(u1.min()) > abs(u1.max()) else 1.0
        ax.plot(xx / np.pi, s1 * u1, color=C_A, label="one-hump")
        u2 = np.real(dd[key2])
        s2 = -1.0 if abs(u2.min()) > abs(u2.max()) else 1.0
        ax.plot(xx / np.pi, s2 * u2, color=C_K, ls="--", lw=1.0,
                label="two-hump")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-0.6, 1.75)
        ax.set_title(lab, loc="left")
        ax.legend(loc="upper center", ncol=2)
        ax.set_xlabel(r"$x/\pi$")
        ax.set_ylabel(r"$\mathrm{Re}\, u$")
    fig.tight_layout()
    save2d(fig, "fig07_humps")


def fig_bands():
    """Band structure of the Mathieu operator (Figure 1)."""
    d = np.load(os.path.join(DATA, "bands.npz"))
    ks = d["k"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, tag, aa in [(axs[0], "flagship", 1.1025), (axs[1], "b1", 1.0)]:
        bands = d["bands_" + tag]
        ax.set_axisbelow(True)
        for n in range(4):
            lo, hi = bands[:, n].min(), bands[:, n].max()
            ax.axhspan(lo, hi, color="0.95", lw=0, zorder=0)
            ax.axhline(lo, color=C_K, lw=0.8, zorder=2)
            ax.axhline(hi, color=C_K, lw=0.8, zorder=2)
            ax.plot(ks, bands[:, n], color=C_K, lw=0.8, zorder=2)
            ax.plot(-ks, bands[:, n], color=C_K, lw=0.8, zorder=2)
        ax.axhline(aa, color=C_A, ls="--", lw=1.2, zorder=4)
        ax.plot([0.0], [aa], marker="o", ms=5, color=C_A, clip_on=False,
                zorder=5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.2, 6.2)
        ax.set_xlabel(r"$k$")
    axs[0].set_ylabel(r"$a$")
    axs[0].set_title("(a)", loc="left")
    axs[1].set_title("(b)", loc="left")
    fig.tight_layout()
    save2d(fig, "fig01_bands")


def fig_comparison():
    """Figure 4: imaginary part of the dipole of family B computed with the
    complex solver S2 (from a seed with a nonzero real part) against the
    odd solution of the flipped local equation computed with S1."""
    d = np.load(os.path.join(DATA, "flagship.npz"))
    x = d["x"]
    gBc = np.imag(d["u_B1c"])
    gB = np.imag(d["u_B1"])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(x / np.pi, gBc, color=C_B, label=r"$\mathrm{Im}\,u$, solver S2")
    ax.plot(x[::28] / np.pi, gB[::28], ls="none", marker="o", ms=3,
            mfc="none", mec=C_K, label=r"$g$, solver S1 at $-c$")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-1.25, 1.6)
    ax.set_xlabel(r"$x/\pi$")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout()
    save2d(fig, "fig02_local_vs_nonlocal")


def fig_profileC():
    """Figure 6: the reference state of family C (two companion panels)."""
    d = np.load(os.path.join(DATA, "flagship.npz"))
    x = d["x"]
    u = d["u_C1"]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.plot(x / np.pi, np.real(u), color=C_A, lw=1.0,
            label=r"$\mathrm{Re}\, u$")
    ax.plot(x / np.pi, np.imag(u), color=C_B, lw=1.0,
            label=r"$\mathrm{Im}\, u$")
    ax.set_xlim(-24, 24)
    ax.set_xlabel(r"$x/\pi$")
    ax.set_ylabel(r"$u$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save2d(fig, "fig03a_pt_profile")
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.plot(x / np.pi, np.abs(u), color=C_K, lw=1.0)
    ax.set_xlim(-24, 24)
    ax.set_xlabel(r"$x/\pi$")
    ax.set_ylabel(r"$|u|$")
    fig.tight_layout()
    save2d(fig, "fig03b_pt_profile")


# ----------------------------------------------------------------------
def fig_homoclinic():
    """Three projections of the homoclinic orbit of family A."""
    d = np.load(os.path.join(DATA, "flagship.npz"))
    x = d["x"]
    L = float(d["L"])
    u = np.real(d["u_A1"])
    up = np.real(spectral_deriv(d["u_A1"], L, 1))
    m = np.abs(x) <= 12 * np.pi
    xs, us, ups = x[m], u[m], up[m]

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(2, 2, (1, 3), projection="3d")
    ax.plot([0, 0], [0, 0], [-1, 1], color=C_K, lw=2.5,
            label=r"projection of $\gamma$")
    left = xs <= 0
    ax.plot(us[~left], ups[~left], np.cos(2 * xs[~left]), color=C_O, lw=0.8,
            label=r"orbit, $0 \leq x \leq 12\pi$")
    ax.set_xlabel(r"$u_1$", labelpad=6)
    ax.set_ylabel(r"$u_2$", labelpad=6)
    ax.set_zlabel(r"$u_5$", labelpad=4)
    ax.view_init(elev=18, azim=-62)
    ax.set_xticks([-0.5, 0.0, 0.5])
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_zticks([-1.0, 0.0, 1.0])
    ax.tick_params(labelsize=8, pad=1)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("(a)", loc="left")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(us[left], ups[left], color=C_A, lw=0.7)
    ax2.plot(us[~left], ups[~left], color=C_O, lw=0.7, ls="--")
    ax2.plot([0], [0], marker="o", color=C_K, ms=3)
    ax2.set_xlabel(r"$u_1$")
    ax2.set_ylabel(r"$u_2$")
    ax2.set_title("(b)", loc="left")

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(xs / np.pi, us, color=C_A, lw=1.0)
    ax3.set_xlabel(r"$x/\pi$")
    ax3.set_ylabel(r"$u_1 = u$")
    ax3.set_xlim(-12, 12)
    ax3.set_title("(c)", loc="left")
    fig.tight_layout()
    savepng(fig, "fig05_homoclinic", tight=True)


# ----------------------------------------------------------------------
def fig_manifold():
    par = manifold_parameterization(A0, B0, C0, N_taylor=21, M=40,
                                    l_scale=0.4)
    W = par["W"]
    M = par["M"]
    modes = np.arange(-M, M + 1)
    th = np.linspace(0, 2 * np.pi, 161)
    sg = np.linspace(-1.0, 1.0, 41)
    TH, SG = np.meshgrid(th, sg)
    E = np.exp(1j * np.outer(th, modes))
    U1 = np.zeros_like(TH)
    U2 = np.zeros_like(TH)
    for n in range(1, W.shape[0]):
        w1 = np.real(E @ W[n, 0])
        w2 = np.real(E @ W[n, 1])
        U1 += SG ** n * w1[None, :]
        U2 += SG ** n * w2[None, :]
    Z = np.cos(2 * TH)
    d = np.load(os.path.join(DATA, "flagship.npz"))
    x = d["x"]
    L = float(d["L"])
    u = np.real(d["u_A1"])
    up = np.real(spectral_deriv(d["u_A1"], L, 1))
    m = (x > 6) & (x < 40)
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(U1, U2, Z, color=C_O, alpha=0.45, lw=0, antialiased=True)
    ax.plot([0, 0], [0, 0], [-1, 1], color=C_K, lw=3.0)
    ax.plot(u[m], up[m], np.cos(2 * x[m]), color=C_A, lw=1.4)
    ax.set_xlabel(r"$u_1$", labelpad=14)
    ax.set_ylabel(r"$u_2$", labelpad=14)
    ax.set_zlabel(r"$u_5$", labelpad=14)
    ax.view_init(elev=20, azim=-55)
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.06, top=0.98)
    savepng(fig, "fig06_manifold")


# ----------------------------------------------------------------------
def fig_ptcheck():
    d = np.load(os.path.join(DATA, "revision_doubled_2304.npz"))
    x = d["x"]
    u = d["u"]
    w = d["w"]
    L = float(d["L"])
    pt = np.conj(reflect(u))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    st = max(1, len(x) // 140)
    axs[0].plot(x / np.pi, np.real(u), color=C_A, lw=1.0,
                label=r"$\mathrm{Re}\, \tilde u(x)$")
    axs[0].plot(x[::st] / np.pi, np.real(pt)[::st], ls="none", marker="o",
                ms=2.6, mfc="none", mec=C_K,
                label=r"$\mathrm{Re}\, \overline{\tilde u(-x)}$")
    axs[0].plot(x / np.pi, np.imag(u), color=C_B, lw=1.0,
                label=r"$\mathrm{Im}\, \tilde u(x)$")
    axs[0].plot(x[::st] / np.pi, np.imag(pt)[::st], ls="none", marker="s",
                ms=2.6, mfc="none", mec=C_G,
                label=r"$\mathrm{Im}\, \overline{\tilde u(-x)}$")
    axs[0].set_xlim(-13, 13)
    axs[0].set_ylim(-1.75, 2.5)
    axs[0].set_xlabel(r"$x/\pi$")
    axs[0].legend(fontsize=7, loc="upper left", ncol=2, columnspacing=0.9,
                  handletextpad=0.4)
    axs[0].set_title("(a)", loc="left")
    axs[1].semilogy(x / np.pi, np.abs(u - pt) + 1e-18, color=C_K, lw=0.9,
                    label=r"$|\tilde u(x) - \overline{\tilde u(-x)}|$")
    axs[1].semilogy(x / np.pi, np.abs(w - pt) + 1e-18, color=C_O, lw=0.9,
                    label=r"$|\tilde w(x) - \overline{\tilde u(-x)}|$")
    axs[1].set_xlim(-L / np.pi, L / np.pi)
    axs[1].set_ylim(1e-18, 1e-8)
    axs[1].set_xlabel(r"$x/\pi$")
    axs[1].legend(fontsize=7, loc="upper right")
    axs[1].set_title("(b)", loc="left")
    fig.tight_layout()
    save2d(fig, "fig04_pt_check")


# ----------------------------------------------------------------------
def fig_branch():
    d = np.load(os.path.join(DATA, "continuation.npz"))
    lo, hi = d["gap"]
    mA = d["A_tail"] < 5e-3 * d["A_amp"]
    mB = d["B_tail"] < 5e-3 * d["B_amp"]
    dc0 = np.load(os.path.join(DATA, "revision_familyC.npz"))
    dc = {"a": dc0["a"], "l2sq": dc0["l2sq"], "Q": dc0["Q"]}
    pl = os.path.join(DATA, "revision_familyC_lower.npz")
    if os.path.exists(pl):
        dl = np.load(pl)
        keep = dl["a"] < dc["a"][0] - 1e-9
        for k in ("a", "l2sq", "Q"):
            dc[k] = np.concatenate([dl[k][keep][::-1], dc[k]])
    pu = os.path.join(DATA, "revision_familyC_upper.npz")
    if os.path.exists(pu):
        du = np.load(pu)
        keep = du["a"] > dc["a"][-1] + 1e-9
        for k in ("a", "l2sq", "Q"):
            dc[k] = np.concatenate([dc[k], du[k][keep]])
    s = json.load(open(os.path.join(DATA, "summary.json")))["flagship"]["diag"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax in axs:
        ax.axvline(lo, color=C_G, ls=":", lw=1.0)
        ax.axvline(hi, color=C_G, ls=":", lw=1.0)
        ax.axvline(A0, color=C_K, ls="-.", lw=0.7)
    axs[0].plot(d["A_a"][mA], d["A_n2"][mA], color=C_A, label="family A")
    axs[0].plot(d["B_a"][mB], d["B_n2"][mB], color=C_B, ls="--",
                label="family B")
    axs[0].plot(dc["a"], dc["l2sq"], color=C_C, ls="-", lw=1.8,
                label="family C")
    axs[1].plot(d["A_a"][mA], d["A_Q"][mA], color=C_A, label="family A")
    axs[1].plot(d["B_a"][mB], d["B_Q"][mB], color=C_B, ls="--",
                label="family B")
    axs[1].plot(dc["a"], dc["Q"], color=C_C, lw=1.8, label="family C")
    pe = os.path.join(DATA, "revision_branchext.npz")
    if os.path.exists(pe):
        de = np.load(pe)
        for fam, col in (("A", C_A), ("B", C_B)):
            if len(de[fam + "_a"]):
                axs[0].plot(de[fam + "_a"], de[fam + "_n2"], color=col, ls=":",
                            lw=1.6)
                axs[1].plot(de[fam + "_a"], de[fam + "_Q"], color=col, ls=":",
                            lw=1.6)
    for tag, col in (("A1", C_A), ("B1", C_B), ("C1", C_C)):
        axs[0].plot([A0], [s[tag]["l2sq"]], marker="*", ms=9, color=col,
                    mec=C_K, mew=0.5, ls="none")
        axs[1].plot([A0], [s[tag]["Q_re"]], marker="*", ms=9, color=col,
                    mec=C_K, mew=0.5, ls="none")
    axs[1].axhline(0.0, color=C_G, lw=0.8)
    axs[0].set_xlabel(r"$a$")
    axs[0].set_ylabel(r"$\|u\|_{2}^2$")
    axs[1].set_xlabel(r"$a$")
    axs[1].set_ylabel(r"$Q$")
    axs[0].legend(fontsize=8, loc="upper left")
    axs[0].set_title("(a)", loc="left")
    axs[1].set_title("(b)", loc="left")
    fig.tight_layout()
    save2d(fig, "fig09_branch")


# ----------------------------------------------------------------------
def fig_existence():
    dm = np.load(os.path.join(DATA, "revision_map.npz"))
    dg = np.load(os.path.join(DATA, "revision_gaps.npz"))
    bs = dg["bs"]
    edges = dg["edges"]
    gap = dg["gap"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    for ax, fam, col, mk, ms, title in ((axs[0], "A", C_A, "s", 2.6,
                                         "(a) family A"),
                                        (axs[1], "B", C_B, "o", 2.6,
                                         "(b) family B")):
        ax.set_axisbelow(True)
        for i in range(0, 8, 2):
            ax.fill_betweenx(bs, edges[:, i], edges[:, i + 1], color="0.80",
                             lw=0, zorder=0)
        ax.plot(gap[:, 0], bs, color=C_K, lw=0.9, zorder=1)
        ax.plot(gap[:, 1], bs, color=C_K, lw=0.9, zorder=1)
        acc = dm[fam + "_acc"]
        non = dm[fam + "_nonloc"]
        fail = dm[fam + "_fail"]
        if len(acc):
            ax.plot(acc[:, 0], acc[:, 1], ls="none", marker=mk, ms=ms,
                    color=col, mew=0, label="accepted", zorder=3)
        if len(non):
            ax.plot(non[:, 0], non[:, 1], ls="none", marker=mk, ms=ms,
                    mfc="none", mec=col, mew=0.5,
                    label="converged, not localized", zorder=3)
        if len(fail):
            ax.plot(fail[:, 0], fail[:, 1], ls="none", marker="x", ms=4,
                    color=C_K, mew=0.8, label="Newton failure", zorder=4)
        ax.axhline(0.0, color=C_G, lw=0.6)
        ax.plot([A0], [B0], marker="*", ms=11, color=C_K, mfc="white",
                mew=0.9, ls="none", label="reference parameters", zorder=5)
        ax.set_xlim(0.05, 2.0)
        ax.set_ylim(-1.55, 1.55)
        ax.set_xlabel(r"$a$")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel(r"$b$")
    h, l = axs[1].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([0], [0], color=C_K, lw=0.9))
    l.append("edges of the first gap")
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save2d(fig, "fig08_existence")


# ----------------------------------------------------------------------
def fig_decay():
    d = np.load(os.path.join(DATA, "flagship.npz"))
    L = float(d["L"])
    dm = np.load(os.path.join(DATA, "revision_manifold.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for tag, col, lab in (("A1", C_A, "family A"), ("B1", C_B, "family B"),
                          ("C1", C_C, "family C")):
        u = d["u_" + tag]
        N = u.size
        uh = np.abs(np.fft.fft(u)) / N
        _, k = fourier_grid(L, N)
        rel = uh / uh.max()
        sel = k >= 0
        kk, cc = k[sel], rel[sel]
        o = np.argsort(kk)
        axs[0].semilogy(kk[o][::3], cc[o][::3] + 1e-20, color=col, lw=0.8,
                        label=lab)
    axs[0].axhline(1e-13, color=C_G, ls=":", lw=0.8)
    axs[0].set_xlim(0, 32)
    axs[0].set_ylim(1e-18, 2)
    axs[0].set_xlabel(r"$k$")
    axs[0].set_ylabel(r"$|\hat u_k| / \max_k |\hat u_k|$")
    axs[0].legend(fontsize=8)
    axs[0].set_title("(a)", loc="left")
    tn = np.array(dm["norms"])
    n = np.arange(len(tn))
    pos = (tn > 0) & (n % 2 == 1)
    axs[1].semilogy(n[pos], tn[pos], ls="none", marker="o", ms=4, color=C_A)
    axs[1].set_xticks(np.arange(1, 22, 2))
    axs[1].set_xlabel(r"Taylor degree $n$ (odd)")
    axs[1].set_ylabel(r"$\|W_n\|$")
    axs[1].set_title("(b)", loc="left")
    fig.tight_layout()
    save2d(fig, "fig10_decay")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["bands", "comparison", "profileC", "humps",
                             "homoclinic", "manifold", "ptcheck",
                             "branch", "existence", "decay"]
    table = {"bands": fig_bands,
             "comparison": fig_comparison, "profileC": fig_profileC,
             "humps": fig_humps, "homoclinic": fig_homoclinic,
             "manifold": fig_manifold, "ptcheck": fig_ptcheck,
             "branch": fig_branch, "existence": fig_existence,
             "decay": fig_decay}
    for w in which:
        table[w]()
