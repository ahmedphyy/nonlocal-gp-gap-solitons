"""
LaTeX tables of the article and its Supplementary Material, generated
from data/revision_summary.json and data/summary.json. In the manuscript
repository (when ../Revised_draft exists) each table goes to the tables
folder of the document that uses it, Revised_draft/Main_manuscript/tables
or Revised_draft/Supplementary_material/tables; otherwise all tables go
to ../tables.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
_DRAFT = os.path.join(HERE, "..", "Revised_draft")
# tables of the article; all others belong to the Supplementary Material
ARTICLE_TABLES = {"tab_convergence"}


def tabdir(name):
    if not os.path.isdir(_DRAFT):
        d = os.path.join(HERE, "..", "tables")
    elif name in ARTICLE_TABLES:
        d = os.path.join(_DRAFT, "Main_manuscript", "tables")
    else:
        d = os.path.join(_DRAFT, "Supplementary_material", "tables")
    os.makedirs(d, exist_ok=True)
    return d


def sci(v, digits=1):
    if v is None:
        return "n/a"
    if v == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    if round(m, digits) >= 10:
        m /= 10
        e += 1
    return "$%.*f\\cdot10^{%d}$" % (digits, m, e)


def load():
    with open(os.path.join(DATA, "revision_summary.json")) as fh:
        return json.load(fh)


def write(name, body):
    with open(os.path.join(tabdir(name), name + ".tex"), "w") as fh:
        fh.write(body)
    print("wrote", name)


def table_manifold(s):
    rows = s["manifold"]["truncation"]
    lines = ["\\begin{tabular}{ccccc}", "\\toprule",
             "modes & degree & $\\sigma_0 = 0.9$ & "
             "$\\sigma_0 = 0.9e^{i\\pi/3}$ & $\\sigma_0 = 0.9i$ \\\\",
             "\\midrule"]
    last = None
    for r in rows:
        if last is not None and r["modes"] != last:
            lines.append("\\midrule")
        last = r["modes"]
        lines.append("%d & %d & %s & %s & %s \\\\" % (
            r["modes"], r["N_taylor"], sci(r["res_real"]),
            sci(r["res_complex"]), sci(r["res_imag"])))
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_manifold", "\n".join(lines) + "\n")


def table_convergence(s):
    rows = s["convergence"]
    order = {"ref": 0, "L": 1, "N": 2}
    lines = ["\\begin{tabular}{cccccccc}", "\\toprule",
             "family & $L/\\pi$ & $N$ & $R_{\\mathrm{nl}}$ & $\\tau$ & "
             "$\\|u\\|_2^2$ & $Q$ & $\\Delta$ \\\\", "\\midrule"]
    for fam in ("A", "B", "C"):
        fr = [r for r in rows if r["fam"] == fam]
        fr.sort(key=lambda r: (r["N"] / r["L_over_pi"] != 64.0,
                               r["L_over_pi"], r["N"]))
        for r in fr:
            mark = "$^{*}$" if r["series"] == "ref" else ""
            lines.append("%s & %d & %d%s & %s & %s & %.10f & %.10f & %s \\\\"
                         % (fam, round(r["L_over_pi"]), r["N"], mark,
                            sci(r["res_nonlocal"]), sci(r["tail_rel"]),
                            r["l2sq"], r["Q"],
                            "ref." if r["series"] == "ref" else sci(r["diff"])))
        if fam != "C":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_convergence", "\n".join(lines) + "\n")


def table_separations(s):
    rows = s["familyC"]["separations"]
    lines = ["\\begin{tabular}{ccccccccc}", "\\toprule",
             "$m$ & $N$ & mixed & $x_f/\\pi$ & $x_g/\\pi$ & $R_{\\mathrm{nl}}$ & "
             "$\\tau$ & $\\|u\\|_2^2$ & $Q$ \\\\", "\\midrule"]
    for r in rows:
        if not r.get("converged"):
            lines.append("%d & %d & \\multicolumn{7}{c}{no convergence} \\\\"
                         % (r["m_seed"], r.get("N", 2304)))
            continue
        lines.append("%d & %d & %s & %.2f & %.2f & %s & %s & %.4f & %.4f \\\\" % (
            r["m_seed"], r.get("N", 2304), "yes" if r["mixed"] else "no", r["pos_re"],
            r["pos_im"], sci(r["res_nonlocal"]), sci(r["tail_rel"]),
            r["l2sq"], r["Q"]))
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_separations", "\n".join(lines) + "\n")


def table_mapcheck(s):
    rows = s.get("mapcheck_refinement") or s["mapcheck"]["refinement"]
    names = ["base step", "half step", "larger window", "finer grid"]
    lines = ["\\begin{tabular}{cccccc}", "\\toprule",
             "family & $b$ & base & $\\Delta a = 0.01$ & $L = 24\\pi$ & "
             "$N = 2048$ \\\\", "\\midrule"]
    for r in rows:
        if not r.get("start"):
            lines.append("%s & $%+.1f$ & \\multicolumn{4}{c}{no start} \\\\"
                         % (r["fam"], r["b"]))
            continue
        cells = []
        for nm in names:
            v = r.get(nm)
            if v is None or v["lower"] is None:
                cells.append("none")
            else:
                cells.append("$[%.3f, %.3f]$" % (v["lower"], v["upper"]))
        lines.append("%s & $%+.1f$ & %s \\\\" % (r["fam"], r["b"],
                                                 " & ".join(cells)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_mapcheck", "\n".join(lines) + "\n")


def table_shooting(s):
    """Method II against method I (stage shooting)."""
    sh = s["shooting"]
    lines = ["\\begin{tabular}{cccccc}", "\\toprule",
             "family & $L$ & $\\sigma_L$ & boundary residual & "
             "$\\Delta$ on $[0,L]$ & $\\Delta$ on $[L,L+20]$ \\\\",
             "\\midrule"]
    for key in ("A1_L30", "A1_L40", "C1_L60", "C1_L70"):
        r = sh[key]
        sr, si = r["sigma_L"]
        if abs(si) < 1e-12:
            sig = "$%.4f$" % sr
        else:
            sig = "$%.4f %s %.4fi$" % (sr, "+" if si >= 0 else "-", abs(si))
        hist = r["bc_history"]
        if len(hist) <= 2:
            bc = sci(hist[-1])
        else:
            # after the second step the residual stagnates; give its range
            tail = hist[1:]
            bc = "%s to %s" % (sci(min(tail)), sci(max(tail)))
        lines.append("%s & $%.2f$ & %s & %s & %s & %s \\\\" % (
            key[0], r["L_m"], sig, bc, sci(r["diff_on_0_L"]),
            sci(r["diff_tail_20"])))
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_shooting", "\n".join(lines) + "\n")


def table_doubled(s):
    """Doubled-system check of family C on L = 36 pi (stage doubled)."""
    runs = sorted(s["doubled_2304"]["runs"], key=lambda r: r["eps"])
    lines = ["\\begin{tabular}{ccccccccc}", "\\toprule",
             "$\\varepsilon$ & seed $\\delta_{\\mathcal{R}}$ & "
             "$\\delta_{\\mathcal{R}}$ & scaled $\\delta_{\\mathcal{R}}$ & "
             "$\\delta_{\\PT}(\\tilde u)$ & $R_{\\mathrm{dbl}}$ & "
             "$R_{\\mathrm{nl}}(\\tilde u)$ & Wronskian & "
             "$\\Delta$ to S2 \\\\", "\\midrule"]
    for r in runs:
        lines.append(" & ".join([
            sci(r["eps"]), sci(r["seed_defect"]),
            sci(r["raw_constraint_defect"]),
            sci(r["aligned_constraint_defect"]),
            sci(r["aligned_pt_defect"]), sci(r["res_doubled"]),
            sci(r["res_nonlocal_aligned"]), sci(r["J_max"]),
            sci(r["diff_to_PT_solver"])]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_doubled", "\n".join(lines) + "\n")


if __name__ == "__main__":
    s = load()
    table_manifold(s)
    table_convergence(s)
    if "familyC" in s:
        table_separations(s)
    if "mapcheck_refinement" in s or "mapcheck" in s:
        table_mapcheck(s)
    if "shooting" in s:
        table_shooting(s)
    if "doubled_2304" in s:
        table_doubled(s)
