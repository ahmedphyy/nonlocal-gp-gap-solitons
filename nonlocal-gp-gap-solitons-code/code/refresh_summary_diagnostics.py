"""Recompute the diagnostics stored in data/summary.json from the saved
profiles, after the correction of the Nyquist splitting in
nonlocal_gp.refine (which is used for the refined-grid residuals)."""
import json
import os
import numpy as np
from run_computations import diagnostics, P1, P2, P3, DATA, SUMMARY_PATH

with open(SUMMARY_PATH) as fh:
    s = json.load(fh)
d = np.load(os.path.join(DATA, "flagship.npz"))
L = float(d["L"])
for tag in s["flagship"]["diag"]:
    old = s["flagship"]["diag"][tag]["res_nonlocal"]
    s["flagship"]["diag"][tag] = diagnostics(d["u_" + tag], P1["a"], P1["b"], P1["c"], L)
    print("flagship %s: res_nonlocal %.4e -> %.4e, res_local %.6f"
          % (tag, old, s["flagship"]["diag"][tag]["res_nonlocal"],
             s["flagship"]["diag"][tag]["res_local"]))
for key, P in (("P2", P2), ("P3", P3)):
    da = np.load(os.path.join(DATA, "alfimov_%s.npz" % key))
    La = float(da["L"])
    for tag in s["alfimov"][key]["diag"]:
        old = s["alfimov"][key]["diag"][tag]["res_nonlocal"]
        s["alfimov"][key]["diag"][tag] = diagnostics(da["u_" + tag], P["a"], P["b"], P["c"], La)
        print("%s %s: res_nonlocal %.4e -> %.4e" % (key, tag, old,
              s["alfimov"][key]["diag"][tag]["res_nonlocal"]))
s["refine_nyquist_corrected"] = True
with open(SUMMARY_PATH, "w") as fh:
    json.dump(s, fh, indent=2, default=float)
