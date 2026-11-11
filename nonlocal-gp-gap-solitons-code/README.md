# Nonlocal parity-time symmetric gap solitons: code and data

Python code and computed data for the article

> H. W. A. Riaz, A. Alshuhail, W. X. Ma, *Nonlocal parity-time symmetric
> gap solitons in the Gross-Pitaevskii equation with a periodic
> potential*, submitted to Qualitative Theory of Dynamical Systems.

The stationary problem is

    u''(x) + (a - b cos 2x) u(x) = c u(x)^2 conj(u(-x)),

solved by Fourier collocation with Newton's method (method I) and by
shooting on a Taylor-Fourier parameterization of the local stable
manifold of the periodic orbit (method II). Figure and table numbers
below refer to the article, and numbers with the prefix S refer to its
Supplementary Material.

## Requirements

Python 3.11 with NumPy, SciPy and Matplotlib (`pip install -r
requirements.txt`). All results were computed with Python 3.11.17,
NumPy 2.4.6, SciPy 1.17.1 (OpenBLAS) and Matplotlib 3.11.2 in double
precision on Linux x86_64. Defects and residuals at the level of 1e-11
and below can change in the last digit with the number of BLAS threads.

## Regenerate the figures and tables from the stored data

    cd code
    python3 make_figures_revised.py      # writes ../figures/
    python3 make_tables_revised.py       # writes ../tables/ (LaTeX)

Both scripts only read `data/` and take a few minutes. The file names
do not follow the numbering of the article:

    figures/fig01_bands.eps             Figure 1
    figures/fig05_homoclinic.png        Figure 2
    figures/fig06_manifold.png          Figure 3
    figures/fig07_humps.eps             Figure 4
    figures/fig03a_pt_profile.eps,
    figures/fig03b_pt_profile.eps       Figure 5 (a) and (b)
    figures/fig09_branch.eps            Figure 6
    figures/fig04_pt_check.eps          Figure 7
    figures/fig10_decay.eps             Figure 8
    figures/fig02_local_vs_nonlocal.eps Figure S1
    figures/fig08_existence.eps         Figure S2
    tables/tab_convergence.tex          Table 1
    tables/tab_doubled.tex              Table S2
    tables/tab_manifold.tex             Table S3
    tables/tab_shooting.tex             Table S4
    tables/tab_separations.tex          Table S5
    tables/tab_mapcheck.tex             Table S6

Table S1 (parameters of every figure and table) is written by hand in
the Supplementary Material.

## Recompute the data

Several hours on four cores. From `code/`, in this order:

    python3 run_computations.py bands flagship alfimov continuation manifold
    python3 refresh_summary_diagnostics.py
    python3 revision_computations.py gaps localres convergence convC4096 \
        fourier manifold shooting branches branchext oddsearch
    python3 revision_computations.py doubled
    DOUBLED_N=2304 DOUBLED_L_OVER_PI=36 python3 revision_computations.py doubled
    python3 revision_computations.py familyC polishsep
    python3 resolve_separations.py       # separations that need N = 3456
    python3 probe_C_ends.py              # ends of the family C branch
    python3 probe_C_ends2.py
    python3 compare_manifold_degrees.py  # degree 21 vs 29, 81 vs 121 modes
    python3 map_parallel.py              # parameter map
    python3 refine_map_parallel.py       # boundary checks of the map
    python3 refine_map_parallel.py "finer grid"
    python3 make_figures_revised.py
    python3 make_tables_revised.py

`run_computations.py manifold` also writes `data/manifold.npz`, a
parameterization of degree 18 that the article does not use.

## Contents

`code/nonlocal_gp.py` is the library (band structure, Floquet analysis,
Newton solvers, manifold parameterization, residuals). The other scripts
in `code/` are the drivers listed above.

Data files, the scripts that write them, and their use in the article:

    data/bands.npz                 run_computations.py bands; Figure 1
    data/flagship.npz              run_computations.py flagship; reference
                                   states on L = 48 pi, N = 3072 (u_A1,
                                   u_B1, u_B2, u_B1c, u_C1 on the grid x);
                                   Figures 2, 3, 4, 5, 8 and S1; Table 1
                                   (reference rows)
    data/alfimov_P2.npz, alfimov_P3.npz
                                   run_computations.py alfimov; Figure 4(c,d)
    data/continuation.npz          run_computations.py continuation;
                                   branches of A and B (Figure 6)
    data/summary.json              run_computations.py and
                                   refresh_summary_diagnostics.py;
                                   diagnostics of the reference states
    data/revision_summary.json     all other reported diagnostics, by key:
                                   gaps, local_residuals, convergence
                                   (Table 1), familyC (Table S5 and the
                                   continuation), familyC_ends,
                                   familyC_ends2, doubled, doubled_2304
                                   (Table S2), manifold (Table S3),
                                   manifold_degree_comparison, shooting
                                   (Table S4), branches, branchext,
                                   oddsearch, fourier, mapcheck_base,
                                   mapcheck_refinement (Table S6)
    data/revision_gaps.npz         Mathieu gap edges against b; Figure S2
    data/revision_convergence.npz  profiles of Table 1, keys
                                   u_<family>_<series>_<L/pi>_<N>,
                                   including u_C_N_48_4096
    data/revision_familyC.npz      family C continuation (a, l2sq, Q, res
                                   on L = 36 pi, N = 2304) and the
                                   separation states u_m<m> (N = 2304) and
                                   u_m<m>_N3456 for m = 3, 4, 5 (Table S5)
    data/revision_familyC_lower.npz, revision_familyC_upper.npz
                                   extension of the family C branch with
                                   step 0.001 (a, l2sq, Q); Figure 6
    data/revision_branchext.npz    A and B on L = 96 pi; Figure 6
    data/revision_doubled.npz, revision_doubled_2304.npz
                                   aligned doubled-system solution (u, w)
                                   and the parity-time solution uPT on
                                   L = 24 pi and 36 pi; Figure 7 uses the
                                   second
    data/revision_manifold.npz     degree-21 parameterization, coefficient
                                   tensor W[n, component, mode] (modes
                                   -M..M, M = 40), lam, l_scale = 0.4, and
                                   the norms of Figure 8(b); Figure 3
                                   recomputes the same parameterization
    data/revision_map.npz          parameter map, accepted, not localized
                                   and failure points (a, b) for A and B;
                                   Figure S2

Profiles are complex grid functions on x_j = -L + 2 L j / N.

The following are stored as numbers and not as profile arrays: the
states along the continuation branches (the files contain a, squared
norm, quasipower and residual), the states of the parameter map and its
boundary checks (only the parameter points and intervals), the shooting
profiles of method II (only the reported differences), and the
doubled-system runs other than the plotted one. The scripts above
recompute all of them.
