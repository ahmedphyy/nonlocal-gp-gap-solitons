"""
Core numerical library for nonlocal PT-symmetric gap solitons in the
Gross-Pitaevskii equation with potential V(x) = b cos(2x).

Local (real) stationary equation:
    u'' + (a - b cos 2x) u = c u^3,           u : R -> R.

Nonlocal PT-symmetric stationary equation:
    u'' + (a - b cos 2x) u = c u(x)^2 conj(u(-x)),   u : R -> C.

Doubled system (w(x) = conj(u(-x))):
    u'' + (a - V) u = c u^2 w,
    w'' + (a - V) w = c w^2 u.

PT-symmetric states satisfy w = u and hence solve the complex analytic
cubic equation
    u'' + (a - V) u = c u^3,   u : R -> C.

All solvers use Fourier collocation on a large periodic domain [-L, L)
with dense Newton iterations.  Verification is by direct substitution of
the computed profile into the governing equations on refined grids.
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
from scipy.linalg import solve, eigh, lstsq


# ----------------------------------------------------------------------
# Grids and spectral differentiation
# ----------------------------------------------------------------------

def fourier_grid(L, N):
    """Periodic grid on [-L, L) with N points and Fourier wavenumbers."""
    x = -L + 2.0 * L * np.arange(N) / N
    k = 2.0 * np.pi * fftfreq(N, d=2.0 * L / N)
    return x, k


def diff2_matrix(L, N):
    """Dense second-derivative Fourier differentiation matrix."""
    _, k = fourier_grid(L, N)
    F = fft(np.eye(N), axis=0)
    D2 = np.real(ifft(-(k ** 2)[:, None] * F, axis=0))
    return D2


def spectral_deriv(u, L, order=2):
    """Spectral derivative of a (possibly complex) periodic grid function."""
    N = u.size
    _, k = fourier_grid(L, N)
    return ifft(((1j * k) ** order) * fft(u))


def reflect(u):
    """Grid representation of u(-x) on the symmetric grid x_j = -L + 2Lj/N."""
    return np.roll(u[::-1], 1)


def even_part(u):
    return 0.5 * (u + reflect(u))


def odd_part(u):
    return 0.5 * (u - reflect(u))


def refine(u, L, factor=4):
    """Trigonometric interpolation of u onto a grid refined by `factor`."""
    N = u.size
    Nf = factor * N
    uh = fft(u)
    uhf = np.zeros(Nf, dtype=complex)
    h = N // 2
    uhf[:h] = uh[:h]
    # negative modes -h+1, ..., -1 (the Nyquist mode -h is treated below)
    uhf[Nf - h + 1:] = uh[N - h + 1:]
    # split the Nyquist coefficient equally between the modes +h and -h
    uhf[h] = 0.5 * uh[h]
    uhf[Nf - h] = 0.5 * uh[h]
    return (Nf / N) * ifft(uhf)


# ----------------------------------------------------------------------
# Mathieu band structure:  u'' + (a - b cos 2x) u = 0
# ----------------------------------------------------------------------

def mathieu_bands(b, k_vals, n_bands=6, M=32):
    """
    Band functions a_n(k) of the Bloch problem for
    L u = u'' + (a - b cos 2x) u = 0, potential period pi.
    Bloch ansatz u = exp(ikx) sum_m c_m exp(2imx) yields
    (k + 2m)^2 c_m + (b/2)(c_{m-1} + c_{m+1}) = a c_m.
    Returns array of shape (len(k_vals), n_bands).
    """
    m = np.arange(-M, M + 1)
    bands = np.zeros((len(k_vals), n_bands))
    for i, kk in enumerate(k_vals):
        H = np.diag((kk + 2.0 * m) ** 2).astype(float)
        off = 0.5 * b * np.ones(2 * M)
        H += np.diag(off, 1) + np.diag(off, -1)
        w = eigh(H, eigvals_only=True)
        bands[i] = w[:n_bands]
    return bands


def band_edges(b, n_bands=6, M=32):
    """Band edge values of a: eigenvalues at k = 0 and k = 1."""
    e0 = mathieu_bands(b, [0.0], n_bands, M)[0]
    e1 = mathieu_bands(b, [1.0], n_bands, M)[0]
    edges = np.sort(np.concatenate([e0, e1]))
    return edges


def first_gap(b, M=32):
    """Endpoints of the first finite spectral gap (between bands 1 and 2)."""
    e0 = mathieu_bands(b, [0.0], 3, M)[0]
    e1 = mathieu_bands(b, [1.0], 3, M)[0]
    # bands: [e0[0], e1[0]] U [e1[1], e0[1]] ... for standard Mathieu ordering
    lo = min(e1[0], e1[1])
    hi = max(e1[0], e1[1])
    return lo, hi


# ----------------------------------------------------------------------
# Floquet analysis of the linearization along the periodic orbit
# ----------------------------------------------------------------------

def monodromy(a, b):
    """Monodromy matrix over one period pi of u'' = -(a - b cos 2x) u."""
    def rhs(x, y):
        y1, y2, z1, z2 = y
        s = -(a - b * np.cos(2.0 * x))
        return [y2, s * y1, z2, s * z1]

    sol = solve_ivp(rhs, [0.0, np.pi], [1.0, 0.0, 0.0, 1.0],
                    rtol=1e-12, atol=1e-14, dense_output=False)
    y = sol.y[:, -1]
    return np.array([[y[0], y[2]], [y[1], y[3]]])


def floquet_exponent(a, b):
    """
    Stable Floquet exponent lambda < 0 (rate of decay exp(lambda x)) for a
    in a spectral gap, together with the stable eigenvector of the
    monodromy matrix and the stable multiplier mu.
    """
    Mmat = monodromy(a, b)
    mu, W = np.linalg.eig(Mmat)
    if np.max(np.abs(np.imag(mu))) > 1e-8:
        return None, None, None   # elliptic: a inside a band
    mu = np.real(mu)
    i_s = np.argmin(np.abs(mu))
    if np.abs(mu[i_s]) >= 1.0 - 1e-12:
        return None, None, None
    lam = np.log(np.abs(mu[i_s])) / np.pi
    w0 = np.real(W[:, i_s])
    return lam, w0, mu[i_s]


def stable_bundle(a, b, M_grid=256):
    """
    2*pi-periodic stable bundle v(theta) with
    u_s(x) = exp(lambda x) P(x), P = (v1, v2) 2*pi-periodic,
    where u_s is the decaying Floquet solution of the Mathieu equation.
    Returns theta grid, P (shape (2, M_grid)), lambda.
    """
    lam, w0, mu = floquet_exponent(a, b)
    if lam is None:
        return None, None, None

    def rhs(x, y):
        return [y[1], -(a - b * np.cos(2.0 * x)) * y[0]]

    theta = 2.0 * np.pi * np.arange(M_grid) / M_grid
    sol = solve_ivp(rhs, [0.0, 2.0 * np.pi], w0, t_eval=theta,
                    rtol=1e-12, atol=1e-14)
    U = sol.y
    P = U * np.exp(-lam * theta)[None, :]
    # normalize: max |P1| = 1
    P = P / np.max(np.abs(P[0]))
    return theta, P, lam


# ----------------------------------------------------------------------
# Newton solvers
# ----------------------------------------------------------------------

def newton_local(a, b, c, L, N, seed, tol=1e-12, maxit=60, verbose=False,
                 damped=True, parity=None):
    """
    Real scalar local equation u'' + (a - b cos 2x) u - c u^3 = 0 by
    Fourier collocation and dense Newton.  Returns u or None.
    If parity is 'even' or 'odd', every iterate is projected onto the
    corresponding symmetry subspace (which the equation preserves).
    """
    x, _ = fourier_grid(L, N)
    D2 = diff2_matrix(L, N)
    V = b * np.cos(2.0 * x)
    Lin = D2 + np.diag(a - V)
    u = seed.copy().astype(float)

    def proj(w):
        if parity == "even":
            return even_part(w).real
        if parity == "odd":
            return odd_part(w).real
        return w

    u = proj(u)

    def res(w):
        return Lin @ w - c * w ** 3

    for it in range(maxit):
        F = res(u)
        r = np.max(np.abs(F))
        if verbose:
            print(f"  newton_local it={it} res={r:.3e}")
        if r < tol:
            return u
        J = Lin - np.diag(3.0 * c * u ** 2)
        try:
            du = solve(J, -F)
        except np.linalg.LinAlgError:
            return None
        if damped:
            # backtracking line search on the sup norm of the residual
            step = 1.0
            for _ in range(10):
                un = proj(u + step * du)
                if np.all(np.isfinite(un)) and np.max(np.abs(res(un))) < r:
                    break
                step *= 0.5
            else:
                return None
            u = un
        else:
            u = proj(u + du)
        if np.max(np.abs(u)) > 1e3 or not np.all(np.isfinite(u)):
            return None
    return u if np.max(np.abs(res(u))) < 1e-8 else None


def newton_complex_pt(a, b, c, L, N, seed, tol=1e-12, maxit=80,
                      enforce_pt=True, verbose=False, damped=True):
    """
    Complex analytic cubic equation u'' + (a - V) u - c u^3 = 0 with
    u = f + i g.  If enforce_pt, the iterates are projected onto the
    PT-symmetric subspace (f even, g odd).  Returns complex u or None.
    """
    x, _ = fourier_grid(L, N)
    D2 = diff2_matrix(L, N)
    V = b * np.cos(2.0 * x)
    Lin = D2 + np.diag(a - V)
    f = np.real(seed).astype(float).copy()
    g = np.imag(seed).astype(float).copy()
    if enforce_pt:
        f = even_part(f).real
        g = odd_part(g).real

    def res(ff, gg):
        F1 = Lin @ ff - c * (ff ** 3 - 3.0 * ff * gg ** 2)
        F2 = Lin @ gg - c * (3.0 * ff ** 2 * gg - gg ** 3)
        return F1, F2

    for it in range(maxit):
        F1, F2 = res(f, g)
        r = max(np.max(np.abs(F1)), np.max(np.abs(F2)))
        if verbose:
            print(f"  newton_complex it={it} res={r:.3e}")
        if r < tol:
            return f + 1j * g
        A11 = Lin - np.diag(3.0 * c * (f ** 2 - g ** 2))
        A12 = np.diag(6.0 * c * f * g)
        A21 = np.diag(-6.0 * c * f * g)
        A22 = Lin - np.diag(3.0 * c * (f ** 2 - g ** 2))
        J = np.block([[A11, A12], [A21, A22]])
        try:
            d = solve(J, -np.concatenate([F1, F2]))
        except np.linalg.LinAlgError:
            return None
        if damped:
            # backtracking line search on the sup norm of the residual
            step = 1.0
            for _ in range(10):
                fn = f + step * d[:N]
                gn = g + step * d[N:]
                if enforce_pt:
                    fn = even_part(fn).real
                    gn = odd_part(gn).real
                G1, G2 = res(fn, gn)
                if np.all(np.isfinite(G1)) and \
                        max(np.max(np.abs(G1)), np.max(np.abs(G2))) < r:
                    break
                step *= 0.5
            else:
                return None
            f, g = fn, gn
        else:
            f = f + d[:N]
            g = g + d[N:]
            if enforce_pt:
                f = even_part(f).real
                g = odd_part(g).real
        if max(np.max(np.abs(f)), np.max(np.abs(g))) > 1e3:
            return None
    F1 = Lin @ f - c * (f ** 3 - 3.0 * f * g ** 2)
    F2 = Lin @ g - c * (3.0 * f ** 2 * g - g ** 3)
    if max(np.max(np.abs(F1)), np.max(np.abs(F2))) < 1e-8:
        return f + 1j * g
    return None


def newton_doubled(a, b, c, L, N, seed_u, seed_w, tol=1e-11, maxit=60,
                   verbose=False):
    """
    Full doubled system (no symmetry imposed):
        u'' + (a - V) u - c u^2 w = 0,
        w'' + (a - V) w - c w^2 u = 0,
    with complex u, w treated as four real fields.  The gauge invariance
    (u, w) -> (exp(i t) u, exp(-i t) w) makes the Jacobian singular at
    solutions; each Newton step therefore solves a bordered square
    system in which the gauge tangent direction is appended as an extra
    column and a phase condition as an extra row.
    Returns (u, w) or (None, None).
    """
    x, _ = fourier_grid(L, N)
    D2 = diff2_matrix(L, N)
    V = b * np.cos(2.0 * x)
    Lin = D2 + np.diag(a - V)
    u = seed_u.astype(complex).copy()
    w = seed_w.astype(complex).copy()

    def split(z):
        return np.real(z), np.imag(z)

    for it in range(maxit):
        Fu = Lin @ u - c * u ** 2 * w
        Fw = Lin @ w - c * w ** 2 * u
        r = max(np.max(np.abs(Fu)), np.max(np.abs(Fw)))
        if verbose:
            print(f"  newton_doubled it={it} res={r:.3e}")
        if r < tol:
            return u, w
        u1, u2 = split(u)
        w1, w2 = split(w)
        Z = np.zeros((N, N))
        # real/imag blocks of d(c u^2 w)/du and /dw etc.
        # d(u^2 w) = 2 u w du + u^2 dw
        A = 2.0 * c * u * w      # complex multiplier of du
        B = c * u ** 2           # complex multiplier of dw
        Cm = 2.0 * c * w * u     # for Fw: d(w^2 u) = 2 w u dw + w^2 du
        Dm = c * w ** 2
        A1, A2 = split(A)
        B1, B2 = split(B)
        C1, C2 = split(Cm)
        D1r, D2r = split(Dm)
        J = np.block([
            [Lin - np.diag(A1), np.diag(A2), -np.diag(B1), np.diag(B2)],
            [-np.diag(A2), Lin - np.diag(A1), -np.diag(B2), -np.diag(B1)],
            [-np.diag(D1r), np.diag(D2r), Lin - np.diag(C1), np.diag(C2)],
            [-np.diag(D2r), -np.diag(D1r), -np.diag(C2), Lin - np.diag(C1)],
        ])
        Fv = np.concatenate([np.real(Fu), np.imag(Fu),
                             np.real(Fw), np.imag(Fw)])
        # the doubled system is invariant under (u, w) -> (z u, z^{-1} w)
        # for complex z; border the Jacobian with both group tangents
        tgauge = np.concatenate([-u2, u1, w2, -w1])     # z on unit circle
        tscale = np.concatenate([u1, u2, -w1, -w2])     # real scaling
        Jb = np.zeros((4 * N + 2, 4 * N + 2))
        Jb[:4 * N, :4 * N] = J
        Jb[:4 * N, 4 * N] = tgauge
        Jb[:4 * N, 4 * N + 1] = tscale
        Jb[4 * N, :4 * N] = tgauge
        Jb[4 * N + 1, :4 * N] = tscale
        Fb = np.concatenate([Fv, [0.0, 0.0]])
        try:
            db = solve(Jb, -Fb)
        except np.linalg.LinAlgError:
            return None, None
        d = db[:4 * N]
        u = u + d[:N] + 1j * d[N:2 * N]
        w = w + d[2 * N:3 * N] + 1j * d[3 * N:]
        if np.max(np.abs(u)) > 1e3:
            return None, None
    return None, None


# ----------------------------------------------------------------------
# Verification by direct substitution
# ----------------------------------------------------------------------

def residual_nonlocal(u, a, b, c, L, factor=2):
    """
    Sup norm of the residual of the nonlocal equation
    u'' + (a - V) u - c u^2 conj(u(-x)) obtained by direct substitution
    of the trigonometric interpolant on a grid refined by `factor`.
    """
    uf = refine(u.astype(complex), L, factor)
    Nf = uf.size
    x = -L + 2.0 * L * np.arange(Nf) / Nf
    upp = spectral_deriv(uf, L, 2)
    V = b * np.cos(2.0 * x)
    res = upp + (a - V) * uf - c * uf ** 2 * np.conj(reflect(uf))
    return np.max(np.abs(res))


def residual_local(u, a, b, c, L, factor=2):
    """Sup norm residual of the local equation u'' + (a-V)u - c|u|^2 u."""
    uf = refine(u.astype(complex), L, factor)
    Nf = uf.size
    x = -L + 2.0 * L * np.arange(Nf) / Nf
    upp = spectral_deriv(uf, L, 2)
    V = b * np.cos(2.0 * x)
    res = upp + (a - V) * uf - c * np.abs(uf) ** 2 * uf
    return np.max(np.abs(res))


def pt_defect(u):
    """Sup norm of u(x) - conj(u(-x))."""
    return np.max(np.abs(u - np.conj(reflect(u))))


def l2_norm_sq(u, L):
    N = u.size
    return np.real(np.sum(np.abs(u) ** 2)) * (2.0 * L / N)


def quasipower(u, L):
    """Nonlocal invariant Q = int u(x) conj(u(-x)) dx."""
    N = u.size
    return np.sum(u * np.conj(reflect(u))) * (2.0 * L / N)


def fourier_coeff_decay(u, L):
    """Normalized magnitudes of Fourier coefficients, sorted by |k|."""
    N = u.size
    uh = fft(u) / N
    _, k = fourier_grid(L, N)
    idx = np.argsort(k)
    return k[idx], np.abs(uh[idx])


def tail_amplitude(u, L, frac=0.05):
    """Max of |u| on the outer `frac` portion of the domain (both ends)."""
    N = u.size
    m = max(2, int(frac * N))
    return max(np.max(np.abs(u[:m])), np.max(np.abs(u[-m:])))


# ----------------------------------------------------------------------
# Seeds
# ----------------------------------------------------------------------

def gauss_hump(x, x0, amp, width):
    return amp * np.exp(-((x - x0) / width) ** 2)


def bloch_hump(x, x0, amp, width):
    """Gap-soliton-like seed: sech envelope times edge Bloch carrier."""
    return amp * np.cos(x - x0) / np.cosh((x - x0) / width)


# ----------------------------------------------------------------------
# Taylor-Fourier parameterization of the stable manifold
# ----------------------------------------------------------------------

def manifold_parameterization(a, b, c, N_taylor=15, M=40, l_scale=0.5):
    """
    Taylor-Fourier parameterization of the local stable manifold of the
    periodic orbit gamma = {u = 0, (cos 2 theta, -2 sin 2 theta)} for the
    (complexified) analytic cubic equation u'' = -(a - b cos 2 theta) u
    + c u^3, written as the first-order u-block
        U' = A(theta) U + (0, c u^3),  A = [[0, 1], [-(a - b cos 2th), 0]].
    The parameterization W(theta, sigma) = sum_n sigma^n W_n(theta)
    satisfies the invariance equation
        d/dtheta W_n + n lambda W_n = A(theta) W_n + N_n,
    with W_0 = 0, W_1 the stable bundle.  All W_n are real; evaluating
    with complex sigma parameterizes the complexified stable manifold.

    Returns dict with lambda and Wn_hat of shape (N_taylor+1, 2, 2M+1)
    (Fourier coefficients, frequencies -M..M in exp(i m theta)).
    """
    theta, P, lam = stable_bundle(a, b, 4 * M)
    if lam is None:
        return None
    modes = np.arange(-M, M + 1)

    def to_hat(fgrid):
        """Fourier coefficients (freqs -M..M) from values on theta grid."""
        n = fgrid.size
        ch = fft(fgrid) / n
        out = np.zeros(2 * M + 1, dtype=complex)
        out[M:] = ch[:M + 1]
        out[:M] = ch[-M:]
        return out

    W = np.zeros((N_taylor + 1, 2, 2 * M + 1), dtype=complex)
    W[1, 0] = to_hat(l_scale * P[0])
    W[1, 1] = to_hat(l_scale * P[1])

    # Fourier-space operator blocks
    Dth = np.diag(1j * modes)
    Id = np.eye(2 * M + 1, dtype=complex)
    # multiplication by cos 2 theta: shift by +-2
    C2 = np.zeros((2 * M + 1, 2 * M + 1), dtype=complex)
    for i in range(2 * M + 1):
        if i + 2 <= 2 * M:
            C2[i + 2, i] += 0.5
        if i - 2 >= 0:
            C2[i - 2, i] += 0.5
    Ablock = np.block([[np.zeros_like(Id), Id],
                       [-a * Id + b * C2, np.zeros_like(Id)]])

    def conv(p, q):
        """Truncated convolution of two coefficient arrays (freqs -M..M)."""
        full = np.convolve(p, q)   # length 4M+1, index s <-> frequency s-2M
        return full[M:3 * M + 1]

    for n in range(2, N_taylor + 1):
        # cubic term coefficient at order n: sum_{i+j+k=n, i,j,k>=1}
        cub = np.zeros(2 * M + 1, dtype=complex)
        for i in range(1, n - 1):
            for j in range(1, n - i):
                k = n - i - j
                if k >= 1:
                    cub += conv(conv(W[i, 0], W[j, 0]), W[k, 0])
        Nn = np.concatenate([np.zeros(2 * M + 1, dtype=complex), c * cub])
        Mn = np.block([[Dth + n * lam * Id, np.zeros_like(Id)],
                       [np.zeros_like(Id), Dth + n * lam * Id]]) - Ablock
        sol = solve(Mn, Nn)
        W[n, 0] = sol[:2 * M + 1]
        W[n, 1] = sol[2 * M + 1:]

    return {"lambda": lam, "W": W, "M": M, "N_taylor": N_taylor,
            "l_scale": l_scale}


def manifold_eval(par, theta, sigma):
    """
    Evaluate the parameterization W(theta, sigma) (both components).
    theta, sigma may be arrays of the same shape; sigma may be complex.
    """
    W = par["W"]
    M = par["M"]
    modes = np.arange(-M, M + 1)
    th = np.atleast_1d(theta)
    E = np.exp(1j * np.outer(th, modes))
    out1 = np.zeros(th.shape, dtype=complex)
    out2 = np.zeros(th.shape, dtype=complex)
    sig = np.atleast_1d(sigma).astype(complex)
    for n in range(1, W.shape[0]):
        w1 = E @ W[n, 0]
        w2 = E @ W[n, 1]
        out1 += sig ** n * w1
        out2 += sig ** n * w2
    return out1, out2


def manifold_orbit_residual(par, a, b, c, sigma0, x_span, n_pts=400):
    """
    Direct-substitution check of the manifold: the curve
    x -> W(x, sigma0 exp(lambda x)) must solve the cubic equation.
    Returns sup norm of u'' + (a - V) u - c u^3 along the curve,
    with derivatives evaluated by the chain rule through the
    parameterization (spectrally exact in theta, exact in sigma).
    """
    lam = par["lambda"]
    W = par["W"]
    M = par["M"]
    modes = np.arange(-M, M + 1)
    xs = np.linspace(x_span[0], x_span[1], n_pts)
    res = 0.0
    for x in xs:
        s = sigma0 * np.exp(lam * x)
        E = np.exp(1j * modes * x)
        u = up = upp = 0.0 + 0.0j
        for n in range(1, W.shape[0]):
            w0 = (E * W[n, 0]).sum()
            w0p = (E * (1j * modes) * W[n, 0]).sum()
            w0pp = (E * (1j * modes) ** 2 * W[n, 0]).sum()
            sn = s ** n
            u += sn * w0
            # d/dx [s^n w(x)] with s = sigma0 e^{lam x}
            up += sn * (n * lam * w0 + w0p)
            upp += sn * ((n * lam) ** 2 * w0 + 2 * n * lam * w0p + w0pp)
        V = b * np.cos(2.0 * x)
        res = max(res, abs(upp + (a - V) * u - c * u ** 3))
    return res
