import numpy as np
from scipy.linalg import solve_triangular


def givens(x: float, y: float) -> np.ndarray:
    if y == 0:
        c, s = 1.0, 0.0
    else:
        if abs(y) >= abs(x):
            cotangent = x / y
            s = 1.0 / np.sqrt(1.0 + cotangent * cotangent)
            c = s * cotangent
        else:
            tangent = y / x
            c = 1.0 / np.sqrt(1.0 + tangent * tangent)
            s = c * tangent
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def srrqr_rank(A: np.ndarray, f: float, k: int):
    """Strong rank-revealing QR with fixed rank k.

    Python translation of `matlab/sRRQR_rank.m` used in this repository.
    Returns Q, R, p where p is 0-based permutation indices.
    """
    if f < 1:
        f = 2.0

    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    k = min(k, m, n)

    # Initial pivoted QR.
    from scipy.linalg import qr

    Q, R, p = qr(A, mode="economic", pivoting=True)

    if k == n:
        return Q[:, :k], R[:k, :], p

    # Make diagonals of R positive.
    if R.shape[0] == 1 or R.shape[1] == 1:
        ss = np.sign(R[0, 0])
    else:
        ss = np.sign(np.diag(R))
    ss = np.where(ss == 0, 1.0, ss)
    R = R * ss.reshape(-1, 1)
    Q = Q * ss.reshape(1, -1)

    AB = solve_triangular(R[:k, :k], R[:k, k:], lower=False)

    gamma = np.zeros(n - k, dtype=np.float64)
    if k != R.shape[0]:
        gamma = np.sqrt(np.sum(R[k:, k:] ** 2, axis=0))

    tmp = solve_triangular(R[:k, :k], np.eye(k), lower=False)
    omega = np.sum(tmp ** 2, axis=1) ** (-0.5)

    Rm = R.shape[0]
    while True:
        mat = (np.outer(1.0 / omega, gamma)) ** 2 + AB ** 2
        idx = np.argwhere(mat > f * f)
        if idx.size == 0:
            break

        i, j = idx[0]

        # MATLAB indexes are 1-based; here i/j are 0-based.
        if j > 0:
            AB[:, [0, j]] = AB[:, [j, 0]]
            gamma[[0, j]] = gamma[[j, 0]]
            R[:, [k, k + j]] = R[:, [k + j, k]]
            p[[k, k + j]] = p[[k + j, k]]

        if i < k - 1:
            p[i:k] = p[np.r_[i + 1:k, i]]
            R[:, i:k] = R[:, np.r_[i + 1:k, i]]
            omega[i:k] = omega[np.r_[i + 1:k, i]]
            AB[i:k, :] = AB[np.r_[i + 1:k, i], :]
            for ii in range(i, k - 1):
                G = givens(R[ii, ii], R[ii + 1, ii])
                if (G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]])) < 0:
                    G = -G
                R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                Q[:, ii:ii + 2] = Q[:, ii:ii + 2] @ G.T
            if R[k - 1, k - 1] < 0:
                R[k - 1, :] = -R[k - 1, :]
                Q[:, k - 1] = -Q[:, k - 1]

        if k < Rm:
            for ii in range(k + 1, Rm):
                G = givens(R[k, k], R[ii, k])
                if (G[0, :] @ np.array([R[k, k], R[ii, k]])) < 0:
                    G = -G
                R[[k, ii], :] = G @ R[[k, ii], :]
                Q[:, [k, ii]] = Q[:, [k, ii]] @ G.T

        p[[k - 1, k]] = p[[k, k - 1]]
        ga = R[k - 1, k - 1]
        mu = R[k - 1, k] / ga
        nu = R[k, k] / ga if k < Rm else 0.0
        rho = np.sqrt(mu * mu + nu * nu)
        ga_bar = ga * rho
        b1 = R[:k - 1, k - 1].copy()
        b2 = R[:k - 1, k].copy()
        c1T = R[k - 1, k + 1:].copy()
        c2T = np.zeros_like(c1T) if k >= Rm else R[k, k + 1:].copy()
        c1T_bar = (mu * c1T + nu * c2T) / rho
        c2T_bar = (nu * c1T - mu * c2T) / rho

        R[:k - 1, k - 1] = b2
        R[:k - 1, k] = b1
        R[k - 1, k - 1] = ga_bar
        R[k - 1, k] = ga * mu / rho
        R[k, k] = ga * nu / rho
        R[k - 1, k + 1:] = c1T_bar
        R[k, k + 1:] = c2T_bar

        u = solve_triangular(R[:k - 1, :k - 1], b1, lower=False)
        u1 = AB[:k - 1, 0].copy()
        AB[:k - 1, 0] = (nu * nu * u - mu * u1) / (rho * rho)
        AB[k - 1, 0] = mu / (rho * rho)
        AB[k - 1, 1:] = c1T_bar / ga_bar
        if AB.shape[1] > 1:
            AB[:k - 1, 1:] = AB[:k - 1, 1:] + (
                np.outer(nu * u, c2T_bar) - np.outer(u1, c1T_bar)
            ) / ga_bar

        gamma[0] = ga * nu / rho
        if gamma.shape[0] > 1:
            gamma[1:] = np.sqrt(gamma[1:] ** 2 + c2T_bar ** 2 - c2T ** 2)

        u_bar = u1 + mu * u
        omega[k - 1] = ga_bar
        omega[:k - 1] = (
            omega[:k - 1] ** (-2) + u_bar ** 2 / (ga_bar * ga_bar) - u ** 2 / (ga * ga)
        ) ** (-0.5)

        if k < Rm:
            Gk = np.array([[mu / rho, nu / rho], [nu / rho, -mu / rho]])
            Q[:, [k - 1, k]] = Q[:, [k - 1, k]] @ Gk.T

    return Q[:, :k], R[:k, :], p
