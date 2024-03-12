import numpy as np


def IRF_simulations_with_GARCH(y: np.ndarray, x: np.ndarray, nlag: int, nboot: int,
                               S: int, C: np.ndarray, Hbar: np.ndarray) -> dict:

    T, neq = y.shape

    rng = np.random.default_rng(np.random.MT19937(seed=0))
    draws = rng.integers(0, T, (T, nboot))  # T * nboot

    # draws: np.ndarray = np.random.randint(0, T, (T, nboot))  # T * nboot

    iQbar: np.ndarray = np.linalg.inv(np.linalg.cholesky(Hbar))  # square matrix of hbar * hbar
    Qt: np.ndarray = np.zeros((S, neq, neq))  # S * neq * neq
    for s in range(S):
        Ht = C[s, :, :]
        H0t = np.dot(Ht, np.linalg.inv(Hbar))
        l = np.linalg.eigvals(H0t)
        ml = np.min(l)
        if ml > 0:
            Qt[s, :, :] = np.linalg.cholesky(H0t)
        else:
            Qt[s, :, :] = np.eye(neq)

    B = [{'IRF': np.zeros((S, neq, neq)), 'IRF_No_GARCH': np.zeros((S, neq, neq)),
          'IRF_GARCH': np.zeros((S, neq, neq))} for _ in range(nboot)]

    for i in range(nboot):
        yi = y[draws[:, i], :]  # T * neq
        xi = x[draws[:, i], :]  # T * (neq * nlag)
        bi = np.linalg.lstsq(xi, yi, rcond=None)[0]  # regression, (neq * nlag) * neq
        AR_Terms = bi[:neq * nlag, :neq].T

        M = np.vstack([AR_Terms, np.eye(neq * nlag, neq * nlag, k=1)])
        M = np.hstack([M, np.zeros((neq * nlag + neq, neq))])

        IRF = np.zeros((S, neq, neq))
        IRF_No_GARCH = IRF.copy()
        IRF_GARCH = IRF_No_GARCH.copy()

        for s in range(S):
            mu_irf = np.linalg.matrix_power(M, s + 1)[:neq, :neq]
            irf_qbar = np.dot(mu_irf, iQbar)
            irf_qt = np.dot(irf_qbar, np.linalg.inv(Qt[s, :, :]))
            IRF[s, :, :] = mu_irf
            IRF_GARCH[s, :, :] = irf_qt
            IRF_No_GARCH[s, :, :] = irf_qbar

        B[i]['IRF'] = IRF
        B[i]['IRF_No_GARCH'] = IRF_No_GARCH
        B[i]['IRF_GARCH'] = IRF_GARCH

    return B
