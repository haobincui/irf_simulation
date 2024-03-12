import numpy as np
import torch


def IRF_simulations_with_GARCH_pytorch(y: np.ndarray, x: np.ndarray, nlag: int, nboot: int,
                               S: int, C: np.ndarray, Hbar: np.ndarray) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    y = torch.tensor(y, dtype=torch.float64).to(device)
    x = torch.tensor(x, dtype=torch.float64).to(device)
    C = torch.tensor(C, dtype=torch.float64).to(device)
    Hbar = torch.tensor(Hbar, dtype=torch.float64).to(device)

    T, neq = y.shape

    torch.manual_seed(0)
    draws = torch.randint(0, T, (T, nboot), device=device)

    # rng = np.random.default_rng(np.random.MT19937(seed=0))
    # draws = rng.integers(0, T, (T, nboot))  # T * nboot
    # draws = torch.tensor(draws, dtype=torch.int64).to(device)

    iQbar = torch.linalg.inv(torch.linalg.cholesky(Hbar))
    Qt = torch.zeros((S, neq, neq), device=device).to(dtype=torch.float64)
    for s in range(S):
        Ht = C[s, :, :]
        H0t = torch.matmul(Ht, torch.linalg.inv(Hbar))
        l, _ = torch.linalg.eig(H0t)
        ml = torch.min(l.real)
        if ml > 0:
            Qt[s, :, :] = torch.linalg.cholesky(H0t)
        else:
            Qt[s, :, :] = torch.eye(neq, device=device)

    B = [{'IRF': torch.zeros((S, neq, neq), device=device).to(dtype=torch.float64),
          'IRF_No_GARCH': torch.zeros((S, neq, neq), device=device).to(dtype=torch.float64),
          'IRF_GARCH': torch.zeros((S, neq, neq), device=device).to(dtype=torch.float64)} for _ in range(nboot)]

    for i in range(nboot):
        yi = y[draws[:, i], :]
        xi = x[draws[:, i], :]
        bi = torch.linalg.lstsq(yi, xi).solution
        AR_Terms = bi[:neq, :neq * nlag]
        M = torch.vstack([AR_Terms, torch.eye(neq * nlag, device=device)])
        M = torch.hstack([M, torch.zeros((neq * nlag + neq, neq), device=device)])

        for s in range(S):
            mu_irf = torch.matrix_power(M, s + 1)[:neq, :neq]
            irf_qbar = torch.matmul(mu_irf, iQbar)
            irf_qt = torch.matmul(irf_qbar, torch.linalg.inv(Qt[s, :, :]).to(dtype=torch.float64))
            B[i]['IRF'][s, :, :] = mu_irf
            B[i]['IRF_GARCH'][s, :, :] = irf_qt
            B[i]['IRF_No_GARCH'][s, :, :] = irf_qbar

    # Convert results back to CPU
    for i in range(nboot):
        for key in B[i]:
            B[i][key] = B[i][key].cpu().numpy()

    return B
