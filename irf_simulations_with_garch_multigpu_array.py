import time

import torch
import numpy as np


def run_simulation_on_device(y, x, C, Hbar, draws, device, nlag, S, neq):
    y = y.to(device)
    x = x.to(device)
    C = C.to(device)
    Hbar = Hbar.to(device)
    draws = draws.to(device)

    iQbar = torch.linalg.inv(torch.linalg.cholesky(Hbar))
    Qt = torch.zeros((S, neq, neq), device=device, dtype=torch.float64)
    for s in range(S):
        Ht = C[s, :, :]
        H0t = torch.matmul(Ht, torch.linalg.inv(Hbar))
        l, _ = torch.linalg.eig(H0t)
        ml = torch.min(l.real)
        if ml > 0:
            Qt[s, :, :] = torch.linalg.cholesky(H0t)
        else:
            Qt[s, :, :] = torch.eye(neq, device=device)

    B = [{'IRF': torch.zeros((S, neq, neq), device=device, dtype=torch.float64),
          'IRF_No_GARCH': torch.zeros((S, neq, neq), device=device, dtype=torch.float64),
          'IRF_GARCH': torch.zeros((S, neq, neq), device=device, dtype=torch.float64)} for _ in range(draws.shape[1])]

    for i in range(draws.shape[1]):
        yi = y[draws[:, i], :]
        xi = x[draws[:, i], :]
        bi = torch.linalg.lstsq(yi, xi).solution
        AR_Terms = bi[:neq, :neq * nlag]
        M = torch.vstack([AR_Terms, torch.eye(neq * nlag, device=device)])
        M = torch.hstack([M, torch.zeros((neq * nlag + neq, neq), device=device)])

        for s in range(S):
            mu_irf = torch.matrix_power(M, s + 1)[:neq, :neq]
            irf_qbar = torch.matmul(mu_irf, iQbar)
            irf_qt = torch.matmul(irf_qbar, torch.linalg.inv(Qt[s, :, :]))
            B[i]['IRF'][s, :, :] = mu_irf
            B[i]['IRF_GARCH'][s, :, :] = irf_qt
            B[i]['IRF_No_GARCH'][s, :, :] = irf_qbar

    return B


def IRF_simulations_with_GARCH_multi_gpu(y: np.ndarray, x: np.ndarray, nlag: int, nboot: int, S: int, C: np.ndarray,
                                       Hbar: np.ndarray):
    if torch.cuda.is_available():
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")
        print('Using device: cuda: 0, 1')
    else:
        device1 = torch.device("cpu")
        device2 = torch.device("cpu")
        print('Using device: cuda')

    y = torch.tensor(y, dtype=torch.float64)
    x = torch.tensor(x, dtype=torch.float64)
    C = torch.tensor(C, dtype=torch.float64)
    Hbar = torch.tensor(Hbar, dtype=torch.float64)
    T, neq = y.shape

    torch.manual_seed(0)
    draws = torch.randint(0, T, (T, nboot))

    # Split draws between two GPUs
    nboot_per_device = nboot // 2
    draws1 = draws[:, :nboot_per_device]
    draws2 = draws[:, nboot_per_device:]

    # Run simulations on each GPU
    B1 = run_simulation_on_device(y, x, C, Hbar, draws1, device1, nlag, S, neq)
    B2 = run_simulation_on_device(y, x, C, Hbar, draws2, device2, nlag, S, neq)

    # Combine results from both GPUs

    # Optionally, convert tensors in results back to CPU for further processing
    B = B1 + B2
    for i in range(len(B)):
        for key in B[i]:
            B[i][key] = B[i][key].cpu().numpy()

    return B
