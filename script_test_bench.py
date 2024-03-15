import time

import scipy.io as sio
import numpy as np

from irf_simulations_with_garch_gpu_arrray import IRF_simulations_with_GARCH_pytorch
from irf_simulations_with_garch import IRF_simulations_with_GARCH
from irf_simulations_with_garch_multigpu_array import IRF_simulations_with_GARCH_mutili_pytorch

# raw_data = h5py.File('./input/testBenchDataSimulations_sample.mat', 'r')

raw_data = sio.loadmat('./input/testBenchDataSimulations_sample.mat')

nboot = 399
# nboot = 10000
# nboot = 1000000
print('Using nboot, ', nboot)

y = raw_data['y']
x = raw_data['x']
nlag = raw_data['nlag'].astype(int)[0][0]
S = raw_data['S'].astype(int)[0][0]
C = raw_data['ConCovMat']

Hbar = raw_data['Hbar']
print('Loaded inputs')
#
s = time.time()
res = IRF_simulations_with_GARCH(y, x, nlag, nboot, S, C, Hbar)
e = time.time()
print(f'original script took {e - s} seconds')

s = time.time()
res_gpu_array = IRF_simulations_with_GARCH_pytorch(y, x, nlag, nboot, S, C, Hbar)
e = time.time()
print(f'gpu array script took {e - s} seconds')


s = time.time()
res_multi_gpu = IRF_simulations_with_GARCH_mutili_pytorch(y, x, nlag, nboot, S, C, Hbar)
e = time.time()
print(f'multi gpu script took {e - s} seconds')