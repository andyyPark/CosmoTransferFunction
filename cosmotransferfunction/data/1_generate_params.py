#!/usr/bin/env/python

import numpy as np
import pyDOE

# Define parameter space
params_list = ['Omega_b', 'Omega_c', 'h']
params_range = [(0.01875, 0.02625), (0.05, 0.255), (0.64, 0.82)]

# Define number of parameters and samples
nparams = len(params_list)
nsamples = 100

# Define parameter range
params = np.vstack([
    np.linspace(prange[0], prange[1], nsamples)
    for prange in params_range
])

lhd = pyDOE.lhs(nparams, samples=nsamples, criterion=None)
idx = (lhd * nsamples).astype(int)

params_samples = np.zeros((nsamples, nparams))
for i in range(nparams):
    params_samples[:, i] = params[i][idx[:, i]]
    
# Save
params = {
    p: params_samples[:, i]
    for i, p in enumerate(params_list)
}

np.savez('parameter_file.npz', **params)
    



