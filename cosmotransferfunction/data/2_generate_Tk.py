#!/usr/bin/env/python

import camb
from camb import model
import numpy as np
import scipy.interpolate
from astropy.io import fits

class interpolate1d(scipy.interpolate.interp1d):
    """Extend scipy interp1d to interpolate/extrapolate per axis in log space"""
    
    def __init__(self, x, y, *args, xspace='linear', yspace='linear', **kwargs):
        self.xspace = xspace
        self.yspace = yspace
        if self.xspace == 'log': x = np.log10(x)
        if self.yspace == 'log': y = np.log10(y)
        super().__init__(x, y, *args, **kwargs)
        
    def __call__(self, x, *args, **kwargs):
        if self.xspace == 'log': x = np.log10(x)
        if self.yspace == 'log':
            return 10**super().__call__(x, *args, **kwargs)
        else:
            return super().__call__(x, *args, **kwargs)

krange1 = np.logspace(np.log10(1e-5), np.log10(1e-4), num=20, endpoint=True) 
krange2 = np.logspace(np.log10(1e-4), np.log10(1e-3), num=40, endpoint=False)
krange3 = np.logspace(np.log10(1e-3), np.log10(1e-2), num=60, endpoint=False)
krange4 = np.logspace(np.log10(1e-2), np.log10(1e-1), num=80, endpoint=False)
krange5 = np.logspace(np.log10(1e-1), np.log10(1), num=100, endpoint=False)
krange6 = np.logspace(np.log10(1), np.log10(10), num=120, endpoint=True)

k = np.concatenate((krange1, krange2, krange3, krange4, krange5, krange6))
nk = len(k) 
np.savetxt('k_modes.txt', k)

parameter_file = np.load('parameter_file.npz')
params = parameter_file.files

if len(params) != 3:
    raise ValueError("Parameter must be of shape [Omega_b, Omega_c, h]")
    
nsamples = len(parameter_file[params[0]])
omega_b = parameter_file[params[0]]
omega_c = parameter_file[params[1]]
h = parameter_file[params[2]]


pars = camb.CAMBparams()
pars.WantTransfer = True
pars.set_matter_power(redshifts=[0], kmax=10)

def gen_Tk(omega_b, omega_c, h):
    H0 = h * 100.
    ombh2 = omega_b * h ** 2
    omch2 = omega_c * h ** 2
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2,
                      mnu=0.06, tau=0.06)
    results = camb.get_results(pars)
    transfer = results.get_matter_transfer_data()
    
    interp = interpolate1d(transfer.q, transfer.transfer_data[model.Transfer_tot-1, :, 0],
                      xspace='log', yspace='log')
    Tk = interp(k)
    return Tk
    
Tks = np.zeros((nsamples, nk))
for i in range(nsamples):
    print('generating', i, 'Tk')
    Tks[i] = gen_Tk(omega_b[i], omega_c[i], h[i])
    
fits.writeto('Tk.fits', Tks, overwrite=True)