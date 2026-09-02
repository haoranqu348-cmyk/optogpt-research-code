import numpy as np
from numpy import pi
import pandas as pd
from tmm import inc_tmm
from scipy.interpolate import interp1d
import os



DATABASE = './nk'
mats = ['Ag', 'Al', 'Al2O3', 'AlN', 'Ge', 'HfO2', 'ITO', 'MgF2', 'MgO', 'Si', 'Si3N4', 'SiO2', 'Ta2O5', 'TiN', 'TiO2', 'ZnO', 'ZnS', 'ZnSe', 'Glass_Substrate']
# NOTE: The default thickness list below (5–250 nm, step 5) may differ from
# the pretrained checkpoint vocabulary (10–500 nm, step 10).
# Data generation for fine-tuning should use the checkpoint's thickness range.
thicks = [str(i) for i in range(5, 255, 5)]

lamda_low = 0.4
lamda_high = 1.1
wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)



def load_materials(all_mats = mats, wavelengths = wavelengths, DATABASE = './nk'):
    '''
    Load material nk and return corresponding interpolators.

    Return:
        nk_dict: dict, key -- material name, value: n, k in the 
        self.wavelength range
    '''
    nk_dict = {}

    for mat in all_mats:
        nk = pd.read_csv(os.path.join(DATABASE, mat + '.csv'))
        nk.dropna(inplace=True)

        wl = nk['wl'].to_numpy()
        index_n = nk['n'].to_numpy()
        index_k = nk['k'].to_numpy()

        n_fn = interp1d(
                wl, index_n,  bounds_error=False, fill_value='extrapolate', kind=3)
        k_fn = interp1d(
                wl, index_k,  bounds_error=False, fill_value='extrapolate', kind=1)
            
        nk_dict[mat] = n_fn(wavelengths) + 1j*k_fn(wavelengths)

    return nk_dict

def spectrum(materials, thickness, pol = 's', theta=0,  wavelengths = wavelengths, nk_dict = {}, substrate = 'Glass_Substrate', substrate_thick = 500000):
    '''
    Input:
        metal materials: list  
        thickness: list
        theta: degree, the incidence angle

    Return:
        All_results: dictionary contains R, T, A, RGB, LAB
    '''
    #aa = time.time()
    degree = pi/180
    theta = theta *degree
    wavess = (1e3 * wavelengths).astype('int')

        
    thickness = [np.inf] + thickness + [substrate_thick, np.inf]

    R, T, A = [], [], []
    inc_list = ['i'] + ['c']*len(materials) + ['i', 'i']
    for i, lambda_vac in enumerate(wavess):

        n_list = [1] + [nk_dict[mat][i] for mat in materials] + [nk_dict[substrate][i], 1]

        res = inc_tmm(pol, n_list, thickness, inc_list, theta, lambda_vac)

        R.append(res['R'])
        T.append(res['T'])

    # thickness = [np.inf] + thickness + [np.inf]

    # R, T, A = [], [], []
    # for i, lambda_vac in enumerate(wavess):

    #     n_list = [1] + [nk_dict[mat][i] for mat in materials] + [nk_dict[substrate][i]]

    #     res = coh_tmm(pol, n_list, thickness, theta, lambda_vac)

    #     R.append(res['R'])
    #     T.append(res['T'])

    return R + T
