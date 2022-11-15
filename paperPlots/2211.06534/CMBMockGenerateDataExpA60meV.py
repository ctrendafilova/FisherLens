from cobaya.model import get_model

import math
import numpy as np

import pickle

###
from cobaya.log import LoggedError
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt
#import MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("rank: "+str(rank))
###

expNumber = 0

Bmodes = False
LensingExtraction = True
unlensed_clTTTEEE = False

f_sky = 0.6
l_mindd = 2
l_min = 30
l_max = 5000
l_maxTT = 3000

import plotTools as pt
jobName = './results/DALI_ExpA_60meV'
fishers, cosmoParams = pt.loadGaussian(jobName = jobName, pythonFlag = 3, returnCosmoParams = True)
fishers = pt.addfsky(fishers,f_sky,gTypes = ['Gaussian'])
covs = pt.invertFishers(fishers,gTypes = ['Gaussian'])
sigmas = pt.getSigmas(covs,gTypes = ['Gaussian'])
mySigmas = sigmas['Gaussian']['lensed'][expNumber]

fiducial_params = {
    'ombh2': 0.0222, 'omch2': 0.1197, 'thetastar': 0.010409, 'tau': 0.06,
    'As': 2.196e-9, 'ns': 0.9655,
    'mnu': 0.06, 'nnu': 3.046}

packages_path = './cobaya'

info_fiducial = {
    'params': fiducial_params,
    'likelihood': {'one': None},
    'theory': {'camb': None},
    'packages_path': packages_path}

model_fiducial = get_model(info_fiducial)

# Declare our desired theory product
# (there is no cosmological likelihood doing it for us)
model_fiducial.add_requirements({"Cl": {'tt': l_max}})
model_fiducial.add_requirements({"Cl": {'te': l_max}})
model_fiducial.add_requirements({"Cl": {'ee': l_max}})
model_fiducial.add_requirements({"Cl": {'bb': l_max}})
model_fiducial.add_requirements({"Cl": {'pp': l_max}})

# Compute and extract the CMB power spectrum
# (In muK^-2, without l(l+1)/(2pi) factor)
# notice the empty dictionary below: all parameters are fixed
model_fiducial.logposterior({})
Cls = model_fiducial.provider.get_Cl(ell_factor=False, units="muK2")

# read noise from saved file
f = open(jobName + '.pkl', 'rb')
data_noise = pickle.load(f)
f.close()
noise_T = data_noise['cmbNoiseSpectra'][expNumber]['cl_TT'][:l_max+1]
noise_P = data_noise['cmbNoiseSpectra'][expNumber]['cl_EE'][:l_max+1]

# default:
numCls = 3
# deal with BB:
if Bmodes:
    index_B = numCls
    numCls += 1
# deal with pp (p = CMB lensing potential):
if LensingExtraction:
    index_pp = numCls
    numCls += 1
    # read the NlDD noise (noise for the extracted deflection field spectrum)
    Nldd = data_noise['deflectionNoises'][expNumber][:l_max+1]

# initialize the fiducial values
Cl_est = np.zeros((numCls, l_max+1), 'float64')
Cl_est[0] = Cls['tt'][:l_max + 1]
Cl_est[2] = Cls['te'][:l_max + 1]
Cl_est[1] = Cls['ee'][:l_max + 1]
# BB:
if Bmodes:
    Cl_est[index_B] = Cls['bb'][:l_max + 1]
# DD (D = deflection field):
if LensingExtraction:
    Cl_est[index_pp] = Cls['pp'][:l_max + 1]
    
def loglkl(_self=None):

    # get Cl's from the cosmological code
    Cl_theo = dict()
    # td -> 0
    cltd_fid = 0
    cltd = 0

    # if we want unlensed Cl's
    if unlensed_clTTTEEE:
        Cl_theo['tt'] = _self.provider.get_unlensed_Cl(ell_factor=False, units="muK2")['tt'][:l_max+1]
        Cl_theo['te'] = _self.provider.get_unlensed_Cl(ell_factor=False, units="muK2")['te'][:l_max+1]
        Cl_theo['ee'] = _self.provider.get_unlensed_Cl(ell_factor=False, units="muK2")['ee'][:l_max+1]
        if LensingExtraction:
            Cl_theo['pp'] = _self.provider.get_Cl(ell_factor=False, units="muK2")['pp'][:l_max+1] # lensing potential power spectrum is always unitless
        if Bmodes:
            Cl_theo['bb'] = _self.provider.get_unlensed_Cl(ell_factor=False, units="muK2")['bb'][:l_max+1]

    # if we want lensed Cl's
    else:
        Cl_theo['tt'] = _self.provider.get_Cl(ell_factor=False, units="muK2")['tt'][:l_max+1]
        Cl_theo['te'] = _self.provider.get_Cl(ell_factor=False, units="muK2")['te'][:l_max+1]
        Cl_theo['ee'] = _self.provider.get_Cl(ell_factor=False, units="muK2")['ee'][:l_max+1]
        if LensingExtraction:
            Cl_theo['pp'] = _self.provider.get_Cl(ell_factor=False, units="muK2")['pp'][:l_max+1] # lensing potential power spectrum is always unitless
        if Bmodes:
            Cl_theo['bb'] = _self.provider.get_Cl(ell_factor=False, units="muK2")['bb'][:l_max+1]

    # compute likelihood

    chi2 = 0

    # cound number of modes.
    # number of modes is different form number of spectra
    # modes = T,E,[B],[D=deflection]
    # spectra = TT,EE,TE,[BB],[DD,TD]
    # default:
    num_modes=2
    # add B mode:
    if Bmodes:
        num_modes += 1
    # add D mode:
    if LensingExtraction:
        num_modes += 1

    Cov_est = np.zeros((num_modes, num_modes), 'float64')
    Cov_the = np.zeros((num_modes, num_modes), 'float64')
    Cov_mix = np.zeros((num_modes, num_modes), 'float64')

    for l in range(l_mindd, l_max+1):
        # piecewise by \ells
        if l < l_min:
            # dd only
            cldd_fid = l*(l+1.)*Cl_est[index_pp, l]
            cldd = l*(l+1.)*Cl_theo['pp'][l]

            Cov_est = np.array([[cldd_fid+Nldd[l]]])
            Cov_the = np.array([[cldd+Nldd[l]]])
            
            num_modes_temp = 1
        elif l < l_maxTT+1:
            # TT, EE, dd
            cldd_fid = l*(l+1.)*Cl_est[index_pp, l]
            cldd = l*(l+1.)*Cl_theo['pp'][l]

            Cov_est = np.array([
                [Cl_est[0, l]+noise_T[l], Cl_est[2, l], cltd_fid],
                [Cl_est[2, l], Cl_est[1, l]+noise_P[l], 0],
                [cltd_fid, 0, cldd_fid+Nldd[l]]])
            Cov_the = np.array([
                [Cl_theo['tt'][l]+noise_T[l], Cl_theo['te'][l], cltd],
                [Cl_theo['te'][l], Cl_theo['ee'][l]+noise_P[l], 0],
                [cltd, 0, cldd+Nldd[l]]])
            
            num_modes_temp = 3
        else:
            # EE, dd
            cldd_fid = l*(l+1.)*Cl_est[index_pp, l]
            cldd = l*(l+1.)*Cl_theo['pp'][l]

            Cov_est = np.array([
                [Cl_est[0, l]+noise_T[l], Cl_est[2, l], cltd_fid],
                [Cl_est[2, l], Cl_est[1, l]+noise_P[l], 0],
                [cltd_fid, 0, cldd_fid+Nldd[l]]])
            Cov_the = np.array([
                [Cl_est[0, l]+noise_T[l], Cl_theo['te'][l], cltd],
                [Cl_theo['te'][l], Cl_theo['ee'][l]+noise_P[l], 0],
                [cltd, 0, cldd+Nldd[l]]])
            
            num_modes_temp = 3
        # add to chi2
        chi2 += (2.*l+1.)*f_sky * \
            (np.trace(np.matmul(Cov_est, np.linalg.inv(Cov_the))) - np.log(np.linalg.det(np.matmul(Cov_est, np.linalg.inv(Cov_the)))) - num_modes_temp)
        
    logp = -chi2/2

    return logp

from cobaya.run import run

propFactor = 10
info = {
    'params': {
        'As': {'prior': {'min': -5*mySigmas[cosmoParams.index('A_s')]+2.196e-9, 'max': 5*mySigmas[cosmoParams.index('A_s')]+2.196e-9},\
            'ref': {'dist':'norm','loc': 2.196e-9, 'scale': mySigmas[cosmoParams.index('A_s')]}, 'latex': 'A_s', \
            'proposal':mySigmas[cosmoParams.index('A_s')]/propFactor},
        'ns': {'prior': {'min': -5*mySigmas[cosmoParams.index('n_s')]+0.9655, 'max': 5*mySigmas[cosmoParams.index('n_s')]+0.9655},\
            'ref': {'dist':'norm','loc': 0.9655, 'scale': mySigmas[cosmoParams.index('n_s')]}, 'latex': 'n_s', \
            'proposal':mySigmas[cosmoParams.index('n_s')]/propFactor},
        'ombh2': {'prior': {'min': -5*mySigmas[cosmoParams.index('omega_b_h2')]+0.0222, 'max': 5*mySigmas[cosmoParams.index('omega_b_h2')]+0.0222},\
            'ref': {'dist':'norm','loc':0.0222,'scale':mySigmas[cosmoParams.index('omega_b_h2')]}, \
            'proposal':mySigmas[cosmoParams.index('omega_b_h2')]/propFactor},
        'omch2': {'prior': {'min': -5*mySigmas[cosmoParams.index('omega_c_h2')]+0.1197, 'max': 5*mySigmas[cosmoParams.index('omega_c_h2')]+0.1197},\
            'ref': {'dist':'norm','loc':0.1197,'scale':mySigmas[cosmoParams.index('omega_c_h2')]}, \
            'proposal':mySigmas[cosmoParams.index('omega_c_h2')]/propFactor},
        'thetastar': {'prior': {'min': -5*mySigmas[cosmoParams.index('theta_s')]+0.010409, 'max': 5*mySigmas[cosmoParams.index('theta_s')]+0.010409},\
            'ref': {'dist':'norm','loc':0.010409,'scale':mySigmas[cosmoParams.index('theta_s')]}, \
            'proposal':mySigmas[cosmoParams.index('theta_s')]/propFactor},
        'tau': {'prior': {'dist':'norm','loc':0.06,'scale':0.007}, \
            'proposal':mySigmas[cosmoParams.index('tau')]/propFactor},
        'mnu': {'prior': {'min': 0, 'max': 5*mySigmas[cosmoParams.index('mnu')]+0.06},\
            'ref': {'dist':'norm','loc':0.06,'scale':mySigmas[cosmoParams.index('mnu')]}, \
            'proposal':mySigmas[cosmoParams.index('mnu')]/propFactor},
        'nnu': {'prior': {'min': -5*mySigmas[cosmoParams.index('N_eff')]+3.046, 'max': 5*mySigmas[cosmoParams.index('N_eff')]+3.046},\
            'ref': {'dist':'norm','loc':3.046,'scale':mySigmas[cosmoParams.index('N_eff')]}, \
            'proposal':mySigmas[cosmoParams.index('N_eff')]/propFactor}
    },
    'likelihood': {'my_cl_like': {
        "external": loglkl,
        # Declare required quantities!
        "requires": {'Cl': {'tt': l_max, 'te': l_max, 'ee': l_max, 'pp':l_max}}}},
    'theory': {'camb': {'stop_at_error': True}},
    'packages_path': packages_path,
    'sampler': {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 1000}},
    'resume': True
    }
    
info["output"] = "chains/CMB_ExpA60meV"
# add output
model = get_model(info)

#run cobaya
success = False
try:
    updated_exactNorm, sampler_exactNorm = run(info)
    success = True
except LoggedError as err:
    pass

# Did it work? (e.g. did not get stuck)
success = all(comm.allgather(success))
if not success and rank == 0:
    print("Sampling failed!")

print(str(rank)+" ran cobaya: "+str(success))
