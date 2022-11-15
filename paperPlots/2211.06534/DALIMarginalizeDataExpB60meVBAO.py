#import basic tools
import numpy as np
import pickle

#import cobaya tools
from cobaya.run import run
from cobaya.log import LoggedError

import plotTools as pt

#import MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("rank: "+str(rank))

#cosmoParamsPretty
cosmoParamsPretty = {'omega_c_h2' : '\Omega_ch^2', \
                             'omega_b_h2' :    '\Omega_bh^2', \
                             'N_eff':   'N_\mathrm{eff}', \
                             'A_s' :  'A_s', \
                             'n_s' : 'n_s', \
                             'tau' :  r'\tau', \
                             'theta_s': r'\theta_s', \
                             'mnu' : r'\sum m_{\nu}', \
                             'Yhe' : r'Y_p'}
cosmoFid = {'omega_c_h2':0.1197, \
                'omega_b_h2': 0.0222, \
                'N_eff': 3.046, \
                'A_s' : 2.196e-9, \
                'n_s' : 0.9655,\
                'tau' : 0.06, \
                #'H0' : 67.5, \
                'theta_s' : 0.010409, \
                #'Yhe' : 0.25, \
                #'r'   : 0.01, \
                'mnu' : 0.06}

#import data to get cosmoParams, nParams, nExps
jobName = './results/DALI_ExpB_60meV'
myFishers, cosmoParams = pt.loadWithDALI(jobName = jobName, pythonFlag = 3, returnCosmoParams = True, gTypes = ['Gaussian'])
fsky = 0.6
myFishers = pt.addfskyWithDALI(myFishers, fsky, gTypes = ['Gaussian'])
baoFile = './results/fisher_BAO_8p_60_Fisher.txt'
baoName = './results/fisher_BAO_8p_60_DALI'
BAOData = pt.loadBAO(baoName)
myFishers = pt.addBAOWithDALI(myFishers, cosmoParams, baoFile, BAOData, gTypes = ['Gaussian'])
myFishers = pt.addTauWithDALI(myFishers, cosmoParams, gTypes = ['Gaussian'])
print(str(rank)+" loaded data from: "+jobName)
nParams = len(cosmoParams)
nExps = len(myFishers['Gaussian']['unlensed']['fisher'])
nExp = 0

covs = pt.invertFishers(myFishers, gTypes = ['Gaussian'])
sigmas = pt.getSigmas(covs, gTypes = ['Gaussian'])

#dictionary of cobaya info
info = dict()

#params
info["params"] = dict()
for k in range(nParams):
    priorWidth = 5*sigmas['Gaussian']['lensed'][nExp][k]
    info["params"][cosmoParams[k]] = {"prior": {"min": -priorWidth+cosmoFid[cosmoParams[k]], "max": priorWidth+cosmoFid[cosmoParams[k]]}, "latex": cosmoParamsPretty[cosmoParams[k]]}

info["params"]['mnu']["prior"]["min"] = 0

info["sampler"] = {
    'mcmc': {
      'Rminus1_stop': 0.01,
      'max_tries': 10000
      }
  }

#likelihood
info["likelihood"] = {
    'DALIGaussian': {
        'python_path': 'DALIGaussian',
        'use_dali': True,
        'jobName': jobName,
        'experiment': nExp,
        'sType': 'lensed',
        'gType': 'Gaussian',
        'tauPrior': True,
        'fsky': fsky,
        'bao': True,
        'baoFile': baoFile,
        'baoName': baoName,
        'cosmoFid': cosmoFid
    }
}

info["resume"] = True
info["output"] = "chains/DALI_lensed_ExpB60meVBAO"

#run cobaya
success = False
try:
    upd_info, mcmc = run(info)
    success = True
except LoggedError as err:
    pass

# Did it work? (e.g. did not get stuck)
success = all(comm.allgather(success))
if not success and rank == 0:
    print("Sampling failed!")

print(str(rank)+" ran cobaya: "+str(success))
