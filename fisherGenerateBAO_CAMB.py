import sys
import cambWrapTools
import classWrapTools
import fisherTools
import pickle
import scipy
import numpy
import os

outputDir = './results/'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
fileBase = 'fisher_BAO_8p'

#######################################################################################3
#LOAD PARAMS AND GET POWER SPECTRA

#Fiducial values and step sizes taken from arXiv:1509.07471 Allison et al
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
#cosmoFid['n_t'] = - cosmoFid['r'] / 8.0 * (2.0 - cosmoFid['n_s'] - cosmoFid['r'] / 8.0)

stepSizes = {'omega_c_h2':0.0030, \
                'omega_b_h2': 0.0008, \
                'N_eff': .080, \
                'A_s' : 0.1e-9, \
                'n_s' : 0.010,\
                'tau' : 0.020, \
                'H0' : 1.2, \
                'theta_s' : 0.000050, \
                'mnu' : 0.02, \
                #'r'   : 0.001, \
                #'n_t' : cosmoFid['n_t'], \
                'Yhe' : 0.0048}

#Redshifts where BAO observables are evaluated.  Those listed here correspond to redshifts where DESI errors are given in arXiv:1509.07471 Allison et al
redshifts = numpy.arange(0.15, 1.95, 0.1)
#rs_dV_errors = numpy.asarray([0.0041, 0.0017, 0.00088, 0.00055, 0.00038, 0.00028, 0.00021, 0.00018, 0.00018, 0.00017, 0.00016, 0.00014, 0.00015, 0.00016, 0.00019, 0.00028, 0.00041, 0.00052])

rs_dV_relative_errors = numpy.asarray([1.89, 1.26, 0.98, 0.80, 0.68, 0.60, 0.52, 0.51, 0.56, \
                                   0.59, 0.60, 0.57, 0.66, 0.75, 0.95, 1.48, 2.28, 3.03])/100

#Current BAO redshifts and errors
#redshifts = [0.106, 0.15, 0.32, 0.57]
#rs_dV_errors = numpy.array([0.0084, 0.015, 0.0023, 0.00071])

cosmoParams = list(cosmoFid.keys())

BAOFid = cambWrapTools.getBAOParams(cosmoFid, redshifts)

rs_dV_errors = BAOFid[:,0]*rs_dV_relative_errors

paramDerivs = fisherTools.getParamDerivsBAOandH0(cosmoFid, stepSizes, redshifts)

nPars = len(cosmoParams)
fisherBAO = numpy.zeros((nPars,nPars))

for cp1, cosmo1 in enumerate(cosmoParams):
    for cp2, cosmo2 in enumerate(cosmoParams):
        fisherBAO[cp1,cp2] = sum(paramDerivs[cosmo1]['BAO'] * paramDerivs[cosmo2]['BAO']/(rs_dV_errors * rs_dV_errors))

header = ' '
for cp, cosmo in enumerate(cosmoParams):
    header += cosmo + '   '

numpy.savetxt(outputDir + fileBase + '_Fisher.txt', fisherBAO, header = header)
numpy.savetxt(outputDir + fileBase + 'rsdV.txt', BAOFid)
