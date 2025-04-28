import sys
import cambWrapTools
import classWrapTools
import fisherTools
import pickle
import scipy
import numpy
import os

classExecDir = '/users/jrmeyers/code/no_nus/FisherLens/CLASS_delens/'
classDataDir = '/users/jrmeyers/code/no_nus/FisherLens/CLASS_delens/'
outputDir = classDataDir + 'results/'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
fileBase = 'fisher_BAO_dAHz_8p'

# ######################################################################################3
# LOAD PARAMS AND GET POWER SPECTRA

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
# redshifts = numpy.arange(0.15, 1.95, 0.1)
# #rs_dV_errors = numpy.asarray([0.0041, 0.0017, 0.00088, 0.00055, 0.00038, 0.00028, 0.00021, 0.00018, 0.00018, 0.00017, 0.00016, 0.00014, 0.00015, 0.00016, 0.00019, 0.00028, 0.00041, 0.00052])

# rs_dV_relative_errors = numpy.asarray([1.89, 1.26, 0.98, 0.80, 0.68, 0.60, 0.52, 0.51, 0.56, \
#                                    0.59, 0.60, 0.57, 0.66, 0.75, 0.95, 1.48, 2.28, 3.03])/100

# Current BAO redshifts and errors
# redshifts = [0.106, 0.15, 0.32, 0.57]
# rs_dV_errors = numpy.array([0.0084, 0.015, 0.0023, 0.00071])

#arXiv:1308.4164
DESI_redshifts = numpy.arange(0.15, 1.95, 0.1)
DESI_dA_over_rs_relative_errors = numpy.asarray([2.78, 1.87, 1.45, 1.19, 1.01, 0.87, 0.77, 0.76, 0.88, 0.91, 0.91, 0.91, 1.00, 1.17, 1.50, 2.36, 3.62, 4.79])/100.
DESI_Hz_rs_relative_errors = numpy.asarray([5.34, 3.51, 2.69, 2.20, 1.85, 1.60, 1.41, 1.35, 1.42, 1.41, 1.38, 1.36, 1.46, 1.66, 2.04, 3.15, 4.87, 6.55])/100.

HETDEX_redshifts = numpy.arange(1.95, 3.55, 0.1)
HETDEX_dA_over_rs_relative_errors = numpy.asarray([5.51, 5.50, 5.49, 5.47, 5.46, 5.45, 5.44, 5.44, 5.43, 5.42, 5.41, 5.40, 5.40, 5.39, 5.39, 5.38])/100.
HETDEX_Hz_rs_relative_errors = numpy.asarray([7.93, 7.90, 7.88, 7.86, 7.84, 7.82, 7.81, 7.81, 7.80, 7.80, 7.79, 7.79, 7.79, 7.79, 7.80, 7.80])/100.



cosmoParams = list(cosmoFid.keys())

DESI_Fid = classWrapTools.getBAOParams(cosmoFid, DESI_redshifts, return_dA_Hz = True)
HETDEX_Fid = classWrapTools.getBAOParams(cosmoFid, HETDEX_redshifts, return_dA_Hz = True)

#rs_dV_errors = BAOFid[:,0]*rs_dV_relative_errors

DESI_paramDerivs = fisherTools.getBAODerivWithParams(cosmoFid, stepSizes, DESI_redshifts,
                            fileNameBase = fileBase, 
                            paramsToDifferentiate = None,
                            return_dA_Hz = True,
                            classExecDir = classExecDir,
                            classDataDir = classDataDir)

HETDEX_paramDerivs = fisherTools.getBAODerivWithParams(cosmoFid, stepSizes, HETDEX_redshifts,
                            fileNameBase = fileBase, 
                            paramsToDifferentiate = None,
                            return_dA_Hz = True,
                            classExecDir = classExecDir,
                            classDataDir = classDataDir)

nPars = len(cosmoParams)

years = [1,3,5,10,20,100]
for year in years:
    DESI_nYears = year
    DESI_dA_over_rs_relative_errors_scaled = numpy.sqrt(5./DESI_nYears) * DESI_dA_over_rs_relative_errors
    DESI_Hz_rs_relative_errors_scaled = numpy.sqrt(5./DESI_nYears) * DESI_Hz_rs_relative_errors
    DESI_dA_over_rs_errors = DESI_dA_over_rs_relative_errors_scaled*DESI_Fid['dA_over_rs']
    DESI_Hz_rs_errors = DESI_Hz_rs_relative_errors_scaled*DESI_Fid['Hz_rs']

    fisherBAO = numpy.zeros((nPars,nPars))
    
    for cp1, cosmo1 in enumerate(cosmoParams):
        for cp2, cosmo2 in enumerate(cosmoParams):
            fisherBAO[cp1,cp2] = sum( DESI_paramDerivs[cosmo1]['Hz_rs'] * DESI_paramDerivs[cosmo2]['Hz_rs'] \
                                     /(DESI_Hz_rs_errors*DESI_Hz_rs_errors) )
            fisherBAO[cp1,cp2] += sum( DESI_paramDerivs[cosmo1]['dA_over_rs'] * DESI_paramDerivs[cosmo2]['dA_over_rs'] \
                                     /(DESI_dA_over_rs_errors*DESI_dA_over_rs_errors) )

    # pickle including DALI terms
    BAOData = dict()
    BAOData['fisher'] = fisherBAO
    BAOData['redshifts'] = DESI_redshifts
    BAOData['HZ_rs_relative_errors'] = DESI_Hz_rs_relative_errors_scaled
    BAOData['dA_over_rs_relative_errors'] = DESI_dA_over_rs_relative_errors_scaled
    BAOData['Hz_rs_errors'] = DESI_Hz_rs_errors
    BAOData['dA_over_rs_errors'] = DESI_dA_over_rs_errors
    BAOData['cosmoParams'] = cosmoParams
    BAOData['BAOFid'] = DESI_Fid
    BAOData['paramDerivs'] = DESI_paramDerivs

    BAOOutput = open(outputDir + fileBase + '_DESI_' + str(year) + 'yr.pkl', 'wb')
    pickle.dump(BAOData, BAOOutput, -1)
    BAOOutput.close()

    header = ' '
    for cp, cosmo in enumerate(cosmoParams):
        header += cosmo + '   '
    
    numpy.savetxt(outputDir + fileBase + '_DESI_' + str(year) + 'yr.txt', fisherBAO, header = header)
    
    HETDEX_nYears = year
    HETDEX_dA_over_rs_relative_errors_scaled = numpy.sqrt(5./HETDEX_nYears) * HETDEX_dA_over_rs_relative_errors
    HETDEX_Hz_rs_relative_errors_scaled = numpy.sqrt(5./HETDEX_nYears) * HETDEX_Hz_rs_relative_errors
    HETDEX_dA_over_rs_errors = HETDEX_dA_over_rs_relative_errors_scaled*HETDEX_Fid['dA_over_rs']
    HETDEX_Hz_rs_errors = HETDEX_Hz_rs_relative_errors_scaled*HETDEX_Fid['Hz_rs']

    fisherBAO = numpy.zeros((nPars,nPars))
    
    for cp1, cosmo1 in enumerate(cosmoParams):
        for cp2, cosmo2 in enumerate(cosmoParams):
            fisherBAO[cp1,cp2] = sum( HETDEX_paramDerivs[cosmo1]['Hz_rs'] * HETDEX_paramDerivs[cosmo2]['Hz_rs'] \
                                     /(HETDEX_Hz_rs_errors*HETDEX_Hz_rs_errors) )
            fisherBAO[cp1,cp2] += sum( HETDEX_paramDerivs[cosmo1]['dA_over_rs'] * HETDEX_paramDerivs[cosmo2]['dA_over_rs'] \
                                     /(HETDEX_dA_over_rs_errors*HETDEX_dA_over_rs_errors) )
    
    # pickle including DALI terms
    BAOData = dict()
    BAOData['fisher'] = fisherBAO
    BAOData['redshifts'] = HETDEX_redshifts
    BAOData['HZ_rs_relative_errors'] = HETDEX_Hz_rs_relative_errors_scaled
    BAOData['dA_over_rs_relative_errors'] = HETDEX_dA_over_rs_relative_errors_scaled
    BAOData['Hz_rs_errors'] = HETDEX_Hz_rs_errors
    BAOData['dA_over_rs_errors'] = HETDEX_dA_over_rs_errors
    BAOData['cosmoParams'] = cosmoParams
    BAOData['BAOFid'] = HETDEX_Fid
    BAOData['paramDerivs'] = HETDEX_paramDerivs

    BAOOutput = open(outputDir + fileBase + '_HETDEX_' + str(year) + 'yr.pkl', 'wb')
    pickle.dump(BAOData, BAOOutput, -1)
    BAOOutput.close()

    numpy.savetxt(outputDir + fileBase + '_HETDEX_' + str(year) + 'yr.txt', fisherBAO, header = header)


