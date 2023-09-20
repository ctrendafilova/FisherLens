import sys
import cambWrapTools
import classWrapTools
import fisherTools
import pickle
from mpi4py import MPI
import scipy
import numpy
import os

#MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print (rank, size)

###  Set of experiments  ###
expNames = list(range(1))
nExps = len(expNames)
lmax = 5000
lmaxTT = 3000
lmin = 30

lbuffer = 1500
lmax_calc = lmax+lbuffer

classExecDir = './CLASS_delens/'
classDataDir = './CLASS_delens/'
outputDir = classDataDir + 'results/'

classDataDirThisNode = classDataDir + 'data/Node_' + str(rank) + '/'
fileBase = 'fisher_Planck_DRDM'
fileBaseThisNode = fileBase + '_' + str(rank)

if not os.path.exists(classDataDirThisNode):
    os.makedirs(classDataDirThisNode)
if not os.path.exists(outputDir):
    os.makedirs(outputDir)


spectrumTypes = ['unlensed', 'lensed', 'delensed', 'lensing']
polCombs = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd']


#######################################################################################3
#LOAD PARAMS AND GET POWER SPECTRA

#Fiducial values and step sizes taken from arXiv:1509.07471 Allison et al
cosmoFid = {'omega_c_h2':0.1197, \
                'omega_b_h2': 0.0222, \
                'N_eff': 3.046, \
                'A_s' : 2.196e-9, \
                'n_s' : 0.9655,\
                'tau' : 0.06, \
                'H0' : 67.5, \
                'mnu' : 0.06, \
                'N_idr': 0.4290, \
                'Gamma_0_nadm': 2.371e-8}
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
                'Yhe' : 0.0048, \
                'varying_alpha' : 0.002, \
                'varying_me' : 0.002, \
                'fEDE' : 0.008, \
                'log10z_c' : 0.04, \
                'thetai_scf' : 0.05, \
                'omk' : 0.01, \
                'N_idr' : 0.080, \
                'Gamma_0_nadm' : 0.37e-8, \
                'bbn_alpha_sensitivity' : 0.002, \
                'varying_transition_redshift' : 0.002}

cosmoParams = list(cosmoFid.keys())
ell = numpy.arange(2,lmax_calc+1+2000)

reconstructionMask = dict()
reconstructionMask['lmax_T'] = 3000

extra_params = dict()
extra_params['f_idm'] = 1
#extra_params['Gamma_0_nadm'] = 0
#extra_params['delensing_verbose'] = 3
#extra_params['output_spectra_noise'] = 'no'
#extra_params['write warnings'] = 'y'

ellsToUse = {'cl_TT': [lmin, lmaxTT], 'cl_TE': [lmin, lmax], 'cl_EE': [lmin, lmax], 'cl_dd': [2, lmax]}
ellsToUseNG = {'cl_TT': [lmin, lmaxTT], 'cl_TE': [lmin, lmax], 'cl_EE': [lmin, lmax], 'cl_dd': [2, lmax], 'lmaxCov': lmax_calc}

cmbNoiseSpectra = dict()
deflectionNoises = dict()
paramDerivs = dict()
powersFid = dict()
invCovDotParamDerivs_delensed = dict()
invCovDotParamDerivs_lensed = dict()
paramDerivStack_delensed = dict()
paramDerivStack_lensed = dict()
fisherGaussian = dict()
fisherNonGaussian_delensed = dict()
fisherNonGaussian_lensed = dict()

doNonGaussian = True
includeUnlensedSpectraDerivatives = True

### Assign task of computing lensed NG covariance to last node       ###
### This is chosen because last node sometimes has fewer experiments ###
if doNonGaussian is True:
    if rank == size-1:

        if includeUnlensedSpectraDerivatives:
            dCldCLd_lensed, dCldCLu_lensed = classWrapTools.class_generate_data(cosmoFid,
                                         cmbNoise = None, \
                                         deflectionNoise = None, \
                                         extraParams = extra_params, \
                                         rootName = fileBaseThisNode, \
                                         lmax = lmax_calc, \
                                         calculateDerivatives = 'lensed', \
                                         includeUnlensedSpectraDerivatives = includeUnlensedSpectraDerivatives,
                                         classExecDir = classExecDir,
                                         classDataDir = classDataDirThisNode)
        else:
            dCldCLd_lensed = classWrapTools.class_generate_data(cosmoFid,
                                         cmbNoise = None, \
                                         deflectionNoise = None, \
                                         extraParams = extra_params, \
                                         rootName = fileBaseThisNode, \
                                         lmax = lmax_calc, \
                                         calculateDerivatives = 'lensed', \
                                         classExecDir = classExecDir,
                                         classDataDir = classDataDirThisNode)
            dCldCLu_lensed = None

        print('Successfully computed derivatives')
        #stop
    else:
        dCldCLd_lensed = None

for k in expNames:
    expName = expNames[k]

    print('Node ' + str(rank) + ' working on experiment ' + str(expName))

    planckNoise = fisherTools.getPlanckInvVarNoise(ells = ell, includePol = True)

    cmbNoiseSpectra[k] = planckNoise
    powersFid[k], deflectionNoises[k] = classWrapTools.class_generate_data(cosmoFid,
                                             cmbNoise = cmbNoiseSpectra[k],
                                             extraParams = extra_params,
                                             rootName = fileBaseThisNode,
                                             lmax = lmax_calc,
                                             classExecDir = classExecDir,
                                             classDataDir = classDataDirThisNode,
                                             reconstructionMask = reconstructionMask)

    paramDerivs[k] = fisherTools.getPowerDerivWithParams(cosmoFid = cosmoFid, \
                            extraParams = extra_params, \
                            stepSizes = stepSizes, \
                            polCombs = polCombs, \
                            cmbNoiseSpectraK = cmbNoiseSpectra[k], \
                            deflectionNoisesK = deflectionNoises[k], \
                            useClass = True, \
                            lmax = lmax_calc, \
                            fileNameBase = fileBaseThisNode, \
                            classExecDir = classExecDir, \
                            classDataDir = classDataDirThisNode)

    fisherGaussian[k] = fisherTools.getGaussianCMBFisher(powersFid = powersFid[k], \
                            paramDerivs = paramDerivs[k], \
                            cmbNoiseSpectra = cmbNoiseSpectra[k], \
                            deflectionNoises = deflectionNoises[k], \
                            cosmoParams = cosmoParams, \
                            spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                            polCombsToUse = polCombs, \
                            ellsToUse = ellsToUse)

    if doNonGaussian:

        ### Overwrite dCldCLd_delensed for each experiment to save memory ###

        if includeUnlensedSpectraDerivatives:
            dCldCLd_delensed, dCldCLu_delensed = classWrapTools.class_generate_data(cosmoFid,
                                                 cmbNoise = cmbNoiseSpectra[k], \
                                                 deflectionNoise = deflectionNoises[k], \
                                                 extraParams = extra_params, \
                                                 rootName = fileBaseThisNode, \
                                                 lmax = lmax_calc, \
                                                 calculateDerivatives = 'delensed', \
                                                 includeUnlensedSpectraDerivatives = includeUnlensedSpectraDerivatives,
                                                 classExecDir = classExecDir,
                                                 classDataDir = classDataDirThisNode)
        else:
            dCldCLd_delensed = classWrapTools.class_generate_data(cosmoFid,
                                                 cmbNoise = cmbNoiseSpectra[k], \
                                                 deflectionNoise = deflectionNoises[k], \
                                                 extraParams = extra_params, \
                                                 rootName = fileBaseThisNode, \
                                                 lmax = lmax_calc, \
                                                 calculateDerivatives = 'delensed', \
                                                 classExecDir = classExecDir,
                                                 classDataDir = classDataDirThisNode)
            dCldCLu_delensed = None


        invCovDotParamDerivs_delensed[k], paramDerivStack_delensed[k] = fisherTools.choleskyInvCovDotParamDerivsNG(powersFid = powersFid[k], \
                                    cmbNoiseSpectra = cmbNoiseSpectra[k], \
                                    deflectionNoiseSpectra = deflectionNoises[k], \
                                    dCldCLd = dCldCLd_delensed,
                                    paramDerivs = paramDerivs[k], \
                                    cosmoParams = cosmoParams, \
                                    dCldCLu = dCldCLu_delensed, \
                                    ellsToUse = ellsToUseNG, \
                                    polCombsToUse = polCombs, \
                                    spectrumType = 'delensed')

        ############################
        ## Seems to hang on bcast ##
        ############################

        if rank != size-1 and dCldCLd_lensed is None:
            classDataDirLastNode = classDataDir + 'data/Node_' + str(size-1) + '/'
            fileBaseLastNode = fileBase + '_' + str(size-1)

            dCldCLd_lensed = classWrapTools.loadLensingDerivatives(rootName = fileBaseLastNode,
                                                                   classDataDir = classDataDirLastNode,
                                                                   dervtype = 'lensed')


            dCldCLu_lensed = None
            if includeUnlensedSpectraDerivatives:
                dCldCLu_lensed = classWrapTools.loadUnlensedSpectraDerivatives(rootName = fileBaseLastNode,
                                                                   classDataDir = classDataDirLastNode,
                                                                   dervtype = 'lensed')

        invCovDotParamDerivs_lensed[k], paramDerivStack_lensed[k] = fisherTools.choleskyInvCovDotParamDerivsNG(powersFid = powersFid[k], \
                                    cmbNoiseSpectra = cmbNoiseSpectra[k], \
                                    deflectionNoiseSpectra = deflectionNoises[k], \
                                    dCldCLd = dCldCLd_lensed,
                                    paramDerivs = paramDerivs[k], \
                                    cosmoParams = cosmoParams, \
                                    dCldCLu = dCldCLu_lensed,
                                    ellsToUse = ellsToUseNG, \
                                    polCombsToUse = polCombs, \
                                    spectrumType = 'lensed')

        fisherNonGaussian_delensed[k] = fisherTools.getNonGaussianCMBFisher(invCovDotParamDerivs = invCovDotParamDerivs_delensed[k], \
                                    paramDerivStack = paramDerivStack_delensed[k], \
                                    cosmoParams = cosmoParams)

        fisherNonGaussian_lensed[k] = fisherTools.getNonGaussianCMBFisher(invCovDotParamDerivs = invCovDotParamDerivs_lensed[k], \
                                    paramDerivStack = paramDerivStack_lensed[k], \
                                    cosmoParams = cosmoParams)

print('Node ' + str(rank) + ' finished all experiments')

forecastData = {'cmbNoiseSpectra' : cmbNoiseSpectra,
                'powersFid' : powersFid,
                'paramDerivs': paramDerivs,
                'fisherGaussian': fisherGaussian,
                'deflectionNoises' : deflectionNoises,
                'cosmoFid' : cosmoFid,
                'cosmoParams' : cosmoParams}
if doNonGaussian:
    forecastData['invCovDotParamDerivs_delensed'] = invCovDotParamDerivs_delensed
    forecastData['paramDerivStack_delensed'] = paramDerivStack_delensed
    forecastData['invCovDotParamDerivs_lensed'] = invCovDotParamDerivs_lensed
    forecastData['paramDerivStack_lensed'] = paramDerivStack_lensed
    forecastData['fisherNonGaussian_delensed'] = fisherNonGaussian_delensed
    forecastData['fisherNonGaussian_lensed'] = fisherNonGaussian_lensed

print('Node ' + str(rank) + ' saving data')

filename = outputDir + fileBase + '.pkl'
delensedOutput = open(filename, 'wb')
pickle.dump(forecastData, delensedOutput, -1)
delensedOutput.close()
print('Node ' + str(rank) + ' saving data complete')
