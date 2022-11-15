import sys
import cambWrapTools
import classWrapTools
import fisherTools
import pickle
from mpi4py import MPI
import scipy
import numpy
import os

import copy

#MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(rank, size)

###  Set of experiments  ###
expNames = list(range(1))
nExps = len(expNames)
lmax = 5000
lmaxTT = 3000
lmin = 30
noiseLevels = numpy.array([5])
beamSizeArcmin = 1.4

lbuffer = 1500
lmax_calc = lmax+lbuffer

expNamesThisNode = numpy.array_split(numpy.asarray(expNames), size)[rank]

classExecDir = '~/code/class_delens-master/'
classDataDir = '/scratch/users/ctrendafilova/CLASS/'
outputDir = classDataDir + 'results/'

classDataDirThisNode = classDataDir + 'data/Node_' + str(rank) + '/'
fileBase = 'DALI_ExpA_60meV'
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

cosmoParams = list(cosmoFid.keys())
delta_l_max = 2000
ell = numpy.arange(2,lmax_calc+1+delta_l_max)

reconstructionMask = dict()
reconstructionMask['lmax_T'] = 3000
reconstructionMask['lmax_E'] = 5000
reconstructionMask['lmax_B'] = 5000

extra_params = dict()
#extra_params['delensing_verbose'] = 3
#extra_params['output_spectra_noise'] = 'no'
#extra_params['write warnings'] = 'y'
extra_params['delta_l_max'] = delta_l_max

ellsToUse = {'cl_TT': [lmin, lmaxTT], 'cl_TE': [lmin, lmax], 'cl_EE': [lmin, lmax], 'cl_dd': [2, lmax]}
ellsToUseNG = {'cl_TT': [lmin, lmaxTT], 'cl_TE': [lmin, lmax], 'cl_EE': [lmin, lmax], 'cl_dd': [2, lmax], 'lmaxCov': lmax_calc}

cmbNoiseSpectra = dict()
deflectionNoises = dict()
paramDerivs = dict()
secondDerivs = dict()
powersFid = dict()
invCovDotParamDerivs_delensed = dict()
invCovDotParamDerivs_lensed = dict()
paramDerivStack_delensed = dict()
paramDerivStack_lensed = dict()
fisherGaussian = dict()
fisherNonGaussian_delensed = dict()
fisherNonGaussian_lensed = dict()

doNonGaussian = False
includeUnlensedSpectraDerivatives = True
doDALI = True

if doDALI is True:
    invCovDotSecondDerivs_delensed = dict()
    invCovDotSecondDerivs_lensed = dict()
    secondDerivStack_delensed = dict()
    secondDerivStack_lensed = dict()
    DALI3Gaussian = dict()
    DALI3NonGaussian_delensed = dict()
    DALI3NonGaussian_lensed = dict()
    DALI4Gaussian = dict()
    DALI4NonGaussian_delensed = dict()
    DALI4NonGaussian_lensed = dict()

### Assign task of computing lensed NG covariance to last node       ###
### This is chosen because last node sometimes has fewer experiments ###
if doNonGaussian is True:
    if rank == size-1:

        if includeUnlensedSpectraDerivatives:
            dCldCLd_lensed, dCldCLu_lensed = classWrapTools.camb_class_generate_data(cosmoFid,
                                     cmbNoise = None, \
                                     deflectionNoise = None, \
                                     extraParams = extra_params, \
                                     rootName = fileBaseThisNode, \
                                     lmax = lmax_calc, \
                                     calculateDerivatives = 'lensed', \
                                     includeUnlensedSpectraDerivatives = includeUnlensedSpectraDerivatives,
                                     classExecDir = classExecDir,
                                     classDataDir = classDataDirThisNode,
                                     doLensedWithCAMB = True)
        else:
            dCldCLd_lensed = classWrapTools.camb_class_generate_data(cosmoFid,
                                         cmbNoise = None, \
                                         deflectionNoise = None, \
                                         extraParams = extra_params, \
                                         rootName = fileBaseThisNode, \
                                         lmax = lmax_calc, \
                                         calculateDerivatives = 'lensed', \
                                         classExecDir = classExecDir,
                                         classDataDir = classDataDirThisNode,
                                         doLensedWithCAMB = True)
            dCldCLu_lensed = None

        print('Successfully computed derivatives')
    else:
        dCldCLd_lensed = None

for k in expNamesThisNode:
    expName = expNames[k]

    print('Node ' + str(rank) + ' working on experiment ' + str(expName))

    cmbNoiseSpectra[k] = classWrapTools.noiseSpectra(l = ell,
                                                noiseLevelT = noiseLevels[k],
                                                useSqrt2 = True,
                                                beamArcmin = beamSizeArcmin)

    powersFid[k], deflectionNoises[k] = classWrapTools.camb_class_generate_data(cosmoFid,
                                         cmbNoise = cmbNoiseSpectra[k],
                                         extraParams = extra_params,
                                         rootName = fileBaseThisNode,
                                         lmax = lmax_calc,
                                         classExecDir = classExecDir,
                                         classDataDir = classDataDirThisNode,
                                         reconstructionMask = reconstructionMask,
                                         doLensedWithCAMB = True)

    paramDerivs[k] = fisherTools.getPowerDerivWithParams(cosmoFid = cosmoFid, \
                            extraParams = extra_params, \
                            stepSizes = stepSizes, \
                            polCombs = polCombs, \
                            cmbNoiseSpectraK = cmbNoiseSpectra[k], \
                            deflectionNoisesK = deflectionNoises[k], \
                            useClass = False, \
                            doLensedWithCAMB = True, \
                            lmax = lmax_calc, \
                            fileNameBase = fileBaseThisNode, \
                            classExecDir = classExecDir, \
                            classDataDir = classDataDirThisNode)
                            
    if doDALI is True:
        print(( 'Node ' + str(rank) + ' calculating second derivatives for experiment ' + str(expName)))
                                
        secondDerivs[k] = fisherTools.getSecondPowerDerivWithParams(cosmoFid = cosmoFid, \
                                stepSizes = stepSizes, \
                                polCombs = polCombs, \
                                cmbNoiseSpectraK = cmbNoiseSpectra[k], \
                                deflectionNoisesK = deflectionNoises[k], \
                                useClass = False, \
                                doLensedWithCAMB = True, \
                                lmax = lmax_calc, \
                                fileNameBase = fileBaseThisNode, \
                                classExecDir = classExecDir, \
                                classDataDir = classDataDirThisNode, \
                                extraParams = extra_params)

    fisherGaussian[k] = fisherTools.getGaussianCMBFisher(powersFid = powersFid[k], \
                            paramDerivs = paramDerivs[k], \
                            cmbNoiseSpectra = cmbNoiseSpectra[k], \
                            deflectionNoises = deflectionNoises[k], \
                            cosmoParams = cosmoParams, \
                            spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                            polCombsToUse = polCombs, \
                            ellsToUse = ellsToUse)
                            
    if doDALI is True:
                            
        DALI3Gaussian[k], DALI4Gaussian[k] = fisherTools.getGaussianDoubletDALI(powersFid = powersFid[k], \
                                paramDerivs = paramDerivs[k], \
                                secondDerivs = secondDerivs[k], \
                                cmbNoiseSpectra = cmbNoiseSpectra[k], \
                                deflectionNoises = deflectionNoises[k], \
                                cosmoParams = cosmoParams, \
                                spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                                polCombsToUse = polCombs, \
                                ellsToUse = ellsToUse)

    if doNonGaussian:

        ### Overwrite dCldCLd_delensed for each experiment to save memory ###

        if includeUnlensedSpectraDerivatives:
            dCldCLd_delensed, dCldCLu_delensed = classWrapTools.camb_class_generate_data(cosmoFid,
                                                 cmbNoise = cmbNoiseSpectra[k], \
                                                 deflectionNoise = deflectionNoises[k], \
                                                 extraParams = extra_params, \
                                                 rootName = fileBaseThisNode, \
                                                 lmax = lmax_calc, \
                                                 calculateDerivatives = 'delensed', \
                                                 includeUnlensedSpectraDerivatives = includeUnlensedSpectraDerivatives,
                                                 classExecDir = classExecDir,
                                                 classDataDir = classDataDirThisNode,
                                                 doLensedWithCAMB = True)
        else:
            dCldCLd_delensed = classWrapTools.camb_class_generate_data(cosmoFid,
                                                 cmbNoise = cmbNoiseSpectra[k], \
                                                 deflectionNoise = deflectionNoises[k], \
                                                 extraParams = extra_params, \
                                                 rootName = fileBaseThisNode, \
                                                 lmax = lmax_calc, \
                                                 calculateDerivatives = 'delensed', \
                                                 classExecDir = classExecDir,
                                                 classDataDir = classDataDirThisNode,
                                                 doLensedWithCAMB = True)
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

        if doDALI is True:
            invCovDotSecondDerivs_delensed[k], secondDerivStack_delensed[k] = fisherTools.choleskyInvCovDotSecondDerivsNG(powersFid = powersFid[k], \
                                        cmbNoiseSpectra = cmbNoiseSpectra[k], \
                                        deflectionNoiseSpectra = deflectionNoises[k], \
                                        dCldCLd = dCldCLd_delensed,
                                        paramDerivs = paramDerivs[k], \
                                        secondDerivs = secondDerivs[k], \
                                        cosmoParams = cosmoParams, \
                                        ellsToUse = ellsToUseNG, \
                                        polCombsToUse = polCombs, \
                                        spectrumType = 'delensed')    

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

        if doDALI is True:
        
            invCovDotSecondDerivs_lensed[k], secondDerivStack_lensed[k] = fisherTools.choleskyInvCovDotSecondDerivsNG(powersFid = powersFid[k], \
                                cmbNoiseSpectra = cmbNoiseSpectra[k], \
                                deflectionNoiseSpectra = deflectionNoises[k], \
                                dCldCLd = dCldCLd_lensed,
                                paramDerivs = paramDerivs[k], \
                                secondDerivs = secondDerivs[k], \
                                cosmoParams = cosmoParams, \
                                ellsToUse = ellsToUseNG, \
                                polCombsToUse = polCombs, \
                                spectrumType = 'lensed')
                                    
            DALI3NonGaussian_delensed[k], DALI4NonGaussian_delensed[k] = fisherTools.getNonGaussianDoubletDALI(invCovDotParamDerivs = invCovDotParamDerivs_delensed[k], \
                                        invCovDotSecondDerivs = invCovDotSecondDerivs_delensed[k], \
                                        secondDerivStack = secondDerivStack_delensed[k], \
                                        cosmoParams = cosmoParams)
                                        
            DALI3NonGaussian_lensed[k], DALI4NonGaussian_lensed[k] = fisherTools.getNonGaussianDoubletDALI(invCovDotParamDerivs = invCovDotParamDerivs_lensed[k], \
                                        invCovDotSecondDerivs = invCovDotSecondDerivs_lensed[k], \
                                        secondDerivStack = secondDerivStack_lensed[k], \
                                        cosmoParams = cosmoParams)
                                    

print('Node ' + str(rank) + ' finished all experiments')

forecastData = {'cmbNoiseSpectra' : cmbNoiseSpectra,
                'powersFid' : powersFid,
                'paramDerivs': paramDerivs,
                'fisherGaussian': fisherGaussian,
                'deflectionNoises' : deflectionNoises}
if doDALI:
    forecastData['secondDerivs'] = secondDerivs
    forecastData['DALI3Gaussian'] = DALI3Gaussian
    forecastData['DALI4Gaussian'] = DALI4Gaussian
if doNonGaussian:
    forecastData['invCovDotParamDerivs_delensed'] = invCovDotParamDerivs_delensed
    forecastData['paramDerivStack_delensed'] = paramDerivStack_delensed
    forecastData['invCovDotParamDerivs_lensed'] = invCovDotParamDerivs_lensed
    forecastData['paramDerivStack_lensed'] = paramDerivStack_lensed
    forecastData['fisherNonGaussian_delensed'] = fisherNonGaussian_delensed
    forecastData['fisherNonGaussian_lensed'] = fisherNonGaussian_lensed
    if doDALI:
        forecastData['invCovDotSecondDerivs_delensed'] = invCovDotSecondDerivs_delensed
        forecastData['secondDerivStack_delensed'] = secondDerivStack_delensed
        forecastData['invCovDotSecondDerivs_lensed'] = invCovDotSecondDerivs_lensed
        forecastData['secondDerivStack_lensed'] = secondDerivStack_lensed
        forecastData['DALI3NonGaussian_delensed'] = DALI3NonGaussian_delensed
        forecastData['DALI3NonGaussian_lensed'] = DALI3NonGaussian_lensed
        forecastData['DALI4NonGaussian_delensed'] = DALI4NonGaussian_delensed
        forecastData['DALI4NonGaussian_lensed'] = DALI4NonGaussian_lensed

print('Node ' + str(rank) + ' saving data')

filename = classDataDirThisNode + fileBaseThisNode + '.pkl'
delensedOutput = open(filename, 'wb')
pickle.dump(forecastData, delensedOutput, -1)
delensedOutput.close()
print('Node ' + str(rank) + ' saving data complete')

comm.Barrier()

if rank==0:
    print('Node ' + str(rank) + ' collecting data')
    for irank in range(1,size):
        print('Getting data from node ' + str(irank))
        filename = classDataDir + 'data/Node_' + str(irank) + '/' + fileBase + '_' + str(irank) + '.pkl'
        nodeData = open(filename, 'rb')
        nodeForecastData = pickle.load(nodeData)
        nodeData.close()
        for key in list(forecastData.keys()):
            forecastData[key].update(nodeForecastData[key])

    print('Node ' + str(rank) + ' reading script')
    f = open(os.path.abspath(__file__), 'r')
    script_text = f.read()
    f.close()

    forecastData['script_text'] = script_text

    forecastData['cosmoFid'] = cosmoFid
    forecastData['cosmoParams'] = cosmoParams

    print('Node ' + str(rank) + ' saving collected data')
    filename = outputDir + fileBase + '.pkl'
    delensedOutput = open(filename, 'wb')
    pickle.dump(forecastData, delensedOutput, -1)
    delensedOutput.close()
