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

print(rank, size)

###  Set of experiments  ###
expNames = list(range(2))
nExps = len(expNames)
lmax = 5000
lmaxTT = 5000
lmin = 30
noiseLevels = [1.0, 5.0]
beamSizeArcmin = 1.4

lbuffer = 1500
lmax_calc = lmax+lbuffer

expNamesThisNode = numpy.array_split(numpy.asarray(expNames), size)[rank]

classExecDir = './CLASS_delens/'
classDataDir = './CLASS_delens/'
outputDir = classDataDir + 'results/'

classDataDirThisNode = classDataDir + 'data/Node_' + str(rank) + '/'
fileBase = 'fisher_baryon_bias'
fileBaseThisNode = fileBase + '_' + str(rank)

if not os.path.exists(classDataDirThisNode):
    os.makedirs(classDataDirThisNode)
if not os.path.exists(outputDir):
    os.makedirs(outputDir)


spectrumTypes = ['unlensed', 'lensed', 'delensed', 'lensing']
polCombs = ['cl_TT', 'cl_TE', 'cl_EE']
polCombsBias = ['cl_TT', 'cl_TE', 'cl_EE']
polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE']

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
                'mnu' : 0.06} #, \
                #'eta_0' : 0.603, \
                #'c_min' : 3.13}
#cosmoFid['n_t'] = - cosmoFid['r'] / 8.0 * (2.0 - cosmoFid['n_s'] - cosmoFid['r'] / 8.0)

cosmoFid_DMO = {'omega_c_h2':0.1197, \
                'omega_b_h2': 0.0222, \
                'N_eff': 3.046, \
                'A_s' : 2.196e-9, \
                'n_s' : 0.9655,\
                'tau' : 0.06, \
                #'H0' : 67.5, \
                'theta_s' : 0.010409, \
                #'Yhe' : 0.25, \
                #'r'   : 0.01, \
                'mnu' : 0.06, \
                'eta_0' : 0.603, \
                'c_min' : 3.13}

cosmoFid_OWLS_AGN = {'omega_c_h2':0.1197, \
                'omega_b_h2': 0.0222, \
                'N_eff': 3.046, \
                'A_s' : 2.196e-9, \
                'n_s' : 0.9655,\
                'tau' : 0.06, \
                #'H0' : 67.5, \
                'theta_s' : 0.010409, \
                #'Yhe' : 0.25, \
                #'r'   : 0.01, \
                'mnu' : 0.06, \
                'eta_0' : 0.76, \
                'c_min' : 2.32}

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
                'eta_0' : 0.01, \
                'c_min' : 0.1}

cosmoParams = list(cosmoFid.keys())
ell = numpy.arange(2,lmax_calc+1+2000)

reconstructionMask = dict()
reconstructionMask['lmax_T'] = 3000

extra_params = dict()
#extra_params['delensing_verbose'] = 3
#extra_params['output_spectra_noise'] = 'no'
#extra_params['write warnings'] = 'y'

ellsToUse = {'cl_TT': [lmin, lmaxTT], 'cl_TE': [lmin, lmax], 'cl_EE': [lmin, lmax], 'cl_dd': [2, lmax]}
ellsToUseNG = {'cl_TT': [lmin, lmaxTT], 'cl_TE': [lmin, lmax], 'cl_EE': [lmin, lmax], 'cl_dd': [2, lmax], 'lmaxCov': lmax_calc}

cmbNoiseSpectra = dict()
deflectionNoises = dict()
paramDerivs = dict()
powersFid = dict()
powersFid_DMO = dict()
powersFid_OWLS_AGN = dict()
invCovDotParamDerivs_delensed = dict()
invCovDotParamDerivs_lensed = dict()
paramDerivStack_delensed = dict()
paramDerivStack_lensed = dict()
fisherGaussian = dict()
fisherNonGaussian_delensed = dict()
fisherNonGaussian_lensed = dict()

sysSpectrum = dict()
biasVectorGaussian = dict()
biasVectorNonGaussian_delensed = dict()
biasVectorNonGaussian_lensed = dict()

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

for k in expNamesThisNode:
    expName = expNames[k]
    sysSpectrum[k] = dict()

    print('Node ' + str(rank) + ' working on experiment ' + str(expName))

    cmbNoiseSpectra[k] = classWrapTools.noiseSpectra(l = ell,
                                                noiseLevelT = noiseLevels[k],
                                                useSqrt2 = True,
                                                beamArcmin = beamSizeArcmin)

    powersFid[k], deflectionNoises[k] = classWrapTools.class_generate_data(cosmoFid,
                                         cmbNoise = cmbNoiseSpectra[k],
                                         extraParams = extra_params,
                                         rootName = fileBaseThisNode,
                                         lmax = lmax_calc,
                                         classExecDir = classExecDir,
                                         classDataDir = classDataDirThisNode,
                                         reconstructionMask = reconstructionMask)

    powersFid_DMO[k], junk = classWrapTools.class_generate_data(cosmoFid_DMO,
                                         cmbNoise = cmbNoiseSpectra[k],
                                         extraParams = extra_params,
                                         rootName = fileBaseThisNode+'DMO',
                                         lmax = lmax_calc,
                                         classExecDir = classExecDir,
                                         classDataDir = classDataDirThisNode,
                                         reconstructionMask = reconstructionMask)
    powersFid_OWLS_AGN[k], junk = classWrapTools.class_generate_data(cosmoFid_OWLS_AGN,
                                         cmbNoise = cmbNoiseSpectra[k],
                                         extraParams = extra_params,
                                         rootName = fileBaseThisNode+'OWLS_AGN',
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

    for st, spectrumType in enumerate(['unlensed', 'lensed', 'delensed']):
        sysSpectrum[k][spectrumType] = dict()
        for pc, polComb in enumerate(['cl_TT', 'cl_TE', 'cl_EE']):
            sysSpectrum[k][spectrumType][polComb] = powersFid_OWLS_AGN[k][spectrumType][polComb] - powersFid[k][spectrumType][polComb]
            sysSpectrum[k][spectrumType][polComb][:lmin-2] = 0
            if polComb == 'cl_TT':
                sysSpectrum[k][spectrumType][polComb][lmaxTT-1:] = 0
            else:
                sysSpectrum[k][spectrumType][polComb][lmax-1:] = 0
    for st, spectrumType in enumerate(['lensing']):
        sysSpectrum[k][spectrumType] = dict()
        for pc, polComb in enumerate(['cl_dd']):
            sysSpectrum[k][spectrumType][polComb] = powersFid_OWLS_AGN[k][spectrumType][polComb] - powersFid[k][spectrumType][polComb]
            sysSpectrum[k][spectrumType][polComb][lmax-1:] = 0

    fisherGaussian[k] = fisherTools.getGaussianCMBFisherWithElls(powersFid = powersFid[k], \
                            paramDerivs = paramDerivs[k], \
                            cmbNoiseSpectra = cmbNoiseSpectra[k], \
                            deflectionNoises = deflectionNoises[k], \
                            cosmoParams = cosmoParams, \
                            spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                            polCombsToUse = polCombsToUse, \
                            ellsToUse = ellsToUse)

    biasVectorGaussian[k] = fisherTools.getBiasVectorGaussian(powersFid = powersFid[k], \
        paramDerivs = paramDerivs[k], \
        cmbNoiseSpectra = cmbNoiseSpectra[k], \
        deflectionNoises = deflectionNoises[k], \
        cosmoParams = cosmoParams, \
        sysSpectrum = sysSpectrum[k], \
        spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
        polCombsToUse = polCombsBias, \
        lmax = lmax)

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


        invCovDotParamDerivs_delensed[k], paramDerivStack_delensed[k] = fisherTools.choleskyInvCovDotParamDerivsNGWithElls(powersFid = powersFid[k], \
                                    cmbNoiseSpectra = cmbNoiseSpectra[k], \
                                    deflectionNoiseSpectra = deflectionNoises[k], \
                                    dCldCLd = dCldCLd_delensed,
                                    paramDerivs = paramDerivs[k], \
                                    cosmoParams = cosmoParams, \
                                    dCldCLu = dCldCLu_delensed, \
                                    ellsToUse = ellsToUseNG, \
                                    polCombsToUse = polCombsToUse, \
                                    spectrumType = 'delensed')

        ############################
        ## Seems to hang on bcast ##
        ############################

        #dCldCLd_lensed = comm.bcast(dCldCLd_lensed, root=size-1)
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


        invCovDotParamDerivs_lensed[k], paramDerivStack_lensed[k] = fisherTools.choleskyInvCovDotParamDerivsNGWithElls(powersFid = powersFid[k], \
                                    cmbNoiseSpectra = cmbNoiseSpectra[k], \
                                    deflectionNoiseSpectra = deflectionNoises[k], \
                                    dCldCLd = dCldCLd_lensed,
                                    paramDerivs = paramDerivs[k], \
                                    cosmoParams = cosmoParams, \
                                    dCldCLu = dCldCLu_lensed,
                                    ellsToUse = ellsToUseNG, \
                                    polCombsToUse = polCombsToUse, \
                                    spectrumType = 'lensed')

        fisherNonGaussian_delensed[k] = fisherTools.getNonGaussianCMBFisher(invCovDotParamDerivs = invCovDotParamDerivs_delensed[k], \
                                    paramDerivStack = paramDerivStack_delensed[k], \
                                    cosmoParams = cosmoParams)

        fisherNonGaussian_lensed[k] = fisherTools.getNonGaussianCMBFisher(invCovDotParamDerivs = invCovDotParamDerivs_lensed[k], \
                                    paramDerivStack = paramDerivStack_lensed[k], \
                                    cosmoParams = cosmoParams)

        biasVectorNonGaussian_delensed[k] = fisherTools.getBiasVectorNonGaussian(invCovDotParamDerivs_delensed[k], \
                                cosmoParams, sysSpectrum[k], lmax = lmax_calc, polCombsToUse = polCombsBias, spectrumType = 'delensed')

        biasVectorNonGaussian_lensed[k] = fisherTools.getBiasVectorNonGaussian(invCovDotParamDerivs_lensed[k], \
                                cosmoParams, sysSpectrum[k], lmax = lmax_calc, polCombsToUse = polCombsBias, spectrumType = 'lensed')

print('Node ' + str(rank) + ' finished all experiments')

forecastData = {'cmbNoiseSpectra' : cmbNoiseSpectra,
                'powersFid' : powersFid,
                'paramDerivs': paramDerivs,
                'fisherGaussian': fisherGaussian,
                'biasVectorGaussian': biasVectorGaussian,
                'deflectionNoises' : deflectionNoises}
if doNonGaussian:
    forecastData['invCovDotParamDerivs_delensed'] = invCovDotParamDerivs_delensed
    forecastData['paramDerivStack_delensed'] = paramDerivStack_delensed
    forecastData['invCovDotParamDerivs_lensed'] = invCovDotParamDerivs_lensed
    forecastData['paramDerivStack_lensed'] = paramDerivStack_lensed
    forecastData['fisherNonGaussian_delensed'] = fisherNonGaussian_delensed
    forecastData['fisherNonGaussian_lensed'] = fisherNonGaussian_lensed
    forecastData['biasVectorNonGaussian_delensed'] = biasVectorNonGaussian_delensed
    forecastData['biasVectorNonGaussian_lensed'] = biasVectorNonGaussian_lensed

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
