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
expNames = list(range(3))
nExps = len(expNames)
lmax = 5000
noiseLevels = [5, 1, 0.1]
beamSizeArcmin = [1.4, 1.4, 0.1]
expLetters = ['A', 'B', 'C']

lbuffer = 1500
lmax_calc = lmax+lbuffer

classExecDir = './CLASS_delens/'
classDataDir = './CLASS_delens/'
outputDir = classDataDir + 'results/'

classDataDirThisNode = classDataDir + 'data/Node_' + str(rank) + '/'

for k in expNames:
    fileBase = 'spectra_Experiment_'+expLetters[k]
    fileBaseThisNode = fileBase + '_' + str(rank)

    if not os.path.exists(classDataDirThisNode):
        os.makedirs(classDataDirThisNode)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)


    spectrumTypes = ['unlensed', 'lensed', 'delensed', 'lensing']

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
    ell = numpy.arange(2,lmax_calc+1+2000)

    reconstructionMask = dict()
    reconstructionMask['lmax_T'] = 3000

    extra_params = dict()
    #extra_params['delensing_verbose'] = 3
    #extra_params['output_spectra_noise'] = 'no'
    #extra_params['write warnings'] = 'y'

    cmbNoiseSpectra = dict()
    deflectionNoises = dict()
    powersFid = dict()


    expName = expNames[k]

    print('Node ' + str(rank) + ' working on experiment ' + str(expName))

    cmbNoiseSpectra[k] = classWrapTools.noiseSpectra(l = ell,
                                                noiseLevelT = noiseLevels[k],
                                                useSqrt2 = True,
                                                beamArcmin = beamSizeArcmin[k])

    powersFid[k], deflectionNoises[k] = classWrapTools.class_generate_data(cosmoFid,
                                         cmbNoise = cmbNoiseSpectra[k],
                                         extraParams = extra_params,
                                         rootName = fileBaseThisNode,
                                         lmax = lmax_calc,
                                         classExecDir = classExecDir,
                                         classDataDir = classDataDirThisNode,
                                         reconstructionMask = reconstructionMask)


    print('Node ' + str(rank) + ' finished experiment '+expLetters[k])

    forecastData = {'cmbNoiseSpectra' : cmbNoiseSpectra,
                    'powersFid' : powersFid,
                    'deflectionNoises' : deflectionNoises,
                    'cosmoFid' : cosmoFid,
                    'cosmoParams' : cosmoParams}

    print('Node ' + str(rank) + ' saving data')

    filename = outputDir + fileBase + '.pkl'
    delensedOutput = open(filename, 'wb')
    pickle.dump(forecastData, delensedOutput, -1)
    delensedOutput.close()
    print('Node ' + str(rank) + ' saving data complete')
