import numpy
from numpy import loadtxt
import pickle

def loadGaussianNG(jobName, pythonFlag = 3, returnCosmoParams = False):
    with open(jobName + ".pkl", 'rb') as f:
        if pythonFlag == 2:
            data = pickle.load(f)
        elif pythonFlag == 3:
            data = pickle.load(f, encoding="latin1")

    nExps = len(data['fisherGaussian'])
    fishers = dict()
    gTypes = ['Gaussian','NonGaussian']
    sTypes = ['delensed','lensed', 'unlensed']
    if returnCosmoParams:
        cosmoParams = data['cosmoParams']

    for gt, gaussianType in enumerate(gTypes):
        fishers[gaussianType] = dict()
        for st, spectrumType in enumerate(sTypes):
            fishers[gaussianType][spectrumType] = dict()
            for i in range(0,nExps):
                if gaussianType == 'Gaussian':
                    fishers[gaussianType][spectrumType][i] = data['fisherGaussian'][i][spectrumType]
                elif gaussianType == 'NonGaussian':
                    if spectrumType == 'unlensed':
                        fishers[gaussianType][spectrumType][i] = data['fisherGaussian'][i][spectrumType]
                    elif spectrumType == 'delensed':
                        fishers[gaussianType][spectrumType][i] = data['fisherNonGaussian_delensed'][i]
                    elif spectrumType == 'lensed':
                        fishers[gaussianType][spectrumType][i] = data['fisherNonGaussian_lensed'][i]

    if returnCosmoParams:
        return fishers, cosmoParams
    else:
        return fishers

def loadBiasVectors(jobName, pythonFlag = 3, returnCosmoParams = False):
    with open(jobName + ".pkl", 'rb') as f:
        if pythonFlag == 2:
            data = pickle.load(f)
        elif pythonFlag == 3:
            data = pickle.load(f, encoding="latin1")

    nExps = len(data['fisherGaussian'])
    biasVectors = dict()
    gTypes = ['Gaussian','NonGaussian']
    sTypes = ['delensed','lensed', 'unlensed']
    cosmoParams = data['cosmoParams']

    for gt, gaussianType in enumerate(gTypes):
        biasVectors[gaussianType] = dict()
        for st, spectrumType in enumerate(sTypes):
            biasVectors[gaussianType][spectrumType] = dict()
            for i in range(0,nExps):
                if gaussianType == 'Gaussian':
                        biasVectors[gaussianType][spectrumType][i] = data['biasVectorGaussian'][i][spectrumType]
                elif gaussianType == 'NonGaussian':
                    if spectrumType == 'unlensed':
                        biasVectors[gaussianType][spectrumType][i] = data['biasVectorGaussian'][i][spectrumType]
                    elif spectrumType == 'delensed':
                        biasVectors[gaussianType][spectrumType][i] = data['biasVectorNonGaussian_delensed'][i]
                    elif spectrumType == 'lensed':
                        biasVectors[gaussianType][spectrumType][i] = data['biasVectorNonGaussian_lensed'][i]

    if returnCosmoParams:
        return biasVectors, cosmoParams
    else:
        return biasVectors

def addfsky(fishers, fsky, gTypes = ['Gaussian','NonGaussian'], sTypes = ['delensed','lensed', 'unlensed']):
    newFishers = dict()
    for gt, gaussianType in enumerate(gTypes):
        newFishers[gaussianType] = dict()
        for st, spectrumType in enumerate(sTypes):
            newFishers[gaussianType][spectrumType] = dict()
            nExps = len(fishers[gaussianType][spectrumType])
            for i in range(0,nExps):
                newFishers[gaussianType][spectrumType][i] = fishers[gaussianType][spectrumType][i]*fsky

    return newFishers

def addTau(fishers, cosmoParams, gTypes = ['Gaussian','NonGaussian'], sTypes = ['delensed','lensed', 'unlensed']):

    nParams = len(cosmoParams)
    # create Tau-only Fisher matrix
    tauFisher = numpy.zeros([nParams,nParams])
    tauPrior2 = 0.007**2
    tauIndex = cosmoParams.index('tau')
    tauFisher[tauIndex][tauIndex] = 1/tauPrior2

    newFishers = dict()
    for gt, gaussianType in enumerate(gTypes):
        newFishers[gaussianType] = dict()
        for st, spectrumType in enumerate(sTypes):
            newFishers[gaussianType][spectrumType] = dict()
            nExps = len(fishers[gaussianType][spectrumType])
            for i in range(0,nExps):
                newFishers[gaussianType][spectrumType][i] = fishers[gaussianType][spectrumType][i] + tauFisher

    return newFishers

def addBAO(fishers, cosmoParams, baoFile, gTypes = ['Gaussian','NonGaussian'], sTypes = ['delensed','lensed', 'unlensed']):

    nParams = len(cosmoParams)

    # import BAO file
    baoArray = loadtxt(baoFile, comments="#", delimiter=" ", unpack=False)
    with open(baoFile) as fp:
        baoParams = fp.readline()
    fp.close()
    baoParams = baoParams[3:-4].split("   ")
    # create BAO matrix
    baoArrayReordered = numpy.zeros([nParams,nParams])
    for i in range(0,nParams):
        for j in range(0,nParams):
            ib = baoParams.index(cosmoParams[i])
            jb = baoParams.index(cosmoParams[j])
            baoArrayReordered[i][j] = baoArray[ib][jb]

    newFishers = dict()
    for gt, gaussianType in enumerate(gTypes):
        newFishers[gaussianType] = dict()
        for st, spectrumType in enumerate(sTypes):
            newFishers[gaussianType][spectrumType] = dict()
            nExps = len(fishers[gaussianType][spectrumType])
            for i in range(0,nExps):
                newFishers[gaussianType][spectrumType][i] = fishers[gaussianType][spectrumType][i] + baoArrayReordered

    return newFishers

def invertFishers(fishers, gTypes = ['Gaussian','NonGaussian'], sTypes = ['delensed','lensed', 'unlensed']):
    newFishers = dict()
    for gt, gaussianType in enumerate(gTypes):
        newFishers[gaussianType] = dict()
        for st, spectrumType in enumerate(sTypes):
            newFishers[gaussianType][spectrumType] = dict()
            nExps = len(fishers[gaussianType][spectrumType])
            for i in range(0,nExps):
                newFishers[gaussianType][spectrumType][i] = numpy.linalg.inv(fishers[gaussianType][spectrumType][i])

    return newFishers

def fixParameters(fisher, cosmoParams, paramsToFix, returnFixedParamList = False):
    cosmoParamsTemp = cosmoParams[:]
    fixed_fisher = fisher.copy()
    if paramsToFix == ['']:
        if returnFixedParamList:
            return fisher, cosmoParamsTemp
        else:
            return fisher
    else:
        for fp, fixedParam in enumerate(paramsToFix):
            for i in range(fisher.ndim):
                fixed_fisher = numpy.delete(fixed_fisher, cosmoParamsTemp.index(fixedParam), axis = i)
            cosmoParamsTemp.remove(fixedParam)
        if returnFixedParamList:
            return fixed_fisher, cosmoParamsTemp
        else:
            return fixed_fisher

def getSigmas(covMats, gTypes = ['Gaussian','NonGaussian'], sTypes = ['delensed','lensed', 'unlensed']):
    sigmas = dict()
    for gt, gaussianType in enumerate(gTypes):
        sigmas[gaussianType] = dict()
        for st, spectrumType in enumerate(sTypes):
            sigmas[gaussianType][spectrumType] = dict()
            nExps = len(covMats[gaussianType][spectrumType])
            for i in range(nExps):
                sigmas[gaussianType][spectrumType][i] = numpy.sqrt(numpy.diag(covMats[gaussianType][spectrumType][i]))
    return sigmas

def getBiases(covMats, biasVectors, gTypes = ['Gaussian','NonGaussian'], sTypes = ['delensed','lensed', 'unlensed']):
    biases = dict()
    for gt, gaussianType in enumerate(gTypes):
        biases[gaussianType] = dict()
        for st, spectrumType in enumerate(sTypes):
            biases[gaussianType][spectrumType] = dict()
            nExps = len(biasVectors[gaussianType][spectrumType])
            for i in range(nExps):
                biases[gaussianType][spectrumType][i] = numpy.einsum('ij, j', \
                                                                     covMats[gaussianType][spectrumType][i], \
                                                                     biasVectors[gaussianType][spectrumType][i])
    return biases

def getFOMs(covMats, gTypes = ['Gaussian','NonGaussian'], sTypes = ['delensed','lensed','unlensed']):
    FOMs = dict()
    for gt, gaussianType in enumerate(gTypes):
        FOMs[gaussianType] = dict()
        for st, spectrumType in enumerate(sTypes):
            FOMs[gaussianType][spectrumType] = dict()
            nExps = len(covMats[gaussianType][spectrumType])
            for i in range(0,nExps):
                FOMs[gaussianType][spectrumType][i] = 1/numpy.sqrt(numpy.linalg.det(covMats[gaussianType][spectrumType][i]))

    return FOMs
