import os
from numpy import *
import numpy
import scipy
import scipy.linalg

import cambWrapTools
import classWrapTools

# utilities for dictionaries

def onedDict(names1):

    output = {}
    for a in names1:
        output[a] = {}
    return output

def twodDict(names1, names2):

    output = {}
    for a in names1:
        output[a] = {}

        for b in names2:
            output[a][b] = {}


    return output

def threedDict(names1, names2, names3):

    output = {}
    for a in names1:
        output[a] = {}

        for b in names2:
            output[a][b] = {}

            for c in names3:
                output[a][b][c] = {}

    return output

def fourdDict(names1, names2, names3, names4):

    output = {}
    for a in names1:
        output[a] = {}

        for b in names2:
            output[a][b] = {}

            for c in names3:
                output[a][b][c] = {}

                for d in names4:
                    output[a][b][c][d] = {}

    return output

def fivedDict(names1, names2, names3, names4, names5):

    output = {}
    for a in names1:
        output[a] = {}

        for b in names2:
            output[a][b] = {}

            for c in names3:
                output[a][b][c] = {}

                for d in names4:
                    output[a][b][c][d] = {}

                    for e in names5:
                        output[a][b][c][d][e] = {}

    return output

# utilities for calculating Fisher forecasts

def noiseSpectra(l, noiseLevelT, useSqrt2 = True, beamArcmin = 1.4, beamFile = None, noiseLevelP = None):
#make a full set of noise spectra.


    if beamFile == None:


        beam_sigma_radians =   (beamArcmin * numpy.pi / (180. * 60.)) / numpy.sqrt(8. * numpy.log(2) )



        beamPower = numpy.exp(l * (l+1) * beam_sigma_radians**2)

        print('** noiseSpectra: using analytic beam')
    else:
        beamVals = numpy.loadtxt(beamFile)
        beamValsOnL = (scipy.interpolate.interp1d(beamVals[:,0], beamVals[:,1], bounds_error = False))(l)


        beamPower = 1/(beamValsOnL**2)
        print('** noiseSpectra: using beam from file')

    noise_ster = (numpy.pi / (180. * 60))**2 * noiseLevelT**2
    nl = len(l)

    cl_TT = numpy.empty(nl)
    cl_TT.fill(noise_ster)
    cl_TT *= beamPower


    cl_EE = numpy.empty(nl)
    if  useSqrt2:
        cl_EE.fill(noise_ster * 2.)
    else:
        noise_sterP = (numpy.pi / (180. * 60))**2 * noiseLevelP**2
        cl_EE.fill(noise_sterP)
    cl_EE *= beamPower


    cl_BB = numpy.empty(nl)
    if  useSqrt2:
        cl_BB.fill(noise_ster * 2.)
    else:
        noise_sterP = (numpy.pi / (180. * 60))**2 * noiseLevelP**2
        cl_BB.fill(noise_sterP)
    cl_BB *= beamPower


    cl_TE = numpy.empty(nl)
    cl_TE.fill(0.)

    output = {'l' : l,\
                  'cl_TT' : cl_TT,\
                  'cl_EE' : cl_EE,\
                  'cl_TE' : cl_TE,\
                  'cl_BB' : cl_BB,
                  'dl_TT' : cl_TT * l * (l + 1) / 2 / numpy.pi,\
                  'dl_EE' : cl_EE * l * (l + 1) / 2 / numpy.pi,\
                  'dl_TE' : cl_TE * l * (l + 1) / 2 / numpy.pi,\
                  'dl_BB' : cl_BB * l * (l + 1) / 2 / numpy.pi
              }
    return output

def getPlanckInvVarNoise(ells, includePol = True):
    #numbers are taken from "Planck_pol.pdf" from https://cosmo.uchicago.edu/CMB-S4workshops/index.php/File:Planck_pol.pdf
    freqs = ["30", "44", "70", "100", "143", "217", "353"]
    nFreqs = len(freqs)
    fwhmsArcmin = [33., 23., 14., 10., 7., 5., 5.]
    tempNoisesUKarcmin = [145., 149., 137., 65., 43., 66., 200]
    polNoisesUKarcmin = [numpy.inf, numpy.inf, 450., 103., 81., 134., 406.]

    cmbNoiseSpectra = onedl(nFreqs)

    #get noise curve for each frequency
    for f, freq in enumerate(freqs):
        cmbNoiseSpectra[f] = noiseSpectra(ells, \
                                                        noiseLevelT = tempNoisesUKarcmin[f], \
                                                        noiseLevelP = polNoisesUKarcmin[f], \
                                                        beamArcmin = fwhmsArcmin[f], \
                                                        useSqrt2 = False)


    #now get the result via inverse-variance sum
    polCombs = list((cmbNoiseSpectra[0]).keys())
    overallResult = onedDict(polCombs)
    for pc, polComb in enumerate(polCombs):
        runningTotal = numpy.zeros(len(ells))
        for f, freq in enumerate(freqs):
            runningTotal += 1/cmbNoiseSpectra[f][polComb]

        overallResult[polComb] = 1/runningTotal

    #the ells got messed up as part of the process ... fix this.
    overallResult['l'] = ells

    if not includePol:

        polCombsPolarization = ['cl_EE', 'cl_TE', 'cl_BB', 'dl_EE', 'dl_TE', 'dl_BB' ]
        for polComb in polCombsPolarization:
            overallResult[polComb] = numpy.inf * numpy.ones(len(ells))

    return overallResult

def onedl(rows):
    return [None] * rows


def getPowerDerivWithParams(cosmoFid, stepSizes, polCombs, cmbNoiseSpectraK, deflectionNoisesK,
                            spectrumTypes = ['unlensed', 'lensed', 'delensed', 'lensing'],
                            lmax = 5000,
                            fileNameBase = 'testing', paramsToDifferentiate = None,
                            accuracy = 2.,
                            useClass = False,
                            doLensedWithCAMB = False,
                            classExecDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/',
                            classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/',
                            extraParams = dict()):

    if paramsToDifferentiate == None:
        paramsToDifferentiate = list(cosmoFid.keys())


    nParams = len(paramsToDifferentiate)

    oneSidedParams = ['DM_Pann']

    if 'mnu' in cosmoFid.keys():
        if cosmoFid['mnu'] < stepSizes['mnu']:
            oneSidedParams.append('mnu')

    oneSidedParamsIso = ['c_ad_cdi', 'c_ad_bi', 'c_ad_nid', 'c_ad_niv', \
                        'c_bi_cdi', 'c_bi_nid', 'c_bi_niv', \
                        'c_cdi_nid', 'c_cdi_niv', \
                        'c_nid_niv'] #could be + or - 1

    powersPlus = dict()
    powersMinus = dict()


    for cosmo in paramsToDifferentiate:
        print(('getting deriv w.r.t. %s' %cosmo))
        cosmoPlus = cosmoFid.copy() #copy all params including those not being perturbed.
        cosmoMinus = cosmoFid.copy()

        cosmoPlus[cosmo] = cosmoPlus[cosmo] + stepSizes[cosmo]
        cosmoMinus[cosmo] = cosmoMinus[cosmo] - stepSizes[cosmo]

        if useClass is True:
            powersPlus[cosmo], junk = classWrapTools.class_generate_data(cosmoPlus,
                        rootName = fileNameBase,
                        cmbNoise = cmbNoiseSpectraK,
                        deflectionNoise = deflectionNoisesK,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        lmax = lmax,
                        extraParams = extraParams)
        else:
            powersPlus[cosmo], junk = classWrapTools.camb_class_generate_data(cosmoPlus,
                        rootName = fileNameBase,
                        cmbNoise = cmbNoiseSpectraK,
                        deflectionNoise = deflectionNoisesK,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        lmax = lmax,
                        extraParams = extraParams,
                        accuracy = accuracy,
                        doLensedWithCAMB = doLensedWithCAMB)


        #### For one-sided derivatives, use fiducial parameters for PowersMinus
        if cosmo in oneSidedParams:
            cosmoMinus = cosmoFid.copy()

        #### isocurvature cross-correlation
        if cosmo in oneSidedParamsIso:
            if cosmoFid[cosmo] == 1.:
                cosmoPlus = cosmoFid.copy()
            elif cosmoFid[cosmo] == -1.:
                cosmoMinus = cosmoFid.copy()

        if useClass is True:
            powersMinus[cosmo], junk = classWrapTools.class_generate_data(cosmoMinus,
                        rootName = fileNameBase,
                        cmbNoise = cmbNoiseSpectraK,
                        deflectionNoise = deflectionNoisesK,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        lmax = lmax,
                        extraParams = extraParams)
        else:
            powersMinus[cosmo], junk = classWrapTools.camb_class_generate_data(cosmoMinus,
                        rootName = fileNameBase,
                        cmbNoise = cmbNoiseSpectraK,
                        deflectionNoise = deflectionNoisesK,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        lmax = lmax,
                        extraParams = extraParams,
                        accuracy = accuracy,
                        doLensedWithCAMB = doLensedWithCAMB)


    #PARAM DERIVATIVES
    polCombsTemp = polCombs.copy()
    if 'cl_dd' in polCombs:
        polCombsTemp.remove('cl_dd')

    paramDerivs = threedDict(paramsToDifferentiate, spectrumTypes, polCombsTemp)

    for  cosmo in paramsToDifferentiate:


        #### Use this for one-sided derivatives (PowersMinus is PowersFid in this case)
        if cosmo in oneSidedParams:
            denom = stepSizes[cosmo]
        else:
        #### Other parameters use two-sided derivatives
            denom = 2 * stepSizes[cosmo]

        for pc, polComb in enumerate(polCombs):
            if polComb == 'cl_dd':
                if 'lensing' in spectrumTypes:
                    paramDerivs[cosmo]['lensing'] = dict()
                    paramDerivs[cosmo]['lensing']['cl_dd'] = \
                        (powersPlus[cosmo]['lensing']['cl_dd'] - powersMinus[cosmo]['lensing']['cl_dd']) / denom
                else:
                    print('lensing must be in spectrum types to calculate cl_dd derivatives.')
            else:
                if 'unlensed' in spectrumTypes:
                    paramDerivs[cosmo]['unlensed'][polComb] = \
                        (powersPlus[cosmo]['unlensed'][polComb] - powersMinus[cosmo]['unlensed'][polComb]) / denom
                if 'lensed' in spectrumTypes:
                    paramDerivs[cosmo]['lensed'][polComb] = \
                        (powersPlus[cosmo]['lensed'][polComb] - powersMinus[cosmo]['lensed'][polComb]) / denom
                if 'delensed' in spectrumTypes:
                    paramDerivs[cosmo]['delensed'][polComb] = \
                        (powersPlus[cosmo]['delensed'][polComb] - powersMinus[cosmo]['delensed'][polComb]) / denom

    return paramDerivs


def getParamDerivsBAOandH0(cosmoFid, stepSizes, redshifts, paramsToDifferentiate = None):
    externalData = ['BAO', 'H0']
    BAOPlus = dict()
    BAOMinus = dict()
    H0Plus = dict()
    H0Minus = dict()
    if paramsToDifferentiate == None:
        paramsToDifferentiate = list(cosmoFid.keys())
    nParams = len(paramsToDifferentiate)
    oneSidedParams = ['DM_Pann']
    if 'mnu' in cosmoFid.keys():
        if cosmoFid['mnu'] < stepSizes['mnu']:
            oneSidedParams.append('mnu')
    for cosmo in paramsToDifferentiate:
        print(('getting deriv w.r.t. %s' %cosmo))
        cosmoPlus = cosmoFid.copy() #copy all params including those not being perturbed.
        cosmoMinus = cosmoFid.copy()
        cosmoPlus[cosmo] = cosmoPlus[cosmo] + stepSizes[cosmo]
        cosmoMinus[cosmo] = cosmoMinus[cosmo] - stepSizes[cosmo]
        #### For one-sided derivatives, use fiducial parameters for PowersMinus
        if cosmo in oneSidedParams:
            cosmoMinus = cosmoFid.copy()
        BAOPlus[cosmo] = cambWrapTools.getBAOParams(cosmoPlus, redshifts)
        BAOMinus[cosmo] = cambWrapTools.getBAOParams(cosmoMinus, redshifts)
        H0Plus[cosmo] = cambWrapTools.get_H0_from_theta(cosmoPlus)
        H0Minus[cosmo] = cambWrapTools.get_H0_from_theta(cosmoMinus)
    #PARAM DERIVATIVES
    paramDerivs = twodDict(paramsToDifferentiate, externalData)
    for  cosmo in paramsToDifferentiate:
        #### Use this for one-sided derivatives (PowersMinus is PowersFid in this case)
        if cosmo in oneSidedParams:
            denom = stepSizes[cosmo]
        else:
        #### Other parameters use two-sided derivatives
            denom = 2 * stepSizes[cosmo]
        paramDerivs[cosmo]['BAO'] = (BAOPlus[cosmo][:,0] - BAOMinus[cosmo][:,0]) / denom
        paramDerivs[cosmo]['H0'] = (H0Plus[cosmo] - H0Minus[cosmo]) / denom

    return paramDerivs

def getBAODerivWithParams(cosmoFid, stepSizes, redshifts,
                            fileNameBase = 'testing', 
                            paramsToDifferentiate = None,
                            classExecDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/',
                            classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/',
                            extraParams = dict()):

    if paramsToDifferentiate == None:
        paramsToDifferentiate = list(cosmoFid.keys())

    nParams = len(paramsToDifferentiate)

    oneSidedParams = ['DM_Pann']

    if 'mnu' in cosmoFid.keys():
        if cosmoFid['mnu'] < stepSizes['mnu']:
            oneSidedParams.append('mnu')

    oneSidedParamsIso = ['c_ad_cdi', 'c_ad_bi', 'c_ad_nid', 'c_ad_niv', \
                        'c_bi_cdi', 'c_bi_nid', 'c_bi_niv', \
                        'c_cdi_nid', 'c_cdi_niv', \
                        'c_nid_niv'] #could be + or - 1

    BAOPlus = dict()
    BAOMinus = dict()

    for cosmo in paramsToDifferentiate:
        #print(('getting deriv w.r.t. %s' %cosmo))
        cosmoPlus = cosmoFid.copy() #copy all params including those not being perturbed.
        cosmoMinus = cosmoFid.copy()

        cosmoPlus[cosmo] = cosmoPlus[cosmo] + stepSizes[cosmo]
        cosmoMinus[cosmo] = cosmoMinus[cosmo] - stepSizes[cosmo]
        
        #### For one-sided derivatives, use fiducial parameters for PowersMinus
        if cosmo in oneSidedParams:
            cosmoMinus = cosmoFid.copy()

        #### isocurvature cross-correlation
        if cosmo in oneSidedParamsIso:
            if cosmoFid[cosmo] == 1.:
                cosmoPlus = cosmoFid.copy()
            elif cosmoFid[cosmo] == -1.:
                cosmoMinus = cosmoFid.copy()

        BAOPlus[cosmo] = classWrapTools.getBAOParams(cosmo = cosmoPlus,
                                                        redshifts = redshifts,
                                                        rootName = fileNameBase,
                                                        classExecDir = classExecDir,
                                                        classDataDir = classDataDir,
                                                        extraParams = extraParams
                                                        )

        BAOMinus[cosmo] = classWrapTools.getBAOParams(cosmo = cosmoMinus,
                                                        redshifts = redshifts,
                                                        rootName = fileNameBase,
                                                        classExecDir = classExecDir,
                                                        classDataDir = classDataDir,
                                                        extraParams = extraParams
                                                        )


    #PARAM DERIVATIVES
    paramDerivs = onedDict(paramsToDifferentiate)
    for  cosmo in paramsToDifferentiate:
        #### Use this for one-sided derivatives (PowersMinus is PowersFid in this case)
        if cosmo in oneSidedParams or cosmo in oneSidedParamsIso:
            denom = stepSizes[cosmo]
        else:
        #### Other parameters use two-sided derivatives
            denom = 2 * stepSizes[cosmo]
        
        paramDerivs[cosmo] = BAOPlus[cosmo] - BAOMinus[cosmo]
        
    return paramDerivs



def getSecondDerivsBAOandH0(cosmoFid, stepSizes, redshifts, paramsToDifferentiate = None):
    externalData = ['BAO', 'H0']
    derivsPlus = dict()
    derivsMinus = dict()
    if paramsToDifferentiate == None:
        paramsToDifferentiate = list(cosmoFid.keys())
    nParams = len(paramsToDifferentiate)
    oneSidedParams = ['DM_Pann']
    if 'mnu' in cosmoFid.keys():
        if cosmoFid['mnu'] < stepSizes['mnu']:
            oneSidedParams.append('mnu')
    for cosmo in paramsToDifferentiate:
        print('getting second deriv w.r.t. %s' %cosmo)
        cosmoPlus = cosmoFid.copy() #copy all params including those not being perturbed.
        cosmoMinus = cosmoFid.copy()

        cosmoPlus[cosmo] = cosmoPlus[cosmo] + stepSizes[cosmo]
        cosmoMinus[cosmo] = cosmoMinus[cosmo] - stepSizes[cosmo]

        derivsPlus[cosmo] = getParamDerivsBAOandH0(cosmoPlus, stepSizes, redshifts, paramsToDifferentiate)


        #### For one-sided derivatives, use fiducial parameters for PowersMinus
        if cosmo in oneSidedParams:
            cosmoMinus = cosmoFid.copy()

        derivsMinus[cosmo] = getParamDerivsBAOandH0(cosmoMinus, stepSizes, redshifts, paramsToDifferentiate)


    #PARAM DERIVATIVES
    secondDerivs = threedDict(paramsToDifferentiate, paramsToDifferentiate, externalData)

    for  cosmo1 in paramsToDifferentiate:


        #### Use this for one-sided derivatives (derivsMinus is derivsFid in this case)
        if cosmo1 in oneSidedParams:
            denom = stepSizes[cosmo1]
        else:
        #### Other parameters use two-sided derivatives
            denom = 2 * stepSizes[cosmo1]

        for cosmo2 in paramsToDifferentiate:
            for ed, exData in enumerate(externalData):
                secondDerivs[cosmo1][cosmo2][exData] = \
                    (derivsPlus[cosmo1][cosmo2][exData] - derivsMinus[cosmo1][cosmo2][exData]) / denom

    return secondDerivs


def getGaussianCov(powersFid, cmbNoiseSpectra, deflectionNoises, \
                   polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                   spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                   lmax = 3000):
    nPolCombsToUse = len(polCombsToUse)
    ell = powersFid['unlensed']['l']
    oneOver2ellPlusOne =  1. / (2. * ell[:lmax-2] + 1.)
    nElls = len(oneOver2ellPlusOne)
    covs = dict()



    for st, spectrumType in enumerate(spectrumTypes):
        covs[spectrumType] = numpy.zeros((nPolCombsToUse, nPolCombsToUse, nElls))

        #on-diagonals for TT and EE
        for polComb1 in ['cl_TT', 'cl_EE', 'cl_BB']:
            if polComb1 in polCombsToUse:
                pc1 = polCombsToUse.index(polComb1)

                covs[spectrumType][pc1, pc1, :] = \
                    2. * oneOver2ellPlusOne \
                    * ((powersFid[spectrumType][polComb1][:lmax-2]  \
                            + cmbNoiseSpectra[polComb1][:lmax-2])**2)

        #on-diagonal for cl_TE
        if ('cl_TE' in polCombsToUse):
            pcTE = polCombsToUse.index('cl_TE')

            covs[spectrumType][pcTE, pcTE, :] = \
                1. * oneOver2ellPlusOne \
                * (powersFid[spectrumType]['cl_TE'][:lmax-2]**2 + \
                       + ((powersFid[spectrumType]['cl_TT'][:lmax-2]  \
                               + cmbNoiseSpectra['cl_TT'][:lmax-2]) \
                              * ((powersFid[spectrumType]['cl_EE'][:lmax-2]  \
                                      + cmbNoiseSpectra['cl_EE'][:lmax-2]))))

        #cross for TT and EE
        if ('cl_TT' in polCombsToUse and 'cl_EE' in polCombsToUse):

            pcEE = polCombsToUse.index('cl_EE')
            pcTT = polCombsToUse.index('cl_TT')

            covs[spectrumType][pcTT, pcEE, :] = \
                2. * oneOver2ellPlusOne \
                    * (powersFid[spectrumType]['cl_TE'][:lmax-2]**2) #no noise on this one

            covs[spectrumType][pcEE, pcTT, :] = \
                covs[spectrumType][pcTT, pcEE, :]

        #cross for TE and {TT, EE}
        if ('cl_TE' in polCombsToUse):# and 'cl_TT' in polCombsToUse):
            pcTE = polCombsToUse.index('cl_TE')

            for polComb1 in ['cl_TT', 'cl_EE']:
                if polComb1 in polCombsToUse:
                    pc1 = polCombsToUse.index(polComb1)

                    covs[spectrumType][pcTE, pc1, :] = \
                        2. * oneOver2ellPlusOne \
                        * powersFid[spectrumType]['cl_TE'][:lmax-2] \
                        * (powersFid[spectrumType][polComb1][:lmax-2] \
                               + cmbNoiseSpectra[polComb1][:lmax-2])

                    covs[spectrumType][pc1, pcTE, :] = \
                        covs[spectrumType][pcTE, pc1, :]

        #on-diagonals for DD
        if ('cl_dd' in polCombsToUse):
            pcDD = polCombsToUse.index('cl_dd')

            covs[spectrumType][pcDD, pcDD, :] = \
                2. * oneOver2ellPlusOne \
                * ((powersFid['lensing']['cl_dd'][:lmax-2] + \
                      deflectionNoises[:lmax-2])**2)

    return covs





def getDiagonalCovariances(polCombsToUse, powersFidES, cmbNoiseSpectraE = None, deflectionNoisesES = None, lmax = None):
#powersFid already indexed by type

    if lmax == None:
        raise ValueError("auto lmax not coded up yet")

    ell = numpy.arange(2,lmax)

    nPolCombsToUse = len(polCombsToUse)
    oneOver2ellPlusOne =  1. / (2. * ell[:lmax-2] + 1.)
    nElls = len(oneOver2ellPlusOne)

    covs = zeros((nPolCombsToUse, nPolCombsToUse, nElls))
    invCovs = zeros((nPolCombsToUse, nPolCombsToUse, nElls))

    #on-diagonals for TT and EE
    for polComb1 in ['cl_TT', 'cl_EE', 'cl_BB']:
        if polComb1 in polCombsToUse:
            pc1 = polCombsToUse.index(polComb1)


            covs[pc1, pc1, :] = \
                2. * oneOver2ellPlusOne \
                * ((powersFidES[polComb1][:lmax-2]  \
                        + cmbNoiseSpectraE[polComb1][:lmax-2])**2)

    #on-diagonal for cl_TE
    if ('cl_TE' in polCombsToUse):
        pcTE = polCombsToUse.index('cl_TE')

        covs[pcTE, pcTE, :] = \
            1. * oneOver2ellPlusOne \
            * (powersFidES['cl_TE'][:lmax-2]**2 + \
                   + ((powersFidES['cl_TT'][:lmax-2]  \
                           + cmbNoiseSpectraE['cl_TT'][:lmax-2]) \
                          * ((powersFidES['cl_EE'][:lmax-2]  \
                                  + cmbNoiseSpectraE['cl_EE'][:lmax-2]))))


    #cross for TT and EE
    if ('cl_TT' in polCombsToUse and 'cl_EE' in polCombsToUse):

        pcEE = polCombsToUse.index('cl_EE')
        pcTT = polCombsToUse.index('cl_TT')

        covs[pcTT, pcEE, :] = \
            2. * oneOver2ellPlusOne \
                * (powersFidES['cl_TE'][:lmax-2]**2) #no noise on this one

        covs[pcEE, pcTT, :] = \
            covs[pcTT, pcEE, :]

    #cross for TE and {TT, EE}
    if ('cl_TE' in polCombsToUse):# and 'cl_TT' in polCombsToUse):
        pcTE = polCombsToUse.index('cl_TE')

        for polComb1 in ['cl_TT', 'cl_EE']:
            if polComb1 in polCombsToUse:
                pc1 = polCombsToUse.index(polComb1)

                covs[pcTE, pc1, :] = \
                    2. * oneOver2ellPlusOne \
                    * powersFidES['cl_TE'][:lmax-2] \
                    * (powersFidES[polComb1][:lmax-2] \
                           + cmbNoiseSpectraE[polComb1][:lmax-2])

                covs[pc1, pcTE, :] = \
                    covs[pcTE, pc1, :]

    #on-diagonals for DD
    if ('cl_dd' in polCombsToUse):
        pcDD = polCombsToUse.index('cl_dd')

        covs[pcDD, pcDD, :] = \
            2. * oneOver2ellPlusOne \
             * ((powersFidES[exp]['lensing']['cl_dd'][:lmax-2] + \
                  deflectionNoises[exp][exp][pcReconst][mcReconst][:lmax-2])**2)

    for l in range(nElls):
        try:
            invCovs[:, :, l] = linalg.inv(covs[:, :, l])
        except:
            print("warning, cov inversion problem " , ell[l])
            invCovs[:, :, l] = full((nPolCombsToUse, nPolCombsToUse), nan)

    return covs , invCovs

def getNonGaussianCov(powersFid, \
                                cmbNoiseSpectra, \
                                deflectionNoiseSpectra, \
                                dCldCLd, \
                                dCldCLu = None, \
                                lmax = 3000, \
                                polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                                spectrumType = 'delensed',\
                                rescaleToCorrelation = False,\
                                rescaleToNoiseless = False):

    gaussianCov = getGaussianCov(powersFid = powersFid, \
                          cmbNoiseSpectra = cmbNoiseSpectra, \
                          deflectionNoises = deflectionNoiseSpectra, \
                          polCombsToUse = polCombsToUse, \
                          spectrumTypes = [spectrumType], \
                          lmax = lmax+1)

    cov = twodDict(polCombsToUse, polCombsToUse)

    for pc1, polComb1 in enumerate(polCombsToUse):
        for pc2, polComb2 in enumerate(polCombsToUse):
            if pc2 <= pc1:
                cov[polComb1][polComb2] = numpy.diag(gaussianCov[spectrumType][pc1, pc2])

    dCldCLd['cl_dd'] = numpy.eye(lmax+1)

    ell = powersFid['unlensed']['l'][:lmax-1]
    oneOver2ellPlusOne =  1. / (2. * ell[:lmax-1] + 1.)
    deflCov = 2*oneOver2ellPlusOne * (powersFid['lensing']['cl_dd'][:lmax-1])**2
    if dCldCLu is not None:
        zeroNoiseSpectra = onedDict(polCombsToUse)
        for polComb in polCombsToUse:
            zeroNoiseSpectra[polComb] = numpy.zeros(lmax)
        zeroDeflectionNoise = numpy.zeros(lmax)
        noiselessCov = getGaussianCov(powersFid = powersFid, \
                          cmbNoiseSpectra = zeroNoiseSpectra, \
                          deflectionNoises = zeroDeflectionNoise, \
                          polCombsToUse = polCombsToUse, \
                          spectrumTypes = ['unlensed'], \
                          lmax = lmax+1)

    for pc1, polComb1 in enumerate(polCombsToUse):
        for pc2, polComb2 in enumerate(polCombsToUse):
            if pc2 <= pc1:
                print('Computing covaraince for ', polComb1, ' and ', polComb2)
                # old - troubleshooting
                # cov[polComb1][polComb2] += numpy.tensordot(dCldCLd[polComb1][2:lmax+1,2:lmax+1], \
                #                             numpy.tensordot(numpy.diag(deflCov), dCldCLd[polComb2][2:lmax+1,2:lmax+1], axes = (1,1)), axes = (1,0))
                if polComb1 == 'cl_dd' and polComb2 == 'cl_dd':
                    ## Avoid double counting on-diagonal covariance for deflection power
                    cov[polComb1][polComb2] += 0.

                else:
                    cov[polComb1][polComb2] += numpy.tensordot(dCldCLd[polComb1][2:lmax+1,2:lmax+1], \
                                                numpy.tensordot(numpy.diag(deflCov), dCldCLd[polComb2][2:lmax+1,2:lmax+1], axes = (1,1)), axes = (1,0))

                if dCldCLu is not None:
                    for pc3, polComb3 in enumerate(polCombsToUse):
                        for pc4, polComb4 in enumerate(polCombsToUse):
                            if (polComb1 + '_' + polComb3 in dCldCLu.keys()
                                    and polComb2 + '_' + polComb4 in dCldCLu.keys()):
                                print('Adding covaraince from d' + polComb1 + '/d' + polComb3 + ' and d' + polComb2 + '/d' + polComb4)
                                tempCov = \
                                    numpy.tensordot(dCldCLu[polComb1 + '_' + polComb3][2:lmax+1,2:lmax+1], \
                                        numpy.tensordot(numpy.diag(noiselessCov['unlensed'][pc3, pc4]), \
                                            dCldCLu[polComb2 + '_' + polComb4][2:lmax+1,2:lmax+1], \
                                            axes = (1,1)), axes = (1,0))
                                ## Need to subtract off on-diagonal piece to avoid doubling it
                                numpy.fill_diagonal(tempCov, 0.)
                                cov[polComb1][polComb2] += tempCov

    if rescaleToCorrelation:
        if rescaleToNoiseless:
            zeroNoiseSpectra2 = onedDict(polCombsToUse)
            for polComb in polCombsToUse:
                zeroNoiseSpectra2[polComb] = numpy.zeros(lmax)
            zeroDeflectionNoise2 = numpy.zeros(lmax)
            noiselessCov2 = getGaussianCov(powersFid = powersFid, \
                              cmbNoiseSpectra = zeroNoiseSpectra2, \
                              deflectionNoises = zeroDeflectionNoise2, \
                              polCombsToUse = polCombsToUse, \
                              spectrumTypes = [spectrumType], \
                              lmax = lmax+1)

        for pc1, polComb1 in enumerate(polCombsToUse):
            for pc2, polComb2 in enumerate(polCombsToUse):
                if pc2 <= pc1:
                    if rescaleToNoiseless:
                        denom = numpy.sqrt(numpy.einsum('i,j->ij', noiselessCov2[spectrumType][pc1,pc1,:], noiselessCov2[spectrumType][pc2,pc2,:]))
                    else:
                        denom = numpy.sqrt(numpy.einsum('i,j->ij', gaussianCov[spectrumType][pc1,pc1,:], gaussianCov[spectrumType][pc2,pc2,:]))
                    cov[polComb1][polComb2] = cov[polComb1][polComb2] / denom

    matrix = list()
    for pc1, polComb1 in enumerate(polCombsToUse):
        row = list()
        for pc2, polComb2 in enumerate(polCombsToUse):
            if pc2 > pc1:
                row.append(cov[polComb2][polComb1].T)
            else:
                row.append(cov[polComb1][polComb2])
        matrix.append(row)

    covMatrix = numpy.block(matrix)

    return covMatrix

def getNonGaussianCMBFisher(invCovDotParamDerivs, paramDerivStack, cosmoParams):
    nParams = len(cosmoParams)
    fisher = numpy.zeros((nParams,nParams))
    for cp1, cosmo1 in enumerate(cosmoParams):
        for cp2, cosmo2 in enumerate(cosmoParams):
            fisher[cp1, cp2] = sum(paramDerivStack[cosmo1] * invCovDotParamDerivs[cosmo2])
    return fisher

def fixParametersInFisher(fisher, cosmoParams, paramsToFix, returnFixedParamList = False):
    cosmoParamsTemp = cosmoParams[:]
    fixed_fisher = fisher.copy()
    for fp, fixedParam in enumerate(paramsToFix):
        fixed_fisher = numpy.delete(numpy.delete(fixed_fisher, cosmoParamsTemp.index(fixedParam), axis = 0), cosmoParamsTemp.index(fixedParam), axis = 1)
        cosmoParamsTemp.remove(fixedParam)
    if returnFixedParamList:
        return fixed_fisher, cosmoParamsTemp
    else:
        return fixed_fisher

def getGaussianCMBFisher(powersFid, paramDerivs, cmbNoiseSpectra, deflectionNoises, cosmoParams,\
                            spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                            polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                            ellsToUse = None, \
                            lmax = None):

    ellsToUse, lmax = createEllsToUseDict(ellsToUse = ellsToUse, \
                        lmax = lmax, \
                        polCombsToUse = polCombsToUse)

    covs = getGaussianCov(powersFid = powersFid, \
                          cmbNoiseSpectra = cmbNoiseSpectra, \
                          deflectionNoises = deflectionNoises, \
                          polCombsToUse = polCombsToUse, \
                          lmax = lmax+1)

    ## Copy data structure of covs to invcovs (values will be overwritten)
    invCovs = covs.copy()
    ell = powersFid[list(powersFid.keys())[0]]['l'][:lmax-1]
    nElls = len(ell)
    nPolCombsToUse = len(polCombsToUse)
    nPars = len(cosmoParams)
    fisherContribs = threedDict(spectrumTypes, cosmoParams, cosmoParams)
    paramDerivArray = twodDict(cosmoParams, spectrumTypes)
    paramDerivToUse = twodDict(cosmoParams, spectrumTypes)
    fisher = dict()

    for spectrumType in spectrumTypes:
        fisher[spectrumType] = numpy.zeros((nPars, nPars))
        for l in range(nElls):
            try:
                invCovs[spectrumType][:, :, l] = linalg.inv(covs[spectrumType][:, :, l])
            except:
                print("warning, cov inversion problem " , spectrumType, ell[l])
                invCovs[spectrumType][:, :, l] = numpy.full((nPolCombsToUse, nPolCombsToUse), numpy.nan)

        for cp1, cosmo1 in enumerate(cosmoParams):
            paramDerivArray[cosmo1][spectrumType] = numpy.zeros((nPolCombsToUse, nElls))
            paramDerivToUse[cosmo1][spectrumType] = numpy.zeros((nPolCombsToUse, nElls))

            for pc, polComb in enumerate(polCombsToUse):
                if polComb == 'cl_dd':
                    #deflection as a special case; use same data regardless of spectrumType
                    if 'cl_dd' in polCombsToUse:
                        pcDD = polCombsToUse.index('cl_dd')
                        paramDerivArray[cosmo1][spectrumType][pcDD, :] = \
                            paramDerivs[cosmo1]['lensing']['cl_dd'][:lmax-1]
                else:
                    paramDerivArray[cosmo1][spectrumType][pc, :] = \
                        paramDerivs[cosmo1][spectrumType][polComb][:lmax-1]
                paramDerivToUse[cosmo1][spectrumType][pc, ellsToUse[polComb][0]-2 : ellsToUse[polComb][1]-1] = 1.
                paramDerivArray[cosmo1][spectrumType][pc, :] = \
                    paramDerivArray[cosmo1][spectrumType][pc, :] * paramDerivToUse[cosmo1][spectrumType][pc, :]



        for cp1, cosmo1 in enumerate(cosmoParams):
            for cp2, cosmo2 in enumerate(cosmoParams):
                fisherContribs[spectrumType][cosmo1][cosmo2] = \
                    numpy.einsum('ik,ijk,jk->k', \
                               paramDerivArray[cosmo1][spectrumType], invCovs[spectrumType], paramDerivArray[cosmo2][spectrumType])

                fisher[spectrumType][cp1, cp2] = sum(fisherContribs[spectrumType][cosmo1][cosmo2])

    return fisher

def getBAOFisher(paramDerivs, rs_dV_errors, cosmoParams):

    nPars = len(cosmoParams)
    fisherBAO = numpy.zeros((nPars,nPars))
    for cp1, cosmo1 in enumerate(cosmoParams):
        for cp2, cosmo2 in enumerate(cosmoParams):
            fisherBAO[cp1,cp2] = sum(paramDerivs[cosmo1] * paramDerivs[cosmo2]/(rs_dV_errors * rs_dV_errors))

    return fisherBAO

def choleskyInvCovDotParamDerivsNG(powersFid, \
                                cmbNoiseSpectra, \
                                deflectionNoiseSpectra, \
                                dCldCLd,
                                paramDerivs, \
                                cosmoParams, \
                                dCldCLu = None, \
                                ellsToUse = None, \
                                lmax = None, \
                                polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                                spectrumType = 'delensed'):

    ellsToUse, lmax = createEllsToUseDict(ellsToUse = ellsToUse, \
                        lmax = lmax, \
                        polCombsToUse = polCombsToUse)
    if 'lmaxCov' in list(ellsToUse.keys()) and ellsToUse['lmaxCov'] > lmax:
        lmax = ellsToUse['lmaxCov']

    covMatrix = getNonGaussianCov(powersFid = powersFid, \
                                cmbNoiseSpectra = cmbNoiseSpectra, \
                                deflectionNoiseSpectra = deflectionNoiseSpectra, \
                                dCldCLd = dCldCLd, \
                                dCldCLu = dCldCLu, \
                                lmax = lmax, \
                                polCombsToUse = polCombsToUse, \
                                spectrumType = spectrumType)

    try:
        covMatrix = scipy.linalg.cho_factor(covMatrix, overwrite_a = True)
    except:
        print('Warning Cholesky decomposition failed')
        covMatrix = scipy.linalg.cho_factor(numpy.eye(len(polCombsToUse)*(lmax-1)))

    paramDerivStack = onedDict(cosmoParams)
    invCovDotParamDerivs = onedDict(cosmoParams)

    for cosmo in cosmoParams:
        paramDerivStack[cosmo] = numpy.empty(0)
        for pc, polComb in enumerate(polCombsToUse):
            paramDerivTemp = numpy.zeros(lmax-1)
            paramDerivTemp[ellsToUse[polComb][0]-2 : ellsToUse[polComb][1]-1] = 1.
            if polComb == 'cl_dd':
                paramDerivStack[cosmo] = numpy.append(paramDerivStack[cosmo], paramDerivTemp * paramDerivs[cosmo]['lensing'][polComb][:lmax-1])
            else:
                paramDerivStack[cosmo] = numpy.append(paramDerivStack[cosmo], paramDerivTemp * paramDerivs[cosmo][spectrumType][polComb][:lmax-1])

        try:
            invCovDotParamDerivs[cosmo] = scipy.linalg.cho_solve(covMatrix, paramDerivStack[cosmo])
        except:
            print('Warning inverse covariance problem with ' + cosmo + ' ' + spectrumType)
            invCovDotParamDerivs[cosmo] = full((lmax-2), nan)

    return invCovDotParamDerivs, paramDerivStack


def getBiasVectorGaussian(powersFid, paramDerivs, cmbNoiseSpectra, deflectionNoises, cosmoParams, sysSpectrum,\
                            spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                            polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                            lmax = 5000):

    covs = getGaussianCov(powersFid = powersFid, \
                          cmbNoiseSpectra = cmbNoiseSpectra, \
                          deflectionNoises = deflectionNoises, \
                          polCombsToUse = polCombsToUse, \
                          lmax = lmax+1)

    ## Copy data structure of covs to invcovs (values will be overwritten)
    invCovs = covs.copy()
    ell = powersFid[list(powersFid.keys())[0]]['l'][:lmax-1]
    nElls = len(ell)
    nPolCombsToUse = len(polCombsToUse)
    nPars = len(cosmoParams)
    biasVectorContribs = twodDict(spectrumTypes, cosmoParams)
    paramDerivArray = twodDict(cosmoParams, spectrumTypes)
    biasVector = dict()

    sysSpectrumArray = dict()
    for spectrumType in spectrumTypes:
        biasVector[spectrumType] = numpy.zeros(nPars)
        sysSpectrumArray[spectrumType] = numpy.zeros((nPolCombsToUse, nElls))

        for l in range(nElls):
            try:
                invCovs[spectrumType][:, :, l] = linalg.inv(covs[spectrumType][:, :, l])
            except:
                print ("warning, cov inversion problem " , spectrumType, ell[l])
                invCovs[spectrumType][:, :, l] = numpy.full((nPolCombsToUse, nPolCombsToUse), numpy.nan)

        for cp1, cosmo1 in enumerate(cosmoParams):
            paramDerivArray[cosmo1][spectrumType] = numpy.zeros((nPolCombsToUse, nElls))

            for pc, polComb in enumerate(polCombsToUse):
                if polComb == 'cl_dd':
                    #deflection as a special case; use same data regardless of spectrumType
                    if 'cl_dd' in polCombsToUse:
                        pcDD = polCombsToUse.index('cl_dd')
                        paramDerivArray[cosmo1][spectrumType][pcDD, :] = \
                            paramDerivs[cosmo1]['lensing']['cl_dd'][:lmax-1]
                        sysSpectrumArray[spectrumType][pcDD, :] = \
                            sysSpectrum['lensing']['cl_dd'][:lmax-1]
                else:
                    paramDerivArray[cosmo1][spectrumType][pc, :] = \
                        paramDerivs[cosmo1][spectrumType][polComb][:lmax-1]
                    sysSpectrumArray[spectrumType][pc, :] = \
                        sysSpectrum[spectrumType][polComb][:lmax-1]

        for cp1, cosmo1 in enumerate(cosmoParams):
            biasVectorContribs[spectrumType][cosmo1] = \
                numpy.einsum('ik,ijk,jk->k', \
                           sysSpectrumArray[spectrumType], invCovs[spectrumType], paramDerivArray[cosmo1][spectrumType])
            biasVector[spectrumType][cp1] = sum(biasVectorContribs[spectrumType][cosmo1])

    return biasVector

def getBiasVectorNonGaussian(invCovDotParamDerivs, cosmoParams, sysSpectrum, polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                             lmax=5000, spectrumType = ['delensed']):
    sysSpectrumStack = numpy.empty(0)
    for pc, polComb in enumerate(polCombsToUse):
        if polComb == 'cl_dd':
            sysSpectrumStack = numpy.append(sysSpectrumStack, sysSpectrum['lensing'][polComb][:lmax-1])
        else:
            sysSpectrumStack = numpy.append(sysSpectrumStack, sysSpectrum[spectrumType][polComb][:lmax-1])

    nParams = len(cosmoParams)
    biasVector = numpy.zeros(nParams)
    for cp1, cosmo1 in enumerate(cosmoParams):
        biasVector[cp1] = sum(sysSpectrumStack * invCovDotParamDerivs[cosmo1])
    return biasVector

def getSecondPowerDerivWithParams(cosmoFid, stepSizes, polCombs, cmbNoiseSpectraK, deflectionNoisesK,
                            spectrumTypes = ['unlensed', 'lensed', 'delensed', 'lensing'],
                            lmax = 5000,
                            fileNameBase = 'testing', paramsToDifferentiate = None,
                            accuracy = 2.,
                            useClass = False,
                            doLensedWithCAMB = False,
                            classExecDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/',
                            classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/',
                            extraParams = dict()):

    if paramsToDifferentiate == None:
        paramsToDifferentiate = list(cosmoFid.keys())


    nParams = len(paramsToDifferentiate)
    oneSidedParams = ['DM_Pann']
    if 'mnu' in cosmoFid.keys():
        if cosmoFid['mnu'] < stepSizes['mnu']:
            oneSidedParams.append('mnu')

    derivsPlus = dict()
    derivsMinus = dict()

    for cosmo in paramsToDifferentiate:
        print('getting second deriv w.r.t. %s' %cosmo)
        cosmoPlus = cosmoFid.copy() #copy all params including those not being perturbed.
        cosmoMinus = cosmoFid.copy()

        cosmoPlus[cosmo] = cosmoPlus[cosmo] + stepSizes[cosmo]
        cosmoMinus[cosmo] = cosmoMinus[cosmo] - stepSizes[cosmo]

        derivsPlus[cosmo] = getPowerDerivWithParams(cosmoPlus, stepSizes, polCombs, cmbNoiseSpectraK, deflectionNoisesK,
                        spectrumTypes = spectrumTypes,
                        lmax = lmax,
                        fileNameBase = fileNameBase,
                        paramsToDifferentiate = paramsToDifferentiate,
                        accuracy = accuracy,
                        useClass = useClass,
                        doLensedWithCAMB = doLensedWithCAMB,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        extraParams = extraParams)


        #### For one-sided derivatives, use fiducial parameters for PowersMinus
        if cosmo in oneSidedParams:
            cosmoMinus = cosmoFid.copy()

        derivsMinus[cosmo] = getPowerDerivWithParams(cosmoMinus, stepSizes, polCombs, cmbNoiseSpectraK, deflectionNoisesK,
                        spectrumTypes = spectrumTypes,
                        lmax = lmax,
                        fileNameBase = fileNameBase,
                        paramsToDifferentiate = paramsToDifferentiate,
                        accuracy = accuracy,
                        useClass = useClass,
                        doLensedWithCAMB = doLensedWithCAMB,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        extraParams = extraParams)


    #PARAM DERIVATIVES
    polCombsTemp = polCombs.copy()
    if 'cl_dd' in polCombs:
        polCombsTemp.remove('cl_dd')
    #PARAM DERIVATIVES
    secondDerivs = fourdDict(paramsToDifferentiate, paramsToDifferentiate, spectrumTypes, polCombsTemp)

    for  cosmo1 in paramsToDifferentiate:


        #### Use this for one-sided derivatives (derivsMinus is derivsFid in this case)
        if cosmo1 in oneSidedParams:
            denom = stepSizes[cosmo1]
        else:
        #### Other parameters use two-sided derivatives
            denom = 2 * stepSizes[cosmo1]

        for cosmo2 in paramsToDifferentiate:
            for pc, polComb in enumerate(polCombs):
                if polComb == 'cl_dd':
                    if 'lensing' in spectrumTypes:
                        secondDerivs[cosmo1][cosmo2]['lensing'] = dict()
                        secondDerivs[cosmo1][cosmo2]['lensing']['cl_dd'] = \
                            (derivsPlus[cosmo1][cosmo2]['lensing']['cl_dd'] - derivsMinus[cosmo1][cosmo2]['lensing']['cl_dd']) / denom
                    else:
                        print('lensing must be in spectrum types to calculate cl_dd second derivatives.')
                else:
                    if 'unlensed' in spectrumTypes:
                        secondDerivs[cosmo1][cosmo2]['unlensed'][polComb] = \
                            (derivsPlus[cosmo1][cosmo2]['unlensed'][polComb] - derivsMinus[cosmo1][cosmo2]['unlensed'][polComb]) / denom
                    if 'lensed' in spectrumTypes:
                        secondDerivs[cosmo1][cosmo2]['lensed'][polComb] = \
                            (derivsPlus[cosmo1][cosmo2]['lensed'][polComb] - derivsMinus[cosmo1][cosmo2]['lensed'][polComb]) / denom
                    if 'delensed' in spectrumTypes:
                        secondDerivs[cosmo1][cosmo2]['delensed'][polComb] = \
                            (derivsPlus[cosmo1][cosmo2]['delensed'][polComb] - derivsMinus[cosmo1][cosmo2]['delensed'][polComb]) / denom

    return secondDerivs

def getGaussianDoubletDALI(powersFid, paramDerivs, secondDerivs, cmbNoiseSpectra, deflectionNoises, cosmoParams,\
                            spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                            polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                            ellsToUse = None, \
                            lmax = None):

    ellsToUse, lmax = createEllsToUseDict(ellsToUse = ellsToUse, \
                        lmax = lmax, \
                        polCombsToUse = polCombsToUse)

    covs = getGaussianCov(powersFid = powersFid, \
                          cmbNoiseSpectra = cmbNoiseSpectra, \
                          deflectionNoises = deflectionNoises, \
                          polCombsToUse = polCombsToUse, \
                          lmax = lmax+1)

    ## Copy data structure of covs to invcovs (values will be overwritten)
    invCovs = covs.copy()
    #ell = powersFid['unlensed']['l']
    ell = powersFid[list(powersFid.keys())[0]]['l'][:lmax-1]
    nElls = len(ell)
    nPolCombsToUse = len(polCombsToUse)
    nPars = len(cosmoParams)
    DALI3Contribs = fourdDict(spectrumTypes, cosmoParams, cosmoParams, cosmoParams)
    DALI4Contribs = fivedDict(spectrumTypes, cosmoParams, cosmoParams, cosmoParams, cosmoParams)
    paramDerivArray = twodDict(cosmoParams, spectrumTypes)
    paramDerivToUse = twodDict(cosmoParams, spectrumTypes)
    secondDerivArray = threedDict(cosmoParams, cosmoParams, spectrumTypes)
    secondDerivToUse = threedDict(cosmoParams, cosmoParams, spectrumTypes)
    DALI3 = dict()
    DALI4 = dict()

    for spectrumType in spectrumTypes:
        DALI3[spectrumType] = numpy.zeros((nPars, nPars, nPars))
        DALI4[spectrumType] = numpy.zeros((nPars, nPars, nPars, nPars))
        for l in range(nElls):
            try:
                invCovs[spectrumType][:, :, l] = linalg.inv(covs[spectrumType][:, :, l])
            except:
                print("warning, cov inversion problem " , spectrumType, ell[l])
                invCovs[spectrumType][:, :, l] = numpy.full((nPolCombsToUse, nPolCombsToUse), numpy.nan)

        for cp1, cosmo1 in enumerate(cosmoParams):
            paramDerivArray[cosmo1][spectrumType] = numpy.zeros((nPolCombsToUse, nElls))
            paramDerivToUse[cosmo1][spectrumType] = numpy.zeros((nPolCombsToUse, nElls))

            for pc, polComb in enumerate(polCombsToUse):
                if polComb == 'cl_dd':
                    #deflection as a special case; use same data regardless of spectrumType
                    if 'cl_dd' in polCombsToUse:
                        pcDD = polCombsToUse.index('cl_dd')
                        paramDerivArray[cosmo1][spectrumType][pcDD, :] = \
                            paramDerivs[cosmo1]['lensing']['cl_dd'][:lmax-1]
                else:
                    paramDerivArray[cosmo1][spectrumType][pc, :] = \
                        paramDerivs[cosmo1][spectrumType][polComb][:lmax-1]
                paramDerivToUse[cosmo1][spectrumType][pc, ellsToUse[polComb][0]-2 : ellsToUse[polComb][1]-1] = 1.
                paramDerivArray[cosmo1][spectrumType][pc, :] = \
                    paramDerivArray[cosmo1][spectrumType][pc, :] * paramDerivToUse[cosmo1][spectrumType][pc, :]

            for cp2, cosmo2 in enumerate(cosmoParams):
                secondDerivArray[cosmo1][cosmo2][spectrumType] = numpy.zeros((nPolCombsToUse, nElls))
                secondDerivToUse[cosmo1][cosmo2][spectrumType] = numpy.zeros((nPolCombsToUse, nElls))

                for pc, polComb in enumerate(polCombsToUse):
                    if polComb == 'cl_dd':
                        #deflection as a special case; use same data regardless of spectrumType
                        if 'cl_dd' in polCombsToUse:
                            pcDD = polCombsToUse.index('cl_dd')
                            secondDerivArray[cosmo1][cosmo2][spectrumType][pcDD, :] = \
                                secondDerivs[cosmo1][cosmo2]['lensing']['cl_dd'][:lmax-1]
                    else:
                        secondDerivArray[cosmo1][cosmo2][spectrumType][pc, :] = \
                            secondDerivs[cosmo1][cosmo2][spectrumType][polComb][:lmax-1]
                    secondDerivToUse[cosmo1][cosmo2][spectrumType][pc, ellsToUse[polComb][0]-2 : ellsToUse[polComb][1]-1] = 1.
                    secondDerivArray[cosmo1][cosmo2][spectrumType][pc, :] = \
                        secondDerivArray[cosmo1][cosmo2][spectrumType][pc, :] * secondDerivToUse[cosmo1][cosmo2][spectrumType][pc, :]



        for cp1, cosmo1 in enumerate(cosmoParams):
            for cp2, cosmo2 in enumerate(cosmoParams):
                for cp3, cosmo3 in enumerate(cosmoParams):
                    DALI3Contribs[spectrumType][cosmo1][cosmo2][cosmo3] = \
                        numpy.einsum('ik,ijk,jk->k', \
                                   secondDerivArray[cosmo1][cosmo2][spectrumType], invCovs[spectrumType], paramDerivArray[cosmo3][spectrumType])

                    DALI3[spectrumType][cp1, cp2, cp3] = sum(DALI3Contribs[spectrumType][cosmo1][cosmo2][cosmo3])

                    for cp4, cosmo4 in enumerate(cosmoParams):
                        DALI4Contribs[spectrumType][cosmo1][cosmo2][cosmo3][cosmo4] = \
                            numpy.einsum('ik,ijk,jk->k', \
                                       secondDerivArray[cosmo1][cosmo2][spectrumType], invCovs[spectrumType], secondDerivArray[cosmo3][cosmo4][spectrumType])

                        DALI4[spectrumType][cp1, cp2, cp3, cp4] = sum(DALI4Contribs[spectrumType][cosmo1][cosmo2][cosmo3][cosmo4])

    return DALI3, DALI4


def choleskyInvCovDotSecondDerivsNG(powersFid, \
                                cmbNoiseSpectra, \
                                deflectionNoiseSpectra, \
                                dCldCLd,
                                paramDerivs, \
                                secondDerivs,
                                cosmoParams, \
                                dCldCLu = None, \
                                ellsToUse = None, \
                                lmax = None, \
                                polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                                spectrumType = 'delensed'):

    ellsToUse, lmax = createEllsToUseDict(ellsToUse = ellsToUse, \
                        lmax = lmax, \
                        polCombsToUse = polCombsToUse)
    if 'lmaxCov' in list(ellsToUse.keys()) and ellsToUse['lmaxCov'] > lmax:
        lmax = ellsToUse['lmaxCov']

    covMatrix = getNonGaussianCov(powersFid = powersFid, \
                                cmbNoiseSpectra = cmbNoiseSpectra, \
                                deflectionNoiseSpectra = deflectionNoiseSpectra, \
                                dCldCLd = dCldCLd, \
                                #includeUnlensedSpectraDerivatives = includeUnlensedSpectraDerivatives, \
                                dCldCLu = dCldCLu, \
                                lmax = lmax, \
                                polCombsToUse = polCombsToUse, \
                                spectrumType = spectrumType)

    try:
        covMatrix = scipy.linalg.cho_factor(covMatrix, overwrite_a = True)
    except:
        print('Warning Cholesky decomposition failed')
        #covMatrix = scipy.linalg.cho_factor(numpy.eye(len(polCombsToUse)*(lmax-2)))
        covMatrix = scipy.linalg.cho_factor(numpy.eye(len(polCombsToUse)*(lmax-1)))

    secondDerivStack = twodDict(cosmoParams, cosmoParams)
    invCovDotSecondDerivs = twodDict(cosmoParams, cosmoParams)

    for cosmo1 in cosmoParams:
        for cosmo2 in cosmoParams:
            secondDerivStack[cosmo1][cosmo2] = numpy.empty(0)
            for pc, polComb in enumerate(polCombsToUse):
                secondDerivTemp = numpy.zeros(lmax-1)
                secondDerivTemp[ellsToUse[polComb][0]-2 : ellsToUse[polComb][1]-1] = 1.
                if polComb == 'cl_dd':
                    secondDerivStack[cosmo1][cosmo2] = numpy.append(secondDerivStack[cosmo1][cosmo2], secondDerivTemp * secondDerivs[cosmo1][cosmo2]['lensing'][polComb][:lmax-1])
                    #paramDerivStack[cosmo] = numpy.append(paramDerivStack[cosmo], paramDerivs[cosmo]['lensing'][polComb][:lmax-2])
                else:
                    secondDerivStack[cosmo1][cosmo2] = numpy.append(secondDerivStack[cosmo1][cosmo2], secondDerivTemp * secondDerivs[cosmo1][cosmo2][spectrumType][polComb][:lmax-1])
                    #paramDerivStack[cosmo] = numpy.append(paramDerivStack[cosmo], paramDerivs[cosmo][spectrumType][polComb][:lmax-2])

            try:
                invCovDotSecondDerivs[cosmo1][cosmo2] = scipy.linalg.cho_solve(covMatrix, secondDerivStack[cosmo1][cosmo2])
            except:
                print('Warning inverse covariance problem with ' + cosmo1 + ' ' + cosmo2 + ' ' + spectrumType)
                #invCovDotParamDerivs[cosmo] = full(len(polCombsToUse)*(lmax-1), nan)
                invCovDotSecondDerivs[cosmo] = full((lmax-2), nan)

    return invCovDotSecondDerivs, secondDerivStack

def getNonGaussianDoubletDALI(invCovDotParamDerivs, invCovDotSecondDerivs, secondDerivStack, cosmoParams):
    nParams = len(cosmoParams)
    DALI3 = numpy.zeros((nParams,nParams,nParams))
    DALI4 = numpy.zeros((nParams,nParams,nParams,nParams))
    for cp1, cosmo1 in enumerate(cosmoParams):
        for cp2, cosmo2 in enumerate(cosmoParams):
            for cp3, cosmo3 in enumerate(cosmoParams):
                DALI3[cp1, cp2, cp3] = sum(secondDerivStack[cosmo1][cosmo2] * invCovDotParamDerivs[cosmo3])
                for cp4, cosmo4 in enumerate(cosmoParams):
                    DALI4[cp1, cp2, cp3, cp4] = sum(secondDerivStack[cosmo1][cosmo2] * invCovDotSecondDerivs[cosmo3][cosmo4])
    return DALI3, DALI4

def createEllsToUseDict(ellsToUse = None, \
                        lmax = None, \
                        polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd']):

    if ellsToUse is not None and lmax is not None:
        # give an error
        raise ValueError("User passed both ellsToUse and lmax.")
    elif ellsToUse is not None and lmax is None:
        print('Using ellsToUse passed by user.')
        lmax = 2
    elif ellsToUse is None and lmax is not None:
        print('Generating ellsToUse from lmax passed by user.')
        ellsToUse = {'cl_TT': [2, lmax], 'cl_TE': [2, lmax], 'cl_EE': [2, lmax], 'cl_BB': [2,lmax], 'cl_dd': [2, lmax], 'lmaxCov': lmax}
    else:
        print('Using default values for ellsToUse.')
        ellsToUse = {'cl_TT': [2, 5000], 'cl_TE': [2, 5000], 'cl_EE': [2, 5000], 'cl_BB': [2,5000], 'cl_dd': [2, 5000], 'lmaxCov': 5000}
        lmax = 5000

    for polComb in polCombsToUse:
        if polComb not in list(ellsToUse.keys()):
            # give an error
            raise ValueError(polComb + " is not in ellsToUse.")

    for polComb in [x for x in list(ellsToUse.keys()) if x != 'lmaxCov']:
        # check that lmins are at least 2; otherwise set to 2
        if ellsToUse[polComb][0] < 2:
            print('Setting lmin for ' + polComb + ' to 2.')
            ellsToUse[polComb][0] = 2
        # pick out the largest lmax
        if ellsToUse[polComb][1] > lmax:
            lmax = ellsToUse[polComb][1]

    return ellsToUse, lmax

def is_float(string):
    try:
        return float(string)
    except ValueError:
        return False