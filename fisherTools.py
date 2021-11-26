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

def getDelensedSpectra(unlensedSpectra, lensedSpectra, cmbNoiseSpectra, lensingSpectra, deflectionNoiseSpectra, \
                           lenspowerDataDir = os.path.dirname(os.path.abspath(__file__)) + '/../data/', lmaxToWrite = 3000, \
                           lenspowerExecDir = os.path.dirname(os.path.abspath(__file__)) + '/../c/', \
                           fileNameBase = 'testing' ):

    ellsToWrite = arange(lmaxToWrite + 1)

    maxInd = (array(where(unlensedSpectra['l'] <= lmaxToWrite))).max() + 1 #+1 to go to 3k

    savetxt(lenspowerDataDir + '/unlensed_power_cmb_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], unlensedSpectra['cl_TT'][:maxInd ] ), append([0,0], unlensedSpectra['cl_EE'][:maxInd ] ), append([0,0], unlensedSpectra['cl_TE'][:maxInd ] ), append([0,0], unlensedSpectra['cl_BB'][:maxInd ] ) ] ))
    savetxt(lenspowerDataDir + '/lensed_power_cmb_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], lensedSpectra['cl_TT'][:maxInd ] ), append([0,0], lensedSpectra['cl_EE'][:maxInd ] ), append([0,0], lensedSpectra['cl_TE'][:maxInd ] ), append([0,0], lensedSpectra['cl_BB'][:maxInd ] ) ] ))

    val = cmbNoiseSpectra['cl_TT'][0]
    savetxt(lenspowerDataDir + '/noise_power_T_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val,val], cmbNoiseSpectra['cl_TT'][:maxInd] ) ] ))

    val = cmbNoiseSpectra['cl_EE'][0]
    savetxt(lenspowerDataDir + '/noise_power_P_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val,val], cmbNoiseSpectra['cl_EE'][:maxInd] ) ] ))


    savetxt(lenspowerDataDir + '/power_defl_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], lensingSpectra['cl_dd'][:maxInd ] ) ] ))


    val = deflectionNoiseSpectra[0]
    savetxt(lenspowerDataDir + '/noise_power_defl_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val, val], deflectionNoiseSpectra[:maxInd ] ) ] ))


    execString = "cd " + lenspowerExecDir + '; ./lenspower ' + '"' + lenspowerDataDir + '" ' +  fileNameBase + ' power'
    print('*** running: ', execString)
    os.system(execString)

    delensedData = loadtxt(lenspowerDataDir + '/delensed_power_cmb_exact_' + fileNameBase + '.txt')

    return delensedData[2:, [1,2,3,4]]


def getEEReconstructionNoise(residualBB, cmbNoiseSpectra, lensingSpectra, deflectionNoiseSpectra, reionizationLensingSpectra, \
                           lenspowerDataDir = os.path.dirname(os.path.abspath(__file__)) + '../data/', lmaxToWrite = 3000, \
                           lenspowerExecDir = os.path.dirname(os.path.abspath(__file__)) + '../c/', \
                           fileNameBase = 'testing' ):

    #note, C code arrays start at  l = 0 so prepend those values.
    ellsToWrite = arange(lmaxToWrite + 1)
    maxInd = (array(where(cmbNoiseSpectra['l'] <= lmaxToWrite))).max() + 1 #+1 to go to 3k
    savetxt(lenspowerDataDir + '/residual_c_ell_B_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], residualBB[:maxInd ] ) ] ))
    val = cmbNoiseSpectra['cl_TT'][0]
    savetxt(lenspowerDataDir + '/noise_power_T_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val,val], cmbNoiseSpectra['cl_TT'][:maxInd] ) ] ))
    val = cmbNoiseSpectra['cl_EE'][0]
    savetxt(lenspowerDataDir + '/noise_power_P_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val,val], cmbNoiseSpectra['cl_EE'][:maxInd] ) ] ))
    savetxt(lenspowerDataDir + '/power_defl_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], lensingSpectra['cl_dd'][:maxInd ] ) ] ))
    val = deflectionNoiseSpectra[0]
    savetxt(lenspowerDataDir + '/noise_power_defl_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val, val], deflectionNoiseSpectra[:maxInd ] ) ] ))
    savetxt(lenspowerDataDir + '/power_phi_reionization_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], reionizationLensingSpectra['cl_phiphi'][:maxInd ] ) ] ))
    savetxt(lenspowerDataDir + '/power_phi_cross_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], reionizationLensingSpectra['cl_phicross'][:maxInd ] ) ] ))

    execString = "cd " + lenspowerExecDir + '; ./lenspower ' + '"' + lenspowerDataDir + '" ' +  fileNameBase + ' rec'
    print('*** running: ', execString)
    os.system(execString)

    reconstructionData = loadtxt(lenspowerDataDir + '/EE_reconstruction_noise_' + fileNameBase + '.txt')
    return reconstructionData[2:, 1]


def getEEReconstructionNoisePatchyTau(screenedBB, cmbNoiseSpectra, patchyTauSpectra, patchyTauNoise, reionizationTauSpectra, meanTau = 0.06, \
                           lenspowerDataDir = os.path.dirname(os.path.abspath(__file__)) + '../data/', lmaxToWrite = 3000, \
                           lenspowerExecDir = os.path.dirname(os.path.abspath(__file__)) + '../c/', \
                           fileNameBase = 'testing' ):

    #note, C code arrays start at  l = 0 so prepend those values.
    ellsToWrite = arange(lmaxToWrite + 1)
    maxInd = (array(where(cmbNoiseSpectra['l'] <= lmaxToWrite))).max() + 1 #+1 to go to 3k
    savetxt(lenspowerDataDir + '/screened_c_ell_B_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], screenedBB[:maxInd ] ) ] ))
    val = cmbNoiseSpectra['cl_TT'][0]
    savetxt(lenspowerDataDir + '/noise_power_T_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val,val], cmbNoiseSpectra['cl_TT'][:maxInd] ) ] ))
    val = cmbNoiseSpectra['cl_EE'][0]
    savetxt(lenspowerDataDir + '/noise_power_P_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val,val], cmbNoiseSpectra['cl_EE'][:maxInd] ) ] ))
    savetxt(lenspowerDataDir + '/c_ell_tau_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], patchyTauSpectra['cl_tautau'][:maxInd ] ) ] ))
    val = patchyTauNoise[0]
    savetxt(lenspowerDataDir + '/n_ell_tau_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val, val], patchyTauNoise[:maxInd ] ) ] ))
    savetxt(lenspowerDataDir + '/power_tau_reionization_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], reionizationTauSpectra['cl_tautau'][:maxInd ] ) ] ))
    savetxt(lenspowerDataDir + '/power_tau_cross_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], reionizationTauSpectra['cl_taucross'][:maxInd ] ) ] ))

    execString = "cd " + lenspowerExecDir + '; ./lenspower ' + '"' + lenspowerDataDir + '" ' +  fileNameBase + ' patchy'
    print('*** running: ', execString)
    os.system(execString)
    EEReconstructionNoise = loadtxt(lenspowerDataDir + '/EE_reconstruction_noise_patchy_tau_' + fileNameBase + '.txt')

    #Need to multiply by e^(2 tau) since it is not included in C code
    return EEReconstructionNoise[2:, 1] * numpy.exp(2.*meanTau)


def delensedArrayToDict(inArr):

    ell = 2. + arange(inArr.shape[0])*1.
    return {'l' : ell, \
                'cl_TT' : inArr[:, 0], \
                'cl_EE' : inArr[:, 1], \
                'cl_TE' : inArr[:, 2], \
                'cl_BB' : inArr[:, 3]}

def getPowerDerivWithParams(cosmoFid, stepSizes, polCombs, cmbNoiseSpectraK, deflectionNoisesK,
                            spectrumTypes = ['unlensed', 'lensed', 'delensed', 'lensing'],
                            lmax = 5000,
                            lenspowerDataDir = os.path.dirname(os.path.abspath(__file__)) + '../data',
                            fileNameBase = 'testing', paramsToDifferentiate = None,
                            accuracy = 2., wantMatterPower = False, redshifts = None,
                            useClass = False,
                            classExecDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/',
                            classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/'):

    if paramsToDifferentiate == None:
        paramsToDifferentiate = list(cosmoFid.keys())


    nParams = len(paramsToDifferentiate)
    oneSidedParams = ['DM_Pann']
    oneSidedParamsIso = ['c_ad_cdi', 'c_ad_bi', 'c_ad_nid', 'c_ad_niv', \
                        'c_bi_cdi', 'c_bi_nid', 'c_bi_niv', \
                        'c_cdi_nid', 'c_cdi_niv', \
                        'c_nid_niv'] #could be + or - 1

    cambPowersPlus = dict()
    cambPowersMinus = dict()
    delensedPowersPlus = dict()
    delensedPowersMinus = dict()


    for cosmo in paramsToDifferentiate:
        print(('getting deriv w.r.t. %s' %cosmo))
        cosmoPlus = cosmoFid.copy() #copy all params including those not being perturbed.
        cosmoMinus = cosmoFid.copy()

        cosmoPlus[cosmo] = cosmoPlus[cosmo] + stepSizes[cosmo]
        cosmoMinus[cosmo] = cosmoMinus[cosmo] - stepSizes[cosmo]

        if useClass is True:
            cambPowersPlus[cosmo], junk = classWrapTools.class_generate_data(cosmoPlus,
                        rootName = fileNameBase,
                        cmbNoise = cmbNoiseSpectraK,
                        deflectionNoise = deflectionNoisesK,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        lmax = lmax)
        else:
            cambPowersPlus[cosmo] = cambWrapTools.getPyCambPowerSpectra(cosmoPlus, accuracy = accuracy, lmaxToWrite = lmax, wantMatterPower = wantMatterPower, redshifts = redshifts)


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
            cambPowersMinus[cosmo], junk = classWrapTools.class_generate_data(cosmoMinus,
                        rootName = fileNameBase,
                        cmbNoise = cmbNoiseSpectraK,
                        deflectionNoise = deflectionNoisesK,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        lmax = lmax)
        else:
            cambPowersMinus[cosmo] = cambWrapTools.getPyCambPowerSpectra(cosmoMinus, accuracy = accuracy, lmaxToWrite = lmax, wantMatterPower = wantMatterPower, redshifts = redshifts)



        # for k in range(iMin, iMax):
        if useClass is False:
            if 'delensed' in spectrumTypes:
                delensedPowersPlus[cosmo] = \
                    delensedArrayToDict( getDelensedSpectra(cambPowersPlus[cosmo]['unlensed'], \
                                                                cambPowersPlus[cosmo]['lensed'], \
                                                                cmbNoiseSpectraK,\
                                                                cambPowersPlus[cosmo]['lensing'], \
                                                                deflectionNoisesK, \
                                                                lmaxToWrite = lmax, \
                                                                lenspowerDataDir = lenspowerDataDir, \
                                                                fileNameBase = fileNameBase) )

                delensedPowersMinus[cosmo] = \
                    delensedArrayToDict( getDelensedSpectra(cambPowersMinus[cosmo]['unlensed'], \
                                                                cambPowersMinus[cosmo]['lensed'], \
                                                                cmbNoiseSpectraK,\
                                                                cambPowersMinus[cosmo]['lensing'], \
                                                                deflectionNoisesK, \
                                                                lmaxToWrite = lmax, \
                                                                lenspowerDataDir = lenspowerDataDir,\
                                                                fileNameBase = fileNameBase) )

    #PARAM DERIVATIVES
    paramDerivs = threedDict(paramsToDifferentiate, spectrumTypes, polCombs)

    for  cosmo in paramsToDifferentiate:


        #### Use this for one-sided derivatives (PowersMinus is PowersFid in this case)
        if cosmo in oneSidedParams:
            denom = stepSizes[cosmo]
        else:
        #### Other parameters use two-sided derivatives
            denom = 2 * stepSizes[cosmo]

        for pc, polComb in enumerate(polCombs):
            if 'unlensed' in spectrumTypes:
                paramDerivs[cosmo]['unlensed'][polComb] = \
                    (cambPowersPlus[cosmo]['unlensed'][polComb] - cambPowersMinus[cosmo]['unlensed'][polComb]) / denom
            if 'lensed' in spectrumTypes:
                paramDerivs[cosmo]['lensed'][polComb] = \
                    (cambPowersPlus[cosmo]['lensed'][polComb] - cambPowersMinus[cosmo]['lensed'][polComb]) / denom
            if 'delensed' in spectrumTypes:
                if useClass is True:
                    paramDerivs[cosmo]['delensed'][polComb] = \
                        (cambPowersPlus[cosmo]['delensed'][polComb] - cambPowersMinus[cosmo]['delensed'][polComb]) / denom
                else:
                    paramDerivs[cosmo]['delensed'][polComb] = \
                        (delensedPowersPlus[cosmo][polComb] - delensedPowersMinus[cosmo][polComb]) / denom
        if 'lensing' in spectrumTypes:
            paramDerivs[cosmo]['lensing'] = dict()
            paramDerivs[cosmo]['lensing']['cl_dd'] = \
                (cambPowersPlus[cosmo]['lensing']['cl_dd'] - cambPowersMinus[cosmo]['lensing']['cl_dd']) / denom
        if 'matter' in spectrumTypes:
            paramDerivs[cosmo]['matter'] = dict()
            paramDerivs[cosmo]['matter']['PK'] = \
                (cambPowersPlus[cosmo]['matter']['PK'] - cambPowersMinus[cosmo]['matter']['PK']) / denom



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


def getDelensedPowerSpectraDerivs(unlensedSpectra, lensedSpectra, cmbNoiseSpectra, lensingSpectra, deflectionNoiseSpectra,
                                  lenspowerDataDir = os.path.dirname(os.path.abspath(__file__)) + '/../data/',
                                  lmaxToWrite = 3000,
                                  lenspowerExecDir = os.path.dirname(os.path.abspath(__file__)) + '../c/',
                                  fileNameBase = 'testing',
                                  includeNoiseDerivs = False ):

    #note, C code arrays start at  l = 0 so prepend those values.
    ellsToWrite = arange(lmaxToWrite + 1)
    maxInd = (array(where(unlensedSpectra['l'] <= lmaxToWrite))).max() + 1 #+1 to go to 3k
    savetxt(lenspowerDataDir + '/unlensed_power_cmb_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], unlensedSpectra['cl_TT'][:maxInd ] ), append([0,0], unlensedSpectra['cl_EE'][:maxInd ] ), append([0,0], unlensedSpectra['cl_TE'][:maxInd ] ), append([0,0], unlensedSpectra['cl_BB'][:maxInd ] ) ] ))
    savetxt(lenspowerDataDir + '/lensed_power_cmb_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], lensedSpectra['cl_TT'][:maxInd ] ), append([0,0], lensedSpectra['cl_EE'][:maxInd ] ), append([0,0], lensedSpectra['cl_TE'][:maxInd ] ), append([0,0], lensedSpectra['cl_BB'][:maxInd ] ) ] ))
    val = cmbNoiseSpectra['cl_TT'][0]
    savetxt(lenspowerDataDir + '/noise_power_T_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val,val], cmbNoiseSpectra['cl_TT'][:maxInd] ) ] ))
    val = cmbNoiseSpectra['cl_EE'][0]
    savetxt(lenspowerDataDir + '/noise_power_P_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val,val], cmbNoiseSpectra['cl_EE'][:maxInd] ) ] ))
    savetxt(lenspowerDataDir + '/power_defl_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([0,0], lensingSpectra['cl_dd'][:maxInd ] ) ] ))
    val = deflectionNoiseSpectra[0]
    savetxt(lenspowerDataDir + '/noise_power_defl_' + fileNameBase + '.txt', \
                transpose([ellsToWrite, append([val, val], deflectionNoiseSpectra[:maxInd ] ) ] ))
    execString = "cd " + lenspowerExecDir + '; ./lenspower ' + '"' + lenspowerDataDir + '" ' +  fileNameBase + ' cov'
    print(('*** running: ', execString))
    os.system(execString)

    spectrumTypes = ['lensed', 'delensed']
    polCombs = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_BB', 'cl_dd']
    dCldCLd = twodDict(spectrumTypes, polCombs)

    dCldCLd['lensed']['cl_TT'] = loadtxt(lenspowerDataDir + '/dClTdCLd_lensed_' + fileNameBase + '.txt')
    dCldCLd['lensed']['cl_EE'] = loadtxt(lenspowerDataDir + '/dClEdCLd_lensed_' + fileNameBase + '.txt')
    dCldCLd['lensed']['cl_TE'] = loadtxt(lenspowerDataDir + '/dClXdCLd_lensed_' + fileNameBase + '.txt')
    dCldCLd['lensed']['cl_dd'] = numpy.eye(lmaxToWrite+1)

    dCldCLd['delensed']['cl_TT'] = loadtxt(lenspowerDataDir + '/dClTdCLd_delensed_' + fileNameBase + '.txt')
    dCldCLd['delensed']['cl_EE'] = loadtxt(lenspowerDataDir + '/dClEdCLd_delensed_' + fileNameBase + '.txt')
    dCldCLd['delensed']['cl_TE'] = loadtxt(lenspowerDataDir + '/dClXdCLd_delensed_' + fileNameBase + '.txt')
    dCldCLd['delensed']['cl_dd'] = numpy.eye(lmaxToWrite+1)

    if includeNoiseDerivs == True:
        dCldCLd['delensed']['Nl_T'] = loadtxt(lenspowerDataDir + '/dNlTdCLd_' + fileNameBase + '.txt')
        dCldCLd['delensed']['Nl_P'] = loadtxt(lenspowerDataDir + '/dNlPdCLd_' + fileNameBase + '.txt')

    return dCldCLd



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





def getInvCovDotParamDerivs(powersFid, \
                                cmbNoiseSpectra, \
                                deflectionNoiseSpectra, \
                                paramDerivs, \
                                cosmoFid, \
                                lenspowerDataDir = os.path.dirname(os.path.abspath(__file__)) + '../data', \
                                lmax = 3000, \
                                lenspowerExecDir = os.path.dirname(os.path.abspath(__file__)) + '../c/', \
                                fileNameBase = 'testing', \
                                doNonGaussian = False, \
                                includeNoiseCovs = False, \
                                polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd']):

    gaussianCov = getGaussianCov(powersFid = powersFid, \
                            cmbNoiseSpectra = cmbNoiseSpectra, \
                            deflectionNoises = deflectionNoiseSpectra, \
                            polCombsToUse = polCombsToUse, \
                            lmax = lmax)

    spectrumTypes = ['unlensed', 'lensed', 'delensed']
    gaussianity = ['gaussian', 'nonGaussian']
    cosmoParams = list(cosmoFid.keys())
    cov = fourdDict(gaussianity, spectrumTypes, polCombsToUse, polCombsToUse)
    choleskyCovMatrix = twodDict(gaussianity, spectrumTypes)
    invCovDotParamDerivs = threedDict(cosmoParams, gaussianity, spectrumTypes)
    paramDerivStack = twodDict(cosmoParams, spectrumTypes)


    for gauss in gaussianity:
        for spectrumType in spectrumTypes:
            for pc1, polComb1 in enumerate(polCombsToUse):
                for pc2, polComb2 in enumerate(polCombsToUse):
                    if pc2 <= pc1:
                        cov[gauss][spectrumType][polComb1][polComb2] = numpy.diag(gaussianCov[spectrumType][pc1, pc2])

    if doNonGaussian:
        dCldCLd = getDelensedPowerSpectraDerivs(unlensedSpectra = powersFid['unlensed'], \
                                                lensedSpectra = powersFid['lensed'], \
                                                cmbNoiseSpectra = cmbNoiseSpectra, \
                                                lensingSpectra = powersFid['lensing'], \
                                                deflectionNoiseSpectra = deflectionNoiseSpectra, \
                                                lenspowerDataDir = lenspowerDataDir, \
                                                lmaxToWrite = lmax, \
                                                lenspowerExecDir = lenspowerExecDir, \
                                                fileNameBase = fileNameBase,
                                                includeNoiseDerivs = includeNoiseCovs)

        ell = powersFid['unlensed']['l']
        nElls = ell.shape
        oneOver2ellPlusOne =  1. / (2. * ell[:lmax-2] + 1.)
        deflCov = 2*oneOver2ellPlusOne * (powersFid['lensing']['cl_dd'][:lmax-2])**2
        nonGaussianSpectrumTypes = ['lensed', 'delensed']

        if includeNoiseCovs == True:
            dCldCLd['delensed']['cl_TT'] += dCldCLd['delensed']['Nl_T']
            dCldCLd['delensed']['cl_EE'] += dCldCLd['delensed']['Nl_P']

        for spectrumType in nonGaussianSpectrumTypes:
            print('Computing', spectrumType, 'non-Gaussian covariances...')
            for pc1, polComb1 in enumerate(polCombsToUse):
                for pc2, polComb2 in enumerate(polCombsToUse):
                    if pc2 <= pc1:
                        print('Computing covaraince for', polComb1, 'and', polComb2)
                        cov['nonGaussian'][spectrumType][polComb1][polComb2] += numpy.tensordot(dCldCLd[spectrumType][polComb1][2:lmax,2:lmax], numpy.tensordot(numpy.diag(deflCov), dCldCLd[spectrumType][polComb2][2:lmax,2:lmax], axes = (1,1)), axes = (1,0))

    for gauss in gaussianity:
        for spectrumType in spectrumTypes:
            try:
                choleskyCovMatrix = scipy.linalg.cho_factor( \
                                    numpy.bmat( [ [cov[gauss][spectrumType]['cl_TT']['cl_TT'], numpy.transpose(cov[gauss][spectrumType]['cl_TE']['cl_TT']), numpy.transpose(cov[gauss][spectrumType]['cl_EE']['cl_TT']), numpy.transpose(cov[gauss][spectrumType]['cl_dd']['cl_TT']) ], \
                                    [cov[gauss][spectrumType]['cl_TE']['cl_TT'], cov[gauss][spectrumType]['cl_TE']['cl_TE'], numpy.transpose(cov[gauss][spectrumType]['cl_EE']['cl_TE']), numpy.transpose(cov[gauss][spectrumType]['cl_dd']['cl_TE'])], \
                                    [cov[gauss][spectrumType]['cl_EE']['cl_TT'], cov[gauss][spectrumType]['cl_EE']['cl_TE'], cov[gauss][spectrumType]['cl_EE']['cl_EE'], numpy.transpose(cov[gauss][spectrumType]['cl_dd']['cl_EE'])], \
                                    [cov[gauss][spectrumType]['cl_dd']['cl_TT'], cov[gauss][spectrumType]['cl_dd']['cl_TE'], cov[gauss][spectrumType]['cl_dd']['cl_EE'], cov[gauss][spectrumType]['cl_dd']['cl_dd'] ]] ) )
            except:
                print('Warning Cholesky decomposition failed for ' + gauss + ' ' + spectrumType)
                choleskyCovMatrix = scipy.linalg.cho_factor(numpy.eye(4*(lmax-2)))
            for cosmo in cosmoParams:
                paramDerivStack[cosmo][spectrumType] = numpy.bmat([[paramDerivs[cosmo][spectrumType]['cl_TT'][:lmax-2], paramDerivs[cosmo][spectrumType]['cl_TE'][:lmax-2], paramDerivs[cosmo][spectrumType]['cl_EE'][:lmax-2], paramDerivs[cosmo]['lensing']['cl_dd'][:lmax-2] ]])
                try:
                    invCovDotParamDerivs[cosmo][gauss][spectrumType] = scipy.linalg.cho_solve(choleskyCovMatrix, numpy.transpose(paramDerivStack[cosmo][spectrumType]))
                except:
                    print('Warning inverse covariance problem with ' + cosmo + ' ' + gauss + ' ' + spectrumType)
                    invCovDotParamDerivs[cosmo][gauss][spectrumType] = full(lmax-2, nan)

    return invCovDotParamDerivs


def getInvCovDotPowerSpectra(powersFid, \
                                cmbNoiseSpectra, \
                                deflectionNoiseSpectra, \
                                lenspowerDataDir = os.path.dirname(os.path.abspath(__file__)) + '../data', \
                                lmax = 3000, \
                                lenspowerExecDir = os.path.dirname(os.path.abspath(__file__)) + '../c/', \
                                fileNameBase = 'testing', \
                                doNonGaussian = False,
                                includeNoiseCovs = False):

    gaussianCov = getGaussianCov(powersFid = powersFid, \
                            cmbNoiseSpectra = cmbNoiseSpectra, \
                            deflectionNoises = deflectionNoiseSpectra, \
                            lmax = lmax)

    spectrumTypes = ['unlensed', 'lensed', 'delensed']
    polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd']
    gaussianity = ['gaussian', 'nonGaussian']
    cov = fourdDict(gaussianity, spectrumTypes, polCombsToUse, polCombsToUse)
    choleskyCovMatrix = twodDict(gaussianity, spectrumTypes)
    invCovDotPowerSpectra = twodDict(gaussianity, spectrumTypes)
    powerSpectraStack = onedDict(spectrumTypes)


    for gauss in gaussianity:
        for spectrumType in spectrumTypes:
            for pc1, polComb1 in enumerate(polCombsToUse):
                for pc2, polComb2 in enumerate(polCombsToUse):
                    if pc2 <= pc1:
                        cov[gauss][spectrumType][polComb1][polComb2] = numpy.diag(gaussianCov[spectrumType][pc1, pc2])

    if doNonGaussian:
        dCldCLd = getDelensedPowerSpectraDerivs(unlensedSpectra = powersFid['unlensed'], \
                                                lensedSpectra = powersFid['lensed'], \
                                                cmbNoiseSpectra = cmbNoiseSpectra, \
                                                lensingSpectra = powersFid['lensing'], \
                                                deflectionNoiseSpectra = deflectionNoiseSpectra, \
                                                lenspowerDataDir = lenspowerDataDir, \
                                                lmaxToWrite = lmax, \
                                                lenspowerExecDir = lenspowerExecDir, \
                                                fileNameBase = fileNameBase,
                                                includeNoiseDerivs = includeNoiseCovs)

        ell = powersFid['unlensed']['l']
        nElls = ell.shape
        oneOver2ellPlusOne =  1. / (2. * ell[:lmax-2] + 1.)
        deflCov = 2*oneOver2ellPlusOne * (powersFid['lensing']['cl_dd'][:lmax-2])**2
        nonGaussianSpectrumTypes = ['lensed', 'delensed']

        if includeNoiseCovs == True:
            dCldCLd['delensed']['cl_TT'] += dCldCLd['delensed']['Nl_T']
            dCldCLd['delensed']['cl_EE'] += dCldCLd['delensed']['Nl_P']

        for spectrumType in nonGaussianSpectrumTypes:
            print('Computing', spectrumType, 'non-Gaussian covariances...')
            for pc1, polComb1 in enumerate(polCombsToUse):
                for pc2, polComb2 in enumerate(polCombsToUse):
                    if pc2 <= pc1:
                        print('Computing covaraince for', polComb1, 'and', polComb2)
                        cov['nonGaussian'][spectrumType][polComb1][polComb2] += numpy.tensordot(dCldCLd[spectrumType][polComb1][2:lmax,2:lmax], numpy.tensordot(numpy.diag(deflCov), dCldCLd[spectrumType][polComb2][2:lmax,2:lmax], axes = (1,1)), axes = (1,0))

    for gauss in gaussianity:
        for spectrumType in spectrumTypes:
            try:
                choleskyCovMatrix = scipy.linalg.cho_factor( \
                                    numpy.bmat( [ [cov[gauss][spectrumType]['cl_TT']['cl_TT'], numpy.transpose(cov[gauss][spectrumType]['cl_TE']['cl_TT']), numpy.transpose(cov[gauss][spectrumType]['cl_EE']['cl_TT']), numpy.transpose(cov[gauss][spectrumType]['cl_dd']['cl_TT']) ], \
                                    [cov[gauss][spectrumType]['cl_TE']['cl_TT'], cov[gauss][spectrumType]['cl_TE']['cl_TE'], numpy.transpose(cov[gauss][spectrumType]['cl_EE']['cl_TE']), numpy.transpose(cov[gauss][spectrumType]['cl_dd']['cl_TE'])], \
                                    [cov[gauss][spectrumType]['cl_EE']['cl_TT'], cov[gauss][spectrumType]['cl_EE']['cl_TE'], cov[gauss][spectrumType]['cl_EE']['cl_EE'], numpy.transpose(cov[gauss][spectrumType]['cl_dd']['cl_EE'])], \
                                    [cov[gauss][spectrumType]['cl_dd']['cl_TT'], cov[gauss][spectrumType]['cl_dd']['cl_TE'], cov[gauss][spectrumType]['cl_dd']['cl_EE'], cov[gauss][spectrumType]['cl_dd']['cl_dd'] ]] ) )
            except:
                print('Warning Cholesky decomposition failed for ' + gauss + ' ' + spectrumType)
                choleskyCovMatrix = scipy.linalg.cho_factor(numpy.eye(4*(lmax-2)))

            powerSpectraStack[spectrumType] = numpy.bmat([[powersFid[spectrumType]['cl_TT'][:lmax-2], powersFid[spectrumType]['cl_TE'][:lmax-2], powersFid[spectrumType]['cl_EE'][:lmax-2], powersFid['lensing']['cl_dd'][:lmax-2] ]])
            try:
                invCovDotPowerSpectra[gauss][spectrumType] = scipy.linalg.cho_solve(choleskyCovMatrix, numpy.transpose(powerSpectraStack[spectrumType]))
            except:
                print('Warning inverse covariance problem with ' + gauss + ' ' + spectrumType)
                invCovDotPowerSpectra[gauss][spectrumType] = full(lmax-2, nan)

    return invCovDotPowerSpectra




def getCovariances(powersFid, \
                                cmbNoiseSpectra, \
                                deflectionNoiseSpectra, \
                                lenspowerDataDir = os.path.dirname(os.path.abspath(__file__)) + '../data', \
                                lmax = 3000, \
                                lenspowerExecDir = os.path.dirname(os.path.abspath(__file__)) + '../c/', \
                                fileNameBase = 'testing', \
                                doNonGaussian = False,
                                includeNoiseCovs = False):

    gaussianCov = getGaussianCov(powersFid = powersFid, \
                            cmbNoiseSpectra = cmbNoiseSpectra, \
                            deflectionNoises = deflectionNoiseSpectra, \
                            lmax = lmax)

    spectrumTypes = ['unlensed', 'lensed', 'delensed']
    polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd']
    gaussianity = ['gaussian', 'nonGaussian']
    cov = fourdDict(gaussianity, spectrumTypes, polCombsToUse, polCombsToUse)


    for gauss in gaussianity:
        for spectrumType in spectrumTypes:
            for pc1, polComb1 in enumerate(polCombsToUse):
                for pc2, polComb2 in enumerate(polCombsToUse):
                    if pc2 <= pc1:
                        cov[gauss][spectrumType][polComb1][polComb2] = numpy.diag(gaussianCov[spectrumType][pc1, pc2])

    if doNonGaussian:
        dCldCLd = getDelensedPowerSpectraDerivs(unlensedSpectra = powersFid['unlensed'], \
                                                lensedSpectra = powersFid['lensed'], \
                                                cmbNoiseSpectra = cmbNoiseSpectra, \
                                                lensingSpectra = powersFid['lensing'], \
                                                deflectionNoiseSpectra = deflectionNoiseSpectra, \
                                                lenspowerDataDir = lenspowerDataDir, \
                                                lmaxToWrite = lmax, \
                                                lenspowerExecDir = lenspowerExecDir, \
                                                fileNameBase = fileNameBase,
                                                includeNoiseDerivs = includeNoiseCovs)

        ell = powersFid['unlensed']['l']
        nElls = ell.shape
        oneOver2ellPlusOne =  1. / (2. * ell[:lmax-2] + 1.)
        deflCov = 2*oneOver2ellPlusOne * (powersFid['lensing']['cl_dd'][:lmax-2])**2
        nonGaussianSpectrumTypes = ['lensed', 'delensed']

        if includeNoiseCovs == True:
            dCldCLd['delensed']['cl_TT'] += dCldCLd['delensed']['Nl_T']
            dCldCLd['delensed']['cl_EE'] += dCldCLd['delensed']['Nl_P']


        for spectrumType in nonGaussianSpectrumTypes:
            print('Computing', spectrumType, 'non-Gaussian covariances...')
            for pc1, polComb1 in enumerate(polCombsToUse):
                for pc2, polComb2 in enumerate(polCombsToUse):
                    if pc2 <= pc1:
                        print('Computing covaraince for', polComb1, 'and', polComb2)
                        cov['nonGaussian'][spectrumType][polComb1][polComb2] += numpy.tensordot(dCldCLd[spectrumType][polComb1][2:lmax,2:lmax], numpy.tensordot(numpy.diag(deflCov), dCldCLd[spectrumType][polComb2][2:lmax,2:lmax], axes = (1,1)), axes = (1,0))

    return cov






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


def getGaussianCMBFisher(powersFid, paramDerivs, cmbNoiseSpectra, deflectionNoises, cosmoParams,\
                            spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                            polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                            lmax = 3000):

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



        for cp1, cosmo1 in enumerate(cosmoParams):
            for cp2, cosmo2 in enumerate(cosmoParams):
                fisherContribs[spectrumType][cosmo1][cosmo2] = \
                    numpy.einsum('ik,ijk,jk->k', \
                               paramDerivArray[cosmo1][spectrumType], invCovs[spectrumType], paramDerivArray[cosmo2][spectrumType])

                fisher[spectrumType][cp1, cp2] = sum(fisherContribs[spectrumType][cosmo1][cosmo2])

    return fisher

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



def choleskyInvCovDotParamDerivsNG(powersFid, \
                                cmbNoiseSpectra, \
                                deflectionNoiseSpectra, \
                                dCldCLd,
                                paramDerivs, \
                                cosmoParams, \
                                dCldCLu = None, \
                                lmax = 3000, \
                                polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                                spectrumType = 'delensed'):

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
            if polComb == 'cl_dd':
                paramDerivStack[cosmo] = numpy.append(paramDerivStack[cosmo], paramDerivs[cosmo]['lensing'][polComb][:lmax-1])
            else:
                paramDerivStack[cosmo] = numpy.append(paramDerivStack[cosmo], paramDerivs[cosmo][spectrumType][polComb][:lmax-1])

        try:
            invCovDotParamDerivs[cosmo] = scipy.linalg.cho_solve(covMatrix, paramDerivStack[cosmo])
        except:
            print('Warning inverse covariance problem with ' + cosmo + ' ' + spectrumType)
            invCovDotParamDerivs[cosmo] = full((lmax-2), nan)

    return invCovDotParamDerivs, paramDerivStack

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

# new methods with \ells

def getGaussianCMBFisherWithElls(powersFid, paramDerivs, cmbNoiseSpectra, deflectionNoises, cosmoParams,\
                            spectrumTypes = ['unlensed', 'lensed', 'delensed'], \
                            polCombsToUse = ['cl_TT', 'cl_TE', 'cl_EE', 'cl_dd'], \
                            ellsToUse = None, \
                            lmax = None):

    if ellsToUse is not None and lmax is not None:
        # give an error
        raise ValueError("User passed both ellsToUse and lmax.")
    elif ellsToUse is not None and lmax is None:
        print('Using ellsToUse passed by user.')
        lmax = 2
    elif ellsToUse is None and lmax is not None:
        print('Generating ellsToUse from lmax passed by user.')
        ellsToUse = {'cl_TT': [2, lmax], 'cl_TE': [2, lmax], 'cl_EE': [2, lmax], 'cl_BB': [2, lmax], 'cl_dd': [2, lmax]}
    else:
        print('Using default values for ellsToUse.')
        ellsToUse = {'cl_TT': [2, 5000], 'cl_TE': [2, 5000], 'cl_EE': [2, 5000], 'cl_BB': [2,5000], 'cl_dd': [2, 5000]}
        lmax = 5000

    for polComb in polCombsToUse:
        if polComb not in list(ellsToUse.keys()):
            # give an error
            raise ValueError(polComb + " is not in ellsToUse.")

    for polComb in list(ellsToUse.keys()):
        # check that lmins are at least 2; otherwise set to 2
        if ellsToUse[polComb][0] < 2:
            print('Setting lmin for ' + polComb + ' to 2.')
            ellsToUse[polComb][0] = 2
        # pick out the largest lmax
        if ellsToUse[polComb][1] > lmax:
            lmax = ellsToUse[polComb][1]

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

def choleskyInvCovDotParamDerivsNGWithElls(powersFid, \
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
