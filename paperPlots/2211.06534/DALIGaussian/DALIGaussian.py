#import basic tools
import sys
import numpy as np
import pickle

#import cobaya tools
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError

#import other tools
import plotTools as pt

class DALIGaussian(Likelihood):
    jobName: str
    use_dali: bool
    sType: str
    gType: str
    experiment: int
    tauPrior: bool
    fsky: float
    bao: bool
    cosmoFid: dict
    baoFile: str
    baoName: str

    def initialize(self):
        self._myFishers, self._cosmoParams = pt.loadWithDALI(jobName = self.jobName, pythonFlag = 3, \
                                           returnCosmoParams = True, gTypes=['Gaussian'])
        self._nParams = len(self._cosmoParams)

        self._myFishers = pt.addfskyWithDALI(self._myFishers, self.fsky, gTypes = ['Gaussian'])
        if self.bao:
            self._BAOData = pt.loadBAO(self.baoName)
            self._myFishers = pt.addBAOWithDALI(self._myFishers, self._cosmoParams, self.baoFile, self._BAOData, gTypes = ['Gaussian'])
        if self.tauPrior:
            self._myFishers = pt.addTauWithDALI(self._myFishers, self._cosmoParams, gTypes = ['Gaussian'])

        self._fisher = self._myFishers[self.gType][self.sType]['fisher'][self.experiment]
        self._DALI3 = self._myFishers[self.gType][self.sType]['DALI3'][self.experiment]
        self._DALI4 = self._myFishers[self.gType][self.sType]['DALI4'][self.experiment]
        self._myGaussianErrors = np.sqrt(np.diag(np.linalg.inv(self._fisher)))
        return super().initialize()

    def get_requirements(self):
        return self._cosmoParams

    def logp(self, _derived=None, **params):
        deviationVector = np.zeros(self._nParams)
        for k in range(self._nParams):
            deviationVector[k] = params[self._cosmoParams[k]] - self.cosmoFid[self._cosmoParams[k]]
        deltaP = deviationVector

        logLike = -0.5*np.einsum('ij,i,j', self._fisher, deltaP, deltaP)
        if self.use_dali:
            logLike += -0.5*np.einsum('ijk,i,j,k', self._DALI3, deltaP, deltaP, deltaP)
            logLike += -0.125*np.einsum('ijkl,i,j,k,l', self._DALI4, deltaP, deltaP, deltaP, deltaP)
        return logLike
