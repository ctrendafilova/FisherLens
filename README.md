

FisherLens: Fisher forecasting code for CLASS_delens
==============================================


Welcome to FisherLens. This code provides a wrapper for the CLASS_delens code to facilitate Fisher forecasting of cosmological parameter constraints from CMB spectra.

For full documentation on CLASS_delens, see:
 https://github.com/selimhotinli/class_delens

Authors: Selim C. Hotinli, Joel Meyers, Cynthia Trendafilova, Daniel Green, Alexander van Engelen

<img src="./FisherLensLogo.svg" width="200" height = "auto" />

Getting started
-----------------------------------

Download and compile the CLASS_delens submodule.

Running the file `fisherGenerateDataClass_example.py` will produce a set of forecasts for LambdaCDM + N_eff + m_nu, for 20 different experimental configurations.

User inputs
-----------------------------------

Within `fisherGenerateDataClass_example.py`, the following options may be adjusted by the user:

`useMPI`: Calculating the full covariance matrices including lensing-induced non-Gaussian covariances demands a lot of memory, O(100 GB), and it is thus recommended to perform these calculations on a high-performance computing cluster. The example forecast file is parallelized using MPI, and this feature can be turned on/off with this flag.

`expNames`: The total number of experiments being included in the forecast.

`lmax`: The highest \ell-modes to include in the Fisher matrix sum.

`lmaxTT`: The highest \ell-modes to include for TT, specifically, in the Fisher matrix sum.

`lmin`: The lowest \ell-modes to include in the Fisher matrix sum.

`noiseLevels`: The white noise levels, in uK-arcmin, of the experiments being included in the forecast.

`beamSizeArcmin`: The beam size, in arcmin, of the experiments being included in the forecast.

`classExecDir`: The directory where you have downloaded and compiled `CLASS_delens`.

`classDataDir`: The directory where you would like the intermediate calculation files and the final result files to be stored. Note that the intermediate files can be **very large** and are best written to e.g. a `scratch` space.

`fileBase`: The base file name to be used for all output files.

`polCombs`: The polarizations to be included in the Fisher matrix sum. BB may also be added.

`cosmoFid`: A dictionary containing the free parameters of interest and their fiducial values. Only the parameters in `cosmoFid` will be varied.

`stepSizes`: A dictionary containing step sizes for parameters to be used for the numerical derivatives.

`reconstructionMask`: A dictionary that can be used to mask certain \ells from being included in the lensing reconstruction. The example shows how to cut off T at \ell of 3000.

`extra_params`: A dictionary where you can specify additional CLASS parameters, included model parameters whose values you would like to specify but keep fixed. Additional `CLASS_delens` options can also be passed within `extra_params`. See the `CLASS_delens` repository for full documentation.

`ellsToUse`: A dictionary specifying which \ells, for each polarization, to include in the Fisher sum.

`ellsToUseNG`: Same as above, but used when non-Gaussian covariances are included in the calculation. The full covariance matrix will be calculated up to `'lmaxCov'`, and then the specified `lmin`, `lmax` ranges will be used in the Fisher sum.

`doNonGaussian`: Whether or not to include calculation of lensing-induced non-Gaussian covariances.

`includeUnlensedSpectraDerivatives`: If `doNonGaussian` is enabled, this option controls whether or not to include the contributions of derivatives of lensed/delensed CMB spectra with respect to unlensed spectra.

Additional files
-----------------------------------

Additional python files and Jupyter notebooks can be found under paperPlots. These will replicate the CMB results and figures from:
 - https://arxiv.org/abs/2111.15036
 - https://arxiv.org/abs/2211.06534
 
 Please place the utility file `plotTools.py` in the same directory as your Jupyter notebooks. This file provides helper functions for loading and manipulating forecast results.

Using the code
-----------------------------------

This code is free to use. If you use it, please cite https://arxiv.org/abs/2111.15036 in your publications.

If you use the DALI functionality released alongside our second paper, please also cite https://arxiv.org/abs/2211.06534.
