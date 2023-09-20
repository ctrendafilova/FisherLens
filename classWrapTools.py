import os
import numpy as np
import cambWrapTools

TCMB = 2.7255  ## CMB temperature in K




def class_generate_data(cosmo,
                        rootName = 'testing',
                        cmbNoise = None,
                        noiseLevel = 1.,
                        beamSizeArcmin = 1.,
                        deflectionNoise = None,
                        externalUnlensedCMBSpectra = None,
                        externalLensedCMBSpectra = None,
                        externalLensingSpectra = None,
                        classExecDir = os.path.dirname(os.path.abspath(__file__)) + '/CLASS_delens/',
                        classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/CLASS_delens/',
                        calculateDerivatives = False,
                        includeUnlensedSpectraDerivatives = False,
                        outputAllReconstructions = False,
                        reconstructionMask = None,
                        lmax = 5000,
                        backgroundOnly = False,
                        extraParams = dict()
                        ):

    if not os.path.exists(classDataDir + 'input/'):
        os.makedirs(classDataDir + 'input/')
    if not os.path.exists(classDataDir + 'output/'):
        os.makedirs(classDataDir + 'output/')

    ########################################################
    ## Convert cosmological parameters to CLASS format    ##
    ########################################################

    cosmoclass = dict()

    if 'H0' in list(cosmo.keys()):
        cosmoclass['H0'] = cosmo['H0']
    elif 'h' in list(cosmo.keys()):
        cosmoclass['h'] = cosmo['h']
    elif 'theta_s' in list(cosmo.keys()):
        cosmoclass['100*theta_s']= 100.*cosmo['theta_s']

    cosmoclass['omega_b'] = cosmo['omega_b_h2']
    cosmoclass['omega_cdm'] = cosmo['omega_c_h2']
    if 'A_s' in list(cosmo.keys()):
        cosmoclass['A_s'] = cosmo['A_s']
    elif 'logA' in list(cosmo.keys()):
        cosmoclass['ln10^{10}A_s'] = cosmo['logA']
    cosmoclass['n_s'] = cosmo['n_s']
    cosmoclass['tau_reio'] = cosmo['tau']
    #omega_k
    if 'omk' in list(cosmo.keys()):
        cosmoclass['Omega_k'] = cosmo['omk']

    ## CLASS treats 'ur' species differently from 'ncdm' species
    ## With one massive neutrino eigenstate, 'N_ur' is 2.0328 to make N_eff 3.046
    ## Subtract off the difference to give consistent N_eff
    cosmoclass['N_ur'] = cosmo['N_eff']-(3.046 - 2.0328) if 'mnu' in list(cosmo.keys()) else cosmo['N_eff'] if 'N_eff' in list(cosmo.keys()) else 3.046

    ## Setup massive neutrinos with a single massive species
    if 'mnu' in list(cosmo.keys()):
        cosmoclass['N_ncdm'] = 1
        cosmoclass['T_ncdm'] = 0.71611
        cosmoclass['m_ncdm'] = cosmo['mnu']

    if 'Yhe' in list(cosmo.keys()):
        cosmoclass['YHe'] = cosmo['Yhe']

    if 'r' in list(cosmo.keys()) and cosmo['r'] != 0.:
        cosmoclass['r'] = cosmo['r']
        if 'n_t' in list(cosmo.keys()):
            cosmoclass['n_t'] = cosmo['n_t']

    ## spectral running
    if 'alpha_s' in list(cosmo.keys()):
        cosmoclass['alpha_s'] = cosmo['alpha_s']


    ## isocurvature parameters

    if 'f_bi' in list(cosmo.keys()):
        cosmoclass['f_bi'] = cosmo['f_bi']
    if 'n_bi' in list(cosmo.keys()):
        cosmoclass['n_bi'] = cosmo['n_bi']
    if 'alpha_bi' in list(cosmo.keys()):
        cosmoclass['alpha_bi'] = cosmo['alpha_bi']

    if 'f_cdi' in list(cosmo.keys()):
        cosmoclass['f_cdi'] = cosmo['f_cdi']
    if 'n_cdi' in list(cosmo.keys()):
        cosmoclass['n_cdi'] = cosmo['n_cdi']
    if 'alpha_cdi' in list(cosmo.keys()):
        cosmoclass['alpha_cdi'] = cosmo['alpha_cdi']

    if 'f_nid' in list(cosmo.keys()):
        cosmoclass['f_nid'] = cosmo['f_nid']
    if 'n_nid' in list(cosmo.keys()):
        cosmoclass['n_nid'] = cosmo['n_nid']
    if 'alpha_nid' in list(cosmo.keys()):
        cosmoclass['alpha_nid'] = cosmo['alpha_nid']

    if 'f_niv' in list(cosmo.keys()):
        cosmoclass['f_niv'] = cosmo['f_niv']
    if 'n_niv' in list(cosmo.keys()):
        cosmoclass['n_niv'] = cosmo['n_niv']
    if 'alpha_niv' in list(cosmo.keys()):
        cosmoclass['alpha_niv'] = cosmo['alpha_niv']

    ## isocurvature covariances
    if 'c_ad_bi' in list(cosmo.keys()):
        cosmoclass['c_ad_bi'] = cosmo['c_ad_bi']
    if 'n_ad_bi' in list(cosmo.keys()):
        cosmoclass['n_ad_bi'] = cosmo['n_ad_bi']
    if 'alpha_ad_bi' in list(cosmo.keys()):
        cosmoclass['alpha_ad_bi'] = cosmo['alpha_ad_bi']

    if 'c_ad_cdi' in list(cosmo.keys()):
        cosmoclass['c_ad_cdi'] = cosmo['c_ad_cdi']
    if 'n_ad_cdi' in list(cosmo.keys()):
        cosmoclass['n_ad_cdi'] = cosmo['n_ad_cdi']
    if 'alpha_ad_cdi' in list(cosmo.keys()):
        cosmoclass['alpha_ad_cdi'] = cosmo['alpha_ad_cdi']

    if 'c_ad_nid' in list(cosmo.keys()):
        cosmoclass['c_ad_nid'] = cosmo['c_ad_nid']
    if 'n_ad_nid' in list(cosmo.keys()):
        cosmoclass['n_ad_nid'] = cosmo['n_ad_nid']
    if 'alpha_ad_nid' in list(cosmo.keys()):
        cosmoclass['alpha_ad_nid'] = cosmo['alpha_ad_nid']

    if 'c_ad_niv' in list(cosmo.keys()):
        cosmoclass['c_ad_niv'] = cosmo['c_ad_niv']
    if 'n_ad_niv' in list(cosmo.keys()):
        cosmoclass['n_ad_niv'] = cosmo['n_ad_niv']
    if 'alpha_ad_niv' in list(cosmo.keys()):
        cosmoclass['alpha_ad_niv'] = cosmo['alpha_ad_niv']

    if 'c_bi_cdi' in list(cosmo.keys()):
        cosmoclass['c_bi_cdi'] = cosmo['c_bi_cdi']
    if 'n_bi_cdi' in list(cosmo.keys()):
        cosmoclass['n_bi_cdi'] = cosmo['n_bi_cdi']
    if 'alpha_bi_cdi' in list(cosmo.keys()):
        cosmoclass['alpha_bi_cdi'] = cosmo['alpha_bi_cdi']

    if 'c_bi_nid' in list(cosmo.keys()):
        cosmoclass['c_bi_nid'] = cosmo['c_bi_nid']
    if 'n_bi_nid' in list(cosmo.keys()):
        cosmoclass['n_bi_nid'] = cosmo['n_bi_nid']
    if 'alpha_bi_nid' in list(cosmo.keys()):
        cosmoclass['alpha_bi_nid'] = cosmo['alpha_bi_nid']

    if 'c_bi_niv' in list(cosmo.keys()):
        cosmoclass['c_bi_niv'] = cosmo['c_bi_niv']
    if 'n_bi_niv' in list(cosmo.keys()):
        cosmoclass['n_bi_niv'] = cosmo['n_bi_niv']
    if 'alpha_bi_nid' in list(cosmo.keys()):
        cosmoclass['alpha_bi_niv'] = cosmo['alpha_bi_niv']

    if 'c_cdi_nid' in list(cosmo.keys()):
        cosmoclass['c_cdi_nid'] = cosmo['c_cdi_nid']
    if 'n_cdi_nid' in list(cosmo.keys()):
        cosmoclass['n_cdi_nid'] = cosmo['n_cdi_nid']
    if 'alpha_cdi_nid' in list(cosmo.keys()):
        cosmoclass['alpha_cdi_nid'] = cosmo['alpha_cdi_nid']

    if 'c_cdi_niv' in list(cosmo.keys()):
        cosmoclass['c_cdi_niv'] = cosmo['c_cdi_niv']
    if 'n_cdi_niv' in list(cosmo.keys()):
        cosmoclass['n_cdi_niv'] = cosmo['n_cdi_niv']
    if 'alpha_cdi_niv' in list(cosmo.keys()):
        cosmoclass['alpha_cdi_niv'] = cosmo['alpha_cdi_niv']

    if 'c_nid_niv' in list(cosmo.keys()):
        cosmoclass['c_nid_niv'] = cosmo['c_nid_niv']
    if 'n_nid_niv' in list(cosmo.keys()):
        cosmoclass['n_nid_niv'] = cosmo['n_nid_niv']
    if 'alpha_nid_niv' in list(cosmo.keys()):
        cosmoclass['alpha_nid_niv'] = cosmo['alpha_nid_niv']

    ## varying fundamental constants
    
    if 'varying_transition_redshift' in list(cosmo.keys()):
        cosmoclass['varying_fundamental_constants'] = 'instantaneous'
        cosmoclass['varying_transition_redshift'] = cosmo['varying_transition_redshift']
    if 'varying_alpha' in list(cosmo.keys()):
        cosmoclass['varying_fundamental_constants'] = 'instantaneous'
        cosmoclass['varying_alpha'] = cosmo['varying_alpha']
    if 'varying_me' in list(cosmo.keys()):
        cosmoclass['varying_fundamental_constants'] = 'instantaneous'
        cosmoclass['varying_me'] = cosmo['varying_me']
    if 'bbn_alpha_sensitivity' in list(cosmo.keys()):
        cosmoclass['varying_fundamental_constants'] = 'instantaneous'
        cosmoclass['bbn_alpha_sensitivity'] = cosmo['bbn_alpha_sensitivity']
    
    ## EDE-edit
    
    if 'f_scf' in list(cosmo.keys()):
        cosmoclass['f_scf'] = cosmo['f_scf']
        cosmoclass['Omega_Lambda'] = 0
        cosmoclass['Omega_fld'] = 0
        cosmoclass['Omega_scf'] = -1
        cosmoclass['scf_parameters'] = '1, 1, 1, 1, 1, 0.0'
        cosmoclass['CC_scf'] = 1
        cosmoclass['n_scf'] = 3
        cosmoclass['scf_tuning_index'] = 3
        cosmoclass['attractor_ic_scf'] = 'no'
    if 'log10f_scf' in list(cosmo.keys()):
        cosmoclass['log10f_scf'] = cosmo['log10f_scf']
        cosmoclass['Omega_Lambda'] = 0
        cosmoclass['Omega_fld'] = 0
        cosmoclass['Omega_scf'] = -1
        cosmoclass['scf_parameters'] = '1, 1, 1, 1, 1, 0.0'
        cosmoclass['CC_scf'] = 1
        cosmoclass['n_scf'] = 3
        cosmoclass['scf_tuning_index'] = 3
        cosmoclass['attractor_ic_scf'] = 'no'
    if 'm_scf' in list(cosmo.keys()):
        cosmoclass['m_scf'] = cosmo['m_scf']
        cosmoclass['Omega_Lambda'] = 0
        cosmoclass['Omega_fld'] = 0
        cosmoclass['Omega_scf'] = -1
        cosmoclass['scf_parameters'] = '1, 1, 1, 1, 1, 0.0'
        cosmoclass['CC_scf'] = 1
        cosmoclass['n_scf'] = 3
        cosmoclass['scf_tuning_index'] = 3
        cosmoclass['attractor_ic_scf'] = 'no'
    if 'log10m_scf' in list(cosmo.keys()):
        cosmoclass['log10m_scf'] = cosmo['log10m_scf']
        cosmoclass['Omega_Lambda'] = 0
        cosmoclass['Omega_fld'] = 0
        cosmoclass['Omega_scf'] = -1
        cosmoclass['scf_parameters'] = '1, 1, 1, 1, 1, 0.0'
        cosmoclass['CC_scf'] = 1
        cosmoclass['n_scf'] = 3
        cosmoclass['scf_tuning_index'] = 3
        cosmoclass['attractor_ic_scf'] = 'no'
    if 'fEDE' in list(cosmo.keys()):
        cosmoclass['fEDE'] = cosmo['fEDE']
        cosmoclass['Omega_Lambda'] = 0
        cosmoclass['Omega_fld'] = 0
        cosmoclass['Omega_scf'] = -1
        cosmoclass['scf_parameters'] = '1, 1, 1, 1, 1, 0.0'
        cosmoclass['CC_scf'] = 1
        cosmoclass['n_scf'] = 3
        cosmoclass['scf_tuning_index'] = 3
        cosmoclass['attractor_ic_scf'] = 'no'
    if 'log10z_c' in list(cosmo.keys()):
        cosmoclass['log10z_c'] = cosmo['log10z_c']
        cosmoclass['Omega_Lambda'] = 0
        cosmoclass['Omega_fld'] = 0
        cosmoclass['Omega_scf'] = -1
        cosmoclass['scf_parameters'] = '1, 1, 1, 1, 1, 0.0'
        cosmoclass['CC_scf'] = 1
        cosmoclass['n_scf'] = 3
        cosmoclass['scf_tuning_index'] = 3
        cosmoclass['attractor_ic_scf'] = 'no'
    if 'thetai_scf' in list(cosmo.keys()):
        cosmoclass['thetai_scf'] = cosmo['thetai_scf']
        cosmoclass['Omega_Lambda'] = 0
        cosmoclass['Omega_fld'] = 0
        cosmoclass['Omega_scf'] = -1
        cosmoclass['scf_parameters'] = '1, 1, 1, 1, 1, 0.0'
        cosmoclass['CC_scf'] = 1
        cosmoclass['n_scf'] = 3
        cosmoclass['scf_tuning_index'] = 3
        cosmoclass['attractor_ic_scf'] = 'no'
        
    ## Dark radiation (7.2.2)
    
    if 'N_idr' in list(cosmo.keys()):
        cosmoclass['N_idr'] = cosmo['N_idr']
    if 'f_idm' in list(cosmo.keys()):
        cosmoclass['f_idm'] = cosmo['f_idm']
    if 'Gamma_0_nadm' in list(cosmo.keys()):
        cosmoclass['Gamma_0_nadm'] = cosmo['Gamma_0_nadm']

    ########################################################
    ## the relevant CLASS commands to call delensing code ##
    ########################################################

    dcode = dict()

    dcode['root'] = classDataDir + 'output/' + rootName #+ '_'
    dcode['output'] = 'tCl,pCl,lCl,dlCl'
    dcode['modes'] = 's'
    dcode['lensing'] = 'yes'
    dcode['non linear'] = 'hmcode'
    if dcode['non linear'] == 'hmcode' and 'eta_0' in list(cosmo.keys()):
        cosmoclass['eta_0'] = cosmo['eta_0']
    if dcode['non linear'] == 'hmcode' and 'c_min' in list(cosmo.keys()):
        cosmoclass['c_min'] = cosmo['c_min']
    dcode['l_max_scalars'] = lmax
    dcode['delta_l_max'] = 2000
    dcode['delta_dl_max'] = 0
    dcode['format'] = 'class'
    dcode['accurate_lensing'] = 1
    dcode['headers'] = 'yes'
    dcode['output_spectra_noise'] = 'yes'
    dcode['delensing_verbose'] = 0
    dcode['write parameters'] = 'yes'
    dcode['delensing derivatives'] = 'no'
    dcode['output_derivatives'] = 'no'
    dcode['overwrite_root'] = 'yes'
    dcode['delensing_verbose'] = 3

    dcode['ic'] = 'ad'

    if 'r' in list(cosmo.keys()) and cosmo['r'] != 0.:
        dcode['modes']+=',t'
        dcode['l_max_tensors'] = 1000

    ## adding isocurvature modes to initial conditions
    if 'f_bi' in list(cosmo.keys()) and cosmo['f_bi'] != 0.:
        dcode['ic']+=',bi'

    if 'f_cdi' in list(cosmo.keys()) and cosmo['f_cdi'] != 0.:
        dcode['ic']+=',cdi'

    if 'f_nid' in list(cosmo.keys()) and cosmo['f_nid'] != 0.:
        dcode['ic']+=',nid'

    if 'f_niv' in list(cosmo.keys()) and cosmo['f_niv'] != 0.:
        dcode['ic']+=',niv'

    if cmbNoise is None:
        ## If T and P noise are not provided, use analytic noise model
        dcode['temperature noise spectra type']  = 'idealized'
        dcode['polarization noise spectra type'] = 'idealized'
        dcode['delta_noise'] = noiseLevel / 60. / 180. * np.pi  ## Convert uK-arcmin to uK-rad
        dcode['sigma_beam']  = beamSizeArcmin / 60. / 180. * np.pi ## Convert arcmin to rad
    else:
        if cmbNoise['l'][-1]<dcode['l_max_scalars']+dcode['delta_l_max']+dcode['delta_dl_max']:
            print('You must provde external CMB noise at least up to l=' +str(dcode['l_max_scalars']+dcode['delta_l_max']+dcode['delta_dl_max']))
            stop
        dcode['temperature noise spectra type'] = 'external'
        dcode['polarization noise spectra type'] = 'external'
        np.savetxt(classDataDir  + 'input/' + rootName + '_temp_noise.dat', np.column_stack((cmbNoise['l'], cmbNoise['cl_TT'])) )
        np.savetxt(classDataDir  + 'input/' + rootName + '_pol_noise.dat', np.column_stack((cmbNoise['l'], cmbNoise['cl_EE'])) )
        dcode['command_for_temperature_noise_spec'] = 'cat ' + classDataDir  + 'input/' + rootName + '_temp_noise.dat'
        dcode['command_for_polarization_noise_spec'] = 'cat ' + classDataDir  + 'input/' + rootName + '_pol_noise.dat'

    if deflectionNoise is None:
        ##  If deflection noise is not provided, use iterative delensing
        dcode['delensing'] = 'iterative'
        dcode['lensing reconstruction noise spectra type'] = 'internal'
        dcode['noise_iteration_type'] = 'diag'
        dcode['min_varr_type']  = 'diag'
        dcode['convergence type'] = 'total'
        dcode['convergence_criterion_itr'] = '1e-5'
        if reconstructionMask is not None:
            dcode['recon_mask_lmin_T'] = reconstructionMask['lmin_T'] if 'lmin_T' in reconstructionMask.keys() else 0
            dcode['recon_mask_lmax_T'] = reconstructionMask['lmax_T'] if 'lmax_T' in reconstructionMask.keys() else 30000
            dcode['recon_mask_lmin_E'] = reconstructionMask['lmin_E'] if 'lmin_E' in reconstructionMask.keys() else 0
            dcode['recon_mask_lmax_E'] = reconstructionMask['lmax_E'] if 'lmax_E' in reconstructionMask.keys() else 30000
            dcode['recon_mask_lmin_B'] = reconstructionMask['lmin_B'] if 'lmin_B' in reconstructionMask.keys() else 0
            dcode['recon_mask_lmax_B'] = reconstructionMask['lmax_B'] if 'lmax_B' in reconstructionMask.keys() else 30000
    else:
        dcode['delensing'] = 'yes'
        dcode['lensing reconstruction noise spectra type'] = 'external'
        ##  Write file containing deflection noise, assuming deflectionNoise is array of NLdd at every L starting from 2
        np.savetxt(classDataDir  + 'input/' + rootName + '_defl_noise.dat', np.column_stack((np.arange(len(deflectionNoise))+2, deflectionNoise)) )
        dcode['command_for_lens_recon_noise_spec'] = 'cat ' + classDataDir  + 'input/' + rootName + '_defl_noise.dat'

    if externalUnlensedCMBSpectra is None:
        ## If unlensed spectra are not provided, calculate spectra with CLASS
        dcode['cmb spectra type']  = 'internal'
    else:
        ## If unlensed spectra are provided, read them in
        if externalLensingSpectra is None:
            print('Need to supply lensing spectrum to use external unlensed spectra')
            stop
        ## Need unlensed spectra and lensing spectra to higher l than lmax
        if dcode['l_max_scalars'] + (dcode['delta_l_max']+dcode['delta_dl_max']) > np.shape(externalUnlensedCMBSpectra['l'])[0]:
            print(('Need to supply unlensed spectrum to lmax >= ' + str( dcode['l_max_scalars'] + (dcode['delta_l_max']+dcode['delta_dl_max'])+1 ) + \
                ' in order to compute delensed spectrum to lmax = ' + str( lmax ) ))
            stop
        if dcode['l_max_scalars'] + (dcode['delta_l_max']+dcode['delta_dl_max']) > np.shape(externalLensingSpectra['cl_phiphi'])[0]:
            print(('Need to supply lensing spectrum to lmax >= ' + str( dcode['l_max_scalars'] + (dcode['delta_l_max']+dcode['delta_dl_max'])+1 ) + \
                ' in order to compute delensed spectrum to lmax = ' + str( lmax ) ))
            stop
        dcode['cmb spectra type']  = 'external'
        np.savetxt(classDataDir  + 'input/' + rootName + '_unlensed_input_spectra.dat', \
            np.column_stack((externalUnlensedCMBSpectra['l'], externalUnlensedCMBSpectra['cl_TT'], \
            externalUnlensedCMBSpectra['cl_TE'], externalUnlensedCMBSpectra['cl_EE'], \
            externalUnlensedCMBSpectra['cl_BB'], externalLensingSpectra['cl_phiphi'] )) )
        dcode['command_for_external_cmb_spectra'] = 'cat ' + classDataDir  + 'input/' + rootName + '_unlensed_input_spectra.dat'

    if externalLensedCMBSpectra is None:
        ## If lensed spectra are not provided, calculate spectra with CLASS
        dcode['lensed cmb spectra type']  = 'internal'
    else:
        print('External lensed spectra not currently working, pass only external unlensed spectra.')
        stop
        ## If lensed spectra are provided, read them in
        if externalLensingSpectra is None:
            print('Need to supply lensing spectrum to use external lensed spectra')
            stop
        ## Need lensed spectra and lensing spectra to higher l than lmax
        if dcode['l_max_scalars'] + (dcode['delta_l_max']+dcode['delta_dl_max']) > np.shape(externalLensedCMBSpectra['l'])[0]:
            print(('Need to supply lensed spectrum to lmax >= ' + str( dcode['l_max_scalars'] + (dcode['delta_l_max']+dcode['delta_dl_max'])+1 ) + \
                ' in order to compute delensed spectrum to lmax = ' + str( lmax ) ))
            stop
        if dcode['l_max_scalars'] + (dcode['delta_l_max']+dcode['delta_dl_max']) > np.shape(externalLensingSpectra['cl_phiphi'])[0]:
            print(('Need to supply lensing spectrum to lmax >= ' + str( dcode['l_max_scalars'] + (dcode['delta_l_max']+dcode['delta_dl_max'])+1 ) + \
                ' in order to compute delensed spectrum to lmax = ' + str( lmax ) ))
            stop
        dcode['lensed cmb spectra type']  = 'external'
        np.savetxt(classDataDir  + 'input/' + rootName + '_lensed_input_spectra.dat', \
            np.column_stack((externalLensedCMBSpectra['l'], externalLensedCMBSpectra['cl_TT'], \
            externalLensedCMBSpectra['cl_TE'], externalLensedCMBSpectra['cl_EE'], \
            externalLensedCMBSpectra['cl_BB'], externalLensingSpectra['cl_phiphi'] )) )
        dcode['command_for_external_lensed_cmb_spectra'] = 'cat ' + classDataDir  + 'input/' + rootName + '_lensed_input_spectra.dat'

########################################################
## CLASS code calculates most of the relevant data,   ##
## including the spectra, reconstruction noise and    ##
## derivatives. It is important to give the correct   ##
## input file and with appropriate options included   ##
########################################################

    if calculateDerivatives is not False:
        dcode['delensing derivatives'] = 'yes'
        dcode['output_derivatives'] = 'yes'
        dcode['derv_binedges'] = '1'
        dervtype = calculateDerivatives
        dcode['derivative type'] = dervtype
        if includeUnlensedSpectraDerivatives is not False:
            dcode['calculate_derviaties_wrt_unlensed'] = 'yes'
            dcode['unlensed derivative type'] = dervtype

    ## extraParams is dictionary allowing for arbitrary additional specifications in CLASS format
    ## e.g. extraParams['tol_ncdm'] = 1.e-3 sets a precision parameter associated with ncdm
    ## extraParams will overwrite any previously set parameters

    if backgroundOnly is True:
        cosmoclass['write_background'] = 'yes'
        cosmoclass['thermodynamics_verbose'] = 1
        dcode['output'] = ''
        dcode['lensing'] = 'no'
        calculateDerivatives = False
        
    
    dcode.update(extraParams)

########################################################
## Running the CLASS code to write the spectra inside ##
## specific files, as well as their derivatives, and  ##
## the CMB and lensing noise used to create these.    ##
########################################################

    with open(classDataDir  + 'input/' + rootName + ".ini", "w") as f:
        f.write( '\n'.join(['%s = %s' % (k, v) for k, v in list(cosmoclass.items())])
                +'\n'
                +'\n'.join(['%s = %s' % (k, v) for k, v in list(dcode.items())]))


    os.system("cd " + classExecDir + " ; ./class " + classDataDir + 'input/' + rootName + ".ini")

########################################################
## Filling the spectra into specific CLASS dict()s.   ##
########################################################

    if calculateDerivatives is False and backgroundOnly is False:

    ########################################################
    ## the calculated spectra to be filled in whats below ##
    ########################################################

        cspec = dict()

        cspec['cl_unlensed'] = None # To be filled with unlensed CMB spectra
        cspec['cl_lensed'] = None # To be filled with lensed CMB spectra
        cspec['cl_delensed'] = None # To be filled with delensed CMB spectra
        cspec['nl_lensing'] = None # To be filled with lensing reconstruction noise
        cspec['nl_cmb'] = None # To be filled with CMB spectra noise

        cspec['cl_unlensed'] = np.loadtxt(classDataDir + "output/" + rootName + "_cl.dat")
        cspec['cl_lensed'] = np.loadtxt(classDataDir + "output/" + rootName + "_cl_lensed.dat")
        cspec['cl_delensed'] = np.loadtxt(classDataDir + "output/" + rootName + "_cl_delensed.dat")
        cspec['nl_cmb'] = np.loadtxt(classDataDir + "output/" + rootName + "_spectra_noise.dat")
        if deflectionNoise is None:
            cspec['nl_lensing'] = np.loadtxt(classDataDir + "output/" + rootName + "_lensing_noise_rcn.dat")

        lvec = cspec['cl_lensed'][:,0]
        nElls = len(lvec)

        ########################################################
        ## Filling dict()s in a way similar to the JM wrapper ##
        ########################################################

        dl_TT = cspec['cl_unlensed'][:nElls,1]*(TCMB*1.e6)**2
        dl_EE = cspec['cl_unlensed'][:nElls,2]*(TCMB*1.e6)**2
        dl_TE = cspec['cl_unlensed'][:nElls,3]*(TCMB*1.e6)**2
        dl_BB = cspec['cl_unlensed'][:nElls,4]*(TCMB*1.e6)**2

        cl_TT = cspec['cl_unlensed'][:nElls,1]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2
        cl_EE = cspec['cl_unlensed'][:nElls,2]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2
        cl_TE = cspec['cl_unlensed'][:nElls,3]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2
        cl_BB = cspec['cl_unlensed'][:nElls,4]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2

        unlensed = {'l' : cspec['cl_unlensed'][:nElls,0],
                    'cl_TT' : cl_TT,
                    'cl_EE' : cl_EE,
                    'cl_TE' : cl_TE,
                    'cl_BB' : cl_BB,

                    'dl_TT' : dl_TT,
                    'dl_EE' : dl_EE,
                    'dl_TE' : dl_TE,
                    'dl_BB' : dl_BB
                   }

        dl_TT_lensed = cspec['cl_lensed'][:nElls,1]*(TCMB*1.e6)**2
        dl_EE_lensed = cspec['cl_lensed'][:nElls,2]*(TCMB*1.e6)**2
        dl_TE_lensed = cspec['cl_lensed'][:nElls,3]*(TCMB*1.e6)**2
        dl_BB_lensed = cspec['cl_lensed'][:nElls,4]*(TCMB*1.e6)**2

        cl_TT_lensed = cspec['cl_lensed'][:nElls,1]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2
        cl_EE_lensed = cspec['cl_lensed'][:nElls,2]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2
        cl_TE_lensed = cspec['cl_lensed'][:nElls,3]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2
        cl_BB_lensed = cspec['cl_lensed'][:nElls,4]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2

        lensed = {'l' : cspec['cl_lensed'][:nElls,0],
                    'cl_TT' : cl_TT_lensed,
                    'cl_EE' : cl_EE_lensed,
                    'cl_TE' : cl_TE_lensed,
                    'cl_BB' : cl_BB_lensed,

                    'dl_TT' : dl_TT_lensed,
                    'dl_EE' : dl_EE_lensed,
                    'dl_TE' : dl_TE_lensed,
                    'dl_BB' : dl_BB_lensed
                 }


        dl_TT_delensed = cspec['cl_delensed'][:nElls,1]*(TCMB*1.e6)**2
        dl_EE_delensed = cspec['cl_delensed'][:nElls,2]*(TCMB*1.e6)**2
        dl_TE_delensed = cspec['cl_delensed'][:nElls,3]*(TCMB*1.e6)**2
        dl_BB_delensed = cspec['cl_delensed'][:nElls,4]*(TCMB*1.e6)**2

        cl_TT_delensed = cspec['cl_delensed'][:nElls,1]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2
        cl_EE_delensed = cspec['cl_delensed'][:nElls,2]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2
        cl_TE_delensed = cspec['cl_delensed'][:nElls,3]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2
        cl_BB_delensed = cspec['cl_delensed'][:nElls,4]*2*np.pi/(lvec*(lvec+1))*(TCMB*1.e6)**2

        delensed = {'l' : cspec['cl_delensed'][:nElls,0],
                    'cl_TT' : cl_TT_delensed,
                    'cl_EE' : cl_EE_delensed,
                    'cl_TE' : cl_TE_delensed,
                    'cl_BB' : cl_BB_delensed,

                    'dl_TT' : dl_TT_delensed,
                    'dl_EE' : dl_EE_delensed,
                    'dl_TE' : dl_TE_delensed,
                    'dl_BB' : dl_BB_delensed
                   }

        cl_TT_noise = cspec['nl_cmb'][:nElls,1]
        cl_EE_noise = cspec['nl_cmb'][:nElls,2]
        cl_TE_noise = np.empty(nElls)
        cl_TE_noise.fill(0.)
        cl_BB_noise = cspec['nl_cmb'][:nElls,2]

        dl_TT_noise = cl_TT_noise*(lvec*(lvec+1))/(2*np.pi)
        dl_EE_noise = cl_EE_noise*(lvec*(lvec+1))/(2*np.pi)
        dl_TE_noise = cl_TE_noise*(lvec*(lvec+1))/(2*np.pi)
        dl_BB_noise = cl_BB_noise*(lvec*(lvec+1))/(2*np.pi)

        cmbNoiseSpectra = {'l' : cspec['nl_cmb'][:nElls,0],
                    'cl_TT' : cl_TT_noise,
                    'cl_EE' : cl_EE_noise,
                    'cl_TE' : cl_TE_noise,
                    'cl_BB' : cl_BB_noise,

                    'dl_TT' : dl_TT_noise,
                    'dl_EE' : dl_EE_noise,
                    'dl_TE' : dl_TE_noise,
                    'dl_BB' : dl_BB_noise
                    }

        cl_phiphi = cspec['cl_unlensed'][:nElls,5]*2*np.pi/(lvec*(lvec+1))
        cl_dd = cl_phiphi*(lvec*(lvec+1))
        cl_kk = cl_phiphi*(lvec*(lvec+1))*(lvec*(lvec+1))/4.

        lensing = {'l' : cspec['cl_unlensed'][:nElls,0],
                    'cl_phiphi' : cl_phiphi,
                    'cl_dd' : cl_dd,
                    'cl_kk' : cl_kk
                 }

        powers=dict()

        powers['unlensed'] = unlensed
        powers['lensed'] = lensed
        powers['lensing'] = lensing
        powers['delensed'] = delensed

        ##  These lines are used to avoid error when not computing deflection noise
        deflection_noise = dict()
        deflection_noise['MV'] = None

        if deflectionNoise is None:
            if dcode['min_varr_type'] == 'diag':
                nl_dd_MV = cspec['nl_lensing'][:nElls,1]*2.*np.pi
                nl_dd_TT = cspec['nl_lensing'][:nElls,2]*2.*np.pi
                nl_dd_TE = cspec['nl_lensing'][:nElls,3]*2.*np.pi
                nl_dd_EE = cspec['nl_lensing'][:nElls,4]*2.*np.pi
                nl_dd_BB = cspec['nl_lensing'][:nElls,5]*2.*np.pi
                nl_dd_EB = cspec['nl_lensing'][:nElls,6]*2.*np.pi
                nl_dd_TB = cspec['nl_lensing'][:nElls,7]*2.*np.pi

                deflection_noise = {'l' :cspec['nl_lensing'][:nElls,0],
                                'MV' : nl_dd_MV,
                                'TT' : nl_dd_TT,
                                'TE' : nl_dd_TE,
                                'EE' : nl_dd_EE,
                                'BB' : nl_dd_BB,
                                'EB' : nl_dd_EB,
                                'TB' : nl_dd_TB
                                }
            elif dcode['min_varr_type'] == 'eb':
                nl_dd_MV = cspec['nl_lensing'][:nElls,1]*2.*np.pi
                nl_dd_EB = cspec['nl_lensing'][:nElls,2]*2.*np.pi

                deflection_noise = {'l' :cspec['nl_lensing'][:nElls,0],
                                'MV' : nl_dd_MV,
                                'EB' : nl_dd_EB
                                }

        if outputAllReconstructions is False:
            reconstructionOutput = deflection_noise['MV']
        if outputAllReconstructions is True:
            reconstructionOutput = deflection_noise


    if calculateDerivatives is not False:
    ########################################################
    ##    Calculate the derivatives of the CMB spectra.   ##
    ########################################################

        dCldCLd = loadLensingDerivatives(rootName = rootName,
                           classDataDir = classDataDir,
                           dervtype = dervtype)

        if includeUnlensedSpectraDerivatives is not False:
            dCldCLu = loadUnlensedSpectraDerivatives(rootName = rootName,
                               classDataDir = classDataDir,
                               dervtype = dervtype)
            
    if backgroundOnly is True:
        # returns rs/dV for desired redshifts
        background = np.loadtxt(classDataDir + "output/" + rootName + "_background.dat")
        thermo = np.loadtxt(classDataDir + "output/" + rootName + "_thermodynamics.dat")
        thermo_summary = np.loadtxt(classDataDir + "output/" + rootName + "_thermodynamics_summary.dat")
        output_data = {'background' : background,
                       'thermo' : thermo,
                       'thermo_summary' : thermo_summary}
        output = output_data
    elif calculateDerivatives is False:
        output = [powers, reconstructionOutput]
    else:
        if includeUnlensedSpectraDerivatives is not False:
            output = [dCldCLd, dCldCLu]
        else:
            output = dCldCLd


    return output

# CST
def camb_class_generate_data(cosmo,
                        rootName = 'testing',
                        cmbNoise = None,
                        noiseLevel = 1.,
                        beamSizeArcmin = 1.,
                        deflectionNoise = None,
                        externalUnlensedCMBSpectra = None,
                        externalLensedCMBSpectra = None,
                        externalLensingSpectra = None,
                        classExecDir = os.path.dirname(os.path.abspath(__file__)) + '/CLASS_delens/',
                        classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/CLASS_delens/',
                        calculateDerivatives = False,
                        includeUnlensedSpectraDerivatives = False,
                        outputAllReconstructions = False,
                        reconstructionMask = None,
                        lmax = 5000,
                        extraParams = dict(),
                        accuracy = 2,
                        doLensedWithCAMB = False):

    cambPowerSpectra = cambWrapTools.getPyCambPowerSpectra(cosmo = cosmo, \
                                                    accuracy = accuracy, \
                                                    lmaxToWrite = lmax+3000)

    if calculateDerivatives == False:
        powersFid, deflectionNoises = class_generate_data(cosmo = cosmo,
                                                        rootName = rootName,
                                                        cmbNoise = cmbNoise,
                                                        noiseLevel = noiseLevel,
                                                        beamSizeArcmin = beamSizeArcmin,
                                                        deflectionNoise = deflectionNoise,
                                                        externalUnlensedCMBSpectra = cambPowerSpectra['unlensed'],
                                                        externalLensedCMBSpectra = externalLensedCMBSpectra,
                                                        externalLensingSpectra = cambPowerSpectra['lensing'],
                                                        classExecDir = classExecDir,
                                                        classDataDir = classDataDir,
                                                        calculateDerivatives = calculateDerivatives,
                                                        includeUnlensedSpectraDerivatives = includeUnlensedSpectraDerivatives,
                                                        outputAllReconstructions = outputAllReconstructions,
                                                        reconstructionMask = reconstructionMask,
                                                        lmax = lmax,
                                                        extraParams = extraParams)
        powersFid['unlensed'] = cambPowerSpectra['unlensed']
        ## If you want to use the lensed spectra from CAMB
        if doLensedWithCAMB == True:
            powersFid['lensed'] = cambPowerSpectra['lensed']
        powersFid['lensing'] = cambPowerSpectra['lensing']
        return powersFid, deflectionNoises
    else:
        return class_generate_data(cosmo = cosmo,
                                    rootName = rootName,
                                    cmbNoise = cmbNoise,
                                    noiseLevel = noiseLevel,
                                    beamSizeArcmin = beamSizeArcmin,
                                    deflectionNoise = deflectionNoise,
                                    externalUnlensedCMBSpectra = cambPowerSpectra['unlensed'],
                                    externalLensedCMBSpectra = externalLensedCMBSpectra,
                                    externalLensingSpectra = cambPowerSpectra['lensing'],
                                    classExecDir = classExecDir,
                                    classDataDir = classDataDir,
                                    calculateDerivatives = calculateDerivatives,
                                    includeUnlensedSpectraDerivatives = includeUnlensedSpectraDerivatives,
                                    outputAllReconstructions = outputAllReconstructions,
                                    reconstructionMask = reconstructionMask,
                                    lmax = lmax,
                                    extraParams = extraParams)

def generate_data(cosmo,
                        rootName = 'testing',
                        cmbNoise = None,
                        noiseLevel = 1.,
                        beamSizeArcmin = 1.,
                        deflectionNoise = None,
                        externalUnlensedCMBSpectra = None,
                        externalLensedCMBSpectra = None,
                        externalLensingSpectra = None,
                        classExecDir = os.path.dirname(os.path.abspath(__file__)) + '/CLASS_delens/',
                        classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/CLASS_delens/',
                        calculateDerivatives = False,
                        includeUnlensedSpectraDerivatives = False,
                        outputAllReconstructions = False,
                        reconstructionMask = None,
                        lmax = 5000,
                        extraParams = dict(),
                        accuracy = 2,
                        useClass = True,
                        doLensedWithCAMB = False):

    if useClass == True:
        return class_generate_data(cosmo,
                        rootName = rootName,
                        cmbNoise = cmbNoise,
                        noiseLevel = noiseLevel,
                        beamSizeArcmin = beamSizeArcmin,
                        deflectionNoise = deflectionNoise,
                        externalUnlensedCMBSpectra = externalUnlensedCMBSpectra,
                        externalLensedCMBSpectra = externalLensedCMBSpectra,
                        externalLensingSpectra = externalLensingSpectra,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        calculateDerivatives = calculateDerivatives,
                        includeUnlensedSpectraDerivatives = includeUnlensedSpectraDerivatives,
                        outputAllReconstructions = outputAllReconstructions,
                        reconstructionMask = reconstructionMask,
                        lmax = lmax,
                        extraParams = extraParams
                        )

    elif useClass == False:
        return camb_class_generate_data(cosmo,
                        rootName = rootName,
                        cmbNoise = cmbNoise,
                        noiseLevel = noiseLevel,
                        beamSizeArcmin = beamSizeArcmin,
                        deflectionNoise = deflectionNoise,
                        externalUnlensedCMBSpectra = externalUnlensedCMBSpectra,
                        externalLensedCMBSpectra = externalLensedCMBSpectra,
                        externalLensingSpectra = externalLensingSpectra,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        calculateDerivatives = calculateDerivatives,
                        includeUnlensedSpectraDerivatives = includeUnlensedSpectraDerivatives,
                        outputAllReconstructions = outputAllReconstructions,
                        reconstructionMask = reconstructionMask,
                        lmax = lmax,
                        extraParams = extraParams,
                        accuracy = accuracy,
                        doLensedWithCAMB = doLensedWithCAMB
                        )

def loadLensingDerivatives(rootName = 'testing',
                           classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/',
                           dervtype = 'lensed'):
    ## Loads dCldCLd from files as computed by class_generate_data and returns dictionary with zero padding as used by Fisher code ##

    dCldCLd = dict()

    dCldCLd['cl_TT'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClTTdCldd_"+dervtype+".dat"), ((2,0),), 'constant')
    dCldCLd['cl_EE'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClEEdCldd_"+dervtype+".dat"), ((2,0),), 'constant')
    dCldCLd['cl_TE'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClTEdCldd_"+dervtype+".dat"), ((2,0),), 'constant')
    dCldCLd['cl_BB'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClBBdCldd_"+dervtype+".dat"), ((2,0),), 'constant')


    return dCldCLd

def loadUnlensedSpectraDerivatives(rootName = 'testing',
                           classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/../../CLASS_delens/',
                           dervtype = 'lensed'):
    ## Loads dCldCLu from files as computed by class_generate_data and returns dictionary with zero padding as used by Fisher code ##

    dCldCLu = dict()

    dCldCLu['cl_TT_cl_TT'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClTTdClTT_"+dervtype+".dat"), ((2,0),), 'constant')
    dCldCLu['cl_TE_cl_TE'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClTEdClTE_"+dervtype+".dat"), ((2,0),), 'constant')
    dCldCLu['cl_EE_cl_EE'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClEEdClEE_"+dervtype+".dat"), ((2,0),), 'constant')
    dCldCLu['cl_EE_cl_BB'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClEEdClBB_"+dervtype+".dat"), ((2,0),), 'constant')
    dCldCLu['cl_BB_cl_EE'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClBBdClEE_"+dervtype+".dat"), ((2,0),), 'constant')
    dCldCLu['cl_BB_cl_BB'] = np.pad(np.loadtxt(classDataDir + "output/" + rootName + "_dClBBdClBB_"+dervtype+".dat"), ((2,0),), 'constant')

    return dCldCLu

###########################
## Copied from biasespol ##
###########################

def noiseSpectra(l, noiseLevelT, useSqrt2 = True, beamArcmin = 1.4, beamFile = None, noiseLevelP = None):
#make a full set of noise spectra.
    if beamFile == None:
        beam_sigma_radians =   (beamArcmin * np.pi / (180. * 60.)) / np.sqrt(8. * np.log(2) )
        beamPower = np.exp(l * (l+1) * beam_sigma_radians**2)
    else:
        beamVals = np.loadtxt(beamFile)
        beamValsOnL = (scipy.interpolate.interp1d(beamVals[:,0], beamVals[:,1], bounds_error = False))(l)
        beamPower = 1/(beamValsOnL**2)
    noise_ster = (np.pi / (180. * 60))**2 * noiseLevelT**2
    nl = len(l)
    cl_TT = np.empty(nl)
    cl_TT.fill(noise_ster)
    cl_TT *= beamPower
    cl_EE = np.empty(nl)
    if  useSqrt2:
        cl_EE.fill(noise_ster * 2.)
    else:
        noise_sterP = (np.pi / (180. * 60))**2 * noiseLevelP**2
        cl_EE.fill(noise_sterP)
    cl_EE *= beamPower
    cl_BB = np.empty(nl)
    if  useSqrt2:
        cl_BB.fill(noise_ster * 2.)
    else:
        noise_sterP = (np.pi / (180. * 60))**2 * noiseLevelP**2
        cl_BB.fill(noise_sterP)
    cl_BB *= beamPower
    cl_TE = np.empty(nl)
    cl_TE.fill(0.)

    output = {'l' : l,\
                  'cl_TT' : cl_TT,\
                  'cl_EE' : cl_EE,\
                  'cl_TE' : cl_TE,\
                  'cl_BB' : cl_BB,
                  'dl_TT' : cl_TT * l * (l + 1) / 2 / np.pi,\
                  'dl_EE' : cl_EE * l * (l + 1) / 2 / np.pi,\
                  'dl_TE' : cl_TE * l * (l + 1) / 2 / np.pi,\
                  'dl_BB' : cl_BB * l * (l + 1) / 2 / np.pi
              }

    return output


def getBAOParams(cosmo,
                    redshifts,
                    rootName = 'testing_bao',
                    classExecDir = os.path.dirname(os.path.abspath(__file__)) + '/CLASS_delens/',
                    classDataDir = os.path.dirname(os.path.abspath(__file__)) + '/CLASS_delens/',
                    extraParams = dict()
                    ):
    
    # returns rs/dV for desired redshifts
    
    bg_data = class_generate_data(cosmo = cosmo,
                        rootName = rootName,
                        classExecDir = classExecDir,
                        classDataDir = classDataDir,
                        backgroundOnly = True,
                        extraParams = extraParams
                        ):
    
    rs = bg_data['thermo_summary'][-1]
    
    zs = bg_data['background'][:,0]
    ang_diam_dist = bg_data['background'][:,5]
    Hz = bg_data['background'][:,3]
    c = 1.

    rs_dV = rs/(((c)*zs*((1+zs) ** 2.) * (ang_diam_dist ** 2.) * (Hz ** -1.)) ** (1./3))

    zs.reverse()
    rs_dV.reverse()
    
    return np.interp(redshifts,zs,rs_dV)