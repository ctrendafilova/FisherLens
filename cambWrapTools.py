import fisherTools
import os
import numpy
import sys
import camb
from camb import model, initialpower

TCMB = 2.7255

def getPyCambPowerSpectra(cosmo, accuracy = 2, lmaxToWrite = None, wantMatterPower = False, \
                              redshifts = None):

    lmax = lmaxToWrite + 1 if lmaxToWrite>0 else None

    #See CAMBDemo.html contained in pycamb/docs for usage information
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()

    #Use BBN consistency if Yhe is not passed
    if 'Yhe' in cosmo and isinstance(cosmo['Yhe'], float):
        Yhe = cosmo['Yhe']
    else:
        Yhe = None

    #Use either H0 or theta_s, but not both
    if 'H0' in cosmo:
        H0 = cosmo['H0']
        theta_s = None
        if 'theta_s' in cosmo:
            raise ValueError('Must pass either H0 or theta_s, not both!')
    elif 'theta_s' in cosmo:
        theta_s = cosmo['theta_s']
        H0 = None
        if 'H0' in cosmo:
            raise ValueError('Must pass either H0 or theta_s, not both!')
    else:
        H0 = None
        theta_s = None

    pars.set_cosmology(H0 = H0, cosmomc_theta = theta_s, ombh2 = cosmo['omega_b_h2'], \
                                omch2 = cosmo['omega_c_h2'], mnu = cosmo['mnu'], tau = cosmo['tau'],
                                nnu = cosmo['N_eff'] if 'N_eff' in list(cosmo.keys()) else 3.046, YHe = Yhe, \
                                omk = cosmo['omk'] if 'omk' in list(cosmo.keys()) else 0.0)
                                #tau_neutron = cosmo['neutron_lifetime'] if 'neutron_lifetime' in cosmo.keys() else 880.3)

    pars.InitPower.set_params(As = cosmo['A_s'], ns = cosmo['n_s'])

    if 'r' in list(cosmo.keys()) and cosmo['r'] != 0.:
        pars.InitPower.set_params(r = cosmo['r'])
        pars.WantTensors = True

    if ('DM_Pann' in list(cosmo.keys()) or 'fine_structure_multiplier' in list(cosmo.keys()) or 'electron_mass_multiplier' in list(cosmo.keys())):
        pars.Recomb.set_params(DM_Pann = cosmo['DM_Pann'] if 'DM_Pann' in list(cosmo.keys()) else 0.0, \
                                   FineS = cosmo['fine_structure_multiplier'] if 'fine_structure_multiplier' in list(cosmo.keys()) else 1.0, \
                                   EMass = cosmo['electron_mass_multiplier'] if 'electron_mass_multiplier' in list(cosmo.keys()) else 1.0)

    # CST changed this by hand from 6000
    pars.set_for_lmax(8102, lens_potential_accuracy = 2 * accuracy, lens_margin = 500, k_eta_fac = 4.)
    pars.set_accuracy(AccuracyBoost = accuracy, lSampleBoost = accuracy, lAccuracyBoost = accuracy,)

    pars.NonLinear = model.NonLinear_both

    if 'w' in list(cosmo.keys()):
        pars.set_dark_energy(w = cosmo['w'])


    if redshifts != None and wantMatterPower:
        pars.set_matter_power(redshifts=redshifts)

        # camb.set_z_outputs(redshifts)
    #calculate results for these parameter
    results = camb.get_results(pars)

    #get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars)
    #Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
    #The differenent CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).

    l = numpy.arange(powers['total'].shape[0])[2:lmax]

    dl_TT_lensed = powers['total'][2:lmax,0]*((TCMB*1.e6)**2)
    dl_EE_lensed = powers['total'][2:lmax,1]*((TCMB*1.e6)**2)
    dl_BB_lensed = powers['total'][2:lmax,2]*((TCMB*1.e6)**2)
    dl_TE_lensed = powers['total'][2:lmax,3]*((TCMB*1.e6)**2)

    cl_TT_lensed = dl_TT_lensed[:]*2*numpy.pi/(l*(l+1))
    cl_EE_lensed = dl_EE_lensed[:]*2*numpy.pi/(l*(l+1))
    cl_TE_lensed = dl_TE_lensed[:]*2*numpy.pi/(l*(l+1))
    cl_BB_lensed = dl_BB_lensed[:]*2*numpy.pi/(l*(l+1))

    lensed = {'l' : l,\
                  'cl_TT' : cl_TT_lensed,\
                  'cl_EE' : cl_EE_lensed,\
                  'cl_TE' : cl_TE_lensed,\
                  'cl_BB' : cl_BB_lensed,\

                  'dl_TT' : dl_TT_lensed,\
                  'dl_EE' : dl_EE_lensed,\
                  'dl_TE' : dl_TE_lensed,\
                  'dl_BB' : dl_BB_lensed,\

              }

    dl_TT_unlensed = powers['unlensed_total'][2:lmax,0]*((TCMB*1.e6)**2)
    dl_EE_unlensed = powers['unlensed_total'][2:lmax,1]*((TCMB*1.e6)**2)
    dl_BB_unlensed = powers['unlensed_total'][2:lmax,2]*((TCMB*1.e6)**2)
    dl_TE_unlensed = powers['unlensed_total'][2:lmax,3]*((TCMB*1.e6)**2)

    cl_TT_unlensed = dl_TT_unlensed[:]*2*numpy.pi/(l*(l+1))
    cl_EE_unlensed = dl_EE_unlensed[:]*2*numpy.pi/(l*(l+1))
    cl_TE_unlensed = dl_TE_unlensed[:]*2*numpy.pi/(l*(l+1))
    cl_BB_unlensed = dl_BB_unlensed[:]*2*numpy.pi/(l*(l+1))

    unlensed = {'l' : l,\
                  'cl_TT' : cl_TT_unlensed,\
                  'cl_EE' : cl_EE_unlensed,\
                  'cl_TE' : cl_TE_unlensed,\
                  'cl_BB' : cl_BB_unlensed,\

                  'dl_TT' : dl_TT_unlensed,\
                  'dl_EE' : dl_EE_unlensed,\
                  'dl_TE' : dl_TE_unlensed,\
                  'dl_BB' : dl_BB_unlensed,\

              }

    cl_phiphi = powers['lens_potential'][2:lmax,0]*2*numpy.pi/((l*(l+1))**2)
    cl_dd = cl_phiphi*(l*(l+1))
    cl_kk = cl_phiphi*(l*(l+1))*(l*(l+1))/4.

    lensing = {'l' : l,\
                  'cl_phiphi' : cl_phiphi,\
                  'cl_dd' : cl_dd,\
                  'cl_kk' : cl_kk, \
              }
    print("Testing H0 output:")
    print(pars.H0)

    output = {'unlensed' : unlensed, 'lensed' : lensed, 'lensing' : lensing}

    if wantMatterPower:


        matter = dict()
        matter['kh'], matter['zs'], matter['PK'] = results.get_matter_power_spectrum()

        output['matter'] = matter

    return output



def getCambPowerSpectra(cosmo, rootName = 'testing' , cambDir = '../CAMB/', useMassiveNeutrinos = False):

    zs = [0.]

    H0 = cosmo['H0']

    if 'Yhe' in cosmo and isinstance(cosmo['Yhe'], float):
        Yhe = cosmo['Yhe']
    else:
        Yp = bbn.yhe_fit(cosmo['omega_b_h2'], cosmo['N_eff']-3.046, 880.3)
        Yhe = bbn.ypBBN_to_yhe(Yp)


    (k, pk) = camb_pks(zs, cosmo, H0, Yhe, rootName = rootName, cambDir = cambDir, useMassiveNeutrinos = useMassiveNeutrinos)

    unlensedSpec = loadUpSpecs(cambDir + rootName + '_scalCls.dat' , isLens = False)

    lensedSpec = loadUpSpecs(cambDir + rootName  + '_lensedCls.dat' , isLens = True)

    lensingSpec = loadUpLensSpec(cambDir + rootName + '_scalCls.dat')



    return {'unlensed' : unlensedSpec, 'lensed' : lensedSpec, 'lensing'  : lensingSpec}


def get_H0_from_theta(cosmo):
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()

    #Use BBN consistency if Yhe is not passed
    if 'Yhe' in cosmo and isinstance(cosmo['Yhe'], float):
        Yhe = cosmo['Yhe']
    else:
        Yhe = None

    #Use either H0 or theta_s, but not both
    if 'H0' in cosmo:
        H0 = cosmo['H0']
        theta_s = None
        if 'theta_s' in cosmo:
            raise ValueError('Must pass either H0 or theta_s, not both!')
    elif 'theta_s' in cosmo:
        theta_s = cosmo['theta_s']
        H0 = None
        if 'H0' in cosmo:
            raise ValueError('Must pass either H0 or theta_s, not both!')
    else:
        H0 = None
        theta_s = None

    pars.set_cosmology(H0 = H0, cosmomc_theta = theta_s, ombh2 = cosmo['omega_b_h2'], \
                                omch2 = cosmo['omega_c_h2'], mnu = cosmo['mnu'], tau = cosmo['tau'],
                                nnu = cosmo['N_eff'] if 'N_eff' in list(cosmo.keys()) else 3.046, YHe = Yhe, \
                                omk = cosmo['omk'] if 'omk' in list(cosmo.keys()) else 0.0)
    if 'w' in list(cosmo.keys()):
        pars.set_dark_energy(w = cosmo['w'])

    return pars.H0


def getBAOParams(cosmo, redshifts):
    # Returns rs/DV, H, DA, F_AP for each requested redshift (as 2D array)
    #Use either H0 or theta_s, but not both
    pars = camb.CAMBparams()

    #Use BBN consistency if Yhe is not passed
    if 'Yhe' in cosmo and isinstance(cosmo['Yhe'], float):
        Yhe = cosmo['Yhe']
    else:
        Yhe = None

    if 'H0' in cosmo:
        H0 = cosmo['H0']
        theta_s = None
        if 'theta_s' in cosmo:
            raise ValueError('Must pass either H0 or theta_s, not both!')
    elif 'theta_s' in cosmo:
        theta_s = cosmo['theta_s']
        H0 = None
        if 'H0' in cosmo:
            raise ValueError('Must pass either H0 or theta_s, not both!')
    else:
        H0 = None
        theta_s = None
    pars.set_cosmology(H0 = H0, cosmomc_theta = theta_s, ombh2 = cosmo['omega_b_h2'], \
                            omch2 = cosmo['omega_c_h2'], mnu = cosmo['mnu'], tau = cosmo['tau'],
                            nnu = cosmo['N_eff'] if 'N_eff' in list(cosmo.keys()) else 3.046, YHe = Yhe, \
                            omk = cosmo['omk'] if 'omk' in list(cosmo.keys()) else 0.0)

    if 'w' in list(cosmo.keys()):
        pars.set_dark_energy(w = cosmo['w'])

    pars.z_outputs=redshifts
    bg = camb.get_background(pars)
    BAO = bg.get_background_outputs()
    return BAO



def camb_pks(zs, cosmo, H0, Yhe, rootName, cambDir = '../CAMB/', justReadFromDisk = False, useMassiveNeutrinos = False):


    if useMassiveNeutrinos:
        Omnuh2 = cosmo['mnu'] / 94.060
        nu_mass_eigenstates = 1
        massive_neutrinos = 1
        massless_neutrinos = cosmo['N_eff'] - 1.
    else:
        Omnuh2 = 0
        nu_mass_eigenstates = 0
        massive_neutrinos = 0
        massless_neutrinos = cosmo['N_eff']



    outputPs = fisherTools.onedl( len(zs))
    outputKs = fisherTools.onedl( len(zs))

    if justReadFromDisk == False:
            #then we run camb.  otherwise the data should just be read in from a previous run.

        camb_ini_head =     """
    #Parameters for CAMB

    #output_root is prefixed to output file names
    output_root = """ + rootName + """

    #What to do
    get_scalar_cls = T
    get_vector_cls = F
    get_tensor_cls = F
    get_transfer   = T

    #if do_lensing then scalar_output_file contains additional columns of l^4 C_l^{pp} and l^3 C_l^{pT}
    #where p is the projected potential. Output lensed CMB Culs (without tensors) are in lensed_output_file below.
    do_lensing     = T

    # 0: linear, 1: non-linear matter power (HALOFIT), 2: non-linear CMB lensing (HALOFIT),
    # 3: both non-linear matter power and CMB lensing (HALOFIT)
    do_nonlinear = 3

    #Maximum multipole and k*eta.
    #  Note that C_ls near l_max are inaccurate (about 5%), go to 50 more than you need
    #  Lensed power spectra are computed to l_max_scalar-100
    #  To get accurate lensed BB need to have l_max_scalar>2000, k_eta_max_scalar > 10000
    #  To get accurate lensing potential you also need k_eta_max_scalar > 10000
    #  Otherwise k_eta_max_scalar=2*l_max_scalar usually suffices, or don't set to use default
    l_max_scalar      = 6500
    k_eta_max_scalar  = 65000

    #  Tensor settings should be less than or equal to the above
    l_max_tensor      = 1500
    k_eta_max_tensor  = 3000

    #Main cosmological parameters, neutrino masses are assumed degenerate
    # If use_phyical set physical densities in baryons, CDM and neutrinos + Omega_k
    use_physical   = T
    ombh2          = """ + str(cosmo['omega_b_h2'])  + """
    omch2          = """ + str(cosmo['omega_c_h2'])  + """
    omnuh2         = """ + str(Omnuh2) + """
    omk            = 0
    hubble         = """ + str(H0)  + """

    #effective equation of state parameter for dark energy
    w              = -1
    #constant comoving sound speed of the dark energy (1=quintessence)
    cs2_lam        = 1

    #varying w is not supported by default, compile with EQUATIONS=equations_ppf to use crossing PPF w-wa model:
    #wa             = 0
    ##if use_tabulated_w read (a,w) from the following user-supplied file instead of above
    #use_tabulated_w = F
    #wafile = wa.dat

    #if use_physical = F set parameters as here
    #omega_baryon   = 0.0462
    #omega_cdm      = 0.2538
    #omega_lambda   = 0.7
    #omega_neutrino = 0

    temp_cmb           = 2.7255
    helium_fraction    = """ + str(Yhe)  + """

    #for share_delta_neff = T, the fractional part of massless_neutrinos gives the change in the effective number
    #(for QED + non-instantaneous decoupling)  i.e. the increase in neutrino temperature,
    #so Neff = massless_neutrinos + sum(massive_neutrinos)
    #For full neutrino parameter details see http://cosmologist.info/notes/CAMB.pdf
    massless_neutrinos = """ + str(massless_neutrinos)  + """

    #number of distinct mass eigenstates
    nu_mass_eigenstates = """ + str(nu_mass_eigenstates) + """
    #array of the integer number of physical neutrinos per eigenstate, e.g. massive_neutrinos = 2 1
    massive_neutrinos  = """ + str(massive_neutrinos) + """
    #specify whether all neutrinos should have the same temperature, specified from fractional part of massless_neutrinos
    share_delta_neff = T
    #nu_mass_fractions specifies how Omeganu_h2 is shared between the eigenstates
    #i.e. to indirectly specify the mass of each state; e.g. nu_mass_factions= 0.75 0.25
    nu_mass_fractions = 1
    #if share_delta_neff = F, specify explicitly the degeneracy for each state (e.g. for sterile with different temperature to active)
    #(massless_neutrinos must be set to degeneracy for massless, i.e. massless_neutrinos does then not include Deleta_Neff from massive)
    #if share_delta_neff=T then degeneracies is not given and set internally
    #e.g. for massive_neutrinos = 2 1, this gives equal temperature to 4 neutrinos: nu_mass_degeneracies = 2.030 1.015, massless_neutrinos = 1.015
    nu_mass_degeneracies =

    #Initial power spectrum, amplitude, spectral index and running. Pivot k in Mpc^{-1}.
    initial_power_num         = 1
    pivot_scalar              = 0.05
    pivot_tensor              = 0.05
    scalar_amp(1)             = """ + str(cosmo['A_s']) + """
    scalar_spectral_index(1)  = """ + str(cosmo['n_s']) + """
    scalar_nrun(1)            = 0
    tensor_spectral_index(1)  = 0
    #ratio is that of the initial tens/scal power spectrum amplitudes
    initial_ratio(1)          = 1
    #note vector modes use the scalar settings above


    #Reionization, ignored unless reionization = T, re_redshift measures where x_e=0.5
    reionization         = T


    re_use_optical_depth = T
    re_optical_depth     = """ + str(cosmo['tau']) + """
    #If re_use_optical_depth = F then use following, otherwise ignored
    re_redshift          = 11
    #width of reionization transition. CMBFAST model was similar to re_delta_redshift~0.5.
    re_delta_redshift    = 1.5
    #re_ionization_frac=-1 sets to become fully ionized using YE to get helium contribution
    #Otherwise x_e varies from 0 to re_ionization_frac
    re_ionization_frac   = -1


    #RECFAST 1.5.x recombination parameters;
    RECFAST_fudge = 1.14
    RECFAST_fudge_He = 0.86
    RECFAST_Heswitch = 6
    RECFAST_Hswitch  = T

    # CosmoMC parameters - compile with RECOMBINATION=cosmorec and link to CosmoMC to use these
    #
    # cosmorec_runmode== 0: CosmoMC run with diffusion
    #                    1: CosmoMC run without diffusion
    #                    2: RECFAST++ run (equivalent of the original RECFAST version)
    #                    3: RECFAST++ run with correction function of Calumba & Thomas, 2010
    #
    # For 'cosmorec_accuracy' and 'cosmorec_fdm' see CosmoMC for explanation
    #---------------------------------------------------------------------------------------
    #cosmorec_runmode        = 0
    #cosmorec_accuracy       = 0
    #cosmorec_fdm            = 0

    #Initial scalar perturbation mode (adiabatic=1, CDM iso=2, Baryon iso=3,
    # neutrino density iso =4, neutrino velocity iso = 5)
    initial_condition   = 1
    #If above is zero, use modes in the following (totally correlated) proportions
    #Note: we assume all modes have the same initial power spectrum
    initial_vector = -1 0 0 0 0

    #For vector modes: 0 for regular (neutrino vorticity mode), 1 for magnetic
    vector_mode = 0

    #Normalization
    COBE_normalize = F
    ##CMB_outputscale scales the output Culs
    #To get MuK^2 set realistic initial amplitude (e.g. scalar_amp(1) = 2.3e-9 above) and
    #otherwise for dimensionless transfer functions set scalar_amp(1)=1 and use
    #CMB_outputscale = 1
    CMB_outputscale = 7.42835025e12

    #Transfer function settings, transfer_kmax=0.5 is enough for sigma_8
    #transfer_k_per_logint=0 sets sensible non-even sampling;
    #transfer_k_per_logint=5 samples fixed spacing in log-k
    #transfer_interp_matterpower =T produces matter power in regular interpolated grid in log k;
    # use transfer_interp_matterpower =F to output calculated values (e.g. for later interpolation)
    transfer_high_precision = T
    transfer_kmax           = 100
    transfer_k_per_logint   = 5
    transfer_num_redshifts  = """ + str(len(zs)) + """
    transfer_interp_matterpower = T"""

    camb_ini_middle = ""
    for j , z, in enumerate(zs[::-1] ):
        camb_ini_middle += """

    transfer_redshift(""" + str(j+1) + ") = " + str(z) + """
    transfer_filename(""" + str(j+1) + ") =  transfer_out%06i.dat" %(j+1) + """
    #Matter power spectrum output against k/h in units of h^{-3} Mpc^3
    transfer_matterpower(""" + str(j+1) + ") =  matterpower%06i.dat" %(j+1) + """


                """
    camb_ini_end =      """

    #Output files not produced if blank. make camb_fits to use the FITS setting.
    scalar_output_file = scalCls.dat
    vector_output_file = vecCls.dat
    tensor_output_file = tensCls.dat
    total_output_file  = totCls.dat
    lensed_output_file = lensedCls.dat
    lensed_total_output_file  =lensedtotCls.dat
    lens_potential_output_file = lenspotentialCls.dat
    FITS_filename      = scalCls.fits

    #Bispectrum parameters if required; primordial is currently only local model (fnl=1)
    #lensing is fairly quick, primordial takes several minutes on quad core
    do_lensing_bispectrum = F
    do_primordial_bispectrum = F

    #1 for just temperature, 2 with E
    bispectrum_nfields = 1
    #set slice non-zero to output slice b_{bispectrum_slice_base_L L L+delta}
    bispectrum_slice_base_L = 0
    bispectrum_ndelta=3
    bispectrum_delta(1)=0
    bispectrum_delta(2)=2
    bispectrum_delta(3)=4
    #bispectrum_do_fisher estimates errors and correlations between bispectra
    #note you need to compile with LAPACK and FISHER defined to use get the Fisher info
    bispectrum_do_fisher= F
    #Noise is in muK^2, e.g. 2e-4 roughly for Planck temperature
    bispectrum_fisher_noise=0
    bispectrum_fisher_noise_pol=0
    bispectrum_fisher_fwhm_arcmin=7
    #Filename if you want to write full reduced bispectrum (at sampled values of l_1)
    bispectrum_full_output_file=
    bispectrum_full_output_sparse=F
    #Export alpha_l(r), beta_l(r) for local non-Gaussianity
    bispectrum_export_alpha_beta=F

    ##Optional parameters to control the computation speed,accuracy and feedback

    #If feedback_level > 0 print out useful information computed about the model
    feedback_level = 1

    #write out various derived parameters
    derived_parameters = T

    # 1: curved correlation function, 2: flat correlation function, 3: inaccurate harmonic method
    lensing_method = 1
    accurate_BB = F


    #massive_nu_approx: 0 - integrate distribution function
    #                   1 - switch to series in velocity weight once non-relativistic
    massive_nu_approx = 1

    #Whether you are bothered about polarization.
    accurate_polarization   = T

    #Whether you are bothered about percent accuracy on EE from reionization
    accurate_reionization   = T

    #whether or not to include neutrinos in the tensor evolution equations
    do_tensor_neutrinos     = T

    #Whether to turn off small-scale late time radiation hierarchies (save time,v. accurate)
    do_late_rad_truncation   = T

    #Computation parameters
    #if number_of_threads=0 assigned automatically
    number_of_threads       = 0

    #Default scalar accuracy is about 0.3% (except lensed BB) if high_accuracy_default=F
    #If high_accuracy_default=T the default target accuracy is 0.1% at L>600 (with boost parameter=1 below)
    #Try accuracy_boost=2, l_accuracy_boost=2 if you want to check stability/even higher accuracy
    #Note increasing accuracy_boost parameters is very inefficient if you want higher accuracy,
    #but high_accuracy_default is efficient

    high_accuracy_default=T

    #Increase accuracy_boost to decrease time steps, use more k values,  etc.
    #Decrease to speed up at cost of worse accuracy. Suggest 0.8 to 3.
    accuracy_boost          = 1

    #Larger to keep more terms in the hierarchy evolution.
    l_accuracy_boost        = 1

    #Increase to use more C_l values for interpolation.
    #Increasing a bit will improve the polarization accuracy at l up to 200 -
    #interpolation errors may be up to 3%
    #Decrease to speed up non-flat models a bit
    l_sample_boost          = 1
            """
    print('*** writing to ', cambDir  + rootName + ".ini")
    textfile = open(cambDir  + rootName + ".ini", "w")


    textfile.write(camb_ini_head + camb_ini_middle + camb_ini_end)
    textfile.close()

    print('*** running:', "cd " + cambDir + " ; ./camb " + rootName + ".ini")
    os.system("cd " + cambDir + " ; ./camb " + rootName + ".ini")
# end of if statement for justReadFromDisk flag.

    for j in range(len(zs)):
        data = numpy.loadtxt(cambDir + '/' + rootName + "_matterpower%06i.dat" %(j + 1))

        outputPs[len(zs) - j - 1] = (data[:,1].copy())
        outputPs[len(zs) - j - 1] *= 1./(H0*0.01)**3

        outputKs[len(zs) - j - 1] = data[:,0].copy()
        outputKs[len(zs) - j - 1] *= (H0*0.01)


    return (outputKs, outputPs)

def loadUpSpecs(filename, isLens = False):

    #camb file
    theoryPower = numpy.loadtxt(filename)
    l=theoryPower[:,0]
    dl_TT=theoryPower[:,1]
    dl_EE=theoryPower[:,2]

#there are different orderings for scalCls.dat and for lensedCls.dat files.
    if not isLens:
        dl_BB = numpy.zeros(len(l))
        dl_TE=theoryPower[:,3]
    else:
        dl_BB = theoryPower[:,3]
        dl_TE = theoryPower[:,4]



    cl_TT=dl_TT[:]*2*numpy.pi/(l*(l+1))
    cl_EE=dl_EE[:]*2*numpy.pi/(l*(l+1))
    cl_TE=dl_TE[:]*2*numpy.pi/(l*(l+1))
    cl_BB=dl_BB[:]*2*numpy.pi/(l*(l+1))

    myzeros = numpy.zeros(len(l))

    output = {'l' : l,\
                  'cl_TT' : cl_TT,\
                  'cl_EE' : cl_EE,\
                  'cl_TE' : cl_TE,\
                  'cl_BB' : cl_BB,\


                  'dl_TT' : dl_TT,\
                  'dl_EE' : dl_EE,\
                  'dl_TE' : dl_TE,\
                  'dl_BB' : dl_BB,\

              }
    return output

def loadUpLensSpec(filename):
    theoryPower = numpy.loadtxt(filename)
    l=theoryPower[:,0]

    cl_phiphi = theoryPower[:,4] / cmbTempUK**2 / l**4

    return {'l' : l, \
            'cl_phiphi' : cl_phiphi,\
            'cl_dd' : l**2 * cl_phiphi,\
            'cl_kk' : l**4 * cl_phiphi / 4}
