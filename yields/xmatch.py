import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from astroquery.gaia import Gaia
from astropy.table import Table
from itertools import chain
from astroquery.mast import Catalogs 
import pdb


'''
Noise model, adopted from a TESS code, and applicable to the SMEX concept.

To use: change global telescope parameters below (specifically
"effective_area", "sys_limit", and "pix_scale" as needed for the SMEX concept).
To generate RMS vs mag plots at the southern ecliptic pole and the galactic
center, run

    $ python noise_model.py

The main contents are in the `noise_model(...)` function.  Given source T mag,
and a coordinate, this function gives predicted RMS for the source.  It does so
by assuming the PSF matches that of TESS, but using the appropriate pixel
scale.  So in practice, it assumes the images are nearly critically sampled.

History:

This code is derivative of Zach Berta-Thompson's SNR calculator, which was
derivative of Josh Winn's IDL TESS SNR calculator (which probably had
comparable provenance down the line).
Zach's is at https://github.com/zkbt/spyffi/blob/master/Noise.py.

Author: Luke Bouma.
Date: Sun Aug  7 12:42:10 2022
'''
from __future__ import division, print_function

import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

resultsdir = './' # directory to which plots are written
if not os.path.exists(resultsdir): os.mkdir(resultsdir)

###############################################################################
# Fixed telescope properties are kept as globals.

global subexptime, e_pix_ro, effective_area, sys_limit, pix_scale

# subexposure time [seconds] (n_exp = exptime/subexptime)
subexptime = 2.0
# rms in no. photons/pixel from readout noise
e_pix_ro = 10.0
# the TESS geometric collecting area.
TESS_effective_area= 86.6
# adopted geometric collecting area. assume 16x TESS
effective_area = 16*TESS_effective_area
# minimum uncertainty in 1 hr of data, in ppm
sys_limit = 60.0
# arcsec per pixel.  21 arcsec/px for TESS.  assume 6 for SMEX.
pix_scale = 6

###############################################################################

def N_pixels_in_aperture_Sullivan(T):
    '''
    optimal n_pixels in aperture according to S+15. Less wonky structure at
    faint end compared to Jenkins.
    '''
    from scipy.interpolate import interp1d
    df = pd.read_csv(
            'Sullivan_2015_optimalnumberofpixels.txt',
            comment='#', delimiter=','
    )
    tmag = np.array(df['tmag']) # it's really I_C
    npix = np.array(df['npix'])
    func = interp1d(tmag, npix)
    return func(T)


def photon_flux_from_source(T_mag):
    '''
    in:
        T_mag (np.ndarray): of the source(s)

    out:
        photon flux from the source in the TESS band [units: ph/s/cm^2].
    '''

    # Zero point stated in Sullivan et al 2015:
    # A T=0 star gives a photon flux of 1.514e6 ph/s/cm^2.

    F_T0 = 1.514e6

    F = 10**(-0.4 * ( T_mag )) * F_T0

    return F


def get_sky_bkgnd(coords, exptime):
    '''
    in:
        input coordinate (astropy SkyCoord instance)

        exposure time (seconds)

    out:
        sky background from zodiacal light at coords [units: e/px]

        (NB. background stars are accounted in post-processing by the TSWG's
        synthetic image procedure)
    '''

    elat = coords.barycentrictrueecliptic.lat.value
    elon = coords.barycentrictrueecliptic.lon.value
    glat = coords.galactic.b.value
    glon = coords.galactic.l.value
    glon -= 180 # Winn's eq 7 (below) requires longitude from -180 to 180
    assert np.all(glon > -180) and np.all(glon < 180)

    # Solid area of a pixel (arcsec^2).
    omega_pix = pix_scale ** 2.

    # Photoelectrons/pixel from zodiacal light.
    dlat = (np.abs(elat) - 90.) / 90.
    vmag_zodi = 23.345 - 1.148 * dlat ** 2.

    # Eqn (3) from Josh Winn's memo on sky backgrounds. This comes from
    # integrating a model ZL spectrum over the TESS bandpass.
    e_pix_zodi = (
      10.0 ** (-0.4 * (vmag_zodi - 22.8)) * 2.39e-3
      * effective_area * omega_pix * exptime
    )

    # You also need the diffuse background from unresolved stars. Eq (7) from
    # the same memo
    a0, a1, a2, a3 = 18.9733, 8.833, 4.007, 0.805
    I_surface_brightness = (
      a0 + a1 * np.abs(glat)/40 + a2 * (np.abs(glat)/180)**a3
    )
    e_pix_faintstars = (
      10.0 ** (-0.4 * I_surface_brightness) * 1.7e6
      * effective_area * omega_pix * exptime
    )

    # for CDIPS, zodi should be small, photon flux due to unresolved stars
    # should dominate

    return e_pix_faintstars


def noise_model(
        T_mags,
        coords,
        exptime=120):
    '''
    ----------
    Mandatory inputs:

    either all floats, or else all 1d numpy arrays of length N_sources.

        T_mags:
            TESS magnitude of the source(s)

        coords:
            target coordinates, a (N_sources * 2) numpy array of (ra, dec),
            specified in degrees.

    ----------
    Optional inputs:


        exptime (float):
            total exposure time in seconds. Must be a multiple of 2 seconds.

    ----------
    Returns:

        [N_sources x 6] array of:
            number of pixels in selected apertures,
            noise for selected number of pixels,
            each of the noise components (star, sky, readout, systematic).

    '''

    # Check inputs. Convert coordinates to astropy SkyCoord instance.
    if not isinstance(T_mags, np.ndarray):
        T_mags = np.array([T_mags])
    assert isinstance(coords, np.ndarray)
    if len(coords.shape)==1:
        coords = coords.reshape((1,2))
    assert coords.shape[1] == 2

    coords = SkyCoord(
                 ra=coords[:,0]*units.degree,
                 dec=coords[:,1]*units.degree,
                 frame='icrs'
                 )

    assert exptime % subexptime == 0, \
            'Exposure time must be multiple of 2 seconds.'
    assert T_mags.shape[0] == coords.shape[0]

    # Basic quantities.
    N_sources = len(T_mags)
    N_exposures = exptime/subexptime

    # Photon flux from source in ph/s/cm^2.
    f_ph_source = np.array(photon_flux_from_source(T_mags))

    # Compute number of photons from source, per exposure.
    ph_source = f_ph_source * effective_area * exptime

    # Load in average PRF produced by `ctd_avg_field_angle_avg.py`.
    # (this is reshaped from 17x17 pixels)
    prf_file = os.path.join('average_PRF.fits')
    hdu = fits.open(prf_file)
    avg_PRF = hdu[0].data

    # Compute cumulative flux fraction, sort s.t. the brightest pixel is first.
    # NOTE: since this is taking the TESS PRF, we're still assuming ~critically
    # sampled PRF.
    CFF = np.cumsum(np.sort(avg_PRF)[::-1])

    # For each source, compute the number of photons collected (in each
    # exposure) as a function of aperture size. Save as array of [N_sources *
    # N_pixels_in_aperture].
    ph_source_all_ap = ph_source[:, None] * CFF[None, :]

    # Convert to number of electrons collected as a function of aperture size.
    # These are the same, since Josh Winn's photon flux formula already
    # accounts for the quantum efficiency.
    e_star_all_ap = ph_source_all_ap

    e_sky = get_sky_bkgnd(coords, exptime)

    # Array of possible aperture sizes: [1,2,...,max_N_ap]
    N_pix_aper = np.array(range(1,len(CFF)+1))

    e_sky_all_ap = e_sky[:, None] * N_pix_aper[None, :]

    ##########################################################################
    # Using the analytic N_pix(T_mag) given to the TSWG by Jon Jenkins, find #
    # the resulting standard deviation in the counts in the aperture.        #
    ##########################################################################

    # N_pix_sel = N_pix_in_aperture_Jenkins(T_mags)
    # use sullivan's instead
    N_pix_sel = N_pixels_in_aperture_Sullivan(T_mags)
    N_pix_sel[(N_pix_sel < 3) | (N_pix_sel > 50)] = 3
    # leave N_pix_sel as float, for smooth display at the end
    N_pix_sel = np.round(
            np.maximum(3*np.ones_like(N_pix_sel),N_pix_sel)).astype(int)

    assert np.max(N_pix_sel) < np.max(N_pix_aper), \
            'maximum aperture size is 17px squared'

    # Indices in the dimension over all possible aperture sizes that correspond to
    # the desired number of pixels in the aperture.
    sel_inds = np.round(N_pix_sel).astype(int) - 1

    # Report the noise and number of pixels for the selected aperture size.
    e_star_sel_ap = []
    e_sky_sel_ap = []
    for ix, sel_ind in enumerate(sel_inds):
        if sel_ind > 50 or sel_ind < 2:
            sel_ind = 2
        e_star_sel_ap.append(e_star_all_ap[ix,sel_ind])
        e_sky_sel_ap.append(e_sky_all_ap[ix,sel_ind])
    e_star_sel_ap = np.array(e_star_sel_ap)
    e_sky_sel_ap = np.array(e_sky_sel_ap)

    noise_star_sel_ap = np.sqrt(e_star_sel_ap) / e_star_sel_ap

    noise_sky_sel_ap = np.sqrt(N_pix_sel * e_sky_sel_ap) / e_star_sel_ap

    noise_ro_sel_ap = np.sqrt(N_pix_sel * N_exposures) * e_pix_ro / e_star_sel_ap

    noise_sys_sel_ap = np.zeros_like(e_star_sel_ap) \
                       + sys_limit / 1e6 / np.sqrt(exptime / 3600.)

    noise_sel_ap = np.sqrt(noise_star_sel_ap ** 2. +
                           noise_sky_sel_ap ** 2. +
                           noise_ro_sel_ap ** 2. +
                           noise_sys_sel_ap ** 2.)

    return np.array(
            [N_pix_sel,
             noise_sel_ap,
             noise_star_sel_ap,
             noise_sky_sel_ap,
             noise_ro_sel_ap,
             noise_sys_sel_ap]
            )



# Young stars from Kerr et al. (Table 2)
ymg=ascii.read('apjac0251t2_mrt.txt')
ix,iy=np.unique(ymg['Gaia'],return_index=True)
ymg=ymg[iy]

# upload table to Gaia server
tables = Gaia.load_tables(only_names=True)
for table in (tables):
    if 'gaiadr3' in table.get_qualified_name():
        print (table.get_qualified_name())

#@title Provide your Gaia archive user name
username = 'dhuber' #@param {type:"string"}
Gaia.login(user = username)

Gaia.delete_user_table('ymg')
Gaia.upload_table(upload_resource=ymg, table_name='ymg')


# cross-match with Gaia
job = Gaia.launch_job_async("SELECT xm.*, ap.*, gaia.*, ymg.* FROM gaiadr3.dr2_neighbourhood AS xm JOIN gaiadr3.gaia_source AS gaia ON xm.dr3_source_id = gaia.source_id JOIN gaiadr3.astrophysical_parameters AS ap ON xm.dr3_source_id = ap.source_id JOIN user_dhuber.ymg AS ymg ON xm.dr2_source_id  = ymg.gaia")


xm = job.get_results()

keep=np.zeros(len(xm))
for i in range(0,len(xm)-1):
	# check if a DR2 ID was matched to more than one DR3 ID
	um=np.where(xm['gaia']==xm['gaia'][i])[0]
	# if yes, keep the star with the closest mag 
	if (len(um) > 1):
		um2=np.where(np.argmin(xm['magnitude_difference'][um]))
		keep[um[um2[0]]]=1
	else:
		keep[i]=1

len(np.unique(xm['gaia']))
um=np.where(keep > 0)[0]
len(um)	
xm=xm[um]

# xmatch with the TESS Input Catalog
result = Catalogs.query_criteria(catalog="Tic", GAIA=xm['dr2_source_id'])
ix,iy=np.unique(result['GAIA'],return_index=True)
result2=result[iy]

ix,iy=match.match(np.asarray(result2['GAIA'],dtype='int'),xm['dr2_source_id'])
tmags=result2['Tmag'][ix]
ticids=result2['ID'][ix]
xm=xm[iy]
xm.add_columns([tmags,ticids],names=['tmag','tic'])


um=np.where(xm['teff_gspphot'] > 0)[0]
print(len(um),' total stars with teff_gspphot')

um=np.where(xm['teff_gspspec'] > 0)[0]
print(len(um),' total stars with teff_gspspec')

um=np.where(xm['teff_esphs'] > 0)[0]
print(len(um),' total stars with teff_esphs')


#flags=np.zeros(len(xm),dtype='int')+9
#for i in range(0,len(xm)):
#	x=xm['flags_gspspec'][i]
#	if len(x) > 0:
#		flags[i]=x[0]
		
	
teff_gspspec=np.zeros(len(ymg))
logg_gspspec=np.zeros(len(ymg))
teff_gspphot=np.zeros(len(ymg))
radius_gspphot=np.zeros(len(ymg))
teff_esphs=np.zeros(len(ymg))
logg_esphs=np.zeros(len(ymg))
tmags=np.zeros(len(ymg))
ticids=np.zeros(len(ymg),dtype='int')

ix,iy=match.match(xm['dr2_source_id'],ymg['Gaia'])

teff_gspspec[iy]=xm['teff_gspspec'][ix]
logg_gspspec[iy]=xm['logg_gspspec'][ix]
teff_gspphot[iy]=xm['teff_gspphot'][ix]
radius_gspphot[iy]=xm['radius_gspphot'][ix]
teff_esphs[iy]=xm['teff_esphs'][ix]
logg_esphs[iy]=xm['logg_esphs'][ix]
tmags[iy]=xm['tmag'][ix]
ticids[iy]=xm['tic'][ix]

# calculate noise using functions by Luke Bouma, assuming 1-min sampling
noise=np.zeros(len(ymg))
for i in range(len(ymg)-1):
	print(i)
	n=noise_model(tmags[i],np.array((ymg['RAdeg'][i],ymg['DEdeg'][i])),exptime=60)
	noise[i]=n[1]


ymg.add_columns([teff_gspphot,radius_gspphot,teff_gspspec,logg_gspspec,teff_esphs,logg_esphs,ticids,tmags,noise],names=['teff_gspphot','radius_gspphot','teff_gspspec','logg_gspspec','teff_esphs','logg_esphs','tic','tmag','noise'])

s=np.argsort(ymg['ID'])

ascii.write(ymg[s],'ymg-gaia-table2.csv',delimiter=',',overwrite=True)


## plots

plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.size']=16
plt.rcParams['mathtext.default']='regular'
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='8'

plt.ion()
plt.clf()
plt.subplot(2,1,1)
plt.loglog(ymg['teff_gspphot'],ymg['teff_gspspec'],'.',label='teff_comp=teff_gspspec')
plt.plot(ymg['teff_gspphot'],ymg['teff_esphs'],'.',label='teff_comp=teff_esphs')
plt.plot([1000,30000],[1000,30000],color='red',ls='dashed')
plt.xlabel('teff_gspphot (K)')
plt.ylabel('teff_comp (K)')
plt.xlim([2500,30000])
plt.ylim([2500,30000])
plt.legend()
plt.subplot(2,1,2)
plt.semilogx(ymg['teff_gspphot'],ymg['teff_gspphot']/ymg['teff_gspspec'],'.',label='teff_comp=teff_gspspec')
plt.plot(ymg['teff_gspphot'],ymg['teff_gspphot']/ymg['teff_esphs'],'.',label='teff_comp=teff_esphs')
plt.xlim([2500,30000])
plt.ylim([0.5,1.5])
plt.xlabel('teff_gspphot (K)')
plt.ylabel('teff_gspphot/teff_comp')
plt.tight_layout()
plt.plot([1000,30000],[1,1],color='red',ls='dashed')
plt.savefig('teff-comp1.png',dpi=200)

steindl=np.array((18000,18236,22919,18074,20473,19000,8375,9500,8000,9750,7500,6750,7300,6784,7170,7520))
gaia=np.array((5793.603,17108.633,18524.545,15180.93,18184.324,24848.676,8849.717,7834.8735,7750.0,10398.14,0,0,6635.4927,6958.36,7210.3154,7877.321))

plt.ion()
plt.clf()
plt.subplot(2,1,1)
plt.loglog(steindl,gaia,'o')
plt.plot([1000,30000],[1000,30000],color='red',ls='dashed')
plt.xlabel('teff_literature (K)')
plt.ylabel('teff_gaia (K)')
plt.xlim([2500,30000])
plt.ylim([2500,30000])
plt.legend()
plt.subplot(2,1,2)
plt.semilogx(steindl,steindl/gaia,'o')
plt.xlim([2500,30000])
plt.ylim([0.5,1.5])
plt.xlabel('teff_literature (K)')
plt.ylabel('teff_literature/teff_gaia')
plt.tight_layout()
plt.plot([1000,30000],[1,1],color='red',ls='dashed')
plt.savefig('teff-comp2.png',dpi=200)


















## MISC

# isoclassify input
'''
job = Gaia.launch_job_async("SELECT xm.*, gaia.*, ymg.* FROM gaiadr3.dr2_neighbourhood AS xm JOIN gaiadr3.gaia_source AS gaia ON xm.dr3_source_id = gaia.source_id JOIN user_dhuber.ymg AS ymg ON xm.dr2_source_id  = ymg.gaia")
xm = job.get_results()

keep=np.zeros(len(xm))
for i in range(0,len(xm)-1):
	# check if a DR2 ID was matched to more than one DR3 ID
	um=np.where(xm['gaia']==xm['gaia'][i])[0]
	# if yes, keep the star with the closest mag 
	if (len(um) > 1):
		um2=np.where(np.argmin(xm['magnitude_difference'][um]))
		keep[um[um2[0]]]=1
	else:
		keep[i]=1
um=np.where((keep == 1) & (xm['parallax'] > 0.) & (xm['phot_rp_mean_mag'] > 0.) & (xm['phot_bp_mean_mag'] > 0.))[0]
xmk=xm[um]

ids=['']*len(xmk)
for i in range(0,len(xmk)):
	ids[i]='GaiaDR2'+str(xmk['dr2_source_id'][i])

pherr=np.zeros(len(xmk))+0.01

dust=['allsky']*len(xmk)
band=['rpmag']*len(xmk)
com=[' ']*len(xmk)

ascii.write([ids,xmk['phot_bp_mean_mag'],pherr,xmk['phot_rp_mean_mag'],pherr,xmk['parallax']/1000,xmk['parallax_error']/1000,xmk['ra'],xmk['dec'],band,dust,com],'../isoclassify/input.csv',names=['id_starname','bpmag','bpmag_err','rpmag','rpmag_err','parallax','parallax_err','ra','dec','band','dust','comments'],delimiter=',',overwrite=True)
'''

'''
plt.clf()

plt.plot(ymg['teff_gspphot'],ymg['teff_gspphot']/ymg['teff_gspspec'],'.',label='teff_comp=teff_gspspec')


gaiasrc=[1,3,3,3,3,3,3,1,2,3]

um=np.where(ymg['teff_gspspec'] > 0)[0]
print(len(um),' total planets with RVS parameters')

um=np.where(xm['teff_gspphot'] > 0)[0]
um=np.where((xm['teff_gspphot'] > 0) & (xm['radius_gspphot'] > 0))[0]

print(len(um),' total planets with RVS parameters')


plt.clf()
plt.plot(np.log10(ymg['teff_gspspec']),ymg['logg_gspspec'],'.')
plt.plot(np.log10(ymg['teff_esphs']),ymg['logg_esphs'],'.')
plt.xlim([4.7,3.3])
plt.ylim([5.7,1.8])


um=np.where((xm['teff_gspspec'] > 0) & (xm['teff_gspphot'] > 0))[0]

um=np.where((ymg['teff_gspphot'] > 0) & (ymg['teff_gspspec'] > 0))[0]
um2=np.where(ymg['TLC'][um] == 22)[0]
um3=np.where(ymg['TLC'] == 22)[0]

um4=np.where(ymg['teff_gspphot'] > 9000)[0]
ymg2=ymg.copy()
ymg2['teff_gspphot'][um4]=ymg2['teff_gspphot'][um4]-5000


um=np.where((ymg['teff_gspphot'] > 0) & (ymg['teff_gspspec'] > 0) & (flags == 0))[0]

plt.ion()
plt.clf()
plt.plot(ymg['teff_gspspec'][um],ymg['teff_gspphot'][um],'.')
plt.plot(ymg['teff_gspspec'][um[um2]],ymg['teff_gspphot'][um[um2]],'.')
plt.plot([3000,10000],[3000,10000],color='red')
plt.xlabel('teff_gspspec')
plt.ylabel('teff_gspphot')
plt.xlim([2500,10000])
plt.ylim([2500,10000])



um=np.where((xm['teff_gspphot'] > 0) & (xm['teff_gspspec'] > 0))[0]
um2=np.where((xm['teff_gspphot'] > 0) & (xm['teff_gspspec'] > 0) & (flags == 0))[0]

plt.ion()
plt.clf()
plt.plot(xm['teff_gspspec'][um],xm['teff_gspphot'][um],'.')
plt.plot(xm['teff_gspspec'][um2],xm['teff_gspphot'][um2],'.')
plt.plot([3000,10000],[3000,10000],color='red')
plt.xlabel('teff_gspspec')
plt.ylabel('teff_gspphot')
plt.xlim([2500,10000])
plt.ylim([2500,10000])


plt.clf()
plt.hist(ymg['teff_gspphot'],bins=50)

plt.ion()
plt.clf()
plt.plot(ymg['teff_gspphot'][um],ymg['teff_gspphot'][um]/ymg['teff_gspspec'][um],'.')



s=np.argsort(tois['Full TOI ID'])

#ascii.write(tois[s],'tois_gaiaDR3.csv',delimiter=',',overwrite=True)

tois=ascii.read('tois_gaiaDR3.csv')

plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size']=18
plt.rcParams['mathtext.default']='regular'
plt.rcParams['lines.markersize']=8
plt.rcParams['xtick.major.pad']='3'
plt.rcParams['ytick.major.pad']='3'
plt.rcParams['ytick.minor.visible'] = 'True'
plt.rcParams['xtick.minor.visible'] = 'True'
plt.rcParams['xtick.direction'] = 'inout'
plt.rcParams['ytick.direction'] = 'inout'
plt.rcParams['ytick.right'] = 'True'
plt.rcParams['xtick.top'] = 'True'

plt.clf()
plt.ion()

upl=0.6
ll=-0.7
snrlim=10
bs=0.1
ss=7

# orange, red, light blue, dark blue
colors=['#FF9408','#DC4D01','#00A9E0','#016795']

import bin
um=np.where((tois['mh_gspspec'] > -99) & (tois['Signal-to-noise'] > 10))[0]
plt.plot(tois['Planet Radius Value'][um],tois['mh_gspspec'][um],'o',alpha=0.5,label='all TOIs',ms=ss,color=colors[3])

um=np.where((tois['mh_gspspec'] > -99) & (tois['Signal-to-noise'] > 10) & ((tois['TOI Disposition'] == 'CP') | (tois['TOI Disposition'] == 'KP') | (tois['TOI Disposition'] == 'PC')))[0]

um=np.where((tois['mh_gspspec'] > -99) & (tois['Signal-to-noise'] > 10) & ((tois['TOI Disposition'] == 'CP') | (tois['TOI Disposition'] == 'KP')))[0]


plt.plot(tois['Planet Radius Value'][um],tois['mh_gspspec'][um],'o',alpha=0.5,label='CPs & KPs',ms=ss,color=colors[1])

x=np.log10(tois['Planet Radius Value'][um])
y=tois['mh_gspspec'][um]
s=np.argsort(x)
binx,biny,binz=bin.bin_time(x[s],y[s],bs)
#plt.errorbar(10**binx,biny,yerr=binz,fmt='-o',color='red')
plt.xlim([0.5,20])
plt.ylim([ll,upl])
plt.title('Gaia DR3 + TESS')
plt.xlabel('Planet Radius (Earth Radii)')
plt.ylabel('[M/H]')
plt.tight_layout()
plt.legend()

#plt.savefig('tois_gaiaDR3.png',dpi=200)

'''
