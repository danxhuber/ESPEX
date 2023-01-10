# rough asteroseismic yields for an ESPEX mission concept

import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import pdb

# SNR limit
lim=10.

# assume 12 months of 1 minute sampling
months=12
sampling=1
npoints=months*30*24*60/sampling

# fraction of stars pulsating within instability region
pulsfrac=0.5

# Distribution of the strongest peak amplitudes of Kepler delta Scuti stars in log10(amp/mmag). 
# This approximates Fig 6a from https://arxiv.org/pdf/1903.00015.pdf
def ampfunc(x):
	x = -(x - 0) / 0.5
	y = 100*np.exp(-x*x/2.0) 
	x2 = -(x - 4.5) / 0.75
	y2 = 1700*np.exp(-x2**2/2) 
	return y+y2

# set seed 
np.random.seed(seed=88)

# Table 2 from https://ui.adsabs.harvard.edu/abs/2021ApJ...917...23K/abstract 
# crossmatched with Gaia DR3 astrophysical parameters
# including expected noise levels from code by Luke Bouma
dat=ascii.read('ymg-gaia-table2.csv')

# assign ages for each subpop. using Tables 4-6 from Kerr et al.
ages=np.zeros(len(dat))
files=['ages_scocen.txt','ages_orion.txt','ages_taurus.txt']
ids=[22,27,18]
for j in range(0,len(files)):
	ages_scocen=ascii.read(files[j])
	ix=np.where(dat['TLC'] == ids[j])[0]
	for i in np.unique(dat['EOM'][ix]):
		iy=np.where(ages_scocen['col1'] == i)[0]
		iz=np.where(dat['EOM'][ix] == i)[0]
		# if no subgroup assigned, draw random age over range of all ages for a given SFR
		if i == -1:
			ages[ix[iz]]=np.random.uniform(low=np.min(ages_scocen['col2']),high=np.max(ages_scocen['col2']),size=len(iz))
		else:
			if len(iy) > 1:
			# if more than 1 subgroup assigned, draw random age over range of all ages for these subgroups
				ages[ix[iz]]=np.random.uniform(low=np.min(ages_scocen['col2'][iy]),high=np.max(ages_scocen['col2'][iy]),size=len(iz))
			else:
				ages[ix[iz]]=ages_scocen['col2'][iy]
		

# adopt Gaia Teff for each star
# teff_esphs > teff_gspspec > teff_gspspec
teff=np.zeros(len(dat))
for i in range(0,len(teff)):
	if ((dat['teff_esphs'][i] > 0) & (np.isfinite(dat['teff_esphs'][i]))):
		teff[i]=dat['teff_esphs'][i]
		continue
	if ((dat['teff_gspspec'][i] > 0) & (np.isfinite(dat['teff_gspspec'][i]))):
		teff[i]=dat['teff_gspspec'][i]
		continue
	if ((dat['teff_gspphot'][i] > 0) & (np.isfinite(dat['teff_gspphot'][i]))):
		teff[i]=dat['teff_gspphot'][i]

# exclude stars without Gaia parameters. 
um=np.where(teff > 0)[0]
dat=dat[um]
teff=teff[um]
ages=ages[um]

# for stars without Gaia DR3 parameters, draw randomly from underlying distribution
#ix=np.where(teff > 0)[0]
#iy=np.where(teff == 0)[0]
#x,y,z=plt.hist(teff[ix],bins=100)
#ds = np.linspace(np.min(teff[ix]),np.max(teff[ix]),len(iy))
#temps = np.random.choice(y[:-1], p=x/np.sum(x), size=len(iy))
#teff[iy]=temps


# define instability regions (logteff)
dsgd_teff=10**np.array((3.8,4.0))
spb_teff=10**np.array((4.1,4.3))
sl_teff=10**np.array((3.75,3.8))
md_teff=10**np.array((3.4,3.6))

#dsgd_cycle=30/sampling
#spb_cycle=12*60/sampling

snrs=np.zeros(len(dat))
clas=np.zeros(len(dat))

amp=np.arange(-3,2,0.01) 
ampf=ampfunc(amp)

#plt.clf()
#plt.plot(amp,ampf)

# loop over all stars
for i in range(0,len(dat)):

	# convert noise to ppm
	noise=(dat['noise'][i]*1e6)/np.sqrt(sampling)

	if ((teff[i] > spb_teff[0]) & (teff[i] < spb_teff[1])):
		# draw amplitude
		uamp = (10**np.random.choice(amp, p=ampf/np.sum(ampf)))*1e3
		snr=uamp/(noise/np.sqrt(npoints))
		#if (snr > 5):
		#	pdb.set_trace()
		if (np.random.uniform(low=0,high=1) > pulsfrac):
			clas[i]=1
			snrs[i]=snr

	if ((teff[i] > dsgd_teff[0]) & (teff[i] < dsgd_teff[1])):
		uamp = (10**np.random.choice(amp, p=ampf/np.sum(ampf)))*1e3
		snr=uamp/(noise/np.sqrt(npoints))
		if (np.random.uniform(low=0,high=1) > pulsfrac):
			clas[i]=2
			snrs[i]=snr

	# not predicting solar-like stars and M dwarfs since pulsator fraction is uncertain
	#if ((teff[i] > sl_teff[0]) & (teff[i] < sl_teff[1])):
	#	snr=sl_amp/(noiseperminute/np.sqrt(npoints))
	#	clas[i]='sl'
	#	snrs[i]=snr

	#if ((teff[i] > md_teff[0]) & (teff[i] < md_teff[1])):
	#	snr=md_amp/(noiseperminute/np.sqrt(npoints))
	#	clas[i]='md'
	#	snrs[i]=snr
		
	if (snrs[i] > lim):
		print('teff,tmag,noise(ppm),amp(ppm)')
		print(teff[i],dat['tmag'][i],dat['noise'][i]*1e6,uamp,snr)
		#print(dat[i])
		print(' ')
	#input(':')


print('ESPEX yields:')
um=np.where((snrs > lim) & ((dat['TLC'] == 22) | (dat['TLC'] == 18) | (dat['TLC'] == 27)))[0]
print('Total:',len(um))
um=np.where((snrs > lim) & (dat['TLC'] == 18))[0]
um2=np.where((snrs > lim) & (dat['TLC'] == 18) & (clas == 2))[0]
um3=np.where((snrs > lim) & (dat['TLC'] == 18) & (clas == 1))[0]
print('Taurus:',len(um),', deltaScuti/gammaDor:',len(um2),', SPB:',len(um3))
um=np.where((snrs > lim) & (dat['TLC'] == 22))[0]
um2=np.where((snrs > lim) & (dat['TLC'] == 22) & (clas == 2))[0]
um3=np.where((snrs > lim) & (dat['TLC'] == 22) & (clas == 1))[0]
print('ScoCen:',len(um),', deltaScuti/gammaDor:',len(um2),', SPB:',len(um3))
um=np.where((snrs > lim) & (dat['TLC'] == 27))[0]
um2=np.where((snrs > lim) & (dat['TLC'] == 27) & (clas == 2))[0]
um3=np.where((snrs > lim) & (dat['TLC'] == 27) & (clas == 1))[0]
print('Orion:',len(um),', deltaScuti/gammaDor:',len(um2),', SPB:',len(um3))

# write simulated yields to file
um=np.where((snrs > lim) & ((dat['TLC'] == 22) | (dat['TLC'] == 18) | (dat['TLC'] == 27)))[0]
ascii.write(dat[um],'espex-seismo-yield.csv',delimiter=',',overwrite=True)


# plots
plt.rcParams['xtick.major.size'] = 9
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 9
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size']=16
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



# use a color-blind friendly palette
# orange, red, light blue, dark blue
colors=['#FF9408','#DC4D01','#00A9E0','#016795']

plt.ion()

plt.clf()
bs=np.arange(1,60,5)
plt.hist(ages[um],bins=bs,label='ESPEX yield',histtype='step',color=colors[1],ls='dashed',lw=3)
plt.hist(ages[um],bins=bs,color=colors[1],alpha=0.1)
known=ascii.read('known.csv')
plt.hist(known['col1'],bins=bs,label='Current yield',histtype='step',color=colors[3],ls='dashed',lw=3)
plt.hist(known['col1'],bins=bs,color=colors[3],alpha=0.1)
plt.title('Young Pulsating Stars with Mode Identification')
plt.xlabel('Age (Myr)')
plt.ylabel('Number of Stars')
plt.tight_layout()
plt.legend()
plt.yscale('log')

plt.savefig('espex-seismo-yield-age.png',dpi=200)

input(':')

bs=np.logspace(np.log10(3000),np.log10(30000),num=50)
plt.clf()
plt.hist(teff,bins=bs,color=colors[1],label='All stars')
plt.hist(teff[um],bins=bs,color=colors[3],label='Predicted detections')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Effective Temperature (K)')
plt.ylabel('Number of Stars')
plt.legend()
plt.tight_layout()
plt.savefig('espex-seismo-yield-teff.png',dpi=200)

input(':')

bs=np.arange(5,20,0.5)
plt.clf()
plt.hist(dat['Gmag'],bins=bs,color=colors[1],label='All stars')
plt.hist(dat['Gmag'][um],bins=bs,color=colors[3],label='Predicted detections')
plt.yscale('log')
plt.xlabel('Gaia G Magnitude')
plt.ylabel('Number of Stars')
plt.legend()
plt.tight_layout()
plt.savefig('espex-seismo-yield-gmag.png',dpi=200)

