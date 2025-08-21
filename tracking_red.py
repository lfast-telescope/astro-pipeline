#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:23:50 2025

@author: petershea
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.gridspec import GridSpec
from astropy.table import Table
from astropy.io import fits, ascii
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.optimize import curve_fit

samp_img = '/Users/petershea/Desktop/Research/LFAST/Data/20250626/000009/20250626T070017_462_Zeta_Herculis.zwo.fits'
save_root = '/Users/petershea/Desktop/Research/LFAST/Data/20250626_red'

tbl = Table.read(f'{save_root}/output_data.txt', format='ascii.fixed_width')

time = (tbl['Time']-tbl['Time'][0]) * 1440 # time in minutes since first observation

bound = 100

hdul = fits.open(samp_img)
imgdata = hdul[0].data

fig = plt.figure(figsize=(9,8), layout='constrained', dpi=1000)
gs = GridSpec(2, 2, figure=fig)

ax0 = fig.add_subplot(gs[0,0])  # Cartesian
ax1 = fig.add_subplot(gs[0,1])  # Cartesian
ax2 = fig.add_subplot(gs[1,0], projection='polar')  # Polar
ax3 = fig.add_subplot(gs[1,1])  # Cartesian

# --- Plot 1: Source Drift ---

norm = ImageNormalize(stretch=SqrtStretch())

norm_time = colors.Normalize(vmin=time.min(), vmax=time.max())
cmap_time = plt.colormaps['viridis']

ax0.imshow(imgdata, norm=norm, origin='lower', cmap='Greys_r',
           interpolation='nearest')
ax0.scatter(tbl['x_0'],tbl['y_0'], c=time, cmap='viridis', marker='.')
ax0.set_xlim(np.min(tbl['x_0'])-bound,np.max(tbl['x_0'])+bound)
ax0.set_ylim(np.min(tbl['y_0'])-bound,np.max(tbl['y_0'])+bound)
ax0.set_aspect('equal', adjustable='box')
ax0.set_title('Tracking Drift')

sm = cm.ScalarMappable(cmap=cmap_time, norm=norm_time)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax0)
cbar.set_label("Time from Initial Observation [min]")

# --- Plot 2: FWHM over time ---

x_mask = tbl['FWHM_x'] >= 1
y_mask = tbl['FWHM_y'] >= 1
tot_mask = x_mask & y_mask

fwhm_x = tbl['FWHM_x'][tot_mask]
fwhm_y = tbl['FWHM_y'][tot_mask]
#masked_ecen = tbl['Eccentricity'][tot_mask]
masked_time = time[tot_mask]

linear = lambda x, m, b: m*x + b

x_popt, x_pcov = curve_fit(linear, masked_time, fwhm_x)
y_popt, y_pcov = curve_fit(linear, masked_time, fwhm_y)
#e_popt, e_pcov = curve_fit(linear, masked_time, masked_ecen)

time_cont = np.linspace(0,25,2500)

ax1.scatter(masked_time, fwhm_x, color='salmon', marker='.', label='X FWHM')
ax1.plot(time_cont,linear(time_cont,*x_popt), color='salmon', linestyle='--')

ax1.scatter(masked_time, fwhm_y, color='#4682B4', marker='.', label='Y FWHM')
ax1.plot(time_cont,linear(time_cont,*y_popt), color='#4682B4', linestyle='--')

ax1.set_title('2D Gaussian FWHM')
ax1.legend(loc='lower right')

# Ecentricity not necesarry
'''
ax1b = ax1.twinx()
ax1b.scatter(masked_time, masked_ecen, color='k', marker='.')
ax1b.plot(time_cont, linear(time_cont,*e_popt), color='k', linestyle='--')
ax1b.set_ylabel('Eccentricity')
ax1b.set_ylim(0,1)
ax1b.tick_params(axis='y')
'''

ax1.set_xlim(0,25)
ax1.set_ylim(0,((np.max(np.maximum(tbl['FWHM_x'][tot_mask],tbl['FWHM_y'][tot_mask]))//5)+1)*5)

ax1.set_xlabel("Time from Initial Observation [min]")
ax1.set_ylabel("FWHM [pixels]")

# --- Plot 3: Rotation over time ---

masked_theta = np.deg2rad(tbl['Rotation'][tot_mask] % 360)

norm_time = colors.Normalize(vmin=time.min(), vmax=time.max())
cmap = cm.viridis

for t, th in zip(masked_time, masked_theta):
    color = cmap(norm_time(t))
    ax2.plot([th, th], [0, 1], color=color) 

sm = cm.ScalarMappable(cmap=cmap, norm=norm_time)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, pad=0.1)
cbar.set_label("Time from Initial Observation [min]")

ax2.set_theta_zero_location('N')  # 0° pointing up
ax2.set_theta_direction(-1)       # clockwise

labels = ax2.get_xticks()   
label_names = [item.get_text() for item in ax2.get_xticklabels()]

ax2.set_title('2D Gaussian Rotation')
ax2.set_yticklabels([])
new_labels = ['' if np.isclose(t, 0) else item for t, item in zip(labels, label_names)]
ax2.set_xticklabels(new_labels)

# --- Plot4: Amplitude over time ---

masked_amp = tbl['Amplitude'][tot_mask]

a_popt, a_pcov = curve_fit(linear, masked_time, masked_amp)

ax3.scatter(masked_time, masked_amp, color='k', marker='.')
ax3.plot(time_cont, linear(time_cont,*a_popt), color='k', linestyle='--')
ax3.set_xlim(0,25)

ax3.set_title('2D Gaussian Amplitude')
ax3.set_xlabel("Time from Initial Observation [min]")
ax3.set_ylabel("Amplitude [ADU]")

exposures = len(tbl['Time'])

print(f'{exposures - np.sum(tot_mask)}/{exposures} Failed')

drift_arr = np.array([])
xdrift_arr = np.array([])
ydrift_arr = np.array([])
for i in range(len(tbl['x_0'])-1):
    x_drift = np.abs(tbl['x_0'][i+1] - tbl['x_0'][i])
    y_drift = np.abs(tbl['y_0'][i+1] - tbl['y_0'][i])
    drift = np.sqrt(x_drift**2 + y_drift**2)
    dt = time[i+1] - time[i]
    
    drift_arr = np.append(drift_arr,drift/dt)
    xdrift_arr = np.append(xdrift_arr,x_drift/dt)
    ydrift_arr = np.append(ydrift_arr,y_drift/dt)

print(np.median(drift_arr),'pixels/min')
print(f'Median X drift: {np.median(xdrift_arr)} pixels/min')
print(f'Median Y drift: {np.median(ydrift_arr)} pixels/min')



plt.suptitle(f'{exposures - np.sum(tot_mask)}/{exposures} Failed; Median X Drift: {np.median(xdrift_arr):.2f} pixels/min; Median Y Drift: {np.median(ydrift_arr):.2f} pixels/min')





