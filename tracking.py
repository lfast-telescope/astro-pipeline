#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 17:06:27 2025

@author: petershea
"""

import numpy as np
import os
from datetime import datetime
import warnings


from astropy.io import fits, ascii
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma, mad_std
from astropy.visualization import simple_norm
from astropy.modeling import models, fitting
from astropy.time import Time
from astropy.table import Table

from photutils.segmentation import SourceCatalog, detect_sources, detect_threshold, detect_sources, deblend_sources, make_2dgaussian_kernel
from photutils.background import Background2D, MMMBackground, MedianBackground, MeanBackground, MADStdBackgroundRMS, SExtractorBackground, BkgZoomInterpolator
from photutils.utils import circular_footprint, NoDetectionsWarning
from photutils.psf import fit_2dgaussian

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize







def detect(image, npixels=10, connectivity=8):
    '''
    Function that preforms rudimentary background estimation and subtraction, 
    detects sources, preforms deblending, and returns an table of sources with 
    data from phot utils, also returns segment map of detected sources and 
    background subtracted image.
    
    Parameters:
        image: 2d ndarray; raw image most often from hdul[0].data
        npixels: int; number of connected pixels to qualify as a source (most 
            used in this fucntion to distinguish between sources and cosmic 
            rays/hot pixels) *Optional
        connectivity: int; 4 or 8 determines type of pixel connectiviy
    
    returns:
        tbl: astropy table; simplistic source characteristics from phot utils
        semgent_map: 2d array; segment map of deblended sources
        imgdata: 2d array; background subtracted image 
    '''
    
    try:    # initial source detection catching images with no source
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=NoDetectionsWarning)
            threshold_init = detect_threshold(image, nsigma=3)
            segment_img_init = detect_sources(image, threshold_init, npixels=npixels)
    
    except NoDetectionsWarning:
        print("\tNo sources detected — skipping this image.")
        return None, None, None
    
    # Initial source mask
    mask_init = segment_img_init.make_source_mask(footprint=circular_footprint(radius=10))
    
    # Initial Background estimation using initial mask 
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (64,64), filter_size=(5,5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask=mask_init)
    
    # Detection threshold for sources based on initial esstimation of background
    threshold = detect_threshold(image, background=bkg.background, error=bkg.background_rms, nsigma=10)

    # Detection of sources
    segment_img = detect_sources(image, threshold, npixels=npixels)
    
    # Creating new mask based on source detection with initial background estimation 
    footprint = circular_footprint(radius=10)
    
    # Mask based on detected sources
    mask = segment_img.make_source_mask(footprint=footprint)

    # Final background estimation 
    bkg = Background2D(image, (64,64), filter_size=(5,5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask=mask)
    
    # Background subtracting image
    imgdata = np.abs(image.astype(float) - (bkg.background))
    
    # Creates a kernel based on a 2d gaussian 
    kernel_sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel = Gaussian2DKernel(kernel_sigma, x_size=3, y_size=3)
    kernel.normalize()  
    
    # Convolves background subtracted image with gaussian kernel
    #kernel = make_2dgaussian_kernel(3.0, size=7)  # FWHM = 3.0
    convolved_data = convolve(imgdata, kernel)
    
    # Detects sources based on the convolved data and final background estimation
    segment_map = detect_sources(convolved_data, threshold, connectivity=connectivity, npixels=npixels)
    
    # Deblends sources to separate sources in close proximity
    segm_deblend = deblend_sources(convolved_data, segment_map, npixels=npixels, progress_bar=False)
    
    # Creates a phot utils catalog basked on the background subtracted image and detected sources
    cat = SourceCatalog(imgdata, segm_deblend, convolved_data=convolved_data)

    # Converts phot utils catalog to astropy table
    tbl = cat.to_table()

    return tbl, segment_map, imgdata

# Path to raw data on my computer 
data_root = '/Users/petershea/Desktop/Research/LFAST/Data/20250626'

save_path = '/Users/petershea/Desktop/Research/LFAST/Data/20250626_red'


### Unfinished
def Process(data_root, save_path):
    '''
    This function itterates through subdirectories within a given directory 
    containinng raw data and reduces raw data files found. Intended to be used 
    to process LFAST tracking drift data.
    
    Parameters:
        data_root: str; path to the directory containing subdirectories with raw data files
        save_path: str; path to the desired directory to save reduced data to
    
    Returns:
        data_table: astropy table; table containing parameters of the fit 2d 
            gaussian and the time of which the exposure was taken in jd
            Columns are: Time, x_0, y_0, FWHM_x, FWHM_y, Eccentricity, Rotation, Amplitude    
    '''

    # Creates a directory based on the raw data path for any saved data 
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    # Conditionals for methods within this code
    photutil_method = False
    astropy_method = True
    
    visualize = False
    verbose = False
    
    # Chages working directory to raw data directory
    os.chdir(data_root)
    
    # Lists subdirectories within the raw data directory  
    subdirs = sorted(os.listdir())
    subdirs.pop(0) # Pops .DS_Store off of list
    
    # Turns list into an ndarry 
    subdirs = np.array(subdirs)
    #print(subdirs)
    
    # Defines column names for data table of 2d gaussian characteristics 
    col_names = np.array(['Time','x_0','y_0','FWHM_x','FWHM_y','Eccentricity','Rotation','Amplitude'])
    
    # Creates empty astropy table with designated column names 
    data_table = Table(names=col_names)
    
    # Interates through subdirs within the main raw data directory
    for subdir in subdirs:
        os.chdir(subdir) # Changes working directory to given sub directory
        if verbose==True: print(f'\nWorking on Directory: {subdir}')
        # Creates list of fiels within the subdirectory
        files = sorted(os.listdir())
        
        i = 0 # variable to rack progress
        if verbose==True:
            if len(files) > 0: print(f'\tProgress: {i/len(files) * 100:.2f}%')
            else: print('Directory is Empty')
        for file in files:
            
            # Determines the type of exposure between sky image and sub lens array
            if '.zwo' not in file:    ### Temporary
            
                ### Work In Progress ###
                
                # Add code to reduce micro lens array data and create new secondary data table
                
                # Move to directly above: if '.zwo' in file:
            
            
                continue
            
            # Opens given file and exracts image data
            hdul = fits.open(file)
            image = hdul[0].data
            
            # Runts detect function and returns source table, segment image, and background subtracted image data
            tbl, segment_map, imgdata = detect(image)
            
            # If detection fails appends nans to data table at the associated time of the exposure
            if tbl == None:
                
                time = hdul[0].header['DATE-OBS']
                
                milliseconds = int(time.split('_')[1])
                dt = datetime.strptime(time[:15], "%Y%m%dT%H%M%S")
                dt = dt.replace(microsecond=milliseconds * 1000)
                t = Time(dt)
                
                data_table.add_row([t.jd,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                                    np.nan])
                
                
                continue
        
            # Plots the image and locations of detected sources 
            if visualize == True:
                fig,ax = plt.subplots()
            
                norm = ImageNormalize(stretch=SqrtStretch())
                
                # Plots the background subtracted image
                ax.imshow(imgdata, norm=norm, origin='lower', cmap='Greys_r',
                           interpolation='nearest')
                
               
                ax.set_title('Background-subtracted Data')
                
                '''
                ax.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
                       interpolation='nearest')
                ax.set_title('Segmentation Image')
                '''
                # Plots the centroid of detected sources
                ax.scatter(tbl['xcentroid'],tbl['ycentroid'],c='blue',marker='.')
            
            # For sky images 
            if '.zwo' in file:
                
                # Due to poor psf quality detected sources are in a ring around true location
                # This method only works for a single source in an exposure
                
                # Estimates true centroid by taking a flux weighted average of detected sources 
                avgk_x = np.average(tbl['xcentroid'],weights=tbl['kron_flux'])
                avgk_y = np.average(tbl['ycentroid'],weights=tbl['kron_flux'])
                
                # Plots the estimated source centroid 
                if visualize == True:
                    ax.scatter(avgk_x, avgk_y, c='r',marker='.')
                
                # Creates a variable for size of cropped image around source
                bound = 50
                
                # Bounds around the source, +1 is necesssary due to astropy 2d gaussian fit requiring image with odd dimensions
                xmin,xmax = int(avgk_x)-bound, int(avgk_x)+bound+1
                ymin,ymax = int(avgk_y)-bound, int(avgk_y)+bound+1
                
                # Phot utils method to fit 2d gaussian, not useful as phot utils gaussian is circular not elliptical 
                if photutil_method == True:
                    fit = fit_2dgaussian(imgdata[ymin:ymax, xmin:xmax], xypos=(avgk_x-xmin, avgk_y-ymin))
                    
                    if visualize == True:
                        cutout = imgdata[ymin:ymax, xmin:xmax]
                        model = fit.make_model_image(cutout.shape)
                        residual = cutout - model
                        
                        vmin = min(cutout.min(), model.min(), residual.min())
                        vmax = max(cutout.max(), model.max(), residual.max())
                        
                        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
                        
                        fig1, ax1 = plt.subplots(1,3)
                        ax1[0].imshow(imgdata[ymin:ymax, xmin:xmax], norm=norm, cmap='Greys_r', interpolation='nearest', origin='lower')   
                        ax1[0].scatter(tbl['xcentroid']-xmin,tbl['ycentroid']-ymin,c='blue',marker='.')
                        ax1[0].scatter(avgk_x-xmin, avgk_y-ymin, c='r',marker='.')
                        
                        ax1[1].imshow(fit.make_model_image(imgdata[ymin:ymax, xmin:xmax].shape), norm=norm, origin='lower', cmap='Greys_r',
                                   interpolation='nearest')
                        
                        ax1[2].imshow(imgdata[ymin:ymax, xmin:xmax]-fit.make_model_image(imgdata[ymin:ymax, xmin:xmax].shape), norm=norm, origin='lower', cmap='Greys_r',
                                   interpolation='nearest')
                
                # Astropy method to fit 2d gaussian
                if astropy_method == True:
                    
                    # Initial Parameter guesses for 2d gaussian model
                    amplitude_guess = np.max(imgdata[ymin:ymax, xmin:xmax])
                    sigma_guess = 2.0
                    theta_guess = 0.0
                
                    # Creates 2d gaussian model
                    g_init = models.Gaussian2D(amplitude=amplitude_guess,
                                               x_mean=avgk_x-xmin,
                                               y_mean=avgk_y-ymin,
                                               x_stddev=sigma_guess,
                                               y_stddev=sigma_guess,
                                               theta=theta_guess)
                    
                    # Creates X and Y arrays the size of image croupped around the source 
                    x,y = np.mgrid[:xmax-xmin, :ymax-ymin]
                    
                    # Fits the 2d gaussian to estimated source centroid
                    fitter = fitting.LevMarLSQFitter() # Determines type of fitter used
                    g_fit = fitter(g_init, x, y, imgdata[ymin:ymax, xmin:xmax])
                    
                    # Variables for fit centroid coordinates
                    x_0 = g_fit.x_mean.value + xmin # adds xmin and ymin to put in 
                    y_0 = g_fit.y_mean.value + ymin # terms of initial image pixel coordinates
                
                    # Variables for standard deviation of gaussian in x and y dimensions
                    sigma_x = g_fit.x_stddev.value
                    sigma_y = g_fit.y_stddev.value
                    
                    # Converts from sigma to FWHM
                    fwhm_x = gaussian_sigma_to_fwhm * sigma_x
                    fwhm_y = gaussian_sigma_to_fwhm * sigma_y
                    
                    # Determines largest of x and y and calculates eccentricity
                    a = max(sigma_x, sigma_y)
                    b = min(sigma_x, sigma_y)
                    eccentricity = np.sqrt(1 - (b/a)**2)
            
                    # Determines rotation angle of the gaussian
                    theta = np.degrees(g_fit.theta.value)
                    
                    # Pulls time from fits header and saves as a julian data
                    time = hdul[0].header['DATE-OBS']
                    
                    milliseconds = int(time.split('_')[1])
                    dt = datetime.strptime(time[:15], "%Y%m%dT%H%M%S")
                    dt = dt.replace(microsecond=milliseconds * 1000)
                    t = Time(dt)
                    
                    # appends a row for the source to the data table
                    data_table.add_row([t.jd,round(x_0,2),round(y_0,2),round(fwhm_x,2),round(fwhm_y,2),
                                        round(eccentricity,2),round(-theta,2),round(g_fit.amplitude.value,2)])
                    
                    # Prints table values if verbose==True
                    if verbose == True:
                        print(f'Time: {t.jd}')
                        print('Centroid:')
                        print(f'\tx_0: {x_0:.2f}')
                        print(f'\ty_0: {y_0:.2f}')
                        print("FWHM:")
                        print(f'\tFWHM_x: {fwhm_y:.2f}')
                        print(f'\tFWHM_y: {fwhm_x:.2f}')
                        print(f"Eccentricity: {eccentricity:.2f}")
                        print(f"-Rotation: {theta:.1f} deg")
                        print(f'Amplitude: {g_fit.amplitude.value:.2f}')
                    
                    # Defines cropped image data
                    cutout = imgdata[ymin:ymax, xmin:xmax]
                    # Creates model based on fit gaussian parameters 
                    model = g_fit(x,y)
                    # Subtracts model from image data to get residuals 
                    residual = cutout - model
                    
                    # Defines maximum and minimum values for consistent image scaling
                    vmin = min(cutout.min(), model.min(), residual.min())
                    vmax = max(cutout.max(), model.max(), residual.max())
                    
                    # Creates image normalization for constant colorvalue
                    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
                    
                    fig1, ax1 = plt.subplots(1,3, figsize=(9,3), layout='constrained')
                    
                    # Plots cropped science image with locations of detected sources and estimated source centroid
                    ax1[0].imshow(imgdata[ymin:ymax, xmin:xmax], norm=norm, cmap='Greys_r', interpolation='nearest', origin='lower')   
                    ax1[0].scatter(tbl['xcentroid']-xmin,tbl['ycentroid']-ymin,c='blue',marker='.')
                    ax1[0].scatter(avgk_x-xmin, avgk_y-ymin, c='r',marker='.')
                    
                    ax1[0].set_title('Detected Source Location')
                    
                    # Creates ellipse patch based on fit gaussian to plot over synthesized image
                    ellipse = Ellipse(
                    (g_fit.y_mean, g_fit.x_mean),                
                    width=3*sigma_y,         
                    height=3*sigma_x,        
                    angle=-theta,             
                    edgecolor='k',
                    facecolor='none',
                    linewidth=2
                    )
                    
                    fig1.suptitle(f'Zeta Herculis Parameters: {time}')
                    
                    for axi in ax1:
                        axi.set_xticklabels([]) 
                        axi.set_yticklabels([])
                        
                    # Plots synthesized image from gaussian model and ellipse of gaussian out to 3sigma
                    ax1[1].imshow(model, norm=norm, origin='lower', cmap='Greys_r',
                               interpolation='nearest')
                    
                    ax1[1].add_patch(ellipse)
                    
                    ax1[1].set_title('Modeled Gaussian')
                    
                    # Plots model residuals
                    ax1[2].imshow(residual, norm=norm, origin='lower', cmap='Greys_r',
                               interpolation='nearest')
                    
                    ax1[2].set_title('Model Residuals')
                        
                    plt.savefig(f'{save_path}/{time}.png')
                    plt.show()
                
                # writes data table to txt file after every source in case script is interupted
                data_table.write(f'{save_path}/output_data.txt', format='ascii.fixed_width', overwrite=True)
            
            # Adds 1 to image processed count
            i += 1
            if verbose==True: print(f'\tProgress: {i/len(files) * 100:.2f}%')     # progress
            
        # Returns to raw data directory
        os.chdir('..')
        
    return data_table


Process(data_root, save_path)



