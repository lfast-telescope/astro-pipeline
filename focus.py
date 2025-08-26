#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:54:45 2025

@author: petershea
"""
import os
from datetime import datetime
import warnings

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits, ascii
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma, mad_std, sigma_clip
from astropy.visualization import simple_norm
from astropy.modeling import models, fitting
from astropy.time import Time
from astropy.table import Table, vstack
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from photutils.segmentation import SourceCatalog, detect_sources, detect_threshold, detect_sources, deblend_sources, make_2dgaussian_kernel
from photutils.background import Background2D, MMMBackground, MedianBackground, MeanBackground, MADStdBackgroundRMS, SExtractorBackground, BkgZoomInterpolator
from photutils.utils import circular_footprint, NoDetectionsWarning
from photutils.psf import fit_2dgaussian

from scipy.optimize import curve_fit



class RED:
    '''
    Class to reduce images: stack and detect sources  
    '''
    
    def __init__(self,path):
        
        os.chdir(path)
        
        files = sorted(os.listdir())
        
        self.fits = [f for f in files if f.endswith('.FIT') or f.endswith('.fits')]
    
    
        
    def stack(self, filename, sigma=3):
        '''
        Function to stack images to increase snr.
        
        Parameters:
            filename: str; name of stacked imaage
            sigma: float or int; sigma level for mad stats 
            
        returns:
            filename of the stacked fits image
        '''

        stack = np.array([fits.getdata(f) for f in self.fits])  
        
        clipped = sigma_clip(stack,sigma=sigma,axis=0)
        
        stacked_image = np.mean(clipped, axis=0)
        
        stacked_image = np.asarray(stacked_image.filled(np.nan)) 
        
        with fits.open(self.fits[0]) as hdul:
            header = hdul[0].header.copy()
        
        if "DATE-OBS" in header:
            header["DATE-OBS"] = header["DATE-OBS"]
        else:
            header["DATE-OBS"] = "UNKNOWN"
        
        hdu = fits.PrimaryHDU(stacked_image, header=header)
        hdu.writeto(f'{filename}.fits', overwrite=True)
        
        return f'{filename}.fits'
        
        
    
    def detect(self, filename, npixels=10, connectivity=8):
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
        
        hdul = fits.open(filename)

        image = hdul[0].data
        
        try:
            self.time = hdul[0].header['DATE-OBS']
        except KeyError:
            print('DATE-OBS keyword not found')
        
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
        self.imgdata = np.abs(image.astype(float) - (bkg.background))
        
        # Creates a kernel based on a 2d gaussian 
        kernel_sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
        kernel = Gaussian2DKernel(kernel_sigma, x_size=3, y_size=3)
        kernel.normalize()  
        
        # Convolves background subtracted image with gaussian kernel
        #kernel = make_2dgaussian_kernel(3.0, size=7)  # FWHM = 3.0
        convolved_data = convolve(self.imgdata, kernel)
        
        # Detects sources based on the convolved data and final background estimation
        self.segment_map = detect_sources(convolved_data, threshold, connectivity=connectivity, npixels=npixels)
        
        # Deblends sources to separate sources in close proximity
        segm_deblend = deblend_sources(convolved_data, self.segment_map, npixels=npixels, progress_bar=False)
        
        # Creates a phot utils catalog basked on the background subtracted image and detected sources
        cat = SourceCatalog(self.imgdata, segm_deblend, convolved_data=convolved_data)
    
        # Converts phot utils catalog to astropy table
        tbl = cat.to_table()
    
        return tbl
        


def fit_2dgauss(imgdata, center_est, time, bound=50, sigma_guess=2, theta_guess=0):
    '''
    This function fits a 2d eliptical gaussian to a source designated by the coordinates
    center_est. To do so, the imaged is cropped to only include the area directy around
    the source and returns the characteristics of the fit gaussian
    
    Parameters:
        imgdata: 2d ndarray; background subtracted image 
        center_est: tuple; two float values coresponding to the estimated centroid of the source
            in the form (x,y)
        time: astropy Time object; time of the start of the exposure or initial exposure if stacked
        bound: int; size of the cropped image, pixels from center_est that the image is cropped
        sigma_guess: int or float; guess or estimation of gaussian rms
        theta_guess: int or float; guess or estimation of gaussian angle of rotation
        
    Returns:
        data_table: astropy table; table containing parameters of the fit 2d 
            gaussian and the time of which the exposure was taken in jd
            Columns are: Time, x_0, y_0, FWHM_x, FWHM_x_err, FWHM_y, FWHM_y_err, 
                Eccentricity, Rotation, Amplitude    
        source_crop: 2d ndarray; cropped image of the source which is being fit
        parameters: tuple; tupple of cropped image bounds (xmin,xmax,ymin,ymax)
        model: 2d ndarray; model imaage of fit 2d gaussian with same dimension as 
            source_crop
    '''
    
    # Estimate for the center of the source
    avg_x, avg_y = center_est
    
    # Define boundaries around source to make fitting quicker
    xmin, xmax = int(avg_x)-bound, int(avg_x)+bound+1
    ymin, ymax = int(avg_y)-bound, int(avg_y)+bound+1
    
    # Save boundaries to a variable to return 
    parameters = (xmin,xmax,ymin,ymax)
    
    # Crop image to only be focusing source
    source_crop = imgdata[ymin:ymax, xmin:xmax]
    
    # Determine size of cropped source image
    x_size, y_size = source_crop.shape
    
    # Create x and y arrays for creating the 2dgaussian model
    x_arr, y_arr = np.mgrid[:x_size, :y_size]
    
    # Initial gaussian model based on estimated parameters
    g_init = models.Gaussian2D(amplitude=np.max(source_crop),
                               x_mean=avg_x-xmin,
                               y_mean=avg_y-ymin,
                               x_stddev=sigma_guess,
                               y_stddev=sigma_guess,
                               theta=theta_guess)
    
    # Fits the model to the given data
    fitter = fitting.LevMarLSQFitter() # Determines type of fitter used
    g_fit = fitter(g_init, x_arr, y_arr, source_crop)
    
    # Model image of the fit 2d gaussian
    model = g_fit(x_arr,y_arr)
    
    # Pulls out uncertainty for the sigma x and y found by the fitting process
    cov = fitter.fit_info.get('param_cov')  # may be None if not estimated
    if cov is not None:
        names = g_fit.param_names
        idx_x = names.index('x_stddev')
        idx_y = names.index('y_stddev')
        dsig_x = np.sqrt(cov[idx_x, idx_x])
        dsig_y = np.sqrt(cov[idx_y, idx_y])
    
    else:
        dsig_x = dsig_y = np.na
    
    # Error propagation from sigma to fwhm
    dy_x = gaussian_sigma_to_fwhm * dsig_x
    dy_y = gaussian_sigma_to_fwhm * dsig_y
    
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
    
    # Determines rotation angle of the gaussian in deg
    theta = np.degrees(g_fit.theta.value)%360
    
    # Determines the amplitude of the gaussian in terms of ADU
    amplitude = g_fit.amplitude.value
    
    # Defines column names for data table of 2d gaussian characteristics 
    col_names = np.array(['Time','x_0','y_0','FWHM_x','FWHM_x_err','FWHM_y','FWHM_y_err','Eccentricity','Rotation','Amplitude'])
    
    # Creates empty astropy table with designated column names 
    data_table = Table(names=col_names)
    
    # Appends a row for the source to the data table
    data_table.add_row([time.jd,round(x_0,2),round(y_0,2),round(fwhm_x,2),round(dy_x,2),round(fwhm_y,2),
                        round(dy_y,2),round(eccentricity,2),round(theta,2),round(amplitude,2)])

    return data_table, source_crop, parameters, model
   
def avg_source(tbl, source=None, bound=50):
    '''
    A function to average the position of sources in the case of poor image 
    focus resulting in toroid psfs.
    
    Parameters:
        tbl: astropy table; table containing source characteristics
        source: row from astropy table; row from tbl containing desired source
        bound: int; size of area in pixels around designated source within which 
        sources' position will be averaged 
    
    Returns:
        coords: tuple; averaged x and y coordinates (avg_x, avg_y)
    
    '''
    
    # If a source is designated, averages sources around designated source, else uses the brightest
    if source==None:
        # Pulls the row of the brightest detected source
        brightest_source = tbl[np.argmax(tbl['kron_flux'])]
        source = brightest_source
    else:
        source = source
    
    # Creates a mask for the tbl of all sources within the designated bound around the brightest source 
    mask = (np.abs(tbl['xcentroid']-source['xcentroid']) <= bound) & (np.abs(tbl['ycentroid']-source['ycentroid']) <= bound)
    
    # Applies mask to source table
    local_sources = tbl[mask]
    
    # Estimates true centroid by taking a flux weighted average of detected sources 
    avg_x = np.average(local_sources['xcentroid'],weights=local_sources['kron_flux'])
    avg_y = np.average(local_sources['ycentroid'],weights=local_sources['kron_flux'])
    
    coords = (avg_x, avg_y)
    
    return coords

def vertical_hyperbola(x, h, k, a, b):
    '''
    Function defining the upper branch of a vertical hyperbola
    '''
    
    return k + np.sqrt(a**2 * (1 + ((x - h)**2) / b**2))


    
def u_curve(focus_arr,x_fwhm_arr,x_fwhm_err,y_fwhm_arr,y_fwhm_err):
    '''
    Function to determine optimal focus point based on minimizing fwhm from
    focus sweep data set
    
    Parameters:
        focus_arr: ndarray [µm]; array of focus positions associated with each image
        x_fwhm_arr: ndarray [pixels]; array of x fwhm of fitted gaussian
        x_fwhm_err: ndarray [pixels]; uncertainty of x fwhm
        y_fwhm_arr: ndarray [pixels]; array of y fwhm of fitted gaussian
        y_fwhm_err: ndarray [pixels]; uncertainty of y fwhm
    
    Returns:
        optimal_focus: float [µm]; the optimal focus location to minimize source fwhm in x and y
    '''
    
    # Fits hyperbola to the x and y fwhm as they change over focus sweep
    xpopt, xpcov = curve_fit(vertical_hyperbola, focus_arr, x_fwhm_arr, sigma=x_fwhm_err, absolute_sigma=True)
    ypopt, ypcov = curve_fit(vertical_hyperbola, focus_arr, y_fwhm_arr, sigma=y_fwhm_err, absolute_sigma=True)
    
    # Defines continuum for proper graphing of fit models
    focus_cont = np.linspace(np.min(focus_arr),np.max(focus_arr),1000)

    # Creates models for x_FWHM(focus_pos) and y_FWHM(focus_pos)
    x_focus = vertical_hyperbola(focus_cont, *xpopt)
    y_focus = vertical_hyperbola(focus_cont, *ypopt)
    
    # Defines magnitude of FWHM in order to determine optimal focus
    tot_foc = np.sqrt(x_focus**2 + y_focus**2)

    # Finds index of the minimum in total focus which is defined as the optimal focus position
    optimal_focus = np.argmin(tot_foc)
    
    # Creates a plot to display focus sweep data
    fig, ax = plt.subplots(layout='constrained')
    
    # Scatter plot with y_err of fwhm at each focus position
    ax.errorbar(focus_arr,x_fwhm_arr,yerr=x_fwhm_err,c='#4682B4',marker='.',linestyle='none')
    ax.errorbar(focus_arr,y_fwhm_arr,yerr=y_fwhm_err,c='salmon',marker='.',linestyle='none')
    
    # Plots the fit hyperbolic models 
    ax.plot(focus_cont, x_focus, c='#4682B4', label='x fwhm')
    ax.plot(focus_cont, y_focus, c='salmon', label='y fwhm')
    
    # Plots a vertical line at the location of optimal focus
    ax.axvline(x=focus_cont[optimal_focus],c='gray',linestyle='--', label=f'Optimal Focus: {focus_cont[optimal_focus]:.2f} [µm]')
    
    # Plot Formatting
    ax.set_title('Focus Sweep: 20250521') # Need to come back and make more modular
    ax.set_xlabel('Focus Position [µm]') # Displays optimal focus found
    ax.set_ylabel('FWHM [pixels]')
    ax.legend()
    
    plt.show()
    
    return optimal_focus


### Start Red ###

# Variables used to reduce 20250521 focus sweep data on my computer 
#raw_data = '/Users/petershea/Desktop/Research/LFAST/Data/20250521/'

#save_data = '/Users/petershea/Desktop/Research/LFAST/Data/20250521_red/'

#focus_arr = ([-400,-320,-240,-160,-80,0,80,160,240,320,400,480,560,640])


def reduce(focus_arr, raw_data, save_data, visualize=False):
    '''
    Main function calling previous functions to reduce data
    
    Parameters:
        focus_arr: ndarray [µm]; array of focus positions associated with each image
        raw_data: str; path to the directory containing subdirectories wiht raw fits data files
        save_data: str; path to new directory in which reduced data is saved 
        visualize: boolean; determines whether source, model, and residuals should be plotted
    
    Returns:
        optimal_focus: float [µm]; the optimal focus location to minimize source fwhm in x and y
    '''

    ### Stack Raw data files ### 

    # Change directory to location of subdirs containing raw data 
    os.chdir(raw_data)
    
    # Creates a new directory to save data to
    if not os.path.isdir(save_data): os.mkdir(save_data)

    # Lists all subdirs which presumably contain data within this directory
    subdirs = sorted(os.listdir())
    subdirs.pop(0) # Pops .DS_Store
    
    # Iterates through subdirs stacking the raw data files
    for subdir in subdirs:
        # Creates Reduction class for the subdir
        directory = RED(subdir)
        
        # Stacks all raw fits files in the subdir
        directory.stack(f'{save_data}/{subdir}_stacked')
        
        # Returns to main raw data directory
        os.chdir('..')
        

    ### Detect Sources and fits 2d Gaussian
    
    # Change working directory to save data location
    os.chdir(save_data)
    
    # Create variable to save 2d gaussian characteristics
    data_table = None
    
    # Create a class object for the save directory data
    reducing_directory = RED(save_data)
    
    # Creates list of all fits type images
    stacked = reducing_directory.fits
    
    # Iterates through detected fits files (showuld be all stacked images)
    for file in stacked:
        
        # Detects sources in the stacked images 
        tbl = reducing_directory.detect(file)
        
        # If more than 1 source detected focuses on the brightest sources 
        # Averages location of sources if multiple detected coresponding to the same source
        if len(tbl) > 1:
            center_est = avg_source(tbl)
        
        # Else takes the detected source's centroid for 2dgaussian center estimate
        else:
            center_est = (tbl['xcentroid'].value[0], tbl['ycentroid'].value[0])
        
        # Stores background subtrtacted image as imgdata
        imgdata = reducing_directory.imgdata
        
        # Saves time of first exposure to a variable
        time = reducing_directory.time
        # Transforms time from string to astropy time object
        t = Time(time, format='isot', scale='utc')
        
        # Fits 2d gaussian to the detected characteristic
        characteristics, source_crop, parameters, model = fit_2dgauss(imgdata, center_est, t)
        
        # Saves 2d gaussian characteristics to main data table
        if data_table == None:
            data_table = characteristics
        else:
            data_table = vstack([data_table,characteristics])
        
        
        if visualize == True:
            # Subtracts model from image data to get residuals 
            residual = source_crop - model
            
            # Defines maximum and minimum values for consistent image scaling
            vmin = min(imgdata.min(), model.min(), residual.min())
            vmax = max(imgdata.max(), model.max(), residual.max())
            
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
            
            fig, ax = plt.subplots(1,3, layout='constrained')
            
            ax[0].imshow(source_crop, norm=norm, cmap='Greys_r', interpolation='nearest', origin='lower')
            ax[0].scatter(tbl['xcentroid']-parameters[0],tbl['ycentroid']-parameters[2],c='b',marker='.')
            if len(tbl) > 1: ax[0].scatter(center_est[0]-parameters[0],center_est[1]-parameters[2],c='r',marker='.')
            ax[0].scatter(characteristics['x_0']-parameters[0],characteristics['y_0']-parameters[2],c='k',marker='.')
            ax[0].set_title('Detected Source Location')
            
            ax[1].imshow(model, norm=norm, cmap='Greys_r', interpolation='nearest', origin='lower')
            
            ax[1].set_title('Modeled Gaussian')
            
            # Plots model residuals
            ax[2].imshow(residual, norm=norm, origin='lower', cmap='Greys_r',
                       interpolation='nearest')
            
            ax[2].set_title('Model Residuals')
            
            for axi in ax:
                axi.set_xticklabels([]) 
                axi.set_yticklabels([])
        
            plt.show()
        

    # Adds Focus position data corresponding to each 2d gaussian fit
    data_table.add_column(focus_arr,name='Focus')
    
    # Raves the data table to txt
    data_table.write(f'{save_data}/output_data.txt', format='ascii.fixed_width', overwrite=True)

    # Runs u_curve fiting to find opttimal focus location
    optimal_focus = u_curve(data_table['Focus'],data_table['FWHM_x'],data_table['FWHM_x_err'],data_table['FWHM_y'],data_table['FWHM_y_err'])

    return optimal_focus    

    
    
    
    
    