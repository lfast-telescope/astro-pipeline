#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 10:09:22 2025

@author: petershea
"""

import os
from datetime import datetime
import warnings

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib import colors, cm

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
    Class for on-sky processing ranging from stacking raw images, detection of
    sources and background subtraction, and psf fitting of select sources to 
    determining the optimal focus position and examining teelscope tracking
    '''
    
    def __init__(self,raw_path,red_path):
        '''
        Initializes the class using the given fits image file name within the 
        designated directory
        
        Parameters:
            raw_path: str; absolute path to the directory containing the desired 
                raw images (should not end with /)
            red_path: str; absolute path to desired save/reduced directory
        
        Returns:
            N/A
            
        Global Variables:
            raw_path: str; absolute path to the directory containing the desired 
                raw images (should not end with /)
            red_path: str; absolute path to desired save/reduced directory
            raw_date: str; name of the directory containing raw data
            fits_imgs: ndarray [str]; list of absolute path to all fits files to be reduced
            subdirs: ndarray [str]; list of the names of subdirs within raw_path
            psf_data_tbl: astropy table; psf characteristics for all requested sources
                across all reduced images
        '''
        # Changes working directory to raw date path
        os.chdir(raw_path)
        
        # Defines raw and red data paths as global variables
        self.raw_path = raw_path
        self.red_path = red_path
        
        # Defines name of raw data directory as global variable
        self.raw_date = raw_path.strip().split('/')[-1]
        
        # Creates red directoryifit does not exist already
        if not os.path.isdir(f'{red_path}/'): os.mkdir(red_path)
        
        # Determines the contents of the supplied raw directory
        contents = sorted(os.listdir())
        contents = [f for f in contents if not f.endswith('.DS_Store')] # Removes .DS_Store
        
        # For the case of a directory with no subdirs
        self.fits_imgs = np.array([f for f in contents if f.endswith('.FIT') or f.endswith('.fits')])
        
        # For the case of a directory with sudirs containing fits imgs
        if len(self.fits_imgs) == 0:
            self.subdirs = np.array(contents) 
        else:
            self.subdirs = np.array([])
        
        # Creates global psf characteristic table for all reduced images
        self.psf_data_tbl = None



    def load_fits(self, filename):
        
        '''
        
        filename: str; absolute path to desired fits image
        '''
        
        # Opens fits file
        hdul = fits.open(filename)
        
        file = filename.strip().split('/')[-1]
        
        file_name, extension = os.path.splitext(file)
        
        self.fits_name = file_name
        
        # Sets header and raw data to be global variables
        self.header = hdul[0].header
        self.image = hdul[0].data
        
        # Trys to extract time at start of exposure from the fits header
        try:
            time = self.header['DATE-OBS']
        except KeyError:
            print('DATE-OBS keyword not found')
        
        if '_' in time:
            time = time.replace('_','.')
            time = datetime.strptime(time, "%Y%m%dT%H%M%S.%f")
            time = time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            time = str(time)
            
        # Transforms time from str to astropy time object in jd
        time = Time(time, format='isot', scale='utc')
        self.time = time.jd



    def stack(self, subdir, stacked_filename, specific_fits=None, sigma=3):
        '''
        Function to stack images to increase snr.
        
        Parameters:
            subdir: str; absolute path to directory that contains the raw images which should be stacked
            stacked_filename: str; name of stacked image (does not include extension .fits)
            specific_fits: ndarray [str]; list of specified fits images to stack if only a subset is desired, absolute paths to files
            sigma: float or int; sigma level for mad stats 
            
        returns:
            N/A
            
        Global Variables: 
            stacked_fits: ndarray [str]; list of absolute paths to all stacked fits files created in this class
            stacked_path: str; path to directory in which all stacked images will be saved (within red_path)
            fits_imgs: ndarray [str]; list of all fits images within the given subdir
        '''
        
        # Defines empty list as global variable 
        self.stacked_fits = np.array([])
        
        # Defines absolute path to location where stacked images are saved
        self.stacked_path = f'{self.red_path}/stacked_imgs'
    
        if not os.path.isdir(self.stacked_path):os.mkdir(self.stacked_path)
        
        # Default; pulls all files from given directory and compiles list of fits images to stack 
        if specific_fits is None:
            files = sorted(os.listdir(subdir))
            self.fits_imgs = np.array([f for f in files if f.endswith('.FIT') or f.endswith('.fits')])
            
        # If specific fits files within the directory are specified only stack those
        else:
            self.fits_imgs = specific_fits

        # Stack all data from fits files in ndarray
        stack = np.array([fits.getdata(f'{subdir}/{f}') for f in self.fits_imgs])  
        
        # Define sigma clip and clip stacked data
        clipped = sigma_clip(stack,sigma=sigma,axis=0)
        
        # Average stacked data
        stacked_image = np.mean(clipped, axis=0)
        
        # Insert nans to ensure array is full
        stacked_image = np.asarray(stacked_image.filled(np.nan)) 
        
        # Variable to store filename
        self.stacked_filename = f'{self.stacked_path}/{stacked_filename}_stacked.fits'
        
        # appends created stacked fits to the global variable to keep track
        self.stacked_fits = np.append(self.stacked_fits, self.stacked_filename)
        
        # Copy header from first exposure 
        with fits.open(f'{subdir}/{self.fits_imgs[0]}') as hdul:
            header = hdul[0].header.copy()
        
        # Open header of last exposure
        with fits.open(f'{subdir}/{self.fits_imgs[-1]}') as hdul:
            header_end = hdul[0].header.copy()
            
        # Copy end time of last exposure to header     ### Something is wrogn with this (possibly header) ###
        if "DATE-END" in header_end:
            header["DATE-END"] = header_end["DATE-END"]
        else:
            header["DATE-END"] = "UNKNOWN"
        
        # Saves stacked image
        hdu = fits.PrimaryHDU(stacked_image, header=header)
        hdu.writeto(self.stacked_filename, overwrite=True)
       
        
       
    def detect(self, npixels=10, connectivity=8):
        '''
        Function that preforms rudimentary background estimation and subtraction, 
        detects sources, preforms deblending, and returns an table of sources with 
        data from phot utils, also returns segment map of detected sources and 
        background subtracted image.
        
        Parameters:
            
            npixels: int; number of connected pixels to qualify as a source (most 
                used in this fucntion to distinguish between sources and cosmic 
                rays/hot pixels) *Optional
            connectivity: int; 4 or 8 determines type of pixel connectiviy
        
        returns:
            tbl: astropy table; source data characteristics from phot utils
            
        Global Variables:
            header: astropy header; header data for given filename
            image: 2d ndarray; raw data from the fits image
            time: astropy time [jd]; time at the start of the exposure
            imgdata: 2d ndarray; background subtracted image 
            semgent_map: 2d ndarray; segment map of deblended sources
            sources_table: astropy table; source data characteristics from phot utils
        '''
        
        try:    # initial source detection catching images with no source
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=NoDetectionsWarning)
                threshold_init = detect_threshold(self.image, nsigma=3)
                segment_img_init = detect_sources(self.image, threshold_init, npixels=npixels)
        
        except NoDetectionsWarning:
            print(f"\tNo sources detected in {self.fits_name}")
            return None
        
        # Initial source mask
        mask_init = segment_img_init.make_source_mask(footprint=circular_footprint(radius=10))
        
        # Initial Background estimation using initial mask 
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(self.image, (64,64), filter_size=(5,5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask=mask_init)
        
        # Detection threshold for sources based on initial esstimation of background
        threshold = detect_threshold(self.image, background=bkg.background, error=bkg.background_rms, nsigma=10)
    
        # Detection of sources
        segment_img = detect_sources(self.image, threshold, npixels=npixels)
        
        # Creating new mask based on source detection with initial background estimation 
        footprint = circular_footprint(radius=10)
        
        # Mask based on detected sources
        mask = segment_img.make_source_mask(footprint=footprint)
    
        # Final background estimation 
        bkg = Background2D(self.image, (64,64), filter_size=(5,5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask=mask)
        
        # Background subtracting image
        self.imgdata = np.abs(self.image.astype(float) - (bkg.background))
        
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
        self.sources_table = tbl
    
        return tbl
        
    
    
    def avg_source(self, source=None, bound=50):
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
            
        Global Variables:
            brightest_source: astropy row object; the source characteristics 
                for the brighest object detected in the image
        '''
        tbl = self.sources_table
        
        # Pulls the row of the brightest detected source
        self.brightest_source = tbl[np.argmax(tbl['kron_flux'])]
        source = self.brightest_source
        
        # If a source is designated, averages sources around designated source, else uses the brightest
        if not source==None: 
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



    def fit_2dgauss(self, center_est, bound=50, sigma_guess=2, theta_guess=0):
        '''
        This function fits a 2d eliptical gaussian to a source designated by the coordinates
        center_est. To do so, the imaged is cropped to only include the area directy around
        the source and returns the characteristics of the fit gaussian
        
        Parameters:
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
            
        Global Variables:
            center_est: tuple; xy position of the estimated center for fitting the 2d gaussian (x,y)
            img_bounds: tuple; tuple of cropped image bounds (xmin,xmax,ymin,ymax)
            source_crop: 2d ndarray; cropped image of the source which is being fit
            model: 2d ndarray; model imaage of fit 2d gaussian with same dimension as 
                source_crop
            residuals: 2d ndarray; residuals of the image and the 
            sigma_x, sigma_y: floats [pixels]; rms stddev of the gaussian in x and y 
            psf_data_tbl: astropy table; appends derived psf characteristics of the given source to
                the data table
        '''
        
        # Estimate for the center of the source
        self.center_est = center_est
        avg_x, avg_y = center_est
        
        # Define boundaries around source to make fitting quicker
        xmin, xmax = int(avg_x)-bound, int(avg_x)+bound+1
        ymin, ymax = int(avg_y)-bound, int(avg_y)+bound+1
        
        # Save boundaries to a variable to return 
        self.img_bounds = (xmin,xmax,ymin,ymax)
        
        # Crop image to only be focusing source
        self.source_crop = self.imgdata[ymin:ymax, xmin:xmax]
        
        # Determine size of cropped source image
        x_size, y_size = self.source_crop.shape
        
        # Create x and y arrays for creating the 2dgaussian model
        x_arr, y_arr = np.mgrid[:x_size, :y_size]
        
        # Initial gaussian model based on estimated parameters
        g_init = models.Gaussian2D(amplitude=np.max(self.source_crop),
                                   x_mean=avg_x-xmin,
                                   y_mean=avg_y-ymin,
                                   x_stddev=sigma_guess,
                                   y_stddev=sigma_guess,
                                   theta=theta_guess)
        
        # Fits the model to the given data
        fitter = fitting.LevMarLSQFitter() # Determines type of fitter used
        g_fit = fitter(g_init, x_arr, y_arr, self.source_crop)
        
        # Model image of the fit 2d gaussian
        self.model = g_fit(x_arr,y_arr)
        
        self.residuals = self.source_crop - self.model
        
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
        self.sigma_x = g_fit.x_stddev.value
        self.sigma_y = g_fit.y_stddev.value
        
        # Converts from sigma to FWHM
        fwhm_x = gaussian_sigma_to_fwhm * self.sigma_x
        fwhm_y = gaussian_sigma_to_fwhm * self.sigma_y
        
        # Determines largest of x and y and calculates eccentricity
        a = max(self.sigma_x, self.sigma_y)
        b = min(self.sigma_x, self.sigma_y)
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
        data_table.add_row([self.time,round(x_0,2),round(y_0,2),round(fwhm_x,2),
                            round(dy_x,2),round(fwhm_y,2),round(dy_y,2),
                            round(eccentricity,2),round(theta,2),round(amplitude,2)])
    
        # if globaal variable for psf data is empty sets created data table to be global
        if self.psf_data_tbl == None:
            self.psf_data_tbl = data_table
        # Otherwise concaternates tables to append psf data of subsuquent images
        else:
            self.psf_data_tbl = vstack([self.psf_data_tbl,data_table])
    
        return data_table
    
    
    
    def visualize_2dgauss(self,psf_characteristics):
        '''
        Function designed to visually examine the results of fit_2dgauss(). Plots
        source image, synthesized model, and model residuals. Intended to be run 
        directly after fit_2dgauss using the returned data from it.
        
        Parameters: 
            psf_characteristics: astropy table; a 1 row astropy table (or row) 
                with psf characteristics of 1 fit source
                
        Returns:
            N/A
        
        '''
        # Defines shorter variable
        tbl = psf_characteristics
        
        # Determines consistent color scale between images
        vmin = min(self.imgdata.min(), self.model.min(), self.residuals.min())
        vmax = max(self.imgdata.max(), self.model.max(), self.residuals.max())
        
        # Defines image norm
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        
        fig, ax = plt.subplots(1,3, layout='constrained')
        
        # Defines image bounds for cropping image to source
        xmin,xmax,ymin,ymax = self.img_bounds
        
        # Plot cropped image of source
        ax[0].imshow(self.source_crop, norm=norm, cmap='Greys_r', interpolation='nearest', origin='lower')
        ax[0].scatter(self.sources_table['xcentroid']-xmin,self.sources_table['ycentroid']-ymin,c='b',marker='.')
        if len(tbl) > 1: ax[0].scatter(self.center_est[0]-xmin,self.center_est[1]-ymin,c='r',marker='.')
        ax[0].scatter(tbl['x_0']-xmin,tbl['y_0']-ymin,c='k',marker='.')
        ax[0].set_title('Detected Source Location')
        
        # Plot model of source
        ellipse = Ellipse(
        (tbl['y_0'], tbl['x_0']),                
        width=3*self.sigma_y,         
        height=3*self.sigma_x,        
        angle=tbl['Rotation'],             
        edgecolor='k',
        facecolor='none',
        linewidth=2)
        
        # plots synthesized image of model and ellips corresponding to 2d gaussian
        ax[1].imshow(self.model, norm=norm, cmap='Greys_r', interpolation='nearest', origin='lower')
        ax[1].add_patch(ellipse)
        ax[1].set_title('Modeled Gaussian')
        
        
        # Plots model residuals
        ax[2].imshow(self.residuals, norm=norm, origin='lower', cmap='Greys_r',
                   interpolation='nearest')
        
        ax[2].set_title('Model Residuals')
        
        for axi in ax:
            axi.set_xticklabels([]) 
            axi.set_yticklabels([])
    
        plt.show()
    
    
    
    def visualize_detected_sources(self):
        
        fig, ax = plt.subplots()
    
        ax.imshow(self.imgdata, norm=norm, cmap='Greys_r', interpolation='nearest', origin='lower')
    
    def phot_reduce_single(self, fits_file, save_bkg_sub):
        '''
        WIP FUNCTION: produces psf chaaracteristics for all sources detected in a given fits file
        
        Parameters:
            fits_file: str; absolute path to desired fits image
            save_bkg_sub: boolean; determines if the background subtracted image 
                created by detect should be saved or not
        
        Returns: 
            N/A
        '''
        # Loads data from the given fits image
        self.load_fits(fits_file)
        
        # Detect sources in the given image 
        tbl = self.detect()
        
        # For each source in the image fit a 2d gaussian
        for i in range(len(tbl['xcentroid'])):
            
            # Defines center estimate of the source to be detected xy coords
            try:
                center_est = (tbl['xcentroid'][i].value[0],tbl['ycentroid'][i].value[0])
                
            except AttributeError:
                center_est = (tbl['xcentroid'][i],tbl['ycentroid'][i])
            
            # Fits 2d gaussian
            psf_characteristics = self.fit_2dgauss(center_est)
        
        # Saves the psf characteristics of all sources in the image to a data table 
        self.psf_data_tbl.write(f'{self.red_path}/{self.fits_name}_psf_data.txt', format='ascii.fixed_width', overwrite=True)
        
        # Saves the background subtracted image if desired copying the header from the raw fits img
        if save_bkg_sub is True:
            
            header = self.header
            
            hdu = fits.PrimaryHDU(self.imgdata, header=header)
            hdu.writeto(f'{self.red_path}/{self.fits_name}_bkgsub.fits', overwrite=True)
            
          
    
    def phot_reduce_all(self, fits_files, save_bkg_sub=False):
        '''
        WIP FUNCTION: designed to reduce a list of fits files and save psf characteristics for all listed files 
        
        Parameters: 
            fits_files: ndarray [str]; list of absolute paths to all fits files 
                which should be photometrically reduced  
            save_bkg_sub: boolean; determines if the background subtracted image 
                created by detect should be saved or not
        
        Returns:
            N/A
        '''
        # Clears PSF data table if another function has been previously called
        self.psf_data_tbl = None
        
        # For each image calls the single image reduction function
        for fits_file in fits_files:
            
            self.phot_reduce_single(fits_file, save_bkg_sub)
            
            # Clears global psf data table for next reduced image
            self.psf_data_tbl = None
            
    
    
    def focus_reduce_one(self, fits_file, visualize=False):
        '''
        Function which calls other functions in the class to detect sources, 
        estimate center if necessry, and fit 2d gaussian to the source of a single
        fits image. Does not stack images.
        
        Parameters:
            fits_file: str; absolute path to fits file desired for reduction
            visualize: boolean; determines if results from 2d gaussian fitting should be shown
        
        Returns:
            N/A
        '''
        # defines source table from detect function
        self.load_fits(fits_file)
        
        tbl = self.detect()
        
        # If more than 1 source is found runs avg_source to determine true center
        if len(tbl) > 1:
            center_est = self.avg_source()
        else:
            center_est = (tbl['xcentroid'].value[0], tbl['ycentroid'].value[0])
        
        # Fits gaussian
        psf_characteristics = self.fit_2dgauss(center_est)
        
        # visualizes result of fit 
        if visualize == True:
            self.visualize_2dgauss(psf_characteristics)
     
        
    
    def reduce_focus_sweep(self, focus_arr, visualize=False):
        '''
        Function to determine optimal focus point based on minimizing fwhm from
        focus sweep data set. Reduces all files in raw_path to do so
        
        Parameters:
            focus_arr: ndarray [µm]; array of focus positions associated with each image
            visualize: boolean; determines if plot of focus sweep data should be created
        
        Returns:
            optimal_focus: float [µm]; the optimal focus location to minimize source fwhm in x and y
        '''
        
        # Determines mode: if there are subdirs then stack images within subdirs and reduce stacked images
        #       otherwise: reduce all fits images within the directory separately (no stacking)
        if not len(self.subdirs) == 0:
            stack_path = f'{red_path}/stacked_imgs'
            if os.path.isdir(stack_path):
                num_stacked = len(np.array([f for f in sorted(os.listdir(stack_path)) if f.endswith('.FIT') or f.endswith('.fits')]))
            
                if num_stacked == len(self.subdirs):
                    fits_names = np.array([f for f in sorted(os.listdir(stack_path)) if f.endswith('.FIT') or f.endswith('.fits')])
                
                    fits_paths = np.array([])
                    
                    for i in range(len(fits_names)):
                        fits_paths = np.append(fits_paths,f'{stack_path}/{fits_names[i]}')
                    
                    self.fits_imgs = fits_paths
                        
                
                else:
                    for subdir in self.subdirs:
                        self.stack(f'{self.raw_path}/{subdir}',subdir)
                    
                    # Defines global variable fits_imgs to be the stacked fits  for reduction
                    self.fits_imgs = self.stacked_fits
        
            else:
                for subdir in self.subdirs:
                    self.stack(f'{self.raw_path}/{subdir}',subdir)
                
                # Defines global variable fits_imgs to be the stacked fits  for reduction
                self.fits_imgs = self.stacked_fits
        
        # Reduces each fits file in fits_imgs global variable
        for fits_img in self.fits_imgs:
            self.focus_reduce_one(fits_img)
        
        self.psf_data_tbl.add_column(focus_arr,name='Focus Position')
        
        # Saves psf characteristic table for all reduced images to a txt
        self.psf_data_tbl.write(f'{self.red_path}/{self.raw_date}_focus_data.txt', format='ascii.fixed_width', overwrite=True)
        
        # Function for upper branch of vertical hyperbola
        vertical_hyperbola = lambda x, h, k, a, b: k + np.sqrt(a**2 * (1 + ((x - h)**2) / b**2))
        
        # Pulls columns for psf fwhm from global data table
        x_fwhm_arr = self.psf_data_tbl['FWHM_x']
        y_fwhm_arr = self.psf_data_tbl['FWHM_y']
        
        # Pulls colums for psf fwhm error from global data table
        x_fwhm_err = self.psf_data_tbl['FWHM_x_err']
        y_fwhm_err = self.psf_data_tbl['FWHM_y_err']
        
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
        
        if visualize == True:
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
        
        return f'{focus_cont[optimal_focus]:.2f} [µm]'
    
    
    
    def reduce_tracking(self, subdirs=None, visualize=False, verbose=False):
        
        self.psf_data_tbl = None
        
        if not os.path.isdir(f'{self.red_path}/stacked_imgs'):
            
            onsky_stacked = np.array([])
            SHWF_stacked = np.array([])
            
            if subdirs is None:
                subdirs = self.subdirs
        
            # Interates through subdirs within the main raw data directory
            for subdir in subdirs:
                
                wkdir = f'{self.raw_path}/{subdir}'
                
                all_files = sorted(os.listdir(wkdir))
                
                onsky_fits = np.array([f for f in all_files if '.zwo' in f])
                
                SHWF_fits = np.array([f for f in all_files if '.zwo' not in f and '.fits' in f])
                
                if len(onsky_fits) > 0:
                    self.stack(wkdir,f'{subdir}_onsky',specific_fits=onsky_fits)
                    onsky_stacked = np.append(onsky_stacked,self.stacked_filename)
                    
                    
                if len(SHWF_fits) > 0:
                    self.stack(wkdir,f'{subdir}_SHWF',specific_fits=SHWF_fits)
                    SHWF_stacked = np.append(SHWF_stacked,self.stacked_filename)
                    
        
        else:
            
            all_stacked = sorted(os.listdir(f'{self.red_path}/stacked_imgs'))
            
            onsky_stacked = np.array([f for f in all_stacked if 'onsky' in f])
            onsky_stacked = np.char.add(f'{self.red_path}/stacked_imgs/',onsky_stacked)
            SHWF_stacked = np.array([f for f in all_stacked if 'SHWF' in f])
            SHWF_stacked = np.char.add(f'{self.red_path}/stacked_imgs/',SHWF_stacked)
        
        ### Shack-Hartman Wavefront
        
        
        
        ### Onsky tracking 
        
        visual = None
        
        if not os.path.isfile(f'{red_path}/tracking_psf_data.txt'):
            
            for file in onsky_stacked:
                
                print(f'Working on {file}')
                
                self.load_fits(file)
                
                tbl = self.detect()
                
                if tbl is None:

                    self.psf_data_tbl.add_row([self.time,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                                    np.nan])
                    
            
                else:
                    
                    if len(tbl) > 1:
                        center_est = self.avg_source()
                    else:
                        center_est = (tbl['xcentroid'].value[0], tbl['ycentroid'].value[0])
                    
                    # Fits gaussian
                    psf_characteristics = self.fit_2dgauss(center_est)
                    
                    if visual is None: visual = self.imgdata
            
                self.psf_data_tbl.write(f'{red_path}/tracking_psf_data.txt', format='ascii.fixed_width', overwrite=True)      
        
        
        else:
            self.psf_data_tbl = Table.read(f'{red_path}/tracking_psf_data.txt', format='ascii.fixed_width')
            
            self.load_fits(onsky_stacked[0])
            
            tbl = self.detect()
            
            if visual is None: visual = self.imgdata
        
        time = (self.psf_data_tbl['Time'] - self.psf_data_tbl['Time'][0]) * 1440
        
        drift_arr = np.array([])
        xdrift_arr = np.array([])
        ydrift_arr = np.array([])
        for i in range(len(self.psf_data_tbl['x_0'])-1):
            x_drift = (self.psf_data_tbl['x_0'][i+1] - self.psf_data_tbl['x_0'][i])
            y_drift = (self.psf_data_tbl['y_0'][i+1] - self.psf_data_tbl['y_0'][i])
            drift = np.sqrt(x_drift**2 + y_drift**2)
            dt = time[i+1] - time[i]
            
            drift_arr = np.append(drift_arr,drift/dt)
            xdrift_arr = np.append(xdrift_arr,x_drift/dt)
            ydrift_arr = np.append(ydrift_arr,y_drift/dt)
            
        mean_drift = np.mean(drift_arr)
        mean_x_drift = np.mean(xdrift_arr)
        mean_y_drift = np.mean(ydrift_arr)
        
        drift_resultls = np.array([mean_drift,mean_x_drift,mean_y_drift])
            
        if visualize == True:
            
            bound = 100
            
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
            
            ax0.imshow(visual, norm=norm, origin='lower', cmap='Greys_r',
                       interpolation='nearest')
            ax0.scatter(self.psf_data_tbl['x_0'],self.psf_data_tbl['y_0'], c=time, cmap='viridis', marker='.')
            ax0.set_xlim(np.min(self.psf_data_tbl['x_0'])-bound,np.max(self.psf_data_tbl['x_0'])+bound)
            ax0.set_ylim(np.min(self.psf_data_tbl['y_0'])-bound,np.max(self.psf_data_tbl['y_0'])+bound)
            ax0.set_aspect('equal', adjustable='box')
            ax0.set_title('Tracking Drift')
            
            sm = cm.ScalarMappable(cmap=cmap_time, norm=norm_time)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax0)
            cbar.set_label("Time from Initial Observation [min]")
            
            # --- Plot 2: FWHM over time ---
            
            x_mask = self.psf_data_tbl['FWHM_x'] >= 1
            y_mask = self.psf_data_tbl['FWHM_y'] >= 1
            tot_mask = x_mask & y_mask
            
            fwhm_x = self.psf_data_tbl['FWHM_x'][tot_mask]
            fwhm_y = self.psf_data_tbl['FWHM_y'][tot_mask]
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
            
            ax1.set_xlim(0,25)
            ax1.set_ylim(0,((np.max(np.maximum(self.psf_data_tbl['FWHM_x'][tot_mask],self.psf_data_tbl['FWHM_y'][tot_mask]))//5)+1)*5)
            
            ax1.set_xlabel("Time from Initial Observation [min]")
            ax1.set_ylabel("FWHM [pixels]")
            
            # --- Plot 3: Rotation over time ---
            
            masked_theta = np.deg2rad(self.psf_data_tbl['Rotation'][tot_mask] % 360)
            
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
            ax2.set_xticks(labels)
            ax2.set_xticklabels(new_labels)
            
            # --- Plot4: Amplitude over time ---
            
            masked_amp = self.psf_data_tbl['Amplitude'][tot_mask]
            
            a_popt, a_pcov = curve_fit(linear, masked_time, masked_amp)
            
            ax3.scatter(masked_time, masked_amp, color='k', marker='.')
            ax3.plot(time_cont, linear(time_cont,*a_popt), color='k', linestyle='--')
            ax3.set_xlim(0,25)
            
            ax3.set_title('2D Gaussian Amplitude')
            ax3.set_xlabel("Time from Initial Observation [min]")
            ax3.set_ylabel("Amplitude [ADU]")
            
            exposures = len(self.psf_data_tbl['Time'])
            
            plt.suptitle(f'{exposures - np.sum(tot_mask)}/{exposures} Failed; Median X Drift: {np.median(xdrift_arr):.2f} pixels/min; Median Y Drift: {np.median(ydrift_arr):.2f} pixels/min')

        if verbose == True: 
            
            print(f'{exposures - np.sum(tot_mask)}/{exposures} Failed')
            
            print(mean_drift,'pixels/min')
            print(f'Mean X drift: {mean_x_drift} pixels/min')
            print(f'Mean Y drift: {mean_y_drift} pixels/min')


        return drift_resultls
    

### TESTING ###

#raw_path = '/Users/petershea/Desktop/Research/LFAST/Data/20250626'

#red_path = '/Users/petershea/Desktop/Research/LFAST/Data/20250626_test'

#test_fit = '/Users/petershea/Desktop/Research/LFAST/Data/20250626_test/stacked_imgs/000009_SHWF_stacked.fits'


#test = RED(raw_path,red_path)

#test.phot_reduce_single(test_fit, True)



