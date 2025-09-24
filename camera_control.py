#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 20:38:13 2025

@author: petershea
"""

import ctypes as ct
import matplotlib.pyplot as plt
import numpy as np
import sys
#from datetime import datetime
import datetime
from PIL import Image
import time
import os
#sys.path.append('./lib')

from astropy.stats import sigma_clipped_stats
from astropy.io import fits, ascii
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import Table

from bokeh.plotting import figure, output_file, show, save, output_notebook
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter, LinearAxis, \
LogAxis, Legend, CustomJS, CheckboxGroup, BooleanFilter, ColorBar, LogTicker, LinearColorMapper, LogColorMapper
from bokeh.layouts import gridplot, row, column
from bokeh.palettes import Reds, d3, Spectral4, inferno, Viridis6, Greys, Greys256

from photutils.centroids import centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic 

from IPython.display import clear_output

from pathlib import Path
import cv2

import zwoasi as asi

from astropy.nddata.utils import Cutout2D

from ids_peak import ids_peak
import ids_peak.ids_peak_ipl_extension as ids_ipl_extension



class ZWOASICamera:
    
    def __init__(self, ASI_filename, camera_id=0, verbose=False):
        '''
        Class to initiate the camera control class for Zwoasi cameras using both 
        the asi filename as well as which camera in the list of zwoasi cameras 
        should be controlled by this instance of the class.
        
        Parameters:
            ASI_filename: str; filename of the ASI control file on the host computer
            camera_id: int; the id of the camera in the list of found cameras, 
                defaults to the first camera found if not specified
            verbose: bool; determines if this object should be verbose in its function 
                and output
        
        Returns:
            N/A
        '''
        
        self.verbose = verbose
        
        if ASI_filename:
            print(ASI_filename)
            asi.init(ASI_filename)
        else:
            raise RuntimeError('The filename of the ASI SDK library is required')
            
        self.num_cameras = asi.get_num_cameras()
        if self.num_cameras == 0:
            raise RuntimeError('No ZWOASI cameras found')
            
        else:
            self.cameras_found = asi.list_cameras()  # Models names of the connected cameras
            if self.verbose: print(f'Cameras found: {self.cameras_found}')
        
        # open camera handle
        self.camera = asi.Camera(camera_id)
        
        self.camera_info = self.camera.get_camera_property()
        
        if verbose:
        # Get all of the current camera control values
            for key,value in self.camera_info.items():
                print(f"{key} :  {value}" )
        
        # Set initial exposure time --- also not sure if necessary for class to function
        set_exptime = 1 # seconds
        set_exptime = int(set_exptime*10**6) # convert to microseconds
        self.camera.set_control_value(asi.ASI_EXPOSURE, set_exptime)  # microseconds
        # confirm exptime was set
        get_exptime = self.camera.get_control_value(asi.ASI_EXPOSURE)
        current_temp = self.camera.get_control_value(asi.ASI_TEMPERATURE)  # returns x10 for precision 
        current_temp[0] = current_temp[0]/10
        self.exptime = get_exptime[0]/10**6
        self.temperature = current_temp[0]


        # Not sure if this is necessary for the class to function
        filename = None
        self.camera.set_image_type(asi.ASI_IMG_RAW16)
        #camera.set_image_type(asi.ASI_IMG_RAW8)
        if filename == None:
            img = self.camera.capture()
        else:
            self.camera.capture(filename=filename)
            print('Saved to %s' % filename)
    
    
    
    def create_timestamp_subfolder(self, homeDir):
        '''
        Creates new folders for the cameras to save their images to based on the 
        time which they are taken
        
        Parameters:
            homeDir: Path; the absolute path to the home save directory within 
                which there are subdirs for ymd and further hms
        
        Returns:
            newpath: Path; the absolute path homeDir/ymd/hms to the directory in
                which the next exposure series will be saved
        '''
        # homeDir
        # └── ymd
        #     ├── hms (exposure series 1)
        #     └── hms (exposure series 2)
        
        # Gets time when this function is called
        tt = datetime.datetime.now()
        # Converts time to ymd/hms
        ymd = tt.strftime("%Y%m%d/%H%M%S/")
        # Creates this new path
        self.newpath = homeDir.joinpath(ymd)
        # Makes new directory if it doesn't already exist
        self.newpath.mkdir(parents=True, exist_ok=True)
        
        return self.newpath
    
    
    
    def capture_imgs(self, object_name='test', exptime=49, nimages=1, sleep=0):
        '''
        The method to capture a series of images using the camera
        
        Parameters:
            object_name: str; the name of the primary object being observed in 
                this exposure series
            exptime: int [sec]; the exposure time for this series in seconds
            nimages: int; the number of images to be taken in this series
            sleep: int [sec]; the number of seconds which will be waited for between
                each exposure in the series
        
        Returns:
            raw_imgs: list [2darrays]; the raw image data of each exposure taken
                in the series
            filenames: str; the filenames of each image taken
            filenamesbmp: str; the filenames for each bitmap corresponding to the 
                images taken
            timestamps: datetime; the time at which each exposure was started
        '''
        # Sets exposure time
        self.exptime = exptime
        # Sets object name
        self.object_name = object_name
        exptime = int(exptime*10**6) # convert to microseconds
        # Sets the camera's exposure time to the specified value
        self.camera.set_control_value(asi.ASI_EXPOSURE, exptime)  # microseconds
        # Sets the image type of the camera to be 16bit
        self.camera.set_image_type(asi.ASI_IMG_RAW16)
        
        # Creates empty lists to store values
        self.raw_imgs = []
        self.filenames = []
        self.filenamesbmp = []
        self.timestamps = []
        
        # Iterates through taking an exposure
        for i in range(nimages):
            
            if self.verbose: print(f'taking ZWO image {i+1}')
            
            # Gets time of the start of the exposure
            t1 = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S_%f")[:-3]
            self.timestamps.append(t1) # appends to time to list
            
            # Creates and appends img and bmp filenames
            self.filenames.append(f'{t1}_{object_name}.zwo.fits')
            self.filenamesbmp.append(f'{t1}_{object_name}.zwo.bmp')
            
            # Captures the image
            img = self.camera.capture()
            self.raw_imgs.append(img) # appends to raw_img list
            
            # if sleep is specified waits that long before continuing
            if(sleep > 0 ):
                if self.verbose==True: print(f'Waiting {sleep}')
                time.sleep(sleep)
    
        return self.raw_imgs, self.filenames, self.filenamesbmp, self.timestamps
    
    
    
    def examine_img(self, image_id=0):
        '''
        Function to examine a specified image taken in the last exposure series
        
        Parameters:
            image_id: int; the id corresponding to the position of the desired 
                exposure within the series
        
        Returns:
            N/A
        '''
        
        # Checks if given value is out of range
        if image_id >= self.nimages:
            
            # Defaults to last image if specified is beyond range
            print('The specified image is out of range, defaulting to last image in series')
            image_id = -1
        
        # Plots the image
        fig, ax = plt.subplots(layout='constrained')
        
        norm = ImageNormalize(stretch=SqrtStretch())
        
        ax.imshow(self.raw_imgs[image_id], norm=norm, cmap='Greys_r', interpolation='nearest', origin='lower')
        
        plt.show()
        
    
    
    def save_data(self, save_fits: bool, save_bmp: bool, savedir=None, imgs=None, filenames=None, filenamesbmp=None, timestamps=None):
        '''
        Function to save data which has been taken 
        
        Parameters:
            save_fits: bool; determines if the function should save fits images
            save_bmp: bool; determines if teh function should save bit maps
            savedir: str; an optional override if this object instance hasn't 
                created a date-time subbdir or if a different savedir is desired
            imgs: list; optional parameter to save a specific list of images instead 
                of the entire last series
            filenames: list; optional parameter equivalent to imgs but for filenames
            fileamesbmp; list optional parameter equivalent to fileames but for bmp
            timestamps; list; optional parameter equivalent to imgs
        
        Returns:
            N/A
        '''
        # Reads in last exposure series save data or override if present
        rawimgs = imgs or self.raw_imgs
        filenames = filenames or self.filenames
        filenamesbmp = filenamesbmp or self.filenamesbmp
        timestamps = timestamps or self.timestamps
        
        # Construct full path to save files
        if savedir is None: 
            filenames = [self.newpath / f for f in filenames]
            filenamesbmp = [self.newpath / f for f in filenamesbmp]
        else:
            filenames = [Path(savedir) / f for f in  filenames]
            filenamesbmp = [Path(savedir) / f for f in filenamesbmp]
    
        # Gets image shape from first image 
        columns,rows = rawimgs[0].shape
        
        # Iterates through given data to save
        for i in range(len(rawimgs)):
            # Save bit map data
            if (save_bmp):
                if self.verbose: print(f'writing number {i+1} to {filenamesbmp[i]}')
                # Creates new image for bitmap
                newImg = Image.new(mode='L', size=[columns, rows])  
                # Inserts data from the raw image
                newImg.putdata(rawimgs[i].flatten())
                # Saves bit map using corresponding bmp filename
                newImg.save(filenamesbmp[i])
            
            # Save Fits data
            if (save_fits):
                if self.verbose: print(f'writing number {i+1} to {filenames[i]}')
                # Saves ndarray version of raw image to fits hdu
                hdu = fits.PrimaryHDU(np.flipud(rawimgs[i]))
                hdul = fits.HDUList([hdu])
                # Creates a header for the fits image metadata
                header = hdul[0].header
                # Writes in meta data for the image
                header['DATE-OBS'] = timestamps[i]
                header['OBJECT'] = self.object_name
                header['EXPTIME'] = (self.exptime, "Exptime in milli-seconds")
                # Saves the fits image to the corresponding filename
                hdul.writeto(filenames[i], overwrite=True)
                hdul.close()
                
    
    
    def __enter__(self):
        '''
        Enter the runtime context for the ZWOASI camera object.

        This method is automatically called when the camera object is used
        in a `with` statement. It initializes the camera session and returns
        the camera instance so that it can be interacted with inside the 
        context.
        
        Parameters:
            N/A
    
        Returns:
            self: ZWOASICamera; The initialized camera object.
        '''
        if self.verbose: print("Starting ZWOASI camera session")
        return self
    
    
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        '''
        Exit the runtime context for the ZWOASI camera object.

        This method is automatically called when the `with` statement
        completes, whether normally or due to an exception. It ensures
        that the camera connection is safely closed.
    
        Parameters
            exc_type : type or None; the exception type if an exception occurred, 
                otherwise `None`.
            exc_value : Exception or None; the exception instance if an exception 
                occurred, otherwise `None`.
            exc_tb : traceback or None; the traceback object if an exception occurred, 
                otherwise `None`.
    
        Returns
            N/A
        '''
        if self.verbose: print("Closing ZWOASI camera session")
        # Closes camera session
        self.camera.close()





class IDSCamera:
    
    def __init__(self, camera_id=0, verbose=False):
        '''
        Class to control an IDS camera. Initializes the camera object, using the 
        given camera_id
        
        Parameters:
            camera_id: int; position in a list of devices corresponding to the desired camera
            verbose: bool; determines whether the functions in this class should be verbose in their output
        
        Returns:
            N/A
        
        Instance Attributes:
            verbose: bool; determines whether the functions in this class should be verbose in their output
            device_manager: DeviceManager object; the ids device manager for the camera
            device: device object; 
        
        '''
        
        self.verbose=verbose
        self.camera_id = camera_id
        
        # Initialize library
        ids_peak.Library.Initialize()
        
        # Create a DeviceManager object
        self.device_manager = ids_peak.DeviceManager.Instance()
        
        # Update device manager
        self.device_manager.Update()
        
        # Not sure if this is necessary
        '''
        # The returned callback object needs to be stored in a variable and
        # needs to live as long as the class that would call it,
        # so as long as the device manager in this case
        # If the callback gets garbage collected it deregisters itself
        device_found_callback = self.device_manager.DeviceFoundCallback(
            lambda found_device: print(
                "Found-Device-Callback: Key={}".format(
                    found_device.Key()), end="\n\n"))
        device_found_callback_handle = self.device_manager.RegisterDeviceFoundCallback(
            device_found_callback)
        # The callback can also be unregistered explicitly using the returned
        # handle
        device_manager.UnregisterDeviceFoundCallback(device_found_callback_handle)
        '''
        
        # Raises an error if no devices are detected
        if self.device_manager.Devices().empty():
            raise RuntimeError("No IDS Camera Found")
        
        # Open the designated device
        self.device = self.device_manager.Devices()[camera_id].OpenDevice(ids_peak.DeviceAccessType_Control)
        if verbose == True: print("Opened Device: " + self.device.DisplayName())
        
        # Nodemap for accessing GenICam nodes
        self.remote_nodemap = self.device.RemoteDevice().NodeMaps()[camera_id]
        
        # Pulls camera pixel dimensions and stores them as instance attributes 
        self.yvalue = self.remote_nodemap.FindNode("SensorPixelHeight").Value()
        self.xvalue = self.remote_nodemap.FindNode("SensorPixelWidth").Value()
    
        # Defines instance attribute data_stream to be None 
        self.data_stream = None
        
        
            
    def create_timestamp_subfolder(self, homeDir):
        '''
        Creates new folders for the cameras to save their images to based on the 
        time which they are taken
        
        Parameters:
            homeDir: Path; the absolute path to the home save directory within 
                which there are subdirs for ymd and further hms
        
        Returns:
            newpath: Path; the absolute path homeDir/ymd/hms to the directory in
                which the next exposure series will be saved
        '''
        # homeDir
        # ├── ymd1
        # │   ├── hms (exposure series 1)
        # │   └── hms (exposure series 2)
        # └── ymd2
        #     ├── hms (exposure series 1)
        #     └── hms (exposure series 2)
        
        
        # Gets time when this function is called
        tt = datetime.datetime.now()
        # Converts time to ymd/hms
        ymd = tt.strftime("%Y%m%d/%H%M%S/")
        # Creates this new path
        self.newpath = homeDir.joinpath(ymd)
        # Makes new directory if it doesn't already exist
        self.newpath.mkdir(parents=True, exist_ok=True)
        
        return self.newpath
        
    
        
    def configure_settings(self, **settings):
        '''
        This function configures the camera settings, defaulting to default setting unless
        other settings are specified
        
        Parameters:
            settings: dict; key-value pairs for specified settings
        
        Returns:
            N/A
        '''
        
        if not settings:
            # Load default camera settings if not otehrwise specified
            self.remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
            self.remote_nodemap.FindNode("UserSetLoad").Execute()
            self.remote_nodemap.FindNode("UserSetLoad").WaitUntilDone() 
        
        else:
            # Apply each provided setting
            for key, value in settings.items():
                node = self.remote_nodemap.FindNode(key)
                if node is None:
                    print(f"Warning: No such setting '{key}' in nodemap")
                    continue

                # Handle enums vs numeric values
                if hasattr(node, "SetValue"):
                    node.SetValue(value)
                elif hasattr(node, "SetCurrentEntry"):
                    node.SetCurrentEntry(value)
                else:
                    print(f"Warning: Could not set {key}")
            print(f"Applied custom settings: {settings}")
        
        
        
    def start_acquisition(self):
        '''
        Configure and start image acquisition for the IDS camera
        
        This method sets up the data stream, allocates and queues the 
        necessary number of image buffers, locks relevant camera parameters,
        and initiates the acquisition sequence.
        
        Parameters:
            N/A
            
        Returns:
            N/A
        '''
        
        # Defines payload size 
        payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
        # Defines data stream variable for the given camera
        self.data_stream = self.device.DataStreams()[self.camera_id].OpenDataStream()
        # Determines the minimum ammount of required buffers needed by the camera
        self.buffer_count_max = self.data_stream.NumBuffersAnnouncedMinRequired()
        # Iterates through buffers to assign payload size and ques them
        for _ in range(self.buffer_count_max):
            buf = self.data_stream.AllocAndAnnounceBuffer(payload_size)
            self.data_stream.QueueBuffer(buf)
        
        # Locks camera parameters for exposure  
        self.remote_nodemap.FindNode("TLParamsLocked").SetValue(1)
        # Prepares the data stream for image acquisition
        self.data_stream.StartAcquisition()
        self.remote_nodemap.FindNode("AcquisitionStart").Execute()
        self.remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()
        
        
    
    # Still need to add the autoexposure setting for when exptime is 0
    def capture_imgs(self, object_name='test', exptime=49, nimages=1, sleep=0):
        '''
        Function to capture images with the camera
        
        Parameters:
            object_name: str; the name of the primary object being observed in 
                this exposure series
            exptime: int [sec]; the exposure time for this series in seconds
            nimages: int; the number of images to be taken in this series
            sleep: int [sec]; the number of seconds which will be waited for between
                each exposure in the series
        
        Returns:
            raw_imgs: list [2darrays]; the raw image data of each exposure taken
                in the series
            filenames: str; the filenames of each image taken
            filenamesbmp: str; the filenames for each bitmap corresponding to the 
                images taken
            timestamps: datetime; the time at which each exposure was started
        '''
        
        # Defines object_name to an instance attribute
        self.object_name = object_name
        
        # Raises an error if the data stream does not exist due to start_acquisition not being used yet
        if self.data_stream is None:
            raise RuntimeError("Acquisition not started. Use start_acquisition() first.")
        
        # Sets exposure value in milliseconds  
        self.remote_nodemap.FindNode("ExposureTime").SetValue(exptime * 1e3)
        
        # Creates empty lists for all information corresponding to teh exposures which should be saved during a series
        self.raw_imgs = []
        self.filenames = []
        self.filenamesbmp = []
        self.timestamps = []
        
        # Stores exptime found in the camera settings to an instance attribute
        self.node_exptime = self.remote_nodemap.FindNode("ExposureTime").Value()/1000
        
        # Loops through capture process taking exposures in a series 
        for i in range(nimages):
            
            # Prints current actions if verbose
            if self.verbose: print(f'taking image {i+1} {self.node_exptime}')
            
            # Determines time at the start of the exposure and saves it to timestamps
            t1 = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S_%f")[:-3]
            self.timestamps.append(t1)
            
            # Appends filename of each exposure to the list of filenames
            self.filenames.append(f'{t1}_{object_name}.fits')
            # Appends filename of each bmp to teh list of bmp filenames
            self.filenamesbmp.append(f'{t1}_{object_name}.bmp')
            
            # Waits for the buffer to finish
            buf = self.data_stream.WaitForFinishedBuffer(1000)
            # Takes image from the buffer
            img = ids_ipl_extension.BufferToImage(buf)
            # Creates a numpy copy of the image 
            self.raw_imgs.append(img.get_numpy().copy())
            # Ques the buffer for continued exposures
            self.data_stream.QueueBuffer(buf)
            
            # If sleep is greater than 0 wait the designated time before the next exposure in the series
            if(sleep > 0 ):
                if self.verbose==True: print(f'Waiting {sleep}')
                time.sleep(sleep)
                
        return self.raw_imgs, self.filenames, self.filenamesbmp, self.timestamps
    
    
    
    def examine_img(self, image_id=0):
        '''
        Function to examine a specified image taken in the last exposure series
        
        Parameters:
            image_id: int; the id corresponding to the position of the desired 
                exposure within the series
        
        Returns:
            N/A
        '''
        
        # Checks if given value is out of range
        if image_id >= self.nimages:
            
            # Defaults to last image if specified is beyond range
            print('The specified image is out of range, defaulting to last image in series')
            image_id = -1
        
        # Plots the image
        fig, ax = plt.subplots(layout='constrained')
        
        norm = ImageNormalize(stretch=SqrtStretch())
        
        ax.imshow(self.raw_imgs[image_id], norm=norm, cmap='Greys_r', interpolation='nearest', origin='lower')
        
        plt.show()
                
                
    
    def save_data(self, save_fits, save_bmp, savedir=None, imgs=None, filenames=None, filenamesbmp=None, timestamps=None):
        '''
        Function to save the desired data taken during the series or afterwards
        
        Parameters:
            save_fits: bool; determines whether fits images should be saved
            save_bmp: bool; determines whether bmp images should be saved
            savedir: path; absolute path to where the fits and bmp files should be saved alternative the directory created created by the camera
            imgs: list [2d ndarray]; all images which should be saved
            filenames: list [str]; filenames of all images which should be saved
            filenamesbmp: list [str]; filenames of all bmp which should be saved
            timestamps: list [str]; hms_f of when each exposure which should be saved
        
        Returns:
            N/A
        '''
        # Determines if outside imgs from previous series or instance attributes should be saved
        rawimgs = imgs or self.raw_imgs
        filenames = filenames or self.filenames
        filenamesbmp = filenamesbmp or self.filenamesbmp
        timestamps = timestamps or self.timestamps
        
        # Construct full path to save files
        if savedir is None: 
            filenames = [self.newpath / f for f in filenames]
            filenamesbmp = [self.newpath / f for f in filenamesbmp]
        else:
            filenames = [Path(savedir) / f for f in  filenames]
            filenamesbmp = [Path(savedir) / f for f in filenamesbmp]
    
        # Determines shape of the images
        columns,rows = rawimgs[0].shape
        
        # Iterates through the designated files
        for i in range(len(rawimgs)):
            # Saves bmp
            if (save_bmp):
                if self.verbose: print(f'writing number {i+1} to {filenamesbmp[i]}')
                newImg = Image.new(mode='L', size=[columns, rows] )  
                newImg.putdata(rawimgs[i].flatten() )
                newImg.save(filenamesbmp[i])
            # Saves fits
            if (save_fits):
                if self.verbose: print(f'writing number {i+1} to {filenames[i]}')
                hdu = fits.PrimaryHDU( np.flipud(rawimgs[i]) )
                hdul = fits.HDUList([hdu])
                header = hdul[0].header
                # Assigns header keywords
                header['xsize'] = (self.xvalue,"pixel width in microns")
                header['ysize'] = (self.yvalue,"pixel height in microns")
                header['DATE-OBS'] = timestamps[i]
                header['OBJECT'] = self.object_name
                header['EXPTIME'] = (self.node_exptime, "Exptime in milli-seconds")
                header['FILTER'] = ("BG39", "FGB39 Thorlabs Filter 360-580nm")
                hdul.writeto(filenames[i], overwrite=True)
                hdul.close()
    
    
    
    def stop_acquisition(self):
        '''
        Function to stop the aquisition process shutting it down properly
        
        Parameters:
            N/A
            
        Returns:
            N/A
        '''
        self.remote_nodemap.FindNode("AcquisitionStop").Execute()
        self.remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()
        self.data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        # FLushes any leftover data from the data_stream buffer
        self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

        for buffer in self.data_stream.AnnouncedBuffers():
            # Remove buffer from the transport layer
            self.data_stream.RevokeBuffer(buffer)
        
        # Unlocks parameters so they can be altered
        self.remote_nodemap.FindNode("TLParamsLocked").SetValue(0)
        
    
    
    def __enter__(self):
        '''
        Enter the runtime context for the IDS camera object.

        This method is automatically called when the camera object is used
        in a `with` statement. It initializes the camera session and returns
        the camera instance so that it can be interacted with inside the 
        context.
        
        Parameters:
            N/A
    
        Returns:
            self: IDSCamera; The initialized camera object.
        '''
        self.configure_settings() # loads default settings
        self.start_acquisition() # prepares for aquisition
        return self
    
    
    
    def __exit__(self,exc_type,exc_value,exc_tb):
        '''
        Exit the runtime context for the ZWOASI camera object.

        This method is automatically called when the `with` statement
        completes, whether normally or due to an exception. It ensures
        that the camera connection is safely closed.
    
        Parameters
            exc_type : type or None; the exception type if an exception occurred, 
                otherwise `None`.
            exc_value : Exception or None; the exception instance if an exception 
                occurred, otherwise `None`.
            exc_tb : traceback or None; the traceback object if an exception occurred, 
                otherwise `None`.
    
        Returns
            N/A
        '''
        self.stop_acquisition() # ends aquisition
        ids_peak.Library.Close() # closes library
        
        
        

## intended implementation
'''
# single camera
with IDSCamera() as cam:
    cam.create_timestamp_subfolder(homeDir) # homeDir need to be Path object
    cam.configure_settings(settings=custom_settings_dict) # customize settings
    images1 = cam.capture_imgs(object_name='test', exptime=5, nimages=10)
    cam.save_data(save_fist=True, save_bmp=True)    # Save can be called after every exposure series
    images2 = cam.capture_imgs(object_name='test', exptime=50, nimages=10)
    cam.save_data(save_fits=True, save_bmp=True, savedir=None, *images1) # or at the end specifying which data to save


# multiple cameras

with IDSCamera() as ids_cam, \
    ZWOASICamera(ASI_filename) as zwo_cam:
        
        ids_cam.configure_settings(settings=custom_Settings_dict) # customize settings of ids camera
        
        
        
        subfolder = ids_cam.create_timestamp_subfolder(homeDir) # for saving images from both cameras to same subfolder
        
        obj1 = 'test1'
        ids_images1 = ids_cam.capture_imgs(object_name=obj1, exptime=5, nimages=10)
        zwo_images1 = zwo_cam.capture_imgs(object_name=obj1, exptime=2.5, nimages=10)
        
        ids_cam.save_data(save_fits=True, save_bmp=True, savedir=subfolder)
        zwo_cam.save_data(save_fits=True, save_bmp=True, savedir=subfolder)
        
        
        
        subfolder = ids_cam.create_timestamp_subfolder(homeDir)
        
        obj2 = 'test2'
        ids_images2 = ids_cam.capture_imgs(object_name=obj2, exptime=10, nimages=10)
        zwo_images2 = zwo_cam.capture_imgs(object_name=obj2, exptime=5, nimages=10)
        
        ids_cam.save_data(save_fits=True, save_bmp=True, savedir=subfolder)
        zwo_cam.save_data(save_fits=True, save_bmp=True, savedir=subfolder)


    
        nseries = 10
        for i in range(nseries):
            
            subfolder = ids_cam.create_timestamp_subfolder(homeDir) # for saving images from both cameras to same subfolder
            
            obj1 = 'test1'
            ids_cam.capture_imgs(object_name=obj1, exptime=5, nimages=10)
            zwo_cam.capture_imgs(object_name=obj1, exptime=2.5, nimages=10)
            
            ids_cam.save_data(save_fits=True, save_bmp=True, savedir=subfolder)
            zwo_cam.save_data(save_fits=True, save_bmp=True, savedir=subfolder)




# multiple cameras multithreading

from concurrent.futures import ThreadPoolExecutor

with IDSCamera() as ids_cam, \
    ZWOASICamera(ASI_filename) as zwo_cam:
        
        ids_cam.configure_settings(settings=custom_Settings_dict) # customize settings of ids camera

        subfolder = ids_cam.create_timestamp_subfolder(homeDir)
        
        Obj1 = 'test1'
        with ThreadPoolExecutor(max_workers=2) as executor:
            
            future1 = executor.submit(ids_cam.capture_imgs, obj1, 5, 10)
            future2 = executor.submit(zwo_cam.capture_imgs, obj1, 5, 10)
            
            results1 = future1.result()
            results2 = future2.result()
        
        ids_cam.save_data(save_fits=True, save_bmp=True, savedir=subfolder)
        zwo_cam.save_data(save_fits=True, save_bmp=True, savedir=subfolder)
        
'''

# scalable multithreading attempt

from concurrent.futures import ThreadPoolExecutor, as_completed



def create_cam_control_file(savedir):
    '''
    Helper function to create the desired txt file for multi camera control. Not
    intended for automation, only as manual data entry to create properly formatted
    txt file.
    
    Parameters:
        savedir: path; absolute path to where the file should be saved 
        
    Retruns:
        Absolute path to the file created for easy access
    '''
    data = {
        "camera": ['ZWO_0', 'IDS_0'],
        "init_file": ['lib\\ASICamera2.dll','N/A'],
        "object_name": ['test1','test1'],
        "exptime": [5,10],
        "nimages": [10,10],
        "sleep": [0,0],
        "savedir": ['/home/steward/lfast/','/home/steward/lfast/']
    }

    tbl = Table(data)

    tbl.write(f'{savedir}cam_control.txt', format='ascii.fixed_width', overwrite=True)

    return Path(f'{savedir}cam_control.txt')

# Mock camera classes to test if camera control table creation was sucessful without
# needing to be connected to the cameras
class MockZWO:
    def __init__(self, init_file, cam_id):
        self.init_file = init_file
        self.cam_id = cam_id
    def __repr__(self):
        return f"<MockZWO init={self.init_file} id={self.cam_id}>"

class MockIDS:
    def __init__(self, cam_id):
        self.cam_id = cam_id
    def __repr__(self):
        return f"<MockIDS id={self.cam_id}>"



def readin_cam_csv(camera_file):
    '''
    Method to read in the camera control txt, initialize camera objects, and store 
    the objects within the table to be used by multi threading function
    
    Parameters:
        camera_file: path; absolute path+filename of the camera conntrol txt which 
            should be read in
    
    Retruns:
        tbl: astropy Table; a table with camera control values after camera objects
            have been initialized
    '''
    
    tbl = Table.read(camera_file, format='ascii.fixed_width')
    
    # Convert camera column to object type so we can store class instances
    tbl['camera'] = tbl['camera'].astype(object)

    for i, row in enumerate(tbl):
        camera, cam_id = row['camera'].split('_')
        
        cam_id = int(cam_id)
        
        if camera == 'ZWO':
            tbl['camera'][i] = ZWOASICamera(row['init_file'],cam_id)
            
        if camera == 'IDS':
            tbl['camera'][i] = IDSCamera(cam_id)
    
    return tbl



def capture_with_context(row):
    '''
    Helper function for proper implementation of camera class in run_multi_cam.
    Takes directions from rows of the camera control table and initializes the 
    object within a with statement for save handling
    
    Parameters:
        row: row; a single row from the camera control table
    
    Returns:
        N/A
    
    '''
    cam = row['camera'] # Defines camera object to a variable
    # Uses with structure to ensure propper __enter__/__exit__ functionality
    with cam:
        return cam.capture_imgs(
            object_name = row['object_name'],
            exptime = row['exptime'],
            nimages = row['nimages'],
            sleep = row['sleep'])



def run_multi_cam(camera_table, save_fits: bool, save_bmp: bool, examine_img_id=None):
    '''
    A method to allow multiple cameras to complete their exposure series simultaneously
    by using multithreading of camera objects. 
    
    Parameters:
        camera_table: astropy.table.Table; control table for a single exposure series 
            using each camera. Must contain:
                'camera': camera object; camera instance (either ZWO or IDS)
                'object_name': str; name of the object targeted by the camera
                'exptime': int; exposure time in sec
                'nimages': int; number of images that should be taken by the camera
                'sleep': int; how long the camera should wait between exposures 
                'savedir': path; path to homedir which the resulting fits images should be saved
        
        save_fits: bool; determines if the function should save fits images
        save_bmp: bool; determines if teh function should save bit maps
        examine_img_id: int; the id of the desired image which should be visualized,
            (The same image id for each camera)
    
    '''
    results = []
    
    # Checks if the provided table has date time subdir for where each camera should save images
    if 'subdir' not in camera_table.colnames:
        camera_table['subdir'] = None
    
    # Creates a savedir map so each given savedir creates only one date-time subdir which all cameras given the savedir save their images to during a series 
    savedir_map = {}
    for row in camera_table:
        # Saves path of the save directory for each row
        savedir = Path(row['savedir'])
        # If save directory hasnt been added to map do so
        if savedir not in savedir_map:
            cam = row['camera']
            subdir = cam.create_timestamp_subfolder(savedir)
            savedir_map[savedir] = subdir

        # All cameras with this savedir share same date-time subdir
        row['subdir'] = savedir_map[savedir]
    
    # Begin multithreading
    with ThreadPoolExecutor(max_workers=len(camera_table)) as executor:
        
        # Dictionary to save 
        futures = {}
        for row in camera_table:
            
            # Submits capture tasks to multithreaded executor
            future = executor.submit(capture_with_context, row) # function to properly begin capturing images
            
            # Assigns the row to a future in a dictionary to link them
            futures[future] = row 
        
        # As tasks are completed the data is saved to the correct file
        for future in as_completed(futures):
            row = futures[future]
            cam = row['camera']
            
            # Visualizes the desired image from each camera exposure series
            if not examine_img_id is None:
                # Trys to examine the desired image from the camera's exposure series after it is completed 
                try:
                    cam.examine_img[examine_img_id]
                
                except Exception as e:
                    print(f'Error displaying image {examine_img_id+1} from {cam}: {e}')
            
            # Tries to save data to the subdir designated to the camera
            try:
                imgs, filenames, filenamesbmp, timestamps = future.result()
                cam.save_data(
                    save_fits = save_fits,
                    save_bmp = save_bmp,
                    savedir = row['subdir']
                    )
                
                # Saves names of all files saved to a dictionary associated with the camera
                saved_files = {}
                if save_fits:
                    saved_files['fits'] = filenames
                if save_bmp:
                    saved_files['bmp'] = filenamesbmp
                
                # Appends saved file dict to results 
                results.append((cam, saved_files))

            # prints Exceptions without exiting if problems occur
            except Exception as e:
                print(f"Error with {cam}: {e}")

    return results
        
  

  
    
  
 