# Purpose: Extract source locations in x and y pixel coordinates from HARMONI datacubes
"""
This script is used to prepare the HSIM output datacube for PampelMuse and extract source locations in x and y pixel
coordinates ready to be imported by PampelMuse.

The boolean variable 'prepare' is used to determine whether the script should do the following:
Load HSIM output datacube and convert into PampelMuse ready cube
1) Convert the CUNIT3 from micron to Angstrom, and other relevant headers (CRVAL3 and CDELT3 x10000)
2) Add a second cube to account for the variances

Input: HARMONI datacube from HSIM, and configuration file
Output: CSV file with source locations in x and y pixel coordinates ready for PampelMuse
"""

# Import necessary libraries and modules
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import numpy as np
from photutils.detection import DAOStarFinder
from matplotlib import pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
import pandas as pd
import time
import math
import sys
import os
from os import path
import json


# Function to load configuration parameters from a file
def load_config(config_file):
    """
    Load a configuration file and return the parameters as a dictionary.
    """
    # Create an empty dictionary to store the parameters
    params = {}

    # Open the configuration file and read the parameters
    with open(config_file) as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue

            # Split the line into a key and value
            key, value = line.split('=')

            # Strip whitespace from the key and value
            key = key.strip()
            value = value.strip()

            # Convert the value to an integer if possible
            try:
                value = int(value)
            except ValueError:
                pass

            # Convert the value to a float if possible
            try:
                value = float(value)
            except ValueError:
                pass

            # Convert the value to a boolean if possible
            if value == 'True' or value == 'yes':
                value = True
            elif value == 'False' or value == 'no':
                value = False

            # Store the parameter in the dictionary
            params[key] = value

    return params


def load_json_config(config_file):
    """
    Load a JSON configuration file and return the parameters as a dictionary.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


def prepare_datacube(flux_file, snr_file, output_dir, quiet):
    """
    Prepare the datacube for PampelMuse by converting the CUNIT3 from micron to Angstrom, and other relevant headers
    (CRVAL3 and CDELT3 x10000). Add a second cube to account for the variances.
        - Changed CTYPE2 from “wavelength” to “AWAV” as per MUSE cubes
        - Multiply signal data by 10^12 to get it in units of '10**(-20)*erg/s/cm**2/Angstrom’
            - whilst considering a 10^-8 reduction which is due to:
                - 10^-4 from converting um to A
                - and 10^-4 from changing units to be /arcsec as each pixel is 10mas
        - The _std file from HSIM is in electrons, we want erg/s/cm**2/Angstrom. Used '_reduced_SNR' cube instead.

    '_reduced_flux_cal.fits' and '_reduced_SNR.fits' are loaded and the stdev cube is squared to get variance.

    """

    # Load the datacubes
    hdulist = fits.open(flux_file)
    stdev = fits.open(snr_file)

    if not quiet:
        hdulist.info()
        stdev.info()

    # Fix header units
    print('Fixing header units...')
    header = hdulist[0].header
    header['CUNIT3'] = 'Angstrom'
    header['CRVAL3'] = header['CRVAL3']*10000   # /1000000#*10000
    header['CDELT3'] = header['CDELT3']*10000   # /1000000#*10000
    header['CTYPE3'] = 'AWAV'
    header['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom'

    # Scale the data cube by 1E12 to get it in units of '10**(-20)*erg/s/cm**2/Angstrom’
    print('Scaling data cube by 1E12...')
    hdulist[0].data = hdulist[0].data*1E12

    # add SNR cube
    print('Adding SNR cube...')
    hdulist.append(stdev[0])

    # Square the stdev cube to get variance (var = (signal/stdev)^2)
    print('Squaring stdev cube to get variance...')
    hdulist[1].data = (hdulist[0].data / hdulist[1].data)**2

    # Change EXTNAME to STAT for the variance cube
    hdulist[1].header['EXTNAME'] = 'STAT'
    hdulist[0].header['EXTNAME'] = 'DATA'

    # Save the datacube to a new file
    hdulist.writeto(path.join(output_dir, 'merged_reduced_SNR_scaled.fits'), overwrite=True)
    print('Datacube saved successfully.')

    cube = hdulist[0].data

    return cube


def cube_stats(config, cube):
    """
    Calculate the mean, median, and standard deviation of the datacube.
    """
    mean_frame = np.asarray(np.mean(cube, axis=0))
    mean, median, std = sigma_clipped_stats(mean_frame, sigma=1.0)
    print('Mean, median, stdev')
    print(mean, median, std)

    return mean_frame, mean, median, std


def source_detection(config, cube):
    """
    Divide cube into 5 chunks, calculate the median frame, and detect sources in each chunk.
    Determine which sources move, and discard.
    """
    # Divide the cube into 5 chunks
    cube1, cube2, cube3, cube4, cube5 = np.array_split(cube, 5, axis=0)

    # Calculate the mean, median, and standard deviation of the datacube
    mean_frame1, mean1, median1, std1 = cube_stats(config, cube1)
    mean_frame2, mean2, median2, std2 = cube_stats(config, cube2)
    mean_frame3, mean3, median3, std3 = cube_stats(config, cube3)
    mean_frame4, mean4, median4, std4 = cube_stats(config, cube4)
    mean_frame5, mean5, median5, std5 = cube_stats(config, cube5)

    # Set the detection threshold
    thresh = config['detection_threshold']
    print('Detection threshold: {}'.format(thresh))

    # Set the FWHM
    fwhm = config['fwhm']
    print('FWHM: {}'.format(fwhm))

    # Detect sources in the image
    print('Detecting sources...')
    std = (std1 + std2 + std3 + std4 + std5) / 5
    daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh*std)
    sources1 = daofind(mean_frame1 - median1)
    sources2 = daofind(mean_frame2 - median2)
    sources3 = daofind(mean_frame3 - median3)
    sources4 = daofind(mean_frame4 - median4)
    sources5 = daofind(mean_frame5 - median5)
    for col in sources1.colnames:
        sources1[col].info.format = '%.8g'
    for col in sources2.colnames:
        sources2[col].info.format = '%.8g'
    for col in sources3.colnames:
        sources3[col].info.format = '%.8g'
    for col in sources4.colnames:
        sources4[col].info.format = '%.8g'
    for col in sources5.colnames:
        sources5[col].info.format = '%.8g'
    print('Sources detected successfully.')

    # Print statistics about detected sources
    print('Number of sources in block 1 detected: {}'.format(len(sources1)))
    print('Number of sources in block 2 detected: {}'.format(len(sources2)))
    print('Number of sources in block 3 detected: {}'.format(len(sources3)))
    print('Number of sources in block 4 detected: {}'.format(len(sources4)))
    print('Number of sources in block 5 detected: {}'.format(len(sources5)))

    # Plot centroids
    plt.figure(figsize=(8, 8))
    plt.scatter(sources1['xcentroid'], sources1['ycentroid'], marker='o', s=100
                , facecolors='none', edgecolors='r', label='Block 1')
    plt.scatter(sources2['xcentroid'], sources2['ycentroid'], marker='o', s=100
                , facecolors='none', edgecolors='b', label='Block 2')
    plt.scatter(sources3['xcentroid'], sources3['ycentroid'], marker='o', s=100
                , facecolors='none', edgecolors='g', label='Block 3')
    plt.scatter(sources4['xcentroid'], sources4['ycentroid'], marker='o', s=100
                , facecolors='none', edgecolors='y', label='Block 4')
    plt.scatter(sources5['xcentroid'], sources5['ycentroid'], marker='o', s=100
                , facecolors='none', edgecolors='m', label='Block 5')
    plt.xlim(0, mean_frame1.shape[1])
    plt.ylim(0, mean_frame1.shape[0])
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.legend()
    #plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 8))
    positions1 = np.transpose((sources1['xcentroid'], sources1['ycentroid']))
    positions2 = np.transpose((sources2['xcentroid'], sources2['ycentroid']))
    positions3 = np.transpose((sources3['xcentroid'], sources3['ycentroid']))
    positions4 = np.transpose((sources4['xcentroid'], sources4['ycentroid']))
    positions5 = np.transpose((sources5['xcentroid'], sources5['ycentroid']))
    apertures1 = CircularAperture(positions1, r=3.0)
    apertures2 = CircularAperture(positions2, r=3.0)
    apertures3 = CircularAperture(positions3, r=3.0)
    apertures4 = CircularAperture(positions4, r=3.0)
    apertures5 = CircularAperture(positions5, r=3.0)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(np.log(mean_frame3 + 1), cmap='Greys', origin='lower')   # , vmin=np.log(6), vmax=np.log(2150))
    apertures1.plot(color='red', lw=1.5, alpha=0.5)
    apertures2.plot(color='blue', lw=1.5, alpha=0.5)
    apertures3.plot(color='green', lw=1.5, alpha=0.5)
    apertures4.plot(color='yellow', lw=1.5, alpha=0.5)
    apertures5.plot(color='magenta', lw=1.5, alpha=0.5)
    plt.show()

    # Combine all sources into columns and save to a CSV file
    print('Saving CSV file...')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dir = os.path.dirname(config['flux_file'])
    dir_output = dir + '/sources'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    filename = os.path.join(dir_output, 'sources_f%s_t%s_%s.csv' % (fwhm, thresh, len(sources1) + len(sources2) + len(sources3) + len(sources4) + len(sources5)))
    np.savetxt(filename, sources1, delimiter=",", fmt='%s')

    # Determine if sources move # TODO - write this function
    # Compare sources1 to sources2, sources2 to sources3, etc.
    # If sources move, discard
    # If sources don't move, keep
    # Return the sources that don't move
    return sources1, sources2, sources3, sources4, sources5


# Function to plot the image with detected sources
def image_plot(config, mean_frame, sources, dir_output):
    """
    Plot the mean frame with detected sources overlaid in red circles.
    """
    plt.figure(figsize=(8, 8))
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=3.0)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(np.log(mean_frame + 1), cmap='Greys', origin='lower')#, vmin=np.log(6), vmax=np.log(2150))
    apertures.plot(color='red', lw=1.5, alpha=0.5)

    basename = os.path.basename(config['flux_file'])
    prefix = basename.split('_reduced')[0]
    plt.savefig(dir_output + '/' + prefix + '_sources.png')
    plt.show()


# Function to plot statistics of detected sources
def stats_plot(config, sources, dir_output):
    """
    Plot histograms of various source properties such as sharpness, peak, magnitude, and flux.
    """
    plt.clf()
    plt.subplot(221)
    plt.hist(sources['sharpness'], bins=20, fc='k', ec='k')
    plt.title("Sharpness")

    plt.subplot(222)
    plt.hist(sources['peak'], range=(0.0, 1000.0), bins=20, fc='k', ec='k')
    plt.title("Peak")

    plt.subplot(223)
    plt.hist(sources['mag'], bins=20, fc='k', ec='k')
    plt.title("Magnitude")

    plt.subplot(224)
    plt.hist(sources['flux'], range=(0.0, 100.0), bins=20, fc='k', ec='k')
    plt.title("Flux")

    plt.tight_layout()
    plt.pause(1)

    basename = os.path.basename(config['flux_file'])
    prefix = basename.split('_reduced')[0]
    plt.savefig(dir_output + '/' + prefix + '_sources_stats.png')
    plt.show()


# Main function that processes the datacube and detects sources
def main(config, output_dir):

    print('---------------------------------')
    print('HARMONI Source Extractor')
    print('---------------------------------')

    # Load the configuration parameters
    # dir = config['dir']
    # prefix = config['prefix']
    flux_file = config['flux_file']
    snr_file = config['snr_file']
    quiet = config['quiet']

    if not quiet:
        print('Configuration parameters:')
        for key, value in config.items():
            print(key, '=', value)

    # Prepare the datacube for PampelMuse
    prepare = config['prepare']
    if prepare:
        cube = prepare_datacube(flux_file, snr_file, output_dir, quiet)
        print('Datacube prepared successfully.')
    else:
        # Check for file in dir ending in _SNR_scaled.fits
        if os.path.isfile(path.join(output_dir, 'merged_reduced_SNR_scaled.fits')):
            print('Datacube already prepared.')
            hdulist = fits.open(path.join(output_dir, 'merged_reduced_SNR_scaled.fits'))
            hdulist.info()
            cube = hdulist[0].data
            print('Datacube loaded successfully.')
        else:
            print('Datacube not prepared. Please set prepare to True in the configuration file.')
            sys.exit(1)

    # Find mean frame and statistics
    mean_frame = np.asarray(np.mean(cube, axis=0))
    mean, median, std = sigma_clipped_stats(mean_frame, sigma=1.0)
    sum_frame = np.sum(cube, axis=0)
    print('Mean, median, stdev')
    print(mean, median, std)

    # Set the detection threshold
    thresh = config['detection_threshold']
    print('Detection threshold: {}'.format(thresh))

    # Set the FWHM
    fwhm = config['fwhm']
    print('FWHM: {}'.format(fwhm))

    # Detect sources with multi-snapshot method
    #sources1, sources2, sources3, sources4, sources5 = source_detection(config, cube)
    # TODO - implement this method

    # Detect sources in the image
    print('Detecting sources...')
    daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh*std, roundlo=-0.75, roundhi=0.75)
    sources = daofind(sum_frame)
    #sources = daofind(mean_frame - median)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'
    print('Sources detected successfully.')

    # Print statistics about detected sources
    print('Number of sources detected: {}'.format(len(sources)))

    # Save source information to a CSV file
    print('Saving CSV files...')
    dir_output = path.join(output_dir, 'sources')

    # Check if the directory exists, if not create it
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    filename = os.path.join(dir_output, 'sources_f%s_t%s_%s.csv' % (fwhm, thresh, len(sources)))
    np.savetxt(filename, sources, delimiter=",", fmt='%s')
    print('All source data CSV file saved successfully:', filename)

    # Create a DataFrame for further processing and save to a CSV file
    names = sources.colnames
    # print(names)
    df = pd.DataFrame(np.asarray(sources), columns=names)

    # Rename and format columns for PampelMuse and save
    df_pm = df[["id", "xcentroid", "ycentroid", "mag"]].copy()
    df_pm.rename(columns={"mag": "H", "xcentroid": "x", "ycentroid": "y"}, inplace=True)
    df_pm.to_csv(dir_output + '/harmoni_stars_f%s_t%s_%s_refcat.csv' % (fwhm, thresh, len(sources)),
                 index=False, header=True, sep=',')
    print('PPM CSV file saved successfully:', dir_output + '/harmoni_stars_f%s_t%s_%s_refcat.csv' %
          (fwhm, thresh, len(sources)))

    # Plot the mean frame with the sources overlaid in red
    image_plot(config, mean_frame, sources, dir_output)
    stats_plot(config, sources, dir_output)


    ## TEMPORARY SPECTRUM EXTRACTOR

if __name__ == '__main__':
    # Read the configuration file from the command line argument
    config_file = sys.argv[1]

    # Check if a configuration file was provided
    if len(sys.argv) < 2:
        print('Usage: python harmoni_source_extractor.py <config_file>')
        exit()

    config_ = load_json_config(config_file)

    global_params = config_.get('global', {})
    config = config_.get('source_extractor', {})

    output_dir = global_params['output_directory']

    if not path.exists(output_dir):
        os.mkdir(output_dir)

    main(config, output_dir)

#%%
