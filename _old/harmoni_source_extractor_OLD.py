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


def prepare_datacube(config):
    """
    Prepare the datacube for PampelMuse by converting the CUNIT3 from micron to Angstrom, and other relevant headers
    (CRVAL3 and CDELT3 x10000). Add a second cube to account for the variances.
    """
    # Load the configuration parameters
    dir = config['dir']
    prefix = config['prefix']

    # Load the datacubes
    hdulist = fits.open(dir + '/'+ prefix +'_reduced_flux_cal.fits')
    stdev = fits.open(dir + '/'+ prefix +'_reduced_SNR.fits')

    # Fix header units
    print('Fixing header units...')
    header = hdulist[0].header
    header['CUNIT3'] = 'Angstrom'
    header['CRVAL3'] = header['CRVAL3']*10000#/1000000#*10000
    header['CDELT3'] = header['CDELT3']*10000#/1000000#*10000
    header['CTYPE3'] = 'AWAV'
    header['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom'

    # Scale the data cube by 1E12
    print('Scaling data cube by 1E12...')
    hdulist[0].data = hdulist[0].data*1E12

    # add SNR cube
    print('Adding SNR cube...')
    hdulist.append(stdev[0])

    # Square the stdev cube to get variance
    print('Squaring stdev cube to get variance...')
    hdulist[1].data = (hdulist[0].data / hdulist[1].data)**2

    # Change EXTNAME to STAT
    hdulist[1].header['EXTNAME'] = 'STAT'
    hdulist[0].header['EXTNAME'] = 'DATA'

    # Save the datacube
    hdulist.writeto(dir+'/'+ prefix +'_reduced_SNR_scaled.fits', overwrite=True)
    print('Datacube saved successfully.')

    cube = hdulist[0].data

    return cube


# Function to plot the image with detected sources
def image_plot(config, mean_frame, sources):
    """
    Plot the mean frame with detected sources overlaid in red circles.
    """
    plt.figure(figsize=(8, 8))
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=3.0)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(np.log(mean_frame + 1), cmap='Greys', origin='lower')#, vmin=np.log(6), vmax=np.log(2150))
    apertures.plot(color='red', lw=1.5, alpha=0.5)
    # Save the figure in dir
    dir = config['dir']
    prefix = config['prefix']
    plt.savefig(dir + '/' + prefix + '_sources.png')
    plt.show()



# Function to plot statistics of detected sources
def stats_plot(config, sources):
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

    # Save the figure in dir
    dir = config['dir']
    prefix = config['prefix']
    plt.savefig(dir + '/' + prefix + '_sources_stats.png')
    plt.show()


# Main function that processes the datacube and detects sources
def main(config):
    # Load the configuration parameters
    dir = config['dir']
    prefix = config['prefix']

    # Prepare the datacube for PampelMuse
    prepare = config['prepare']
    if prepare:
        cube = prepare_datacube(config)
        print('Datacube prepared successfully.')
    else:
        # Check for file in dir ending in _SNR_scaled.fits
        if os.path.isfile(dir + '/' + prefix + '_reduced_SNR_scaled.fits'):
            print('Datacube already prepared.')
            hdulist = fits.open(dir + '/' + prefix + '_reduced_SNR_scaled.fits')
            hdulist.info()
            cube = hdulist[0].data
            print('Datacube loaded successfully.')
        else:
            print('Datacube not prepared. Please set prepare to True in the configuration file.')
            sys.exit(1)

    # Find mean frame and statistics
    mean_frame = np.asarray(np.mean(cube, axis=0))
    mean, median, std = sigma_clipped_stats(mean_frame, sigma=1.0)
    print('Mean, median, stdev')
    print(mean, median, std)

    # Set the detection threshold
    thresh = config['detection_threshold']
    print('Detection threshold: {}'.format(thresh))

    # Set the FWHM
    fwhm = config['fwhm']
    print('FWHM: {}'.format(fwhm))

    # Detect sources in the image
    print('Detecting sources...')
    daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh*std)
    sources = daofind(mean_frame - median)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'
    print('Sources detected successfully.')

    # Print statistics about detected sources
    print('Number of sources detected: {}'.format(len(sources)))

    # Save source information to a CSV file
    print('Saving CSV file...')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dir = os.path.dirname(config['datacube_file'])
    dir = dir + '/sources'

    # Check if the directory exists, if not create it
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = os.path.join(dir, 'sources_f%s_t%s_%s.csv' % (fwhm, thresh, len(sources)))
    np.savetxt(filename, sources, delimiter=",", fmt='%s')
    print('CSV file saved successfully:', filename)

    # Create a DataFrame for further processing and save to a CSV file
    names = sources.colnames
    print(names)
    df = pd.DataFrame(np.asarray(sources), columns=names)

    # Rename and format columns for PampelMuse and save
    df_pm = df[["id", "xcentroid", "ycentroid", "mag"]].copy()
    df_pm.rename(columns={"mag": "H", "xcentroid": "x", "ycentroid": "y"}, inplace=True)
    df_pm.to_csv(dir + '/harmoni_stars_f%s_t%s_%s_refcat.csv' % (fwhm, thresh, len(sources)),
                 index=False, header=True, sep=',')

    # Plot the mean frame with the sources overlayed in red
    image_plot(config, mean_frame, sources)
    stats_plot(config, sources)


if __name__ == '__main__':
    # Read the configuration file from the command line argument
    config_file = sys.argv[1]
    config = load_config(config_file)
    # Call the main function with the loaded configuration
    main(config)
