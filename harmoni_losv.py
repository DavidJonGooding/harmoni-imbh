# A programme which uses pPXF to determine the line-of-sight velocity (losv) of stars from PampelMuse spectra

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
import ppxf.ppxf_util as util
import ppxf.ppxf as ppxf
from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rcParams


# Set plotting parameters
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['font.size'] = 12
rcParams['text.usetex'] = True
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.minor.width'] = 1.5
rcParams['ytick.minor.width'] = 1.5


# Define a function to load spectra from PampelMuse prm file
def load_spectra(filename):
    """
    Load the spectra from a PampelMuse prm file.
    """
    # Open the prm file
    with fits.open(filename) as hdu:
        # Extract the spectra
        spectra = hdu[2].data
    # THIS IS INCOMPLETE
    # Return the spectra
    return spectra


# Define a function to load the configuration parameters from a file
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

    # Return the parameters
    return params

# Define a function to plot the image with detected sources
def image_plot(mean_frame, sources):
    """
    Plot the mean frame with detected sources overlaid in red circles.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(mean_frame, cmap='Greys_r', origin='lower', norm=LogNorm())
    plt.scatter(sources['xcentroid'], sources['ycentroid'], marker='o', s=100, facecolors='none', edgecolors='r')
    plt.xlim(0, mean_frame.shape[1])
    plt.ylim(0, mean_frame.shape[0])
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.tight_layout()
    plt.show()

# Define a function to plot the spectrum
def spectrum_plot(wavelength, flux, error, redshift, title, output_file):
    """
    Plot the spectrum.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(wavelength, flux, color='k', lw=1)
    plt.plot(wavelength, error, color='r', lw=1)
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux')
    plt.title(title)
    plt.xlim(wavelength[0], wavelength[-1])
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Define a function to open PampelMuse data
def open_data(filename):
    """
    Open a PampelMuse data file and return the wavelength, flux and error.
    """
    # Open the data file
    with fits.open(filename) as hdu:
        # Extract the wavelength, flux and error
        wavelength = hdu[1].data
        flux = hdu[2].data
        error = hdu[3].data

    # Return the wavelength, flux and error
    return wavelength, flux, error



dir = '/Users/gooding/Desktop/IMBH_PPM/imbh/'
filename = '6093_sources_0l_imbh_reduced_SNR_scaled.prm.fits'
# Load the spectra
spectra = load_spectra(os.path.join(dir, filename))
print('Spectra loaded successfully.')

plt.plot(spectra[0])
plt.show()

#%%

#%%
