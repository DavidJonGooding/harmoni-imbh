# A programme which uses pPXF to determine the line-of-sight velocity (losv) of stars from PampelMuse spectra

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
import ppxf.ppxf_util as util
import ppxf as ppxf_package
from ppxf.ppxf import ppxf, rebin
from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rcParams
import json
from os import path


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


def fits_to_array(folder_path):
    """
    Converts a folder of FITS files into a single NumPy array with wavelengths as the first column
    :param folder_path:
    :return: the NumPy array
    """
    data_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.fits') and 'id' in filename:
            file_path = os.path.join(folder_path, filename)

            with fits.open(file_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header

                # Check if data is 1D and if it needs byte order conversion
                if data.ndim == 1:
                    # Check if data is big-endian and convert it to little-endian if needed
                    if data.dtype.byteorder == '>':
                        data = data.byteswap().newbyteorder()

                    data_list.append(data)

    wavelengths = np.arange(header['CRVAL1'], header['CRVAL1'] + header['CDELT1'] * header['NAXIS1'], header['CDELT1'])

    # Check if all data arrays have the same length
    if not all(len(data) == len(data_list[0]) for data in data_list):
        print("Data arrays have different lengths.")
        return

    # Combine all data arrays into a single NumPy array
    numpy_array = np.column_stack(data_list)

    # Check if the column length in the array matches the number of wavelengths
    if numpy_array.shape[1] != len(wavelengths):
        print("Data array column length does not match the number of wavelengths.")
        print("Data array shape:", numpy_array.shape)
        print("Wavelengths length:", len(wavelengths))

    # QUICK FIX - make wavelengths the same length as the data
    wavelengths = wavelengths[0:len(data_list[0])]

    # Add wavelengths to array as first column
    numpy_array = np.column_stack((wavelengths, numpy_array))

    # Print some statistics about the array
    print("Data shape:", numpy_array.shape)
    print("Data type:", numpy_array.dtype)
    print("Data size:", numpy_array.nbytes, "bytes")

    return numpy_array


def open_marcs_spectra(directory):
    # Get the list of files in the directory
    file_list = os.listdir(directory)

    # Filter out directories and non-ASC files from the list
    file_list = [f for f in file_list if os.path.isfile(os.path.join(directory, f)) and f.endswith('.flx')]
    print(file_list)

    if len(file_list) == 0:
        print("No FLX files found in the directory.")
        return

    # Select a random file from the list
    random_file = random.choice(file_list)

    # Get the full file path
    file_path = os.path.join(directory, random_file)

    # Open the ASC file
    try:
        with open(file_path) as f:
            # Access the data from the ASC file
            data = np.loadtxt(f, skiprows=0, usecols=0)
            # header = f.readline()
            print("Data shape:", data.shape)
    except IOError:
        print("Error opening the FLX file.")
    return data


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

def load_json_config(config_file):
    """
    Load a JSON configuration file and return the parameters as a dictionary.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


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


def main(config, output_dir):
    dir = '/Users/gooding/Desktop/IMBH_PPM/imbh/'
    filename = '6093_sources_0l_imbh_reduced_SNR_scaled.prm.fits'
    # Load the spectra
    spectra = load_spectra(os.path.join(dir, filename))
    print('Spectra loaded successfully.')

    plt.plot(spectra[0])
    plt.show()


if __name__ == '__main__':
    # Check if a configuration file was provided
    if len(sys.argv) < 2:
        print('Usage: python auto_hsim.py <config_file>')
        exit()

    config_file = sys.argv[1]

    config_ = load_json_config(config_file)

    global_params = config_.get('global', {})
    config = config_.get('losv', {})

    output_dir = global_params['output_directory']

    if not path.exists(output_dir):
        os.mkdir(output_dir)

    main(config, output_dir)
#%%
