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
from ppxf.ppxf import ppxf, robust_sigma, rebin
from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rcParams
import json
from os import path
import random
from scipy import ndimage
from astropy.stats import biweight_scale
import math
import losv_fitting

np.set_printoptions(threshold=sys.maxsize)


def fits_to_array(folder_path):
    """
    Converts a folder of FITS files into a single NumPy array with wavelengths as the first column
    :param folder_path:
    :return: the NumPy array
    """

    data_list = []
    coords_list = []
    id_list = []
    mag_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.fits') and 'id' in filename:
            file_path = os.path.join(folder_path, filename)
            print(file_path)

            with fits.open(file_path) as hdul:
                ignore_missing_simple=True
                data = hdul[0].data
                header = hdul[0].header

                # Check if data is 1D and if it needs byte order conversion
                if data.ndim == 1:
                    # Check if data is big-endian and convert it to little-endian if needed
                    if data.dtype.byteorder == '>':
                        data = data.byteswap().newbyteorder()

                    # Append the data to the list
                    data_list.append(data)

                    # Get the coordinates of the star
                    x_coord = header['HIERARCH SPECTRUM XCUBE']
                    y_coord = header['HIERARCH SPECTRUM YCUBE']
                    coords_list.append((x_coord, y_coord))

                    # Get the star ID
                    star_id = header['HIERARCH STAR ID']
                    id_list.append(star_id)

                    # Get the star magnitude
                    star_mag = header['HIERARCH STAR MAG']
                    mag_list.append(star_mag)

    wavelengths = np.arange(header['CRVAL1'], header['CRVAL1'] + header['CDELT1'] * header['NAXIS1'], header['CDELT1'])

    # Check if all data arrays have the same length
    if not all(len(data) == len(data_list[0]) for data in data_list):
        print("Data arrays have different lengths.")
        return

    # Combine all data arrays into a single NumPy array
    numpy_array = np.column_stack(data_list)

    # Check if the column length in the array matches the number of wavelengths
    if numpy_array.shape[0] != len(wavelengths):
        print("Data array column length does not match the number of wavelengths.")
        print("Data array shape:", numpy_array.shape)
        print("Wavelengths length:", len(wavelengths))

    # QUICK FIX - make wavelengths the same length as the data
    # wavelengths = wavelengths[0:len(data_list[0])]

    # Add wavelengths to array as first column
    numpy_array = np.column_stack((wavelengths, numpy_array))

    # Print some statistics about the array
    print("Data shape:", numpy_array.shape)
    print("Data type:", numpy_array.dtype)
    print("Data size:", numpy_array.nbytes, "bytes")

    # Rescale coordinate to be 0-indexed and in arcseconds
    coords = (np.asarray(coords_list)/100)-1

    # Order everything in increasing magnitude
    #numpy_array = numpy_array[:, np.argsort(mag_list)]
    #coords = coords[np.argsort(mag_list)]
    #id_list = [x for _, x in sorted(zip(mag_list, id_list))]
    #mag_list = sorted(mag_list)

    return numpy_array, coords, id_list, mag_list


def open_marcs_spectra(directory, quiet):
    # Get the list of files in the directory
    file_list = os.listdir(directory)

    # Filter out directories and non-FLX files from the list
    file_list = [f for f in file_list if os.path.isfile(os.path.join(directory, f)) and f.endswith('.flx')]
    if not quiet:
        print('Template spectra file list:', file_list)

    if len(file_list) == 0:
        print("No FLX files found in the directory.")
        return

    # Select a random file from the list
    random_file = random.choice(file_list)

    # Get the full file path
    file_path = os.path.join(directory, random_file)

    # Open the .FLX file
    try:
        with open(file_path) as f:
            # Access the data from the .FLX file
            data = np.loadtxt(f, skiprows=0, usecols=0)
            # header = f.readline()
            print("Data shape:", data.shape)
    except IOError:
        print("Error opening the FLX file.")

    # Open wavelength data
    # Get MARCS wavelength data from standard file (A)
    marcs_wavelengths = np.loadtxt(directory + '/flx_wavelengths.vac', skiprows=0, usecols=0)

    # Define H band limits (A)
    h_blue = 14000#350
    h_red = 18500#200

    # Select only the H band region of the MARCS spectra
    w = (marcs_wavelengths > h_blue) & (marcs_wavelengths < h_red)
    templ = data[w]
    wavelengths = marcs_wavelengths[w]

    return data, marcs_wavelengths #templ, wavelengths


def clip_outliers(galaxy, bestfit, mask):
    """
    Repeat the fit after clipping bins deviants more than 3*sigma in relative
    error until the bad bins don't change any more. This function uses eq.(34)
    of Cappellari (2023) https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C
    """
    while True:
        scale = galaxy[mask] @ bestfit[mask]/np.sum(bestfit[mask]**2)
        resid = scale*bestfit[mask] - galaxy[mask]
        err = robust_sigma(resid, zero=1)
        ok_old = mask
        mask = np.abs(bestfit - galaxy) < 3*err
        if np.array_equal(mask, ok_old):
            break

    return mask


def ppxf_fit_and_clean(templates, galaxy, velscale, start, mask0, lam, lam_temp, plot=True, quiet=False):
    """
    This is a simple pPXF wrapper. It performs two pPXF fits: the first one
    serves to estimate the scatter in the spectrum and identify the outlier
    pixels. The second fit uses the mask obtained from the first fit to exclude
    the outliers. The general approach used in this function is described in
    Sec.6.5 of Cappellari (2023) https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C
    """
    #pp = ppxf(template, star, t_noise[:, k], velscale, start,
    #          goodpixels=goodpixel, plot=False, moments=2,
    #          degree=4, vsyst=0)

    mask = mask0.copy()
    print(lam_temp.shape, templates.shape)
    pp = ppxf(templates, galaxy, np.ones_like(galaxy), velscale, start,
              moments=2, degree=20, mdegree=-1, fixed=[0, 1], lam=lam, lam_temp=lam_temp,
              mask=mask, quiet=quiet)

    if plot:
        plt.figure(figsize=(20, 3))
        plt.subplot(121)
        pp.plot()
        plt.title("Initial pPXF fit before outliers removal")

    mask = clip_outliers(galaxy, pp.bestfit, mask)

    # Add clipped pixels to the original masked emission lines regions and repeat the fit
    mask &= mask0
    pp = ppxf(templates, galaxy, np.ones_like(galaxy), velscale, start,
              moments=2, degree=20, mdegree=-1, fixed=[0, 1], lam=lam, lam_temp=lam_temp,
              mask=mask, quiet=quiet)

    optimal_template = templates.reshape(templates.shape[0], -1) @ pp.weights

    resid = (pp.galaxy - pp.bestfit)[pp.goodpixels]
    sn = np.nanmedian(pp.galaxy[pp.goodpixels])/robust_sigma(resid)

    if plot:
        plt.subplot(122)
        pp.plot()

    return pp, optimal_template, sn, pp.chi2


def ppxf_stars(shape1, targets, template, coords, id_list, velscale, t_noise, folder_path, lam, lam_temp, mag_list,
               plot=True):
    # set up arrays for output parameters
    # velocities = np.empty(all_spectra.shape[1]-1)
    sigma = np.empty(shape1-1)
    h3 = np.empty(shape1-1)
    h4 = np.empty(shape1-1)
    velocities = []
    snr = []
    snr_ppxf = []
    chi2_ppxf = []

    # Dubugging printing
    print('Shape1:', shape1)
    print('Targets:', targets.shape)
    print('Template:', template.shape)
    print('Coords:', coords.shape)
    print('ID List:', len(id_list))
    print('Velscale:', velscale)
    print('T_noise:', t_noise.shape)
    print('Folder Path:', folder_path)
    print('Lam:', lam.shape)
    print('Lam Temp:', lam_temp.shape)

    plot_path = f'%s/ppxf_plots' % folder_path

    redshift = 0.00093               # Initial redshift estimate of the galaxy

    # Initial estimate of the galaxy velocity in km/s
    c = 299792.458
    vel = c*np.log(1 + redshift)   # eq.(8) of Cappellari (2017, MNRAS)
    # gives 279, LMC estimate based on redshift of 0.00093 and literature stating 287 km s-1
    start = [vel, 0.19]  # (km/s), starting guess for [V, sigma] -- start = [velStart, sigmaStart]

    goodpixel = np.arange(200, 3500)

    # make initial mask based on goodpixel
    mask0 = np.zeros_like(targets[:, 0], dtype=bool)
    mask0[goodpixel] = True

    # looping ppxf for each star
    for k in range(0, shape1-1):
        print("Star %s / %s" % (k, shape1))
        if math.isnan(t_noise[0][k]) is True:
            continue

        star = targets[:, k]#ndimage.gaussian_filter1d(targets[:, k], 4)

        #pp = ppxf(template, star, t_noise[:, k], velscale, start,
        #          goodpixels=goodpixel, plot=False, moments=2,
        #          degree=4, vsyst=0)

        #
        pp, snr_star = losv_fitting.get_losv(star, lam, template, lam_temp)
        #pp, optimal_template, sn, chi2 = ppxf_fit_and_clean(template, star, velscale, start, mask0, lam=lam, lam_temp=lam_temp, plot=False)

        velocities = np.append(velocities, pp.sol[0])
        sigma[k-1] = pp.sol[1]

        # SNR calc
        #residuals = star[goodpixel] - pp.bestfit[goodpixel]
        #median_flux = np.median(star[goodpixel])
        #biweight_sigma = biweight_scale(residuals)
        #snr_star = median_flux / biweight_sigma
        snr = np.append(snr, snr_star)
        snr_ppxf = np.append(snr_ppxf, snr_star)
        chi2_ppxf = np.append(chi2_ppxf, pp.chi2)

        #print('SNR:', snr_star, sn)

        if plot and snr_star > 5:
            # plot and print results
            pp.plot()
            plt.title(f'Star {id_list[k]} - velocity = {pp.sol[0]:.2f} km/s, snr = {snr_star:.2f}')
            # add coordinates as text to plot
            plt.text(0.6, 0.95, f'x = {coords[k][0]:.4f}, y = {coords[k][1]:.4f}', transform=plt.gca().transAxes)
            # save plot and create directory
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            plt.savefig(f'%s/mag{mag_list[k]}_star_{id_list[k]}.png' % plot_path)
            # clear plot for next iteration
            plt.clf()

    return velocities, sigma, h3, h4, snr, snr_ppxf, chi2_ppxf


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

    print("Running HARMONI LOSV routine with pPXF...")

    # Load the configuration parameters
    folder_path = config['spectra_path']
    template_path = config['template_path']
    quiet = config['quiet']
    plot = config['plot']

    if not quiet:
        print('Configuration parameters:')
        for key, value in config.items():
            print(key, '=', value)

    # Load all spectra into an array
    # folder_path = '/Users/gooding/Desktop/IMBH_PPM/R136_5/spectra/'
    all_spectra, coords, id_list, mag_list = fits_to_array(folder_path)

    # Open MARCS spectra as a template, and wavelength data
    # template_path = '/Users/gooding/Desktop/IMBH/templates/MARCS_library/'
    templ, lam_templ = open_marcs_spectra(template_path, quiet)

    # pPXF preparation

    lamRange1 = np.transpose([np.min(all_spectra[:, 0]), np.max(all_spectra[:, 0])])

    # define empty arrays for targets and noise
    targets = np.empty((all_spectra.shape[0], all_spectra.shape[1]))
    t_noise = np.empty((all_spectra.shape[0], all_spectra.shape[1]))

    shape1 = all_spectra.shape[1]

    # loop through all stars - log rebinning and normalising spectra
    #for i in range(0, shape1-1):  # edited : 8/8/24
    #    gal_lin = np.transpose(all_spectra[:, i+1])
    #    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_lin.ravel())
    #    galaxy = galaxy/np.median(galaxy)       # Normalize spectrum to avoid numerical issues
    #    noise = galaxy*0 + 0.0049               # Assume constant noise per pixel here - new value needed?
    #
    #    targets[:, i] = galaxy   # store galaxy spectra in targets array
    #    t_noise[:, i] = noise    # store noise in t_noise array
    #
    #lam = np.exp(logLam1)

    # log_rebin the template spectra and normalize
    #template, ln_lambda = util.log_rebin(lam_templ, templ, velscale=velscale)[:2]
    #template /= np.median(template[template > 0])  # Normalizes template
    #lam_templ = np.exp(ln_lambda)
    template = templ
    velscale = 1
    lam = all_spectra[:, 0]

    targets = all_spectra[:, 1:]

    # PPXF
    velocities, sigma, h3, h4, snr, sn_pp, chi2_pp = ppxf_stars(
        shape1, targets, template, coords, id_list, velscale, t_noise, folder_path, lam, lam_templ, mag_list, plot=plot)

    # save results to file
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    results = pd.DataFrame(
        {'id': id_list, 'x': x_coords, 'y': y_coords, 'velocity': velocities, 'sigma': sigma,
         'snr': snr, 'sn_pp': sn_pp, 'mag': mag_list, 'chi2_pp': chi2_pp})

    # Results with SNR >3
    results_bright = results[results['snr'] > 5]
    results.to_csv(f'%s/ppxf_results_test.csv' % folder_path, index=False)
    results_bright.to_csv(f'%s/ppxf_results_SNR5test.csv' % folder_path, index=False)
    print("Results saved to %s/ppxf_results.csv" % folder_path)

    print("HARMONI LOSV routine complete.")


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
