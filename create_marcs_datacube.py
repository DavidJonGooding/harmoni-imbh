# create_marcs_datacube.py is the script that generates the datacube with MARCS spectra.
# The script is called with a configuration file as an argument, which contains the parameters for the simulation.
# The script reads the configuration file, creates an empty datacube, applies MARCS spectra to the datacube,
# scales the magnitudes to match Vega, and saves the datacube to a FITS file.
# The script also splits the datacube into multiple parts if specified in the configuration file.
# The configuration file is a JSON file with the following parameters:
# {
#     "marcs_directory": "path/to/marcs/directory",
#     "fov": 1.0,
#     "ang_res": 0.01,
#     "sources": 100,
#     "x_start": -0.5,
#     "x_stop": 0.5,
#     "y_start": -0.5,
#     "y_stop": 0.5,
#     "band": "H",
#     "nbody_data": "path/to/nbody/data",
#     "spec_step": 0.26,
#     "quiet": false,
#     "plot": false,
#     "split_parts": 2,
#     "overlap": 10,
#     "output_directory": "path/to/output/directory"
# }
# The parameters are as follows:
# - marcs_directory: Path to the directory containing the MARCS spectra.
# - fov: Field of view in arcseconds.
# - ang_res: Angular resolution in arcseconds per pixel.
# - sources: Number of sources to apply spectra to in the field of view.
# - x_start, x_stop, y_start, y_stop: Coordinates of the field of view in arcseconds.
# - band: HARMONI band to simulate (e.g., "H", "K", "H+K").
# - nbody_data: Path to the file containing the pixel locations and line-of-sight velocities.
# - spec_step: Spectral step size in Angstroms.
# - quiet: Boolean indicating whether to print debug information.
# - plot: Boolean indicating whether to plot the spectra.
# - split_parts: Number of parts to split the datacube into.
# - overlap: Number of overlapping pixels between split parts.
# - output_directory: Path to the output directory to save the datacube and split parts.

# The script uses the following functions:
# - create_harmoni_datacube: Creates an empty datacube with the specified dimensions, wavelength range, and header information.
# - open_marcs_spectra: Opens a random MARCS spectrum file from the specified directory.
# - rebin_spectrum_bspline: Rebins a spectrum to a constant spectral resolution in terms of wavelength using a B-spline fit.
# - scale_vega: Scales a spectrum to match a target magnitude in J band using Vega as a reference.
# - apply_spectra_from_file_vega: Applies spectra from a file to the datacube, scales magnitudes to Vega, and adds them to the datacube.
# - add_stellar_spectrum: Adds a stellar spectrum to a specific pixel location in the datacube.
# - crop_spectrum: Crops a spectrum to a specific wavelength range.
# - harmoni_band: Returns the start and end wavelengths of a HARMONI band.
# - load_config: Loads a configuration file and returns the parameters as a dictionary.
# - split_fits_datacube: Splits a datacube into multiple parts with overlapping pixels.
# - marcs_condition: Finds the wavelength step size of MARCS in the middle of the HARMONI band.
# - main: Main function to create a datacube with MARCS spectra.

import sys
import numpy as np
from astropy.io import fits
import os
from os import path
import random
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline, make_interp_spline
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
import json

ppxf_dir = path.dirname(path.realpath(lib.__file__))


def create_harmoni_datacube(cube_shape, spec_step, angular_resolution, band_start):
    # Create an empty datacube with the specified dimensions, wavelength range, and header information
    # Create an empty datacube
    datacube = np.zeros(cube_shape)

    # Create a FITS header
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = -32

    # Datacube dimensions
    header['NAXIS'] = 3
    header['NAXIS1'] = cube_shape[2]
    header['NAXIS2'] = cube_shape[1]
    header['NAXIS3'] = cube_shape[0]

    # Wavelength axis information
    header['CRPIX3'] = 1
    header['CRVAL3'] = band_start
    header['CDELT3'] = spec_step
    header['CUNIT3'] = 'angstrom'
    header['CTYPE3'] = 'wavelength'

    # Spatial axis information
    header['CTYPE1'] = 'x'
    header['CTYPE2'] = 'y'
    header['CUNIT1'] = 'arcsec'
    header['CUNIT2'] = 'arcsec'
    header['CDELT1'] = angular_resolution
    header['CDELT2'] = angular_resolution

    # Other information
    header['BUNIT'] = 'erg/s/cm2/AA/arcsec2'
    band_middle = 16250
    marcs_resolution = 20000
    spec_res = band_middle / marcs_resolution
    header['SPECRES'] = spec_res

    return datacube, header


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
    return data


def rebin_spectrum_bspline(wavelengths, spectrum, hsim_lam):
    """
    Rebin a spectrum to a constant spectral resolution in terms of wavelength using a B-spline fit.

    Parameters:
    - wavelengths: 1D numpy array with the original wavelengths in Angstroms.
    - spectrum: 1D numpy array with the flux values of the original spectrum.
    - hsim_lam: 1D numpy array with the wavelengths of the rebinned spectrum.

    Returns:
    - new_spectrum: 1D numpy array with the rebinned spectrum.
    """
    # Create a B-spline representation of the spectrum
    # t, c, k = make_interp_spline(wavelengths, spectrum, k=3).tck
    # bspline = BSpline(t, c, k)

    # Evaluate the B-spline at the HSIM wavelengths
    # new_spectrum = bspline(hsim_lam)

    new_spectrum = np.interp(hsim_lam, wavelengths, spectrum)   # TODO - test in notebook, compare to bspline

    return new_spectrum


def scale_vega(wavelengths, spectrum, a, quiet, target_mag=23):
    """
    Scale a spectrum to match a target magnitude in J band using Vega as a reference.

    Parameters:
    - wavelengths: 1D numpy array with the wavelengths of the spectrum.
    - spectrum: 1D numpy array with the flux values of the spectrum.
    - a: Dictionary with the Vega spectrum data.
    - quiet: Boolean indicating whether to print debug information.
    - target_mag: Target magnitude in the H band (default is 23).

    Returns:
    - wavelengths: 1D numpy array with the wavelengths of the scaled spectrum.
    - spectrum_new: 1D numpy array with the flux values of the scaled spectrum.
    """

    # Define constants
    c = 299792.458  # speed of light in km/s
    velscale = c * np.log(wavelengths[1] / wavelengths[0])  # eq.(8) of Cappellari (2017)
    band = ['2MASS/H']  # ['Johnson-Cousins_I', 'SDSS/i', '2MASS/J', '2MASS/H', '2MASS/K']
    z = 0.00093  # group redshift of LMC

    # Ppxf log rebin function
    ln_flux, ln_lambda, vs = util.log_rebin(wavelengths, spectrum)

    # Calculate flux in defined band
    result = util.synthetic_photometry(ln_flux, np.exp(ln_lambda), band, redshift=z, quiet=1)
    # result = util.synthetic_photometry(spectrum, wavelengths, band, redshift=z, quiet=1)

    # Load Vega flux
    flux_vega = a["flux_vega"]

    # Rebin Vega spectrum to match input spectrum
    flux_vega_new, ln_lam_vega, velscale = util.log_rebin(a["lam"], flux_vega, velscale)

    # Calculate flux of Vega in defined band
    result_vega = util.synthetic_photometry(flux_vega_new, np.exp(ln_lam_vega), band, redshift=z, quiet=1)
    # result_vega = util.synthetic_photometry(flux_vega, a["lam"], band, redshift=z, quiet=1)

    # Calculate magnitude of input spectrum
    mag = -2.5 * np.log10(result.flux / result_vega.flux)

    # Calculate delta magnitude
    delta_m = target_mag - mag

    # Scale input spectrum to match target magnitude
    ln_flux_new = ln_flux * 10 ** (delta_m / -2.5)
    spectrum_new = spectrum * 10 ** (delta_m / -2.5)  # TODO: check this is correct

    if not quiet:
        # Calculate flux of scaled spectrum in defined band - just to check
        result_new = util.synthetic_photometry(ln_flux_new, np.exp(ln_lambda), band, redshift=z, quiet=1)
        # result_new = util.synthetic_photometry(spectrum_new, wavelengths, band, redshift=z, quiet=1)

        # Calculate magnitude of scaled spectrum - just to check
        mag_new = -2.5 * np.log10(result_new.flux / result_vega.flux)

        print('Target magnitude: ', target_mag)
        print('Magnitude of input spectrum: ', mag)
        print('Delta to Vega: ', delta_m)
        print('Magnitude of scaled spectrum: ', mag_new)

        print(spectrum_new)

        # plt.plot(ln_lambda, spectrum_new)
        # plt.show()
        # plt.plot(wavelengths, spectrum_new)
        # plt.show()

    return spectrum_new    # ln_lambda, spectrum_new


def apply_spectra_from_file_vega(datacube, header, p, n_body_file, spec_step, hsim_lam, quiet, plot=False):
    """
    Apply spectra from a file to the datacube, scale magnitudes to Vega, and add them to the datacube.

    Parameters:
    - datacube: 3D numpy array with the datacube to which the spectra will be added.
    - header: FITS header with the datacube's metadata.
    - p: Dictionary with the parameters of the simulation.
    - n_body_file: Path to the file containing the pixel locations and line-of-sight velocities.
    - spec_res: Spectral resolution in Angstroms.
    - plot: Boolean indicating whether to plot the spectra (default is False).

    Returns:
    - datacube: 3D numpy array with the updated datacube.
    - header: FITS header with the updated metadata.
    - nsources: Number of sources in the window.
    """

    # Load the parameters from the dictionary
    marcs_directory, fov, angular_resolution, sources, x_start, x_stop, y_start, y_stop, band = (p['marcs_directory'],
                                                                                           p['fov'], p['ang_res'],
                                                                                           p['sources'], p['x_start'],
                                                                                           p['x_stop'], p['y_start'],
                                                                                           p['y_stop'], p['band'])
    # Print out basic info
    print('Loading datacube with shape: ', datacube.shape)

    # Load all stars from the file
    all_stars = np.loadtxt(n_body_file, usecols=(3, 4, 5, 6, 7, 8, 9))

    print('Loading pixel coordinates from %s sources' % len(all_stars))

    # Filter stars within the field of view
    window_stars = np.asarray([(x, y, i, j, h, k, v) for x, y, i, j, h, k, v in all_stars
                               if x_start <= x <= x_stop and y_start <= y <= y_stop])

    # Get the magnitudes and line-of-sight velocities
    i_mag, j_mag, h_mag, k_mag, losv = np.transpose(window_stars)[2:]

    # Get the pixel coordinates
    # window_pixels = np.transpose(window_stars)[:2]
    window_pixels = window_stars[:,:2]
    print('Loading pixel coordinates from %s sources out of %s' % (len(window_stars), len(all_stars)))

    # limit the number of sources to use if sources is less than the number of stars in the window
    if sources < len(window_stars):
        j = int(sources)
        print('Applying spectra to first %s sources in window' % j)
    else:
        j = len(window_stars)
        print('Applying spectra to all %s sources in window' % j)

    # Angular resolution scaling factor (sf) for pixel locations
    sf = 1 / angular_resolution

    # Load the spectrum from the MARCS library
    raw_spectrum = open_marcs_spectra(marcs_directory, quiet)
    print('Template spectrum of shape:', raw_spectrum.shape)
    # TODO: move into the loop when using different spectra for each star

    # Get MARCS wavelength data from standard file
    marcs_wavelengths = np.loadtxt(marcs_directory + '/' + 'flx_wavelengths.vac', skiprows=0, usecols=0)

    # Load Vega spectrum
    a = np.load(ppxf_dir + '/sps_models/spectra_sun_vega.npz')  # Spectrum in cgs/A

    # Iterate over each pixel location and apply the spectrum
    for i in range(len(window_stars[:j])):

        # Debug printing and pixel coordinates
        print('Applying spectrum to star %s' % i)
        pixel_x = int((window_pixels[i][0] - x_start) * sf)
        pixel_y = int((window_pixels[i][1] - y_start) * sf)
        if not quiet:
            print('Pixel coordinates were: %s' % window_pixels[i])
            print('Pixel coordinates are: x %s, y %s' % (pixel_x, pixel_y))

        # Apply redshift of star to the raw MARCS wavelengths
        if not quiet:
            print('----- Redshifting -----')
        wavelengths_z = redshift_star(marcs_wavelengths, losv[i])

        # Debug printing
        if not quiet:
            print('Original wavelengths: ', marcs_wavelengths)

        # Adjust template spectrum to match Vega - wavelengths stay the same
        if not quiet:
            print('----- Scaling to Vega -----')
        spectrum_s = scale_vega(wavelengths_z, raw_spectrum, a, quiet, target_mag=h_mag[i])
        # wavelengths_s, spectrum_s = scale_vega(wavelengths_r, spectrum_r, a, quiet, target_mag=j_mag[i])

        if not quiet:
            print('----- Interpolating -----')
            print('Spectrum shape:', np.shape(spectrum_s))

        # Rebin spectrum with bspline interpolation to get HSIM expected resolution - using hsim_lam
        # This is where the log spectrum is scaled to match the hsim linear wavelength resolution
        # wavelengths, spectrum = rebin_spectrum_bspline(np.exp(wavelengths_z), spectrum_z, spec_step)
        spectrum_r = rebin_spectrum_bspline(wavelengths_z, spectrum_s, hsim_lam)

        if plot:
            plt.plot(spectrum_r, 'k.')
            plt.xlabel('Pixel')
            plt.ylabel('Flux [erg/s/cm^2/Å]')
            plt.title('MARCS Spectrum')
            plt.show()

        # Crop spectra to chosen HARMONI band
        #if not quiet:
        #    print('----- Cropping spectrum -----')
        #    print('Spectrum shape:', np.shape(spectrum_r))
        #    print('Target length', len(hsim_lam))

        # OLD - used to crop spectrum here, now done in rebin_spectrum_bspline()
        # spectrum_c, crval3 = crop_spectrum(spectrum_s, wavelengths_s, quiet, band)
        # print(spectrum_c, crval3)

        # Add the spectrum to the pixel location
        if not quiet:
            print('----- Adding spectrum to datacube -----')
        # datacube, header = add_stellar_spectrum(datacube, header, pixel_x, pixel_y,
        #                                        spectrum_c, angular_resolution, quiet)
        datacube = add_stellar_spectrum(datacube, pixel_x, pixel_y, spectrum_r,
                                                angular_resolution, quiet)

    # print info about output
    print("Datacube created with shape:", datacube.shape)
    print("Number of sources: ", j)

    # Change CRVAL3 to match the first wavelength in the datacube
    # header['CRVAL3'] = crval3

    return datacube, header, len(window_pixels)


def redshift_star(wavelengths, v_star, z_LMC=0.00093, c=299792.458):
    """
    Apply redshift to a stellar spectrum considering both the LMC's group redshift
    and the star's line-of-sight velocity.

    Parameters:
    - wavelengths: A 1D array or list containing the wavelengths of the spectrum.
    - spectrum: A 1D array or list containing the corresponding flux values.
    - v_star: Line-of-sight velocity of the star relative to the LMC in km/s.
    - z_LMC: Redshift of the LMC (default is 0.00093).
    - c: Speed of light in km/s (default is 299792.458).

    Returns:
    - redshifted_spectrum: A 2D array or list with the same structure as the input spectrum,
                           where the wavelengths have been redshifted.
    """

    # Calculate the redshift of the star from its velocity - RELATIVISTIC
    # z_star = np.sqrt((1 + v_star / c) / (1 - v_star / c)) - 1

    # Calculate the total redshift combining the LMC's and the star's redshifts
    z_total = (1 + z_LMC) * (1 + v_star / c) - 1

    # Apply the total redshift to the spectrum
    wavelengths_rest = np.array(wavelengths)
    wavelengths_observed = wavelengths_rest * (1 + z_total)

    return wavelengths_observed


def add_stellar_spectrum(datacube, pixel_x, pixel_y, spectrum, angular_resolution, quiet):
    # Validate pixel coordinates
    if pixel_x < 0 or pixel_x >= datacube.shape[2]:
        raise ValueError(f"Invalid x-coordinate. Must be within the range 0 to {datacube.shape[2] - 1}.")
    if pixel_y < 0 or pixel_y >= datacube.shape[1]:
        raise ValueError(f"Invalid y-coordinate. Must be within the range 0 to {datacube.shape[1] - 1}.")

    # Validate spectrum shape
    if not quiet:
        if spectrum.shape[0] != datacube.shape[0]:
            print("WARNING - Spectrum shape might not match the datacube.")

    # convert ergs/cm2/s/Å to erg/s/cm2/AA/arcsec2
    spectrum = spectrum / (angular_resolution ** 2)

    # find the middle of the band, and add the spectrum to the datacube from this point
    # useful only when using a subsection of the band
    spectrum_middle = int(len(spectrum) / 2)
    z = int(datacube.shape[0])

    # Debug printing
    if not quiet:
        print('Adding spectrum of shape:', np.shape(spectrum))
        print('Datacube shape:', np.shape(datacube))
        print('Pixel y:', pixel_y)
        print('Pixel x:', pixel_x)

    # Add the spectrum to the datacube
    datacube[:, pixel_y, pixel_x] += spectrum   # [int(spectrum_middle - (z / 2)):int(spectrum_middle + (z / 2))]

    return datacube


def crop_spectrum(spectrum, wavelengths, quiet, band='H'):

    # Get band start and end wavelengths in Angstroms
    band_start, band_end = harmoni_band(band)

    if not quiet:
        print('Cropping the spectrum from %s to %s Angstroms' % (band_start, band_end))
        print('Spectrum originally of shape:', np.shape(spectrum))
        print('Template wavelengths')
        print('Shape', np.shape(wavelengths), 'min', np.min(wavelengths), 'max', np.max(wavelengths))

    # Find extent of band in MARCS
    marcs_band_blue = len(wavelengths[wavelengths < band_start])
    marcs_band_red = len(wavelengths[wavelengths < band_end])

    # limit to HARMONI band wavelengths
    spectrum = spectrum[marcs_band_blue:marcs_band_red]

    if not quiet:
        print('Cropped to:')
        print(np.shape(spectrum))
        print('Using limit indices of %s and %s' % (marcs_band_blue, marcs_band_red))
    return spectrum, wavelengths[-1]


def harmoni_band(band):
    # Get band start and end wavelengths in Angstroms
    if band == 'Iz+J' or band == 'LR1':
        band_start = 8110
        band_end = 13690
    elif band == 'H+K' or band == 'LR2':
        band_start = 14500
        band_end = 24500
    elif band == 'Iz' or band == 'MR1':
        band_start = 8300
        band_end = 10500
    elif band == 'J' or band == 'MR2':
        band_start = 10460
        band_end = 13240
    elif band == 'H' or band == 'MR3':
        band_start = 14350
        band_end = 18150
    elif band == 'K' or band == 'MR4':
        band_start = 19510
        band_end = 25000
    elif band == 'z-high' or band == 'HR1':
        band_start = 8270
        band_end = 9030
    elif band == 'H-high' or band == 'HR2':
        band_start = 15380
        band_end = 16780
    elif band == 'K-short' or band == 'HR3':
        band_start = 20170
        band_end = 22010
    elif band == 'K-long' or band == 'HR4':
        band_start = 21990
        band_end = 23990
    else:
        raise ValueError(f"Invalid band: {band}")

    return band_start, band_end


def load_config(config_file):
    """
    Load a configuration file and return the parameters as a dictionary, handling inline comments.
    """
    # Create an empty dictionary to store the parameters
    params = {}

    # Open the configuration file and read the parameters
    with open(config_file) as f:
        for line in f:
            # Ignore empty lines
            if not line.strip():
                continue

            # Remove comments from the line (if present)
            line, *comment = line.split('#', 1)
            line = line.strip()  # Remove leading/trailing whitespace

            # Skip lines that became empty after removing comments
            if not line:
                continue

            # Split the line into a key and value
            try:
                key, value = line.split('=', 1)
            except ValueError:
                # Handle lines that don't have a '=' character, if any
                print(f"Skipping malformed line: {line}")
                continue

            # Strip whitespace from the key and value
            key = key.strip()
            value = value.strip()

            # Attempt to convert the value to an int, then float, and finally check for booleans
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value in ['True', 'true', 'yes']:
                        value = True
                    elif value in ['False', 'false', 'no']:
                        value = False
                    # If the value is not a recognized boolean, leave it as a string

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


def split_fits_datacube(datacube, header, hsim_lam, file_name, split_parts, overlap, output_directory):
    # Define split points depending on the number of parts and the overlap
    naxis3 = header['NAXIS3']
    split_points = [0] + [naxis3 // split_parts * i for i in range(1, split_parts + 1)]
    print('Split points:', split_points)

    # create folder called 'cubeparts' in output_directory, check if it exists first
    parts_directory = path.join(output_directory, 'cubeparts')
    if not path.exists(parts_directory):
        os.mkdir(parts_directory)
        print('Directory created for split datacubes')
    else:
        print('Directory already exists, overwriting files...')

    # Create the split datacubes
    for i in range(split_parts):
        # Calculate the split points
        start = max(split_points[i], 0)
        end = min(split_points[i + 1] + overlap, naxis3)
        print(f'Splitting datacube from {start} to {end} for part {i + 1} of {split_parts}')

        # Create the split datacube
        datacube_split = datacube[start:end, :, :]

        # Update the header
        header_split = header.copy()
        header_split['NAXIS3'] = datacube_split.shape[0]
        header_split['CRPIX3'] = - split_points[i + 1] + 1
        # header_split['CRVAL3'] = header['CRVAL3'] + start * header['CDELT3']  # stays the same

        # Save the split datacube to a FITS file
        part_file_name = f'{file_name}_part{i + 1}of{split_parts}.fits'
        fits.writeto(path.join(parts_directory, part_file_name), datacube_split, header_split, overwrite=True)

        print(f'Datacube saved as {part_file_name}')


def marcs_condition(marcs_directory, harmoni_red, harmoni_blue):
    """
    Find the wavelength step size of MARCS in the middle of the HARMONI band, this defines the best spectral resolution
    :param marcs_directory:
    :param harmoni_red:
    :param harmoni_blue:
    :return:
    """
    # Get MARCS wavelength data from standard file (A)
    marcs_wavelengths = np.loadtxt(marcs_directory + '/flx_wavelengths.vac', skiprows=0, usecols=0)

    # Find extent of HARMONI band in MARCS (pixel)
    harmoni_blue = float(harmoni_blue)
    harmoni_red = float(harmoni_red)
    marcs_harmoni_blue = len(marcs_wavelengths[marcs_wavelengths < harmoni_blue])
    marcs_harmoni_red = len(marcs_wavelengths[marcs_wavelengths < harmoni_red])

    # limit to HARMONI wavelengths (A)
    marcs_wavelengths = marcs_wavelengths[marcs_harmoni_blue:marcs_harmoni_red]

    # getting the wavelength step size

    # find middle of HARMONI band (A)
    harmoni_middle = (harmoni_red - harmoni_blue) / 2 + harmoni_blue

    # find the closest wavelength in MARCS to middle of HARMONI band (A)
    marcs_middle = min(marcs_wavelengths, key=lambda x: abs(x - harmoni_middle))

    # find index of closest wavelength in MARCS to middle of HARMONI band (pixel)
    marcs_middle_index = np.where(marcs_wavelengths == marcs_middle)[0][0]

    # find wavelength step size in middle (A)
    marcs_wavelength_step = marcs_wavelengths[marcs_middle_index + 1] - marcs_wavelengths[marcs_middle_index]

    # round to 2 decimal places
    spec_res = round(marcs_wavelength_step, 2)

    return spec_res, marcs_middle


def main(config, output_dir):
    # Main function to create a datacube with MARCS spectra

    # Load the configuration file
    p = config  # p is a dictionary of parameters

    # Determine if quiet or not (for debug)
    quiet = p['quiet']

    # IMBH present or not
    if p['imbh_present']:
        nbody_file = p['nbody_data_imbh']
        imbh_label = 'imbh'
    else:
        nbody_file = p['nbody_data']
        imbh_label = 'noimbh'

    # Get wavelength info of the template library
    # spec_res, lam_centre = marcs_condition(p['marcs_directory'], p['harmoni_red'], p['harmoni_blue'])

    # Get the spectral step size from the config file
    spec_step = p['spec_step']     # Config file specifies the step size in Angstroms as 0.26 A
    # OLD -- Calculated by: del_lambda = lam_centre / R  = 16250/7104 = 2.287A, then oversampled by 4 to 0.57515A
    # check the pixel step in wavelength for HARMONI - Consider 2.35 factor for FWHM
    # HSIM output wavelength step: 0.000103974713349815 micron = 1.039747 A, therefore, 4x oversampling = 0.26 A
    spec_step = (0.000103974713349815 * 10000) / 4

    # Get the band start and end wavelengths in Angstroms
    band_start, band_end = harmoni_band(p['band'])

    # Calculate the number of pixels in the band
    spec_pixels = int((band_end - band_start) / spec_step)

    # Create the wavelength axis for the datacube
    hsim_lam = np.linspace(band_start, band_end, spec_pixels)

    if not quiet:
        print('Spectral pixel size: ', spec_step)
        print('Number of spectral pixels: ', spec_pixels)

    # Define the shape of the datacube
    fov_x = p['x_stop'] - p['x_start']  # FoV in arcsec in x direction
    fov_y = p['y_stop'] - p['y_start']  # FoV in arcsec in y direction
    cube_shape = (int(spec_pixels), int(fov_x / p['ang_res']), int(fov_y / p['ang_res']))   # ang_res is 10mas / 5
    print('Blank datacube created with shape:', cube_shape)

    # Create an empty datacube with the specified dimensions, wavelength range, and header information
    datacube, header = create_harmoni_datacube(cube_shape, spec_step, p['ang_res'], band_start)

    # Apply spectra from a file, scale magnitudes to Vega, and add them to the datacube
    datacube, header, nsources = apply_spectra_from_file_vega(datacube, header, p, nbody_file, spec_step, hsim_lam,
                                                              quiet, plot=p['plot'])
    print('Datacube generated')

    # Save the datacube to a FITS file
    print('Saving datacube')

    datacube_name = 'rawcube_%sstars_%sfov_%s' % (nsources, fov_x, imbh_label)

    fits.writeto(output_dir + '/' + datacube_name + '.fits', datacube, header, overwrite=True)
    print('Datacube saved as', datacube_name)

    # Split datacube into x number of parts
    if p['split_parts'] > 1:
        print('Splitting datacube into %s parts' % p['split_parts'])
        split_fits_datacube(datacube, header, hsim_lam, datacube_name, p['split_parts'], p['overlap'], output_dir)


if __name__ == '__main__':
    # Check if a configuration file was provided
    if len(sys.argv) < 2:
        print("Usage: python3 create_marcs_datacube.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    # config = load_config(config_file) # OLD METHOD

    config_ = load_json_config(config_file)

    global_params = config_.get('global', {})
    config = config_.get('create_marcs_datacube', {})

    # Load configuration
    # with open(config_file, 'r', encoding='utf-8') as f:
    #     config = json.load(f)['create_marcs_datacube']
    #     # global_config = json.load(f)['global']

    # output_dir = config['output_directory']
    # output_dir = global_config['output_directory']
    output_dir = global_params['output_directory']

    if not path.exists(output_dir):
        os.mkdir(output_dir)

    main(config, output_dir)
