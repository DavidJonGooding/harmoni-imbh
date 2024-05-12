import sys

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.constants import c
import os
from os import path
import random
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline, make_interp_spline
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

ppxf_dir = path.dirname(path.realpath(lib.__file__))


def create_harmoni_datacube(cube_shape, wavelength_center, spec_resolution, angular_resolution):
    # Create an empty datacube with the specified dimensions, wavelength range, and header information
    # Create an empty datacube
    datacube = np.zeros(cube_shape)

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
    header['CRVAL3'] = wavelength_center
    header['CDELT3'] = spec_resolution
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
    header['SPECRES'] = spec_resolution

    return datacube, header


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


def rebin_spectrum_bspline(wavelengths, spectrum, delta_lambda_target):
    """
    Rebin a spectrum to a constant spectral resolution in terms of wavelength using a B-spline fit.

    Parameters:
    - wavelengths: 1D numpy array with the original wavelengths in Angstroms.
    - spectrum: 1D numpy array with the flux values of the original spectrum.
    - delta_lambda_target: Target spectral resolution in Angstroms.

    Returns:
    - new_wavelengths: 1D numpy array with the rebinned wavelengths.
    - new_spectrum: 1D numpy array with the rebinned spectrum.
    """
    # Create a B-spline representation of the spectrum
    t, c, k = make_interp_spline(wavelengths, spectrum, k=3).tck
    bspline = BSpline(t, c, k)

    # Define the new wavelength grid
    new_wavelengths = np.arange(wavelengths[0], wavelengths[-1], delta_lambda_target)

    # Evaluate the B-spline on the new wavelength grid
    new_spectrum = bspline(new_wavelengths)

    return new_wavelengths, new_spectrum


def scale_vega(wavelengths, spectrum, target_mag=23, quiet=False):
    """
    Scale a spectrum to match a target magnitude.
    """
    c = 299792.458  # speed of light in km/s
    velscale = c * np.log(wavelengths[1] / wavelengths[0])  # eq.(8) of Cappellari (2017)

    bands = ['2MASS/J']  # ['Johnson-Cousins_I', 'SDSS/i', '2MASS/J', '2MASS/H', '2MASS/K']
    z = 0.00093  # group redshift of LMC

    # ppxf logrebin function
    ln_flux, ln_lambda, vs = util.log_rebin(wavelengths, spectrum)

    # calculate flux in defined band
    result = util.synthetic_photometry(ln_flux, np.exp(ln_lambda), bands, redshift=z, quiet=1)

    # load Vega spectrum
    a = np.load(ppxf_dir + '/sps_models/spectra_sun_vega.npz')  # Spectrum in cgs/A
    flux_vega = a["flux_vega"]

    # rebin vega spectrum to match input spectrum
    flux_vega_new, ln_lam_vega, velscale = util.log_rebin(a["lam"], flux_vega, velscale)

    # calculate flux of Vega in defined band
    result_vega = util.synthetic_photometry(flux_vega_new, np.exp(ln_lam_vega), bands, redshift=z, quiet=1)

    # calculate magnitude of input spectrum
    mag = -2.5 * np.log10(result.flux / result_vega.flux)

    # calculate delta magnitude
    delta_m = target_mag - mag

    # scale input spectrum to match target magnitude
    ln_flux_new = ln_flux * 10 ** (delta_m / -2.5)
    spectrum_new = spectrum * 10 ** (delta_m / -2.5)  # TODO: check this is correct

    # calculate flux of scaled spectrum in defined band - just to check
    result_new = util.synthetic_photometry(ln_flux_new, np.exp(ln_lambda), bands, redshift=z, quiet=1)

    # calculate magnitude of scaled spectrum - just to check
    mag_new = -2.5 * np.log10(result_new.flux / result_vega.flux)

    if not quiet:
        print('Target magnitude: ', target_mag)  # try with 0 and plot to see if same as Vega
        print('Magnitude of input spectrum: ', mag)
        print('Delta to Vega: ', delta_m)
        print('Magnitude of scaled spectrum: ', mag_new)

    return ln_lambda, spectrum_new


def apply_spectra_from_file_vega(datacube, header, p, n_body_file, spec_res, plot=False):
    marcs_directory, fov, angular_resolution, sources, x_start, x_stop, y_start, y_stop, band = (p['marcs_directory'],
                                                                                           p['fov'], p['ang_res'],
                                                                                           p['sources'], p['x_start'],
                                                                                           p['x_stop'], p['y_start'],
                                                                                           p['y_stop'], p['band'])
    # print out basic info
    print('Loading datacube with shape: ', datacube.shape)

    # Load the pixel locations from the file within the field of view
    pixel_data = np.loadtxt(n_body_file, usecols=(3, 4))
    window_pixels = np.asarray([(x, y) for x, y in pixel_data if x_start <= x <= x_stop and y_start <= y <= y_stop])
    print('Loading pixel coordinates from %s sources out of %s' % (len(window_pixels), len(pixel_data)))

    # Load line of sight velocities and magnitudes from the file
    i_mag, j_mag, h_mag, k_mag, losv = np.loadtxt(n_body_file, usecols=(5, 6, 7, 8, 9), unpack=True)

    # limit the number of sources to use
    j = int(sources)
    print('Applying spectra to first %s sources in window' % j)

    # Angular resolution scaling factor (sf) for pixel locations
    #fov_x = x_stop - x_start
    #fov_y = y_stop - y_start
    #fov_scale_x = fov_x / 10
    #fov_scale_y = fov_y / 10
    #fov_scale = fov / 10
    #scale_factor = fov_scale / angular_resolution
    sf = 1 / angular_resolution
    #scale_factor_x = fov_scale_x / angular_resolution
    #scale_factor_y = fov_scale_y / angular_resolution

    # Load the spectrum for the pixel
    raw_spectrum = open_marcs_spectra(marcs_directory)
    # TODO: move into the loop when using different spectra for each star

    # Get MARCS wavelength data from standard file
    marcs_wavelengths = np.loadtxt(marcs_directory + '/' + 'flx_wavelengths.vac', skiprows=0, usecols=0)

    # Iterate over each pixel location and apply the spectrum
    for i in range(len(window_pixels[:j])):
        print('Applying spectrum to star %s' % i)
        print('Pixel coordinates were: %s' % window_pixels[i])
        pixel_x = int((window_pixels[i][0] - x_start) * sf)
        pixel_y = int((window_pixels[i][1] - y_start) * sf)
        print('Pixel coordinates are: x %s, y %s' % (pixel_x, pixel_y))

        # Adjust spectrum to match Vega
        wavelengths, spectrum = scale_vega(marcs_wavelengths, raw_spectrum, target_mag=j_mag[i], quiet=1)

        # rebin spectrum with bspline interpolation to get HSIM expected resolution
        wavelengths, spectrum = rebin_spectrum_bspline(marcs_wavelengths, spectrum, spec_res)

        if plot:
            plt.plot(spectrum, 'k.')
            plt.xlabel('Pixel')
            plt.ylabel('Flux [erg/s/cm^2/Å]')
            plt.title('MARCS Spectrum')
            plt.show()

        if i == 0:
            print('Template spectrum of shape:', spectrum.shape)

        # Apply redshift to the spectrum
        spectrum = apply_redshift(spectrum, losv[i])

        if i == 0:
            print('Redshifted spectrum of shape:', spectrum.shape)

        # crop spectra to HARMONI bands (8000-25000A)
        spectrum, crval3 = crop_spectrum(spectrum, marcs_directory, band)

        # Add the spectrum to the pixel location
        datacube, header = add_stellar_spectrum(datacube, header, pixel_x, pixel_y, spectrum, angular_resolution)

    # print info about output
    print("Datacube created with shape:", datacube.shape)
    print("Number of sources: ", j)

    # Change CRVAL3 to match the first wavelength in the datacube
    band_start, band_end = harmoni_band(band)
    header['CRVAL3'] = crval3

    return datacube, header, len(window_pixels)


def apply_redshift(spectrum, v):
    # redshift  z* =  v/c + zLMC

    c = 299792458  # m/s
    z_lmc = 0.00093  # group redshift of LMC
    # z_star = (v/c.to('km/s')) + z_lmc
    z_star = (v / (c / 1000)) + z_lmc
    # spectrum = spectrum * (1 + z_star)  # linear space
    spectrum = spectrum * np.sqrt((1 + z_star) / (1 - z_star))  # log space
    return spectrum


def add_stellar_spectrum(datacube, header, pixel_x, pixel_y, spectrum, angular_resolution):
    # Validate pixel coordinates
    if pixel_x < 0 or pixel_x >= datacube.shape[2]:
        raise ValueError(f"Invalid x-coordinate. Must be within the range 0 to {datacube.shape[2] - 1}.")
    if pixel_y < 0 or pixel_y >= datacube.shape[1]:
        raise ValueError(f"Invalid y-coordinate. Must be within the range 0 to {datacube.shape[1] - 1}.")

    # Validate spectrum shape
    # if spectrum.shape[0] != datacube.shape[0]:
    #    raise ValueError("Spectrum shape does not match the datacube.")

    # convert ergs/cm2/s/Å to erg/s/cm2/AA/arcsec2
    spectrum = spectrum / (angular_resolution ** 2)

    # Add the stellar spectrum to the specified pixel location
    # select 5 pixels from H-band
    # H_band_centre = 16000
    # datacube[:, pixel_y, pixel_x] += spectrum[10000:10005]  # TODO change to length of full spectrum
    # Before attempting to add, check if the slice is non-empty

    # take 50 pixels from the middle of the spectrum
    spectrum_middle = int(len(spectrum) / 2)
    z = int(datacube.shape[0])

    datacube[:, pixel_y, pixel_x] += spectrum[int(spectrum_middle - (z / 2)):int(spectrum_middle + (z / 2))]

    # if spectrum[9000:9005].size > 0:
    #    datacube[:, pixel_y, pixel_x] += spectrum[9000:9005]
    # else:
    #    # Handle the case where the slice is empty
    #    # For example, you might log a warning or use a default value
    #    print(f"Warning: Spectrum slice for indices 10000:10005 is empty.")

    return datacube, header


def crop_spectrum(spectrum, marcs_directory, band='H'):
    # Get MARCS wavelength data from standard file
    # marcs_directory = '/Users/gooding/Desktop/IMBH/templates/MARCS_library'
    marcs_wavelengths = np.loadtxt(marcs_directory + '/' + 'flx_wavelengths.vac', skiprows=0, usecols=0)

    # Define limits to wavelengths with HARMONI # TODO: confirm these are correct
    # harmoni_blue = 8000
    # harmoni_red = 25000

    # Get band start and end wavelengths in Angstroms
    band_start, band_end = harmoni_band(band)

    # Find extent of HARMONI band in MARCS
    # marcs_harmoni_blue = len(marcs_wavelengths[marcs_wavelengths < harmoni_blue])
    # marcs_harmoni_red = len(marcs_wavelengths[marcs_wavelengths < harmoni_red])

    # Find extent of band in MARCS
    marcs_band_blue = len(marcs_wavelengths[marcs_wavelengths < band_start])
    marcs_band_red = len(marcs_wavelengths[marcs_wavelengths < band_end])

    # limit to HARMONI wavelengths
    # spectrum = spectrum[marcs_harmoni_blue:marcs_harmoni_red]
    spectrum = spectrum[marcs_band_blue:marcs_band_red]
    return spectrum, marcs_wavelengths[marcs_band_blue]


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

            # Convert the value to an integer
            try:
                value = int(value)
            except ValueError:
                pass

            # Convert the value to a float
            try:
                value = float(value)
            except ValueError:
                pass

            # Convert the value to a boolean
            if value == 'True' or value == 'yes':
                value = True
            elif value == 'False' or value == 'no':
                value = False

            # Store the parameter in the dictionary
            params[key] = value

    return params


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


def main(config):
    # Load the configuration file
    p = config  # load_config('config.txt')   # p is a dictionary of parameters
    if p['imbh_present']:
        nbody_file = p['nbody_data_imbh']
        imbh_label = 'imbh'
    else:
        nbody_file = p['nbody_data']
        imbh_label = 'noimbh'

    # Get wavelength info
    spec_res, lam_centre = marcs_condition(p['marcs_directory'], p['harmoni_red'], p['harmoni_blue'])

    # if p['wavelength_planes'] == 0 then get the number of wavelength planes from the band and the spec res
    if p['wavelength_planes'] == 0:
        band_start, band_end = harmoni_band(p['band'])
        spec_pixels = int((band_end - band_start) / spec_res)
    else:
        spec_pixels = p['wavelength_planes']

    # Define the shape of the datacube
    fov_x = p['x_stop'] - p['x_start']  # FoV in arcsec in x direction
    fov_y = p['y_stop'] - p['y_start']  # FoV in arcsec in y direction
    cube_shape = (int(spec_pixels), int(fov_x / p['ang_res']),
                  int(fov_y / p['ang_res']))  # TODO: check x and y are correct
    print('Blank datacube created with shape:', cube_shape)

    # Create an empty datacube with the specified dimensions, wavelength range, and header information
    datacube, header = create_harmoni_datacube(cube_shape, lam_centre, spec_res, p['ang_res'])

    # Apply spectra from a file
    # datacube, header = apply_spectra_from_file_vega(datacube, header, p['marcs_directory'], nbody_file, p['fov'], p['ang_res'], spec_res, p['sources'], plot=p['plot'])
    datacube, header, nsources = apply_spectra_from_file_vega(datacube, header, p, nbody_file, spec_res, plot=p['plot'])
    print('Datacube generated')

    # Save the datacube to a FITS file
    print('Saving datacube')
    fits.writeto(p['output_directory'] + '/' + '%s_sources_%sl_%s.fits' %
                 (nsources, int(p['wavelength_planes']), imbh_label), datacube, header, overwrite=True)
    print('Datacube saved as %s_sources_%sl_%s.fits' % (int(p['sources']), int(spec_pixels), imbh_label))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 create_marcs_datacube.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)
    main(config)
