import numpy as np
from ppxf.ppxf import ppxf
from astropy.stats import biweight_scale
import ppxf.ppxf_util as util

def get_losv(spectrum, wavelengths, template, template_wavelengths):
    """
    Fits the line-of-sight velocity and signal-to-noise ratio of the input spectrum using the pPXF method.

    Parameters:
    spectrum: array_like
        The observed spectrum to fit.
    wavelengths: array_like
        The wavelengths corresponding to the spectrum.
    template: array_like
        The template spectrum for fitting.
    template_wavelengths: array_like
        The wavelengths corresponding to the template spectrum.

    Returns:
    tuple
        The line-of-sight velocity (km/s), the signal-to-noise ratio of the fit, and the chi-squared of the fit.
    """

    # Define the starting velocity guess
    z = 0.00093
    c = 299792.458
    vel = c * np.log(1 + z)
    start = [vel, 0]  # (km/s), starting guess for [V, sigma]

    # Define the good pixels and wavelength ranges
    goodpixels = np.arange(500, 3500)

    # Log rebin the spectrum
    star, logLam1, velscale = util.log_rebin(wavelengths, spectrum)
    star = star / np.median(star)
    lam = np.exp(logLam1)

    # Define noise (you may need to customize this depending on your data)
    noise = np.full_like(star, 1)

    # Log rebin the template
    templ, logLam2, velscale = util.log_rebin(template_wavelengths, template, velscale=velscale)
    templ = templ / np.median(templ)
    lam_temp = np.exp(logLam2)

    # Perform the fit
    pp = ppxf(templ, star, noise, velscale, start, goodpixels=goodpixels, plot=False, moments=2, degree=8, lam=lam, lam_temp=lam_temp, quiet=True)

    # Calculate residuals and signal-to-noise ratio
    residuals = star[goodpixels] - pp.bestfit[goodpixels]
    median_flux = np.median(star[goodpixels])
    biweight_sigma = biweight_scale(residuals)
    snr_star = median_flux / biweight_sigma

    return pp, snr_star
