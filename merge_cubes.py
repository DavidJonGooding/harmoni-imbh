import os
from astropy.io import fits
import numpy as np


def merge_fits_datacubes(part1_file_path, part2_file_path, output_file_path):
    # Load the two parts of the datacube
    datacube1, header1 = fits.getdata(part1_file_path, header=True)
    datacube2, header2 = fits.getdata(part2_file_path, header=True)

    # Determine the overlap
    overlap = datacube1.shape[0] + datacube2.shape[0] - header1['NAXIS3']

    # Check if overlap is correctly calculated
    if overlap <= 0:
        print("Error: No overlap detected or incorrect file sizes.")
        return

    # Calculate new datacube dimensions
    naxis3_new = header1['NAXIS3'] + header2['NAXIS3'] - overlap
    naxis2_new = header1['NAXIS2']  # Assuming the other dimensions remain constant
    naxis1_new = header1['NAXIS1']

    # Merge data arrays
    merged_data = np.zeros((naxis3_new, naxis2_new, naxis1_new), dtype=datacube1.dtype)

    # Fill in the non-overlapping parts
    merged_data[:header1['NAXIS3'] - overlap] = datacube1[:header1['NAXIS3'] - overlap]
    merged_data[-header2['NAXIS3'] + overlap:] = datacube2[overlap:]

    # Average the overlapping regions
    merged_data[header1['NAXIS3'] - overlap:header1['NAXIS3']] = (
                                                                         datacube1[-overlap:] + datacube2[:overlap]) / 2

    # Adjust the header information to reflect the merged datacube
    header = header1.copy()
    header['NAXIS3'] = naxis3_new
    header['CRVAL3'] = header1['CRVAL3']  # Assuming the reference value does not need adjustment
    header['CDELT3'] = header1['CDELT3']  # Assuming the delta does not need adjustment

    # Save the merged datacube to a new FITS file
    fits.writeto(output_file_path, merged_data, header, overwrite=True)
    print(f'Merged datacube saved as {output_file_path}')


if __name__ == '__main__':
    # Full paths for the parts
    #part1_file_path = input('Enter the full path of the first part of the FITS file: ')
    #part2_file_path = input('Enter the full path of the second part of the FITS file: ')

    part1_file_path = \
        '/Users/gooding/Desktop/IMBH_PPM/IMBH_5hours_new/rawcube_6093_sources_imbh_part1of2_reduced_flux_cal.fits'
    part2_file_path = \
        '/Users/gooding/Desktop/IMBH_PPM/IMBH_5hours_new/rawcube_6093_sources_imbh_part2of2_reduced_flux_cal.fits'

    # Full path for the output file
    output_file_path = input('Enter the name for the merged FITS file: ')

    # Check if files exist
    print(f'Checking if {part1_file_path} and {part2_file_path} exist...')
    if os.path.isfile(part1_file_path) and os.path.isfile(part2_file_path):
        merge_fits_datacubes(part1_file_path, part2_file_path, output_file_path)
    else:
        print('One or both files not found. Please check the paths and try again.')
