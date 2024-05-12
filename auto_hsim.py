# Programme to open different datacube parts and automatically run HSIM on them with a set of given parameters

import os
from os import path
import subprocess
import sys
import json
from astropy.io import fits
import numpy as np


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


def main(config, output_dir):
    # Define the directory containing the FITS files and the output directory
    input_dir = path.join(output_dir, config['parts_dir'])
    hsim_config_path = config['hsim_config']
    hsim_output_dir = path.join(output_dir, 'hsim_output/')

    # Create the output directory if it does not exist -- seems to be overwriting on glamdring
    if not os.path.exists(hsim_output_dir):
        os.makedirs(hsim_output_dir)

    # List all FITS files in the directory
    fits_files = [f for f in os.listdir(input_dir) if f.endswith('.fits')]
    if not fits_files:
        print('No FITS files found in the directory. Please ensure the files are in FITS format and try again.')
        exit()

    # Print the list of FITS files found
    print('Found the following FITS files:')
    for file in fits_files:
        print(file)

    # Print the number of FITS files found
    print(f'Found {len(fits_files)} FITS files in the directory.')

    # Load the configuration file
    hsim_config = load_config(hsim_config_path)

    # Print the configuration file
    print('HSIM configuration file:')
    print(hsim_config)

    # Modify output_dir in the config file
    hsim_config['output_dir'] = hsim_output_dir

    # Iterate over each FITS file
    for file in fits_files:
        file_path = os.path.join(input_dir, file)

        # Modify 'input_cube' in the config file to point to the current FITS file
        hsim_config['input_cube'] = file_path

        # Write the modified config file
        with open(hsim_config, 'w') as f:
            f.write('[HSIM]\n')
            for key, value in hsim_config.items():
                f.write(f'{key} = {value}\n')
        print(f'Processing {file}...')

        # Process the file with HSIM
        # the configuration file is specified with -c
        # -b forces HSIM to start the simulation not showing the GUI
        command = f'/usr/bin/python3.10 /mnt/zfsusers/goodingd/HSIM/hsim/hsim3.py -b -c {hsim_config_path}'
        print(f'Running command: {command}')
        subprocess.run(command, shell=True)

        print(f'Processed {file} and saved output')

    print('All files processed successfully.')

    # Merge output cubes, named such as '_part1of5_reduced_flux_cal.fits'



    print('Merging output cubes...')
    output_flux_files = [f for f in os.listdir(hsim_output_dir) if f.endswith('_reduced_flux_cal.fits')]
    output_snr_files = [f for f in os.listdir(hsim_output_dir) if f.endswith('_reduced_SNR.fits')]
    if len(output_flux_files) < 2:
        print('At least two output files are needed to merge.')
        exit()

    # Sort the files to ensure the correct order
    output_flux_files.sort()
    output_snr_files.sort()

    # Merge the output cubes in a loop, merge 2 to 1, 3 to the merged 1, 4 to the merged 1, and so on
    merged_flux_file = output_flux_files[0]
    merged_snr_file = output_snr_files[0]

    for flux_file, snr_file in zip(output_flux_files[1:], output_snr_files[1:]):
        part1_file_path = os.path.join(hsim_output_dir, merged_flux_file)
        part2_file_path = os.path.join(hsim_output_dir, flux_file)
        output_file_path = os.path.join(hsim_output_dir, f'merged_{merged_flux_file}')

        # Merge the cubes
        merge_fits_datacubes(part1_file_path, part2_file_path, output_file_path)

        # Update the merged file name for the next iteration
        merged_flux_file = f'merged_{merged_flux_file}'
        merged_snr_file = f'merged_{merged_snr_file}'



    print('All files processed and merged successfully.')


if __name__ == '__main__':
    # Check if a configuration file was provided
    if len(sys.argv) < 2:
        print('Usage: python auto_hsim.py <config_file>')
        exit()

    config_file = sys.argv[1]

    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)['auto_hsim']
        global_config = json.load(f)['global']

    output_dir = global_config['output_directory']

    main(config, output_dir)
