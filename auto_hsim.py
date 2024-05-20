# Auto_HSIM: Automates the process of running HSIM on a set of FITS files and merging the output datacubes.
#
# The script reads a configuration file that specifies the input directory containing the FITS files, the HSIM configuration file, and the output directory.
# It processes each FITS file using HSIM and then merges the output datacubes.
#
# The script also includes a function to merge two datacubes with an overlapping region.
# The merged datacube is saved to a new FITS file.
#
# Usage: python auto_hsim.py <config_file>
# The configuration file should be in JSON format and contain the following keys:
# - parts_dir: The directory containing the input FITS files
# - hsim_config: The path to the HSIM configuration file
# - output_directory: The directory to save the output datacubes
#
# Example configuration file:
# {
#     "auto_hsim": {
#         "parts_dir": "datacubes",
#         "hsim_config": "hsim_config.ini",
#         "output_directory": "output"
#     }
# }
# The script assumes that the HSIM configuration file is in INI format and contains the following sections:
# [HSIM]
# input_cube = <path_to_input_cube>
# output_dir = <output_directory>
# ...
# The script modifies the input_cube and output_dir parameters in the HSIM configuration file for each FITS file processed.
# The output datacubes are saved in the specified output directory.
#
# The script also merges the output datacubes by averaging the overlapping regions.
# The merged datacubes are saved with the prefix 'merged_' in the output directory.
#
# The script uses the Astropy library to read and write FITS files and NumPy for array manipulation.
# The subprocess module is used to run the HSIM script with the modified configuration file.
#


import os
from os import path
import subprocess
import sys
import json
from astropy.io import fits
import numpy as np
import shutil


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


def merge_fits_datacubes(part1_file_path, part2_file_path):
    """
    Merge two FITS datacubes with an overlapping region.
    The overlapping region is averaged to create a smooth transition between the two parts.

    Args:
        part1_file_path:
        part2_file_path:

    Returns:

    """
    # Load the two parts of the datacube
    datacube1, header1 = fits.getdata(part1_file_path, header=True)
    datacube2, header2 = fits.getdata(part2_file_path, header=True)

    # Determine the overlap
    # overlap = 12 # Assumes 12 pixel overlap, simplification based on original overlap of 50 oversampled by 4x
    # --> 50/4 = 12.5. Have changed original overlap to 60, so in future 60/4 = 15.
    overlap = 15
    # TODO - change overlap to be calculated based on the oversampling factor from the config file

    # Check if overlap is correctly calculated
    if overlap <= 0:
        print("Error: No overlap detected or incorrect file sizes.")
        return

    print(f'Overlap: {overlap} pixels')

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
    merged_data[header1['NAXIS3'] - overlap:header1['NAXIS3']] = (datacube1[-overlap:] + datacube2[:overlap]) / 2

    # Adjust the header information to reflect the merged datacube
    header = header1.copy()
    header['NAXIS3'] = naxis3_new
    header['CRVAL3'] = header1['CRVAL3']  # Assuming the reference value does not need adjustment
    header['CDELT3'] = header1['CDELT3']  # Assuming the delta does not need adjustment

    return merged_data, header


def main(config, output_dir, split_parts):
    # Define the directory containing the FITS files and the output directory
    if split_parts > 1:
        input_dir = path.join(output_dir, config['parts_dir'])
    else:
        input_dir = output_dir
    hsim_config_path = config['hsim_config']
    hsim_output_dir = path.join(output_dir, 'hsim_output/')

    # Create the output directory if it does not exist
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
        with open(hsim_config_path, 'w') as f:
            f.write('[HSIM]\n')
            for key, value in hsim_config.items():
                f.write(f'{key} = {value}\n')
        print(f'Processing {file}...')

        # Process the file with HSIM
        # the configuration file is specified with -c
        # -b forces HSIM to start the simulation not showing the GUI
        command = f'/usr/bin/python3.10 /mnt/zfsusers/goodingd/HSIM/hsim/hsim3.py -b -c {hsim_config_path}'
        # TODO - make generic to run on any machine
        print(f'Running command: {command}')
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f'Processed {file} and saved output')

    print('All files processed successfully.')

    # Check if merging is needed
    if split_parts < 2:
        print('No merging needed.')
        sys.exit(1) # Exit with error code 1

    # Merge output cubes, named such as '_part1of5_reduced_flux_cal.fits'
    hsim_output_dir = hsim_config['output_dir']

    print('Merging output cubes...')
    output_flux_files = [f for f in os.listdir(hsim_output_dir) if f.endswith('_reduced_flux_cal.fits')]
    output_flux_files = [f for f in output_flux_files if 'merged' not in f]
    output_snr_files = [f for f in os.listdir(hsim_output_dir) if f.endswith('_reduced_SNR.fits')]
    output_snr_files = [f for f in output_snr_files if 'merged' not in f]

    if len(output_flux_files) < 2:
        print('At least two output files are needed to merge.')
        exit()

    # Sort the files to ensure the correct order
    output_flux_files.sort()
    output_snr_files.sort()
    print('Cubes to merge:', output_flux_files)

    # Define the merged file names
    merged_flux_file = 'merged_reduced_flux_cal.fits'
    merged_snr_file = 'merged_reduced_SNR.fits'

    # Duplicate the first cube to merge with the rest
    shutil.copy2(os.path.join(hsim_output_dir, output_flux_files[0]), os.path.join(hsim_output_dir, merged_flux_file))
    shutil.copy2(os.path.join(hsim_output_dir, output_snr_files[0]), os.path.join(hsim_output_dir, merged_snr_file))

    for flux_file, snr_file in zip(output_flux_files[1:], output_snr_files[1:]):
        print(f'Merging {flux_file}...')
        part1_flux_path = path.join(hsim_output_dir, merged_flux_file) # Takes first / original cube
        part2_flux_path = path.join(hsim_output_dir, flux_file) # Takes second cube or next cube to merge
        part1_snr_path = path.join(hsim_output_dir, merged_snr_file)
        part2_snr_path = path.join(hsim_output_dir, snr_file)

        # Merge the cubes
        merged_data, header = merge_fits_datacubes(part1_flux_path, part2_flux_path)
        merged_snr, header_snr = merge_fits_datacubes(part1_snr_path, part2_snr_path)

        # Save the merged datacube to a FITS file
        merged_flux_path = path.join(hsim_output_dir, merged_flux_file)
        fits.writeto(merged_flux_path, merged_data, header, overwrite=True)
        merged_snr_path = path.join(hsim_output_dir, merged_snr_file)
        fits.writeto(merged_snr_path, merged_snr, header_snr, overwrite=True)
        print(f'Merged {flux_file} into {merged_flux_file} and {snr_file} into {merged_snr_file}.')

    print('Merging complete.')


if __name__ == '__main__':
    # Check if a configuration file was provided
    if len(sys.argv) < 2:
        print('Usage: python auto_hsim.py <config_file>')
        exit()

    config_file = sys.argv[1]

    config_ = load_json_config(config_file)

    global_params = config_.get('global', {})
    config = config_.get('auto_hsim', {})
    create_config = config_.get('create_marcs_datacube', {})

    output_dir = global_params['output_directory']
    split_parts = create_config['split_parts']

    if not path.exists(output_dir):
        os.mkdir(output_dir)

    main(config, output_dir, split_parts)
