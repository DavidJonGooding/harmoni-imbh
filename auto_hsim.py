# Programme to open different datacube parts and automatically run HSIM on them with a set of given parameters

import os
import subprocess
import sys
import json


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


def main(config):
    # Define the directory containing the FITS files and the output directory
    input_dir = '/mnt/zfsusers/goodingd/imbh/input_cubes/cubeparts/'     # for glamdring use
    # input_dir = './cubeparts/' # for local use
    config_file = '/mnt/zfsusers/goodingd/imbh/input_cubes/config_auto_hsim.json'    # for glamdring use
    # config_file = './config_auto_hsim.json' # for local use
    output_dir = input_dir + 'hsim_output/'

    # Create the output directory if it does not exist -- seems to be overwriting on glamdring
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)

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
    config = load_config(config_file)

    # Print the configuration file
    print('Configuration file:')
    print(config)

    # Modify output_dir in the config file
    config['output_dir'] = output_dir

    # Iterate over each FITS file
    for file in fits_files:
        file_path = os.path.join(input_dir, file)

        # Modify 'input_cube' in the config file to point to the current FITS file
        config['input_cube'] = file_path

        # Write the modified config file
        with open(config_file, 'w') as f:
            f.write('[HSIM]\n')
            for key, value in config.items():
                f.write(f'{key} = {value}\n')
        print(f'Processing {file}...')

        # Process the file with HSIM
        # the configuration file is specified with -c
        # -b forces HSIM to start the simulation not showing the GUI
        command = f'/usr/bin/python3.10 /mnt/zfsusers/goodingd/HSIM/hsim/hsim3.py -b -c {config_file}'
        print(f'Running command: {command}')
        subprocess.run(command, shell=True)

        print(f'Processed {file} and saved output')

    print('All files processed successfully.')


if __name__ == '__main__':
    # Check if a configuration file was provided
    if len(sys.argv) < 2:
        print('Usage: python auto_hsim.py <config_file>')
        exit()

    config_file = sys.argv[1]
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)['auto_hsim']

    main(config)
