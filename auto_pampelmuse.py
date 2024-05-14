# A programme which loads a datacube and automatically runs all PampelMuse routines on it

# Import modules
import json
import subprocess
from astropy.io import fits
import sys
import os
from os import path


# Load the configuration file
def load_json_config(config_file):
    """
    Load a JSON configuration file and return the parameters as a dictionary.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


# Run a command using subprocess and handle output
def run_command(command):
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        raise Exception(f"Command failed: {' '.join(command)}")
    print(f"Output: {result.stdout}")
    return result.stdout


# Main function to run PampelMuse routines
def main(config, output_dir):

    # Load datacube
    datacube_path = config['datacube_path']
    print(f"Loading datacube from {datacube_path}")
    with fits.open(datacube_path) as hdul:
        datacube = hdul[0].data

    # Load source locations
    source_locations = config['source_locations']

    # Run PampelMuse routines
    pampelmuse_config = config['pampelmuse']
    for routine in pampelmuse_config['routines']:
        command = ['pampelmuse', routine] + pampelmuse_config['args']
        run_command(command)

    # Save the output if needed
    output_path = config['output_path']
    print(f"Saving results to {output_path}")
    # Assuming the results are in a variable called `results`
    # with fits.writeto(output_path, results)


if __name__ == '__main__':
    # Read the configuration file from the command line argument
    config_file = sys.argv[1]

    # Check if a configuration file was provided
    if len(sys.argv) < 2:
        print('Usage: python auto_pampelmuse.py <config_file>')
        exit()

    config_ = load_json_config(config_file)

    global_params = config_.get('global', {})
    config = config_.get('auto_pampelmuse', {})

    output_dir = global_params['output_directory']

    if not path.exists(output_dir):
        os.mkdir(output_dir)

    main(config, output_dir)
