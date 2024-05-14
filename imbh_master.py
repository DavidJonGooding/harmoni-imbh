import subprocess


def run_script(script_path, args=[]):
    """Function to run a python script using subprocess."""
    command = ['/usr/bin/python3.10', script_path] + args
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running {script_path}: {result.stderr}")
        raise Exception(f"Script {script_path} failed")
    print(f"Output from {script_path}: {result.stdout}")
    return result


def main():
    # Paths to your scripts
    path_create = 'create_marcs_datacube.py'
    path_hsim = 'auto_hsim.py'
    path_merge = 'merge_cubes.py'
    path_hsextractor = 'harmoni_source_extractor.py'
    master_config = 'configs/imbh-config.json'

    # 1. Create the datacube parts
    print("------------------------------------------------------")
    print("Creating datacube...")
    run_script(path_create, [master_config])
    print("------------------------------------------------------")
    print("RAW DATACUBES CREATED")

    # 2. Process each part through HSIM
    print("------------------------------------------------------")
    print("Processing datacube parts through HSIM...")
    run_script(path_hsim, [master_config])
    print("------------------------------------------------------")
    print("AS-OBSERVED DATACUBES CREATED")

    # 3. Prepare cube for PampelMuse and extract sources
    print("------------------------------------------------------")
    print("Preparing cube for PampelMuse and extracting sources...")
    run_script(path_hsextractor, [master_config])
    print("------------------------------------------------------")

    print("Process completed successfully.")


if __name__ == "__main__":
    main()
