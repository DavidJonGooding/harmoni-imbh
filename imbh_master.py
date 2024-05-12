import subprocess


def run_script(script_path, args=[]):
    """Function to run a python script using subprocess."""
    command = ['python', script_path] + args
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running {script_path}: {result.stderr}")
        raise Exception(f"Script {script_path} failed")
    print(f"Output from {script_path}: {result.stdout}")
    return result


def main():
    # Paths to your scripts
    path_create = './create_marcs_datacube.py'
    path_hsim = './auto_hsim.py'
    path_merge = './merge_cubes.py'
    path_hsextractor = './harmoni_source_extractor.py'

    # 1. Create the datacube parts
    print("---------------------------")
    print("Creating datacube...")
    run_script(path_create)
    print("---------------------------")

    # 2. Process each part through HSIM
    print("---------------------------")
    print("Processing datacube parts through HSIM...")
    run_script(path_hsim)
    print("---------------------------")

    # 3. Merge the processed parts
    print("---------------------------")
    print("Merging processed datacubes...")
    run_script(path_merge)
    print("---------------------------")

    # 4. Prepare cube for PampelMuse and extract sources
    print("---------------------------")
    print("Preparing cube for PampelMuse and extracting sources...")
    run_script(path_hsextractor)
    print("---------------------------")

    print("Process completed successfully.")


if __name__ == "__main__":
    main()
