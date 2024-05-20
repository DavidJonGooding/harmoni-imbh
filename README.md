# harmoni-imbh
Tools to simulate ELT/HARMONI observations of star clusters.

Build datacubes based on N-body simulation data, simulate observations, and perform data reduction to extract stellar kinematics.

### Installation

1. Clone the repository
2. Install the required packages using pip:
```
pip install -r requirements.txt
```
3. Install HSIM (https://github.com/HARMONI-ELT/HSIM)
4. Install PampelMuse (https://pampelmuse.readthedocs.io)
5. Install pPXF (https://www-astro.physics.ox.ac.uk/~cappellari/software/#ppxf)

### Workflow

1. Creates raw datacubes with MARCS spectra using MICADO N-Body simulation data.
2. Uses the output in HSIM to create as-observed datacubes.
3. Finds source locations using daophot routine in as-observed datacubes and conditions data for PampelMuse.
4. Extracts the spectra for each source using PampelMuse.
5. Gets line-of-sight velocities for each source using pPXF.

### Manual use

1. Run 'create_marcs_datacube.py' to build raw datacubes with MARCS spectra using MICADO N-Body simulation data.
2. Use the output in HSIM to create as-observed datacubes.
3. Run 'harmoni_source_extractor.py' to find source locations using daophot routine in as-observed datacubes and condition data for PampelMuse.
4. Use PampelMuse to extract the spectra for each source.
5. Run 'harmoni_losv.py' to get LOSVs for each sources, this uses pPXF routine.

### NEW Automated use

1. Define all parameters in 'imbh-config.json'
2. Run 'imbh_master.py', which will:
   - call 'create_marcs_datacube.py' as above
   - call 'auto_hsim.py' to run HSIM sims through each datacube using 'hsim-config.json'
   - call 'harmoni_source_extractor.py' as above.
   - IN DEVELOPMENT - automated PampelMuse routine

### Inputs

1. Data: outputs from MICADO N-body simulations of a star cluster with and without a central IMBH
2. Templates: spectra from MARCS library
3. Configs: files which are used to choose settings on python routines

### Dependencies

The key major software dependencies for this repo are:

1. HSIM - the HARMONI instrument simulator
2. PampelMuse - PSF source extraction from integral field spectroscopy datacubes
3. pPXF - stellar kinematics via full spectrum fitting with photometry

The N-body simulation data is courtesy of Fiorentino et al. (2020), found at: https://academic.oup.com/mnras/article/494/3/4413/5813442

### Author

Written by David Gooding, DPhil student at the Department of Astrophysics, University of Oxford.

Contact: david.gooding@physics.ox.ac.uk