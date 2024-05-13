# harmoni-imbh
Tools to simulate HARMONI observations and data reduction using HSIM, daophot, PampelMuse and pPXF to assess IMBH detection feasibility in globular clusters using the integral field spectrograph, HARMONI on the E-ELT.

## Manual use

1. Run 'create_marcs_datacube.py' to build raw datacubes with MARCS spectra using MICADO N-Body simulation data.
2. Use the output in HSIM to create as-observed datacubes.
3. Run 'harmoni_source_extractor.py' to find source locations using daophot routine in as-observed datacubes and condition data for PampelMuse.
4. Use PampelMuse to extract the spectra for each source.
5. Run 'harmoni_losv.py' to get LOSVs for each sources, this uses pPXF routine.

## NEW Automated use

1. Define all parameters in 'imbh-config.json'
2. Run 'imbh_master.py', which will:
   - call 'create_marcs_datacube.py' as above
   - call 'auto_hsim.py' to run HSIM sims through each datacube using 'hsim-config.json'
   - call 'harmoni_source_extractor.py' as above.
   - IN DEVELOPMENT - automated PampelMuse routine

## Inputs

1. Data: outputs from MICADO N-body simulations of a star cluster with and without a central IMBH
2. Templates: spectra from MARCS library
3. Configs: files which are used to choose settings on python routines

## Dependencies

1. HSIM - the HARMONI instrument simulator (https://github.com/HARMONI-ELT/HSIM)
2. PampelMuse - PSF source extraction from integral field spectroscopy datacubes (https://pampelmuse.readthedocs.io)
3. pPXF - stellar kinematics via full spectrum fitting with photometry
 (https://www-astro.physics.ox.ac.uk/~cappellari/software/#ppxf)