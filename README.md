# harmoni-imbh
Tools to simulate HARMONI observations and data reduction using HSIM, daophot, PampelMuse and pPXF to assess IMBH detection feasibility

## Use

1. Run 'create_marcs_datacube.py' to build raw datacubes with MARCS spectra using MICADO N-Body simulation data.
2. Use the output in HSIM to create as-observed datacubes.
3. Run 'harmoni_source_extractor.py' to find source locations using daophot routine in as-observed datacubes and condition data for PampelMuse.
4. Use PampelMuse to extract the spectra for each source.
5. Run 'harmoni_losv.py' to get LOSVs for each sources, this uses pPXF routine.

## Inputs

Data: outputs from MICADO N-body simulations of a star cluster with and without a central IMBH
Templates: spectra from MARCS library
Configs: files which are used to choose settings on python routines
