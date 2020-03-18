# P2_farm_biochar

This repository contains the python code used in the manuscript "Small-scale biochar production on Swedish farms: a model for estimating potential, variability, and environmental performance".

**Short summary**


## Structure

It is structured as follows:


    .
    ├── farm_biochar_model                   # Core scripts
    │   ├ farm_demand.py                        # calculate farm energy demand
    │   ├ farm_supply.py                        # define equipment available on farm
    │   ├ farm_ficus.py                         # unit commitment solver (adapted from ficus - https://github.com/yabata/ficus)
    │   ├ farm_bw2.py                           # calculate indicators and generalte LCA inventory (based on brightway2 - https://github.com/brightway-lca)
    │   ├ lcopt_multi_tagged.py                 # useful functions (from lcopt - https://github.com/pjamesjoyce/lcopt)
    │   ├ farm_ficus_input_template.xlsx        # template input file to ficus
    │   └ farm_supply_plants.xlsx               # list of equipment available for scenarios, new plant can be added here 
    │     
    ├── ex0_template                         # Template for new project
    │
    ├── ex1_Lindeborg                        # Case study at Lindeborg's farm
    │   ├ input_files                           # e.g. electricity data, weather data
    │   ├ output_files                          # e.g. ficus generated results, figures, logger
    │   ├ 1_run scenarios.ipynb                 # notebook where simulations are defined and run
    │   ├ 2_analyse scenarios.ipynb             # notebook where simulations are analysed, figures plotted
    │   └ 3_bonus_dimensioning analysis.ipynb   # notebook for other purposes
    │
    ├── requirements.txt
    └── README.md


## Snapshots


## Dependencies & credits

This work relies on other libraries, including:
- ficus, https://github.com/yabata/ficus
- brightway2, https://github.com/brightway-lca
- lcopt, https://github.com/pjamesjoyce/lcopt



 

