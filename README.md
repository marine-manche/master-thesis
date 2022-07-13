# Master’s project 

Exploration of geospatial modelling approaches to assess the effect of urban green-blue areas
on physical activity and health: Case study for Swiss cities

## Description

This repository provides the code of the work done in the Master's project. The data folder provides baseline data to use in the scripts. Confidential data is not provided. 
The folder `code_land-cover` contains the code used to extract Open Street Maps data (see Section 2.2.3 from report) and the NDVI raster. 
The folders `code_lausanne` and `code_swiss-cities` contain the code used for the data treatment of the MOBIS and the CoLaus datasets, at the local and the multi-city scale, respectively (see Sections 2.2.1, 2.2.2 and 2.2.4 from report).
The data treatment is done step by step and the folders are numbered accordingly. For the data treatment at the local scale, outputs from the step 6 at the swiss-city scale (`code_swiss-cities/6_linestrings.py`) are used as an input of the local-scale step `code_lausanne/7_use`.

## Dependencies

The code is written in Python and the libraries listed in the file `requirement.txt` are needed. One exception to this is the file `code_land-cover/ndvi.txt` which was run with Google Earth Engine (GEE).

## Author

Marine Manche 

## License

This project is licensed under the CC-BY-SA License.
