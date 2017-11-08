# Data contained herein

## Combination of ACS5 and decennial census data:

* ```prediction-geojson```
Directory containing files for each state with block population estimation based on 2015 block group estimations, congressional district for each block, centroids, etc.


## Data from the ACS5 census:

* ```acs5-population```
Directory in which population for each block group for each state is saved.
These are downloaded from the API: http://api.census.gov/data/2015/acs5/examples.html

* ```acs5-shapefiles```
Directory containing the shapefiles for each state.
Download manually from: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2015&layergroup=Block+Groups

## Shapefiles for whole USA:

* ```usa-cd-shapefile```
This directory contains the congressional districts (shapefiles, etc.) for the whole USA.
Download manually from: https://www.census.gov/cgi-bin/geo/shapefiles/index.php.

## Data from 2010 decennial census:

* ```2010-census-shapefiles```
Directory with shapefiles of the blocks (not block groups), as well as the population, per state.
Download manually from section _Population & Housing Unit Counts — Blocks_ at: https://www.census.gov/geo/maps-data/data/tiger-data.html
Files should have name ```tabblock2010_*_pophu``` where ```*``` is the FIPS code per state.

* ```block-assignments```
Directory with the congressional district each block group is assigned to per state.
Download manually from section _115th Congressional Block Equivalency Files_ at: https://www.census.gov/rdo/data/113th_congressional_and_2012_state_legislative_district_plans.html

