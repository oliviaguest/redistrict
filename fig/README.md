# Directories

This directory contains the files used to create the figures in the journal article. Each figure (the final product) in directories ```./fig/svg``` (original created using output from QGIS in Inkscape) and  ```./fig/pdf``` (for inserting into LaTeX) has a QGIS project associated with it in ```./fig/QGIS```. The QGIS projects contain links to all the relevant data, like cities, geographies, improvement from clustering, populaltion, etc.

## csv
Contains file ```improvement.csv``` generated from notebook ```Regression.ipynb``` in ```./notebooks/```.

## geojson
Best results per state saved as geojson. These are created by running ```get_geojson_per_state.py```.

## QGIS
The QGIS projects with all the layers to create the figures in the journal article.

## shapefile
Shapefiles to help with making the figures — see ```./fig/shapefile/README.md``` for more details.

## results
Output from ```./notebooks/Whole USA figure.ipynb```. I have tried very hard to make geopandas accept that I prefer a different projection, but it ignores me. I have used ```./notebooks/Whole USA figure.ipynb``` to produce the figure I want (roughly) but it's not the projection I would like. To fix this I'm using QGIS (see directory ```./fig/QGIS```).
