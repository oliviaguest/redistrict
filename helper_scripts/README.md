# Scripts

## How to Run
Run from outside the ```helper_scripts``` directory, e.g.:
```
python helper_scripts/check.py
```

## block_stats.py
Calculate the descriptive statistics for blocks in our dataset.
Save result in ```./results/block_stats/```

## check.py
Check if all states have results.

## estimate_population.py
Take the input files from the decennial census and the ACS5 2015 census and predicts the populaltion in each state.
Give a rough estimation of the populaltion per block (not block group) for the year 2015.
This same calculation is done from within ```run.py``` if needed.

## find_min_and_max.py
Discover which congressional districts in which states have the maximum and minimum congressional districts in the new predicted populations.
This means that states with a single congressional district (e.g., 'at Large') are included.
Save result in ```./data/census/overall/```.

## create_geojson.py
Creates GeoJSON files which have rounded values for lat and lon and only the required columns.
These are used for the figures for the journal article and on the website (as they are appropriate for converting to TopoJSON).
