"""Get descriptives for the census blocks in our data."""
from __future__ import print_function

import os
import glob
import pandas as pd
import geopandas as gpd

from utils.functions import update_best_stats_dir
from utils.settings import (best_stats_dir,
                            clustered_geojson_results_dir,
                            results_dir)

# Update the best stats and get their names:
update_best_stats_dir()
filenames = []
for name in glob.glob(best_stats_dir + '*'):
    filenames.append(os.path.splitext(os.path.split(name)[1])[0])

for n, name in enumerate(filenames):
    print(name, n, 'of', len(filenames))
    temp = gpd.read_file(clustered_geojson_results_dir + name + '.geojson')
    temp = temp.to_crs({'init': 'epsg:3857'})
    try:
        population = population.append(temp['Predicted 2015 Population'])
        area = area.append(temp['geometry'].area / 10**6)
    except NameError:
        population = temp['Predicted 2015 Population']
        area = temp['geometry'].area / 10**6
    del temp

pop_desc = pd.DataFrame({'mean': pd.Series(population.mean()),
                         'median': pd.Series(population.median()),
                         'mode': pd.Series(population.mode()),
                         'min': pd.Series(population.min()),
                         'max': pd.Series(population.max())})
pop_desc = pop_desc.reindex(columns=["mean", "median", "mode",
                                     "min", "max"],
                            copy=True)

pop_desc.to_csv(results_dir + 'block_stats/population_descriptives.csv',
                index=False)
area_desc = pd.DataFrame({'mean': pd.Series(area.mean()),
                          'median': pd.Series(area.median()),
                          'mode': pd.Series(area.mode()),
                          'min': pd.Series(area.min()),
                          'max': pd.Series(area.max())})
area_desc = area_desc.reindex(columns=["mean", "median", "mode",
                                       "min", "max"],
                              copy=True)
area_desc.to_csv(results_dir + 'block_stats/area_descriptives.csv',
                 index=False)
