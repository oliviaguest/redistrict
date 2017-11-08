"""Get real districts and best solutions for each state in GeoJSON format."""
from __future__ import print_function

import os
import us
import glob
import numpy as np
import geopandas as gpd

from shapely.geometry import shape, mapping
from utils.functions import update_best_stats_dir
from utils.settings import (best_stats_dir,
                            clustered_geojson_results_dir,
                            site_dir, fig_dir, usa_shapefile_path)


def save_geojson_congressional_districts():
    """Create GeoJSONs with real districts for website and journal article."""
    usa = gpd.read_file(usa_shapefile_path)
    with open('USA.geojson', 'w') as outfile:
        outfile.write(usa.to_json())

    for state in us.STATES:
        print(state)
        cong_dist = usa[usa['STATEFP'].apply(int) == int(state.fips)]
        # cong_dist = cong_dist.to_crs({'init': 'epsg:3395'})
        with open(state.abbr + '.geojson', 'w') as outfile:
            outfile.write(cong_dist.to_json())


def save_geojson_best_solutions():
    """Create GeoJSONs with best solutions for website and journal article."""
    # We are saving the output to be used on the website (interactive maps)...
    output_dir = site_dir + 'json/geo/'
    # and to be used to make the figures for the article:
    output_dir_2 = fig_dir + '/geojson/'

    # Update the best stats and get their names:
    update_best_stats_dir()
    filenames = []
    for name in glob.glob(best_stats_dir + '*'):
        filenames.append(os.path.splitext(os.path.split(name)[1])[0])

    # Get the geoJSON files which correspond to the best results:
    skip = False
    for name in filenames:
        output_file = output_dir + name + '.geo.json'
        output_file_2 = output_dir_2 + name + '.geo.json'

        # Check if file is already present for small speed-up.
        if os.path.isfile(output_file):
            skip = True
        if os.path.isfile(output_file_2):
            if not skip:
                skip = False
            else:
                skip = True
                print('Files: ' + output_file + ' and ' + output_file_2 +
                      ', already exist. Skipping!')
                continue

        print(name)
        state = gpd.read_file(
            clustered_geojson_results_dir + name + '.geojson')
        mini_state = state[['geometry', 'Cluster']]
        del state
        clusters_df = mini_state.dissolve(by='Cluster')
        del mini_state

        # Now we have gotten rid of the columns we do not need, we can round
        # the coordinates:
        for index in clusters_df.index:
            print(index)
            geojson = mapping(clusters_df['geometry'].iloc[index])
            t = np.asarray(geojson['coordinates'])
            try:
                t = np.round(t, 6)
            except TypeError:
                pass
            try:
                for i, points in enumerate(t):
                    new_points = []
                    for p in points[0]:
                        p = list(p)
                        p[0] = np.round(p[0], 6)
                        p[1] = np.round(p[1], 6)
                        p = tuple(p)
                        new_points.append(p)
                    points[0] = new_points
            except TypeError:
                for i, points in enumerate(t):
                    new_points = []
                    for p in points:
                        p = list(p)
                        p[0] = np.round(p[0], 6)
                        p[1] = np.round(p[1], 6)
                        p = tuple(p)
                        new_points.append(p)
                    points = new_points
            geojson['coordinates'] = list(t)
            clusters_df.iloc[index, clusters_df.columns.get_loc(
                'geometry')] = shape(geojson)

        with open(output_file, 'w') as outfile:
            outfile.write(clusters_df.to_json())
        with open(output_file_2, 'w') as outfile:
            outfile.write(clusters_df.to_json())


if __name__ == '__main__':
    save_geojson_congressional_districts()
    save_geojson_best_solutions()
