"""Default values for parameters, filenames, and directories."""

import pandas as pd
########################################################
# Global values, mainly for filenames and directories. #
########################################################
# Directory in which shapefiles have been downloaded
# d/l from:
# https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2015&layergroup=Block+Groups
# in link above remember to match year (2015) to the year of the census data if
# changed.

# Data directory:
data_dir = './data/'

acs5_shapefile_dir = './data/census/acs5-shapefiles/'
# Shapefiles for each block from the 2010 census:
blocks_2010_shapefile_dir = './data/census/2010-census-shapefiles/'
# Block assignments from 2010 census used for redistricting:
block_assignments_dir = './data/census/block-assignments/'
# Directory in/from which geoJSON files with everything we need will be
# saved/loaded to/from:
prediction_geojson_dir = './data/census/prediction-geojson/'
# Directory in which to save population data:
acs5_population_dir = './data/census/acs5-population/'
# Relative path for the shapefile of the whole USA with congressional
# districts:
usa_shapefile_path = './data/census/usa-cd-shapefile/tl_2016_us_cd115.shp'
# Quicker to open version of above (without geometry, etc.):
usa_csv_path = './data/census/usa-cd-shapefile/tl_2016_us_cd115.csv'
# Path to file where the min and max populations for all states are:
min_max_populations_path = './data/census/predicted-district-populations/' + \
                           'overall/min_max_populations.csv'
# The year for the data from the ACS5 census:
year = '2015'
# Column name for the number of people in each block:
population = 'Predicted ' + year + ' Population'

# Results directory:
results_dir = './results/'

# Directory to save the best results per state after clustering multiple times:
best_stats_dir = './results/best_stats/'
# Same as above but for the plots:
best_plots_dir = './results/best_plots/'

# Diretory in which to save results of clustering:
clustered_geojson_results_dir = './results/geojson_results/'
clustered_csv_results_dir = './results/csv_results/'
# Directory in which to save stats on the clustering:
clustered_stats_dir = './results/csv_stats/'
clustered_means_dir = './results/csv_means/'
# Directory in which to save plots that show results of clustering:
plots_dir = './results/cluster_plots/'
# Directory in which to save plots of the current congressional districts:
cong_dist_plot_dir = './results/congressional_district_plots/'

# Directory to load corruption data from:
corruption_data_dir = './data/corruption/'

# Exponent used to calculate the scaling factor:
alpha = 0.0
initial_alpha = alpha
alpha_max = 50
# Stickiness parameter used during time-averaging (a constant):
beta = 0.5
# How much to increment value of alpha when iterating over weighted k-means:
alpha_increment = 1.0
# How much to increment value of beta when iterating over weighted k-means:
beta_increment = 0.1
# When to stop clustering (to avoid infinite loop)
max_runs = 200
# We need population constraints per district (from 2015 predicted
# population):
try:
    min_max_df = pd.read_csv(min_max_populations_path)
except NameError:
    print('Please run find_min_and_max.py!')
    exit()
# District with the most people: Montana At-large
max_population = min_max_df[population].max()
# District with the fewest people: Rhode Island's 1st
min_population = min_max_df[population].min()
del min_max_df

# Website directory:
site_dir = './gerrymandering-site/'
# Figures directory:
fig_dir = './fig/'
