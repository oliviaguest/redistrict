"""Find the min and max populations per district."""

import os
import pandas as pd
import geopandas as gpd
from utils.settings import prediction_geojson_dir, population

# Directory to read/write the predicted population by district for each state:
predicted_district_population_dir =\
    './census-data/predicted-district-populations/'
# This loop will not do anything if all the files in prediction_geojson_dir are
# present in predicted_district_population_dir. So if every state has a
# predicted geojson and every state has a predicted population by distrct,
# nothing will happen in this loop:
for filename in os.listdir(prediction_geojson_dir):
    # Ignore the README file:
    if filename == 'README.md':
        continue
    # Get the state:
    state = os.path.splitext(filename)[0]
    # This is the file we will save:
    population_by_district_file = predicted_district_population_dir + \
        'state/' + state + '.csv'
    # If it already exists, go to the next state:
    if os.path.isfile(population_by_district_file):
        print state, 'already done!'
        continue
    # state = 'RI'
    # filename = state+'.geojson'
    blocks_geojson_filename = prediction_geojson_dir + filename
    print 'Opening:', filename
    df = gpd.read_file(blocks_geojson_filename)
    print 'Done!'
    population_by_district = df.groupby(['Congressional District'])[
        [population]].sum()
    # del df
    population_by_district['State'] = state
    # print population_by_district.head()

    # del population_by_district
    population_by_district.to_csv(population_by_district_file)

for filename in os.listdir(predicted_district_population_dir + 'state/'):
    # Go through again this time in order to create a df with all states'
    # populations in a single file:
    state = os.path.splitext(filename)[0]
    # The file with the populaltion of a state by district:
    population_by_district_file = predicted_district_population_dir + 'state/'\
        + state + '.csv'

    # print population_by_district_file
    df = pd.read_csv(population_by_district_file)
    df['Congressional District'] = df['Congressional District'].astype(str)
    if df[df['Congressional District'] == 'ZZ'].empty:
        None
    else:
        df.drop(df.index[
            df[df['Congressional District']
               == 'ZZ'].index], inplace=True)
    # print df.head()
    try:
        district_populations_df = district_populations_df.append(df)
    except NameError:
        district_populations_df = df
    # print district_populations_df.head()
del df
district_populations_df.reset_index(inplace=True)
district_populations_df.to_csv(predicted_district_population_dir +
                               'overall/all_states_district_populations.csv',
                               index=False)

max_population = district_populations_df[population].max()
min_population = district_populations_df[population].min()
print max_population, min_population

max_df = district_populations_df.loc[district_populations_df[population]
                                     .idxmax()]
min_df = district_populations_df.loc[district_populations_df[population]
                                     .idxmin()]
print 'max:', max_df
print 'min:', min_df

# exit()
min_max_df = pd.concat([min_df, max_df], axis=1).transpose()
# Create a df with just the min and max values of the previous df which has
# all:
# min_max_df = min_df.join(max_df, on='Congressional District', how='left')
min_max_df.to_csv(predicted_district_population_dir +
                  'overall/min_max_populations.csv', index=False)
