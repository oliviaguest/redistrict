"""Utility functions."""

from __future__ import division, print_function
import os
import us
import glob
import time
import requests

import numpy as np
import pandas as pd
# import seaborn as sns
from pdist.pdist import c_mean_dist
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import Fore, Style

import weighted_k_means.wkmeans as km


class State():
    """State of the USA to be redistricted."""

    def __init__(self, state, year, alpha, alpha_increment, alpha_max, beta,
                 beta_increment, dist, max_runs,
                 blocks_2010_shapefile_dir=None, block_assignments_dir=None,
                 acs5_shapefile_dir=None, acs5_population_dir=None,
                 usa_shapefile_path=None, usa_csv_path=None,
                 prediction_geojson_dir=None, clustered_stats_dir=None,
                 clustered_means_dir=None, clustered_csv_results_dir=None,
                 clustered_geojson_results_dir=None, plots_dir=None,
                 cong_dist_plot_dir=None, unique_filename_for_state=None,
                 min_population=None, max_population=None, verbose=True):
        """Initialise State with estimated population for each block."""
        self.state = state
        self.year = year
        self.alpha = alpha
        self.alpha_increment = alpha_increment
        self.alpha_max = alpha_max
        self.beta = beta
        self.beta_increment = beta_increment
        self.dist = dist
        self.max_runs = max_runs
        self.blocks_2010_shapefile_dir = blocks_2010_shapefile_dir
        self.block_assignments_dir = block_assignments_dir
        self.acs5_shapefile_dir = acs5_shapefile_dir
        self.acs5_population_dir = acs5_population_dir
        self.usa_shapefile_path = usa_shapefile_path
        self.usa_csv_path = usa_csv_path
        self.prediction_geojson_dir = prediction_geojson_dir
        self.clustered_stats_dir = clustered_stats_dir
        self.clustered_means_dir = clustered_means_dir
        self.clustered_csv_results_dir = clustered_csv_results_dir
        self.clustered_geojson_results_dir = clustered_geojson_results_dir
        self.plots_dir = plots_dir
        self.cong_dist_plot_dir = cong_dist_plot_dir
        self.unique_filename_for_state = unique_filename_for_state
        self.min_population = min_population
        self.max_population = max_population
        self.verbose = verbose

        if blocks_2010_shapefile_dir is None:
            from settings import blocks_2010_shapefile_dir
        self.blocks_2010_shapefile_dir = blocks_2010_shapefile_dir
        if acs5_shapefile_dir is None:
            from settings import acs5_shapefile_dir
        self.acs5_shapefile_dir = acs5_shapefile_dir
        if acs5_population_dir is None:
            from settings import acs5_population_dir
        self.acs5_population_dir = acs5_population_dir
        if usa_shapefile_path is None:
            from settings import usa_shapefile_path
        self.usa_shapefile_path = usa_shapefile_path
        if usa_csv_path is None:
            from settings import usa_csv_path
        self.usa_csv_path = usa_csv_path
        if prediction_geojson_dir is None:
            from settings import prediction_geojson_dir
        self.prediction_geojson_dir = prediction_geojson_dir
        if clustered_geojson_results_dir is None:
            from settings import clustered_geojson_results_dir
        self.clustered_geojson_results_dir = clustered_geojson_results_dir
        if clustered_csv_results_dir is None:
            from settings import clustered_csv_results_dir
        self.clustered_csv_results_dir = clustered_csv_results_dir
        if plots_dir is None:
            from settings import plots_dir
        self.plots_dir = plots_dir
        if cong_dist_plot_dir is None:
            from settings import cong_dist_plot_dir
        self.cong_dist_plot_dir = cong_dist_plot_dir
        if clustered_stats_dir is None:
            from settings import clustered_stats_dir
        self.clustered_stats_dir = clustered_stats_dir
        if clustered_means_dir is None:
            from settings import clustered_means_dir
        self.clustered_means_dir = clustered_means_dir
        if max_population is None:
            from settings import max_population
        self.max_population = max_population
        if min_population is None:
            from settings import min_population
        self.min_population = min_population

        # Column name for the number of people in each block:
        self.population = 'Predicted ' + self.year + ' Population'

        try:
            self.state = us.states.lookup(self.state).abbr
        except AttributeError:
            self.state = self.state.zfill(2)
            try:
                self.state = us.states.lookup(self.state).abbr
            except AttributeError:
                print('FIPS code not assigned to a state!')
                exit()

        if self.unique_filename_for_state is None:
            self.create_unique_filename_for_state()

        self.clustered_geojson_filename = (
            self.clustered_geojson_results_dir +
            self.unique_filename_for_state + '.geojson')
        self.clustered_csv_filename = (self.clustered_csv_results_dir +
                                       self.unique_filename_for_state + '.csv')
        self.clusters_plot_filename = self.plots_dir + \
            self.unique_filename_for_state  # no extension!
        self.stats_filename = (self.clustered_stats_dir +
                               self.unique_filename_for_state + '.csv')
        self.means_filename = (self.clustered_means_dir +
                               self.unique_filename_for_state + '.csv')
        # The filename for the end product (a whole state with block groups and
        # their population and their congressional district):
        self.blocks_geojson_filename = (self.prediction_geojson_dir +
                                        self.state + '.geojson')
        # Get the FIPS code, used in some lookups:
        self.state_fips = us.states.lookup(self.state).fips

        # Load up the congressional districts for the whole of the USA:
        try:
            usa = pd.read_csv(self.usa_csv_path)
        except IOError:
            usa = gpd.read_file(self.usa_shapefile_path)
            usa = usa[['STATEFP', 'CD115FP', 'NAMELSAD', 'GEOID']]
            usa.to_csv(self.usa_csv_path)
            usa = pd.read_csv(self.usa_csv_path)
        # Extract the congressional districts for the current state:
        self.cong_dist = usa[usa['STATEFP'].apply(
            int) == int(self.state_fips)]
        # Throw it away since we only need the congressional districts for one
        # state:
        del usa

    def load_estimation(self):
        print('Attempting to load predicted population by block for:',
              Style.BRIGHT, self.state, self.blocks_geojson_filename,
              Style.RESET_ALL)
        self.df = gpd.read_file(self.blocks_geojson_filename)

    def get_estimation(self):
        """Load the data for each block for the state."""
        try:
            self.load_estimation()
        except IOError:
            print('\tNot found on disk...')
            print('\tDownloading and processing blocks and block groups for:',
                  self.state)
            # Open 2010 blocks for state (the population):
            print('\tOpening 2010 blocks shapefile for:', self.state)
            self.blocks_2010_shapefile_name = (self.blocks_2010_shapefile_dir,
                                               'tabblock2010_',
                                               self.state_fips,
                                               '_pophu.shp')
            blocks_df = gpd.read_file(self.blocks_2010_shapefile_name)
            blocks_df['BLOCKID10'] = blocks_df['BLOCKID10'].apply(str)

            def get_group_GEOID(blockGEOID):
                return blockGEOID[0:-3]
            blocks_df['Block Group GEOID'] = blocks_df['BLOCKID10'].apply(
                get_group_GEOID)
            assert (blocks_df['Block Group GEOID'].apply(len)).unique() == [12]
            # The filename for the whole state with block groups (without
            # population):
            print('\tOpening ACS5 block groups shapefile for:', self.state)
            block_groups_shapefile_name = (self.acs5_shapefile_dir + 'tl_' +
                                           self.year + '_' +
                                           self.state_fips + '_bg.shp')
            block_groups_df = gpd.read_file(block_groups_shapefile_name)
            # We will add to this the population per block group from the API.
            # Check GEOIDs are of length 12 because they have to be by
            # definition, see:
            # https://www.census.gov/geo/reference/geoidentifiers.html
            assert (block_groups_df['GEOID'].apply(len)).unique() == [12]
            # Check that blocks from 2010 fit inside block groups from 2015:
            # non_matching = 0
            # for bg_index, bg_row in block_groups_df.iterrows():
            #     for block_index, block_row in blocks_df.iterrows():
            #         intersection = (block_row['geometry'].intersection(
            #             bg_row['geometry'])).area
            #         if intersection > 0.0001:
            #             print(intersection, block_row['Block Group GEOID'],
            #                   bg_row['GEOID'])
            #             assert (block_row['Block Group GEOID'] ==
            #                     bg_row['GEOID'])
            #             non_matching += 1
            # print('non-matching block groups:', non_matching)
            # Check that the block groups and the blocks have the same exact
            # GEOIDs:
            try:
                # Assert that GEOIDs haven't changed, because if they have...
                assert set(block_groups_df['GEOID'].unique()) == set(
                    blocks_df['Block Group GEOID'].unique())
            except AssertionError:
                # then we need to correct that somehow!
                # Find the difference between the two lists of GEOIDs. These
                # are the block groups that have changed name:
                diff = list(set(block_groups_df['GEOID'].unique()).
                            symmetric_difference(set(blocks_df[
                                'Block Group GEOID'].
                                unique())))
                # Collect them up into a dataframe:
                for i, d in enumerate(diff):
                    try:
                        diff_block_groups = pd.concat(
                            [diff_block_groups,
                             block_groups_df[block_groups_df['GEOID'].
                                             astype(str).str.contains(d)]])
                    except NameError:
                        diff_block_groups = \
                            block_groups_df[block_groups_df['GEOID'].astype(
                                str).str.contains(d)]

            # There are block group changes in the following states:
            # Arizona, California, New York, Virginia
            # Code to get the total estimated population from the census files:
            # http://api.census.gov/data/2015/acs5/variables.html
            api_population = u'B01003_001E'
            # NB: this is not the same code in the shapefiles for each block
            # from the 2010 census, which instead uses 'POP10'.

            with open("census_api_key", "r") as key_file:
                api_key = key_file.read().replace('\n', '')

            # Get all the counties' IDs so we can d/l each county's population
            # per block group:
            def block_group_to_county_geoid(geoid):
                return geoid[2:5]
            block_groups_df['County'] = block_groups_df['GEOID'].apply(
                block_group_to_county_geoid)
            block_groups_for_state = (self.acs5_population_dir + self.state +
                                      '.json')
            try:
                print('\tAttempting to load ACS5 population ',
                      'for block groups by county for:', self.state)
                with open(block_groups_for_state, 'r') as infile:
                    population_df = pd.read_json(infile)
            except IOError:
                print('\t\tNot found on disk...')
                counties = block_groups_df['County'].unique()
                for county in counties:
                    print('\t\tDownloading block groups for county:', county)
                    url = ('http://api.census.gov/data/' + self.year +
                           '/acs5?get=NAME,' + api_population +
                           '&for=block+group:*&in=state:' +
                           self.state_fips + '+county:' + county +
                           '&key=' + api_key)
                    # Make a get request to get the population of each county
                    # per block group:
                    response = requests.get(url)
                    try:
                        data = response.json()
                    except ValueError:
                        # This is what happens when the server times out etc.
                        # This includes simplejson.decoder.JSONDecodeError
                        print('Decoding JSON has failed. Server response:',
                              response.status_code)
                        exit()
                    # Merge the df to get one huge df for the whole state which
                    # contains all the group blocks and their populations:
                    # columns as list and leaves data intact
                    columns = data.pop(0)
                    try:
                        # Try appending to previous population_df:
                        population_df = population_df.append(
                            pd.DataFrame(data, columns=columns),
                            ignore_index=True)
                    except NameError:
                        # There is no previous, so create it:
                        population_df = pd.DataFrame(data, columns=columns)
                population_df.to_json(block_groups_for_state)
            print('\tDone!')
            # Create the GEOID to associate population with shapefile for each
            # block group:

            def create_geoid(row):
                # GEOID Structure is defined as
                # STATE + COUNTY + TRACT + BLOCK GROUP = 2 + 3 + 6 + 1 = 12
                # see: https://www.census.gov/geo/reference/geoidentifiers.html
                # zero padding to conform to GEOID
                STATE = str(row['state']).zfill(2)
                COUNTY = str(row['county']).zfill(3)
                TRACT = str(row['tract']).zfill(6)
                BLOCK_GROUP = str(row['block group'])
                assert len(STATE + COUNTY + TRACT + BLOCK_GROUP) == 12
                return STATE + COUNTY + TRACT + BLOCK_GROUP
            population_df['GEOID'] = population_df.apply(create_geoid, axis=1)
            assert (block_groups_df['GEOID'].shape ==
                    population_df['GEOID'].shape)
            block_groups_df = block_groups_df.merge(population_df, on='GEOID')
            assert (block_groups_df['GEOID'].shape ==
                    population_df['GEOID'].shape)
            # Associate the appropriate population with the diff_block_groups
            # too:
            try:
                diff_block_groups = diff_block_groups.merge(
                    population_df, on='GEOID')
                # print('merging')
            except NameError:
                pass
            del population_df
            # Get rid of other columns, we are just interested in below for
            # each block group:
            block_groups_df = block_groups_df.loc[:, ['GEOID', api_population]]
            # We need to calculate the proportion of the populalion of 2010
            # block group that a constituent 2010 block has:
            # a. get the total population for each block group from 2010 and
            # associate each 2010 block with its total:
            blocks_df['POP10'] = pd.to_numeric(blocks_df['POP10'])
            blocks_df = blocks_df.join(blocks_df.groupby('Block Group GEOID')[
                                       'POP10'].sum(), on='Block Group GEOID',
                                       rsuffix=' for 2010 Block Group')
            blocks_df.rename(columns={'POP10 for 2010 Block Group':
                                      'Population for 2010 Block Group'},
                             inplace=True)
            # b. get the ratio of block population to respective block group
            # population:

            def calculate_ratio(row):
                try:
                    return row['POP10'] / \
                        row['Population for 2010 Block Group']
                except ZeroDivisionError:
                    return row['Population for 2010 Block Group']
            blocks_df['Population Ratio'] = blocks_df.apply(
                calculate_ratio, axis=1)
            # c. get the population for the 2015 ACS5 block groups and
            # associate them with their respective blocks:
            blocks_df = pd.merge(blocks_df, block_groups_df,
                                 left_on='Block Group GEOID',
                                 right_on='GEOID')
            blocks_df.rename(columns={api_population:
                                      'Population for ' + self.year +
                                      ' Block Group'},
                             inplace=True)
            assert (blocks_df['GEOID'] == blocks_df['Block Group GEOID']).all()
            del blocks_df['GEOID']
            blocks_df['Population for ' + self.year + ' Block Group'] =\
                pd.to_numeric(blocks_df['Population for ' + self.year +
                                        ' Block Group'])
            # assert (block_groups_df[api_population].sum() >
            #         blocks_df['POP10'].sum())
            # assert blocks_df['Population for 2010 Block Group'].sum() <\
            # blocks_df['Population for ' + year + ' Block Group'].sum()
            del block_groups_df  # we do not need this anymore
            # d. calculate the predicted population for the block based on the
            # above:
            blocks_df[self.population] = blocks_df['Population Ratio'] *\
                blocks_df['Population for ' + self.year + ' Block Group']
            # Calculate the centroids for each block required for clustering:

            def get_x(p): return p.x

            def get_y(p): return p.y
            blocks_df['Centroid Longitude'] = \
                blocks_df['geometry'].centroid.apply(get_x)
            blocks_df['Centroid Latitude'] = \
                blocks_df['geometry'].centroid.apply(get_y)
            try:
                # Do the same for the block groups which changed name, if
                # applicable:
                diff_block_groups['Centroid Longitude'] =\
                    diff_block_groups['geometry'].centroid.apply(get_x)
                diff_block_groups['Centroid Latitude'] =\
                    diff_block_groups['geometry'].centroid.apply(get_y)
                diff_block_groups.rename(
                    columns={'B01003_001E':
                             'Predicted ' + self.year + ' Population'},
                    inplace=True)
                diff_block_groups = diff_block_groups[[
                    'GEOID', 'geometry', 'Predicted ' + self.year +
                    ' Population', 'Centroid Longitude', 'Centroid Latitude']]
                # Also find which congressional district each of these new
                # block groups belong to:
                usa = gpd.read_file(self.usa_shapefile_path)
                cong_dist = usa[usa['STATEFP'].apply(
                    int) == int(self.state_fips)]
                del usa
                diff_block_groups_cong_dist = pd.DataFrame(
                    columns=[['GEOID', 'Congressional District']])
                # print(diff_block_groups_cong_dist.head())
                geoid = []
                cd = []
                for bg_index, bg_row in diff_block_groups.iterrows():
                    for dist_index, dist_row in cong_dist.iterrows():
                        bg_area = bg_row['geometry'].area
                        intersection_area = bg_row['geometry'].intersection(
                            dist_row['geometry']).area
                        if bg_area - intersection_area < 1e-10:
                            geoid.append(bg_row['GEOID'])
                            cd.append(dist_row['CD115FP'])
                diff_block_groups_cong_dist['GEOID'] = geoid
                diff_block_groups_cong_dist['Congressional District'] = cd
                del geoid
                del cd
                diff_block_groups = diff_block_groups.merge(
                    diff_block_groups_cong_dist, on='GEOID')
            except NameError:
                pass
            # Find which congressional district each block group for this state
            # belongs to. This is done using the block assignments files
            # downloaded from:
            # https://www.census.gov/geo/maps-data/data/baf.html
            # print '\tCalculating congressional districts for:', state
            block_assignments_df = pd.read_csv(self.block_assignments_dir +
                                               'National_CD115.txt',
                                               dtype={"BLOCKID": str,
                                                      "CD115": str})
            block_assignments_df.rename(
                columns={'BLOCKID': 'GEOID'}, inplace=True)
            blocks_df.rename(columns={'BLOCKID10': 'GEOID'}, inplace=True)
            blocks_df = blocks_df.merge(block_assignments_df, how='left',
                                        indicator=True)
            assert blocks_df['_merge'].unique() == ['both']
            del blocks_df['_merge']
            del block_assignments_df
            # We are done now, check that columns are what we expect before we
            # remove useless ones:
            assert list(blocks_df) ==\
                [u'BLOCKCE', 'GEOID', u'COUNTYFP10', u'HOUSING10', u'PARTFLG',
                 u'POP10', u'STATEFP10', u'TRACTCE10', 'geometry',
                 'Block Group GEOID', 'Population for 2010 Block Group',
                 'Population Ratio',
                 'Population for ' + self.year + ' Block Group',
                 'Predicted ' + self.year + ' Population',
                 'Centroid Longitude', 'Centroid Latitude', 'CD115']

            # Get rid of colums we do not need:
            blocks_df = blocks_df.loc[:, ['GEOID', 'CD115', 'geometry',
                                          ('Predicted ' + self.year +
                                           ' Population'),
                                          'Centroid Longitude',
                                          'Centroid Latitude']]
            # Give columns better names:
            blocks_df.rename(columns={'CD115': 'Congressional District',
                                      'Centroid Longitude':
                                      'Centroid Longitude',
                                      'Centroid Latitude':
                                      'Centroid Latitude'},
                             inplace=True)

            # Now to add the missing ones, if applicable:
            try:
                blocks_df = blocks_df.append(
                    diff_block_groups, ignore_index=True)
            except NameError:
                pass
            blocks_df['Congressional District'] = \
                blocks_df['Congressional District'].apply(str)
            blocks_df['Centroid Longitude'] = \
                blocks_df['Centroid Longitude'].apply(float)
            blocks_df['Centroid Latitude'] = \
                blocks_df['Centroid Latitude'].apply(float)
            blocks_df['Predicted ' + self.year + ' Population'] = \
                blocks_df['Predicted ' + self.year +
                          ' Population'].apply(float)
            # We are finally done, save the file!
            print('Saving file...')
            blocks_df = gpd.GeoDataFrame(blocks_df)
            with open(self.blocks_geojson_filename, 'w') as outfile:
                outfile.write(blocks_df.to_json())
            # In the rest of this code the block-wise dataframe with predicted
            # populations is called just df:
            self.df = blocks_df
        print('Done!')

    def create_run_and_return_kmeans(self):
        """
        Create a new instance of kmeans and run clustering on it.

        Provided the clustering did not result in any empty clusters, return a
        kmeans object instance. Otherwise, return None to indicate that
        clustering has been aborted.
        """
        print(Fore.CYAN + '\nParameter values:')
        print(Style.NORMAL + '\tCurrent Alpha:',
              Style.BRIGHT, self.alpha)
        print(Style.NORMAL + '\tCurrent Beta:',
              Style.BRIGHT,  self.beta)
        print(Style.NORMAL, '\tAlpha Increment:',
              Style.BRIGHT,  self.alpha_increment)
        print(Style.NORMAL, '\tBeta Increment:',
              Style.BRIGHT,  self.beta_increment,
              Style.RESET_ALL)
        # print(Style.RESET_ALL)
        # Initialise the class with some default values:
        self.kmeans = km.KPlusPlus(K=self.K, X=self.X, c=self.counts,
                                   alpha=self.alpha, beta=self.beta,
                                   dist=self.dist, max_runs=self.max_runs,
                                   label=self.unique_filename_for_state,
                                   verbose=self.verbose)
        # Initialise centroids using k-means++...
        self.kmeans.init_centers()
        # and run to find clusters:
        try:
            self.kmeans.find_centers(method='++')
        except ValueError:
            print('Cluster(s) cannot be empty!')
            return None
        return self.kmeans

    def get_clusters(self):
        print('Attempting to load k-means clusters for: ' + Style.BRIGHT +
              self.unique_filename_for_state + Style.RESET_ALL)
        # If we have it already from a previous run:
        self.df = gpd.read_file(self.clustered_geojson_filename)

    def cluster(self):
        """
        Run clustering until a stable solution is found.

        Take the required filenames, the congressional districts, and the
        parameters for a state and cluster until a stable solution is found.
        The returned dataframe contains the alpha and beta parameters as well
        as the cluster and congressional district each block belongs to.
        """
        # Print verbose information:
        if (self.verbose):
            print(Fore.GREEN + '\nFiles to be created (if not present):')
            print(Style.BRIGHT + '\n\t' + self.clustered_geojson_filename,
                  Style.NORMAL)
            print('\t ' +
                  'Results file from which plotting can occur \
                  (i.e., with geometry).')
            print(Style.BRIGHT + '\n\t' + self.clustered_csv_filename,
                  Style.NORMAL)
            print('\t ' +
                  'Results with assignments per block, no geometry \
                  (quicker to open).')
            print(Style.BRIGHT + '\n\t' + self.stats_filename, Style.NORMAL)
            print('\t Mean pairwise distances for clusters and districts.')
            print(Style.BRIGHT + '\n\t' + self.means_filename, Style.NORMAL)
            print('\t Mean of mean pairwise distances for state \
            (for comparison).')
            print(Style.BRIGHT + '\n\t' + self.clusters_plot_filename +
                  '.png and ' +
                  self.clusters_plot_filename + '.svg', Style.NORMAL)
            print('\t', 'Map of the state with clusters as districts.')
            print(Style.RESET_ALL)
            print(Fore.CYAN + 'Parameter values:')
            print(Style.NORMAL + '\tInitial Alpha:', Style.BRIGHT, self.alpha)
            print(Style.NORMAL + '\tInitial Beta:', Style.BRIGHT,  self.beta)
            print(Style.NORMAL + '\tAlpha Increment:',
                  Style.BRIGHT, self.alpha_increment)
            print(Style.NORMAL + '\tBeta Increment:',
                  Style.BRIGHT, self.beta_increment)
            print(Style.RESET_ALL)

        #######################################################################
        # Weighted k-means ####################################################
        #######################################################################

        # Perform our own clustering on the state at the block level.
        try:
            self.get_clusters()
        except IOError:
            ##################################################################
            # Get the estimated population for each block using the 2010 census
            # and the 2015 ACS5 census. The former decennial census from 2010
            # contains population per block because they actually go round the
            # USA and count essentially everybody. The ACS5 data is an
            # estimated population at the block group level (one above blocks).
            # Combining the two we can create estimations per block for more
            # recent years than 2010.
            self.get_estimation()
            ###################################################################
            # If the state's congressional district contains the string
            # 'at Large', then that means the state cannot be clustered, so we
            # can skip it:
            if 'at Large' in str(self.cong_dist['NAMELSAD']):
                print(self.state,
                      'has a single congressional distrct. No point \
                      clustering!')
                return None
            else:
                # Now to cluster the centroids for each block group for this
                # state:
                self.startTime = datetime.now()
                # Centroids for each census block group, the data we wish to
                # cluster:
                self.X = np.stack((np.asarray(self.df['Centroid Latitude'].
                                              apply(float)),
                                   np.asarray(self.df['Centroid Longitude'].
                                              apply(float))),
                                  axis=1)
                # Number of congressional districts that a state has:
                cong_dist_index = set(
                    self.df['Congressional District'].unique())
                # We need to get rid of ZZ because:
                # "In Connecticut, Illinois, and Michigan the state participant
                # did not assign the current (113th) congressional districts to
                # cover all of the state or equivalent area. The code "ZZ" has
                # been assigned to areas with no congressional district defined
                # (usually large water bodies). These unassigned areas are
                # treated within state as a single congressional district for
                # purposes of data presentation." from:
                # https://www.census.gov/rdo/data/113th_congressional_and_2012_state_legislative_district_plans.html
                cong_dist_index.discard('ZZ')

                # These are the valid congressional districts from:
                # https://www.census.gov/rdo/data/113th_congressional_and_2012_state_legislative_district_plans.html
                valid_cong_dist = set([str(i).zfill(2) for i in range(1, 54)])
                assert cong_dist_index <= valid_cong_dist
                self.K = len(cong_dist_index)
                del valid_cong_dist, cong_dist_index
                # How many people live in each block:
                self.counts = list(self.df[self.population].apply(float))
                self.people_per_cluster = 0  # init value
                while self.cluster_population_is_out_of_bounds():
                    # While we need to keep clustering (because the number of
                    # people in the clusters so far are out of bounds)...
                    # a) (re)initialise the class with some default values:
                    # b) (re)nitialise centroids using k-means++...
                    # and c) (re)run to find clusters.
                    # This does all of a-c above:
                    self.kmeans = self.create_run_and_return_kmeans()
                    if self.kmeans is None or self.alpha > self.alpha_max:
                        # It has not given us an answer and instead aborted so
                        # increment beta:
                        self.beta += self.beta_increment
                        if self.beta >= 1.0:
                            self.beta = 1.0 - self.beta_increment
                        # and we reset alpha, to alpha_increment because no
                        # point of going back to zero:
                        self.alpha = self.initial_alpha + self.alpha_increment
                        # and we need to spoof the
                        # cluster_population_is_out_of_bounds function to
                        # certainly loop again by giving it:
                        self.people_per_cluster = 0
                    else:
                        # It has returned a viable kmeans object.
                        # We also need to keep track of the people we have
                        # assigned to each cluster in order to be in bounds as
                        # defined by the real congressional district
                        # populations.
                        self.people_per_cluster = \
                            self.kmeans.counts_per_cluster
                        # We might as well increment alpha, in case we need to
                        # loop again.
                        self.alpha += self.alpha_increment
                # We're done so print some useful info:
                print('\tRun time: ', Style.BRIGHT,
                      datetime.now() - self.startTime, Style.RESET_ALL)
                # Below is the running total (all runs) of times that
                # clustering has been carried out:
                print('\tGrand Total Runs: ', Style.BRIGHT,
                      self.kmeans._cluster_points.calls, Style.RESET_ALL)
                print('\tAlpha: ', Style.BRIGHT,
                      self.kmeans.alpha, Style.RESET_ALL)
                print('\tBeta: ', Style.BRIGHT,
                      self.kmeans.beta, Style.RESET_ALL)
                print('\tState: ', Style.BRIGHT,
                      self.unique_filename_for_state, Style.RESET_ALL)
                # Save with extra column denoting the cluster membership of the
                # block group
                self.df['Cluster'] = self.kmeans.cluster_indices
                self.df['Alpha'] = [
                    self.kmeans.alpha for i in range(len(self.df))]
                self.df['Beta'] = [
                    self.kmeans.beta for i in range(len(self.df))]
                # Write it as a geojson file:
                print('Writing to: ' + self.clustered_geojson_filename)
                with open(self.clustered_geojson_filename, 'w') as outfile:
                    outfile.write(self.df.to_json())

        if os.path.isfile(self.clustered_csv_filename):
            print('Did not create CSV version for',
                  self.unique_filename_for_state,
                  'because pre-existing file found!')
        else:
            # Write it as a csv too:
            print('Writing to: ' + self.clustered_csv_filename)
            try:
                csv_df = self.df.drop(['id', 'geometry'], 1)
            except ValueError:
                csv_df = self.df.drop(['geometry'], 1)
            csv_df.to_csv(self.clustered_csv_filename)
        print('Done!')

    def create_unique_filename_for_state(self):
        """Create unique filename using state's name, and date and time."""
        unique_filename_for_state = (self.state +
                                     datetime.fromtimestamp(time.time()).
                                     strftime('_%Y_%m_%d_%H_%M_%S'))
        clustered_geojson_filename = (self.clustered_geojson_results_dir +
                                      unique_filename_for_state + '.geojson')
        # In the very unlikely case the file already exists:
        while os.path.exists(clustered_geojson_filename):
            unique_filename_for_state = (self.state +
                                         datetime.fromtimestamp(time.time()).
                                         strftime('_%Y_%m_%d_%H_%M_%S'))
            clustered_geojson_filename = (self.clustered_geojson_results_dir +
                                          unique_filename_for_state +
                                          '.geojson')
        self.unique_filename_for_state = unique_filename_for_state

    def cluster_population_is_out_of_bounds(self):
        """Check clusters are within bounds set by congressional districts."""
        if np.max(self.people_per_cluster) > self.max_population:
            return True
        elif np.min(self.people_per_cluster) < self.min_population:
            return True
        return False

    def run_stats(self):
        """Calculate and save stats on clusters and congressional districts."""
        try:
            print('Attempting to load stats for:', Style.BRIGHT,
                  self.unique_filename_for_state, Style.RESET_ALL)
            # If we have it already from a previous run (possible in the past
            # cos I did not save the means_df, so used this function to get
            # it):
            stats_df = pd.read_csv(self.stats_filename)
        except IOError:
            print('Calculating stats for:', Style.BRIGHT,
                  self.unique_filename_for_state, Style.RESET_ALL)
            stats_df = self.get_stats()
            stats_df.to_csv(self.stats_filename)
        stats_df['Weighted'] = stats_df['Population'] * stats_df[
            'Mean Pairwise Distance']
        means_df = stats_df.groupby(['Is Cluster'])['Weighted'].sum()\
            / stats_df.groupby(['Is Cluster'])['Population'].sum()
        means_df = means_df.to_frame()
        means_df.rename(columns={0: 'Mean Pairwise Distance'}, inplace=True)
        means_df.to_csv(self.means_filename)
        print('Done!')

    def get_stats(self, subsample_step=1, subsample_start=0):
        """Calculate stats for districts and clusters."""
        # Calculate some stats with respect to congressional districts and the
        # clusters and save them to a new separate dataframe:
        # cluster_ids = df['Cluster'].unique()
        try:
            self.df[[self.population]] = \
                self.df[[self.population]].apply(pd.to_numeric)
        except AttributeError:
            self.get_clusters()
        except IOError:
            print('Run clustering before computing stats!')
            return IOError
        # Stats for clusters:
        df1 = self.df.groupby(['Cluster'])[[self.population]].sum()
        df2 = self.df.groupby(['Cluster'])[['GEOID']].count()
        cluster_stats_df = pd.concat([df1, df2], axis=1)
        del df1, df2
        cluster_stats_df.columns = ['Population', 'Number of Blocks']
        is_cluster = [True for i in range(cluster_stats_df.shape[0])]
        cluster_stats_df['Is Cluster'] = is_cluster
        # Stats for congressional districts:
        df1 = self.df.groupby(['Congressional District'])[
            [self.population]].sum()
        df2 = self.df.groupby(['Congressional District'])[['GEOID']].count()
        cong_dist_stats_df = pd.concat([df1, df2], axis=1)
        del df1, df2
        cong_dist_stats_df.columns = ['Population', 'Number of Blocks']
        # We need to get rid of ZZ because:
        # "In Connecticut, Illinois, and Michigan the state participant did not
        # assign the current (113th) congressional districts to cover all of
        # the state or equivalent area. The code "ZZ" has been assigned to
        # areas with no congressional district defined (usually large water
        # bodies). These unassigned areas are treated within state as a single
        # congressional district for purposes of data presentation.
        # from:
        # https://www.census.gov/rdo/data/113th_congressional_and_2012_state_legislative_district_plans.htm
        try:
            cong_dist_stats_df.drop('ZZ', inplace=True)
        except ValueError:
            pass
        is_cluster = [False for i in range(cong_dist_stats_df.shape[0])]
        cong_dist_stats_df['Is Cluster'] = is_cluster

        def pairwise_distance(centroids):
            """Use centroids to calculate the average pairwise distance."""
            # Centroids for each census block group:
            X = np.stack(
                (np.asarray(centroids['Centroid Latitude'].apply(float)),
                 np.asarray(centroids['Centroid Longitude'].apply(float))),
                axis=1)
            mean = \
                c_mean_dist(X[subsample_start::subsample_step],
                            np.asarray(centroids[self.population].
                                       apply(float)
                                       [subsample_start::subsample_step]))
            return mean
        cluster_distance = self.df.groupby(['Cluster'])[
            ['Centroid Latitude', 'Centroid Longitude', self.population]].\
            apply(pairwise_distance)
        cong_dist_distance = self.df.groupby(['Congressional District'])[
            ['Centroid Latitude',
             'Centroid Longitude', self.population]].apply(pairwise_distance)
        try:
            cong_dist_distance.drop('ZZ', inplace=True)
        except ValueError:
            pass
        distance_df = pd.concat([cluster_distance, cong_dist_distance], axis=0)
        stats_df = pd.concat([cluster_stats_df, cong_dist_stats_df], axis=0)
        stats_df = pd.concat([stats_df, distance_df], axis=1)
        stats_df = stats_df.rename(columns={0: 'Mean Pairwise Distance'})
        stats_df['Cluster/Congressional District ID'] = stats_df.index
        return stats_df

    def plot(self):
        """Plot results and real congressional districts (if applicable)."""
        # If we haven't plotted the congressional districts already, plot them.
        # Extract the congressional districts for the current state:
        cong_dist_plot_filename = (self.cong_dist_plot_dir + self.state +
                                   '_cong_dist')
        if glob.glob(cong_dist_plot_filename + '*'):
            print('Did not create congressional districts plots for',
                  Style.BRIGHT,
                  self.state, Style.RESET_ALL,
                  'because pre-existing plots found!')
        else:
            print('Saving plot with congressional districts for:',
                  Style.BRIGHT,
                  self.state, Style.RESET_ALL)
            usa = gpd.read_file(self.usa_shapefile_path)
            cong_dist = usa[usa['STATEFP'].apply(int) == int(self.state_fips)]
            del usa
            cong_dist = cong_dist.to_crs({'init': 'epsg:3395'})
            cong_dist.plot(column='GEOID', linewidth=0.1)
            plt.axis('off')
            plt.savefig(cong_dist_plot_filename + '.png', bbox_inches='tight')
            plt.savefig(cong_dist_plot_filename + '.pdf', bbox_inches='tight')
            print('Done!')
        #######################################################################
        # We have a cluster ID associated with each block group, so now we can
        # plot the state with new congressional districts:
        if glob.glob(self.clusters_plot_filename + '*'):
            print('Did not create cluster plots for', Style.BRIGHT,
                  self.unique_filename_for_state, Style.RESET_ALL,
                  'because pre-existing plots found!')
        else:
            print('Saving cluster result plots by cluster for:', Style.BRIGHT,
                  self.unique_filename_for_state, Style.BRIGHT)
            clusters_df = self.df.dissolve(by='Cluster')
            clusters_df.crs = {'init': 'epsg:3395'}
            clusters_df = clusters_df.to_crs({'init': 'epsg:3395'})
            clusters_df.plot(linewidth=0.1)
            plt.axis('off')
            plt.savefig(self.clusters_plot_filename + '.png',
                        bbox_inches='tight')
            plt.savefig(self.clusters_plot_filename + '.pdf',
                        bbox_inches='tight')
            print('Done!')
