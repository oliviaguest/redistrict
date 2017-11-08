"""Utility functions that were in use before I made the State class.

Assume this is completely deprecated. I do not reccomend using it. I'm keeping
it here only for posterity and legacy reasons."""

from __future__ import division, print_function
import os
import us
import glob
import time
import shutil
import requests

import numpy as np
import pandas as pd
from pdist import dist
import geopandas as gpd
from datetime import datetime
from colorama import Fore, Style

import weighted_k_means.wkmeans as km


def cluster_population_is_out_of_bounds(points_per_cluster,
                                        min_population=None,
                                        max_population=None):
    """Check clusters are within bounds set by congressional districts."""
    if max_population is None:
        from settings import max_population
    if min_population is None:
        from settings import min_population

    if np.max(points_per_cluster) > max_population:
        return True
    elif np.min(points_per_cluster) < min_population:
        return True
    return False


def create_run_and_return_kmeans(K, X, c,
                                 alpha, beta,
                                 alpha_increment, beta_increment,
                                 max_runs, dist, label, verbose):
    """
    Create a new instance of kmeans, initialise it, and run clustering on it.

    Provided the clustering did not result in any empty clusters, return a
    kmeans object instance. Otherwise, return None to indicate that clustering
    has been aborted.
    """
    print(Fore.CYAN + '\nParameter values:')
    print(Style.NORMAL + '\tCurrent Alpha:', Style.BRIGHT, alpha)
    print(Style.NORMAL + '\tCurrent Beta:', Style.BRIGHT,  beta)
    print(Style.NORMAL, '\tAlpha Increment:', Style.BRIGHT,  alpha_increment)
    print(Style.NORMAL, '\tBeta Increment:', Style.BRIGHT,  beta_increment,
          Style.RESET_ALL)
    # print(Style.RESET_ALL)
    # Initialise the class with some default values:
    kmeans = km.KPlusPlus(K, X=X, c=c, alpha=alpha, beta=beta,
                          dist=dist, max_runs=max_runs, label=label,
                          verbose=verbose)
    # Initialise centroids using k-means++...
    kmeans.init_centers()
    # and run to find clusters:
    try:
        kmeans.find_centers(method='++')
    except ValueError:
        print('Cluster(s) cannot be empty!')
        return None
    return kmeans


def create_unique_filename_for_state(state, clustered_geojson_results_dir):
    """Create unique filename using the state's name, and the date and time."""
    unique_filename_for_state = (state +
                                 datetime.fromtimestamp(time.time()).
                                 strftime('_%Y_%m_%d_%H_%M_%S'))
    clustered_geojson_filename = (clustered_geojson_results_dir +
                                  unique_filename_for_state + '.geojson')
    # In the very unlikely case the file already exists:
    while os.path.exists(clustered_geojson_filename):
        unique_filename_for_state = (state +
                                     datetime.fromtimestamp(time.time()).
                                     strftime('_%Y_%m_%d_%H_%M_%S'))
        clustered_geojson_filename = (clustered_geojson_results_dir +
                                      unique_filename_for_state + '.geojson')
    return unique_filename_for_state


def run_clustering_on_state(unique_filename_for_state,
                            clustered_geojson_filename, clustered_csv_filename,
                            state, year, population,
                            blocks_geojson_filename, cong_dist,
                            alpha, beta, initial_alpha,
                            alpha_increment, beta_increment, max_runs, dist,
                            verbose, alpha_max, blocks_2010_shapefile_dir,
                            block_assignments_dir,
                            acs5_shapefile_dir, acs5_population_dir,
                            usa_shapefile_path):
    """
    Run clustering until a stable solution is found.

    Take the required filenames, the congressional districts, and the
    parameters for a state and cluster until a stable solution is found.
    The returned dataframe contains the alpha and beta parameters as well as
    the cluster and congressional district each block belongs to.
    """
    ##########################################################################
    # Weighted k-means #######################################################
    ##########################################################################

    # Perform our own clustering on the state at the block level.
    try:
        print('Attempting to load k-means clusters for: ' + Style.BRIGHT +
              unique_filename_for_state + Style.RESET_ALL)
        # If we have it already from a previous run:
        df = gpd.read_file(clustered_geojson_filename)
    except IOError:
        #######################################################################
        # Get the estimated population for each block using the 2010 census and
        # the 2015 ACS5 census. The former decennial census from 2010 contains
        # population per block because they actually go round the USA and count
        # essentially everybody. The ACS5 data is an estimated population at
        # the block group level (one above blocks). Combining the two we can
        # create estimations per block for more recent years than 2010.
        df = get_estimation(state, year, blocks_geojson_filename,
                            blocks_2010_shapefile_dir, block_assignments_dir,
                            acs5_shapefile_dir, acs5_population_dir,
                            usa_shapefile_path)
        #######################################################################
        # If the state's congressional district contains the string 'at Large',
        # then that means the state cannot be clustered, so we can skip it:
        if 'at Large' in str(cong_dist['NAMELSAD']):
            print(state,
                  'has a single congressional distrct! No point clustering.')
            exit()
        else:
            # Now to cluster the centroids for each block group for this state:
            startTime = datetime.now()
            # Centroids for each census block group:
            X = np.stack((np.asarray(df['Centroid Latitude'].apply(float)),
                          np.asarray(df['Centroid Longitude'].apply(float))),
                         axis=1)
            # Number of congressional districts that a state has:
            cong_dist_index = set(df['Congressional District'].unique())
            # We need to get rid of ZZ because:
            # "In Connecticut, Illinois, and Michigan the state participant did
            # not assign the current (113th) congressional districts to cover
            # all of the state or equivalent area. The code "ZZ" has been
            # assigned to areas with no congressional district defined (usually
            # large water bodies). These unassigned areas are treated within
            # state as a single congressional district for purposes of data
            # presentation." from:
            # https://www.census.gov/rdo/data/113th_congressional_and_2012_state_legislative_district_plans.html
            cong_dist_index.discard('ZZ')

            # These are the valid congressional districts from:
            # https://www.census.gov/rdo/data/113th_congressional_and_2012_state_legislative_district_plans.html
            valid_cong_dist = set([str(i).zfill(2) for i in range(1, 54)])
            assert cong_dist_index <= valid_cong_dist
            K = len(cong_dist_index)
            # How many people live in each block:
            counts = list(df[population].apply(float))
            people_per_cluster = 0  # init value
            while cluster_population_is_out_of_bounds(people_per_cluster):
                # While we need to keep clustering (because the number of
                # people in the clusters so far are out of bounds)...
                # a) (re)initialise the class with some default values:
                # b) (re)nitialise centroids using k-means++...
                # and c) (re)run to find clusters.
                # This does all of a-c above:
                kmeans = \
                    create_run_and_return_kmeans(K, X, counts,
                                                 alpha, beta,
                                                 alpha_increment,
                                                 beta_increment, max_runs,
                                                 dist,
                                                 unique_filename_for_state,
                                                 verbose)
                if kmeans is None or alpha > alpha_max:
                    # It has not given us an answer and instead aborted so
                    # increment beta:
                    beta += beta_increment
                    if beta >= 1.0:
                        beta = 1.0 - beta_increment
                    # and we reset alpha, to alpha_increment because no point
                    # of going back to zero really
                    alpha = initial_alpha + alpha_increment
                    # and we need to spoof the
                    # cluster_population_is_out_of_bounds function to certainly
                    # loop again by giving it:
                    people_per_cluster = 0
                else:
                    # It has returned a viable kmeans object.
                    # We also need to keep track of the people we have assigned
                    # to each cluster in order to be in bounds as defined by
                    # the real congressional district populations.
                    people_per_cluster = kmeans.counts_per_cluster
                    # We might as well increment alpha, in case we need to loop
                    # again.
                    alpha += alpha_increment
            # We're done so print some useful info:
            print('\tRun time: ', Style.BRIGHT,
                  datetime.now() - startTime, Style.RESET_ALL)
            # Below is the running total (all runs) of times that clustering
            # has been carried out:
            print('\tGrand Total Runs: ', Style.BRIGHT,
                  kmeans._cluster_points.calls, Style.RESET_ALL)
            print('\tAlpha: ', Style.BRIGHT, kmeans.alpha, Style.RESET_ALL)
            print('\tBeta: ', Style.BRIGHT, kmeans.beta, Style.RESET_ALL)
            print('\tState: ', Style.BRIGHT,
                  unique_filename_for_state, Style.RESET_ALL)
            # Save with extra column denoting the cluster membership of the
            # block group
            df['Cluster'] = kmeans.cluster_indices
            df['Alpha'] = [kmeans.alpha for i in range(len(df))]
            df['Beta'] = [kmeans.beta for i in range(len(df))]
            # Write it as a geojson file:
            with open(clustered_geojson_filename, 'w') as outfile:
                outfile.write(df.to_json())

    if os.path.isfile(clustered_csv_filename):
        print('Did not create CSV version for', unique_filename_for_state,
              'because pre-existing file found!')
    else:
        # Write it as a csv too:
        try:
            csv_df = df.drop(['id', 'geometry'], 1)
        except ValueError:
            csv_df = df.drop(['geometry'], 1)
        csv_df.to_csv(clustered_csv_filename)
        del csv_df
    print('Done!')
    return df


def merge_pairwise_distances(dir):
    """Return dataframe with mean pairwise distance per state in directory."""
    for state in us.states.STATES:
        state_abbr = state.abbr
        try:
            # The reason there is a try here is because not all states have
            # been run yet (so bear in mind this try needs to actually not pass
            # but raise error during runtime soon).
            filename = glob.glob(dir + state_abbr + '*.csv')
            if len(filename) > 0:
                filename = filename[0]
            stats_df = pd.read_csv(filename, index_col=False)

            # The following is to tidy up the df as they are not exactly
            # identical. This *should* be now fixed, but some files are older:
            try:
                stats_df['Cluster/Congressional District ID']
            except KeyError:
                stats_df = stats_df.rename(
                    columns={u'Unnamed: 0':
                             'Cluster/Congressional District ID'})
            try:
                del stats_df[u'Unnamed: 0']
            except KeyError:
                pass
            stats_df = stats_df[[u'Cluster/Congressional District ID',
                                 u'Population', u'Number of Blocks',
                                 u'Is Cluster', u'Mean Pairwise Distance']]
            stats_df['Cluster/Congressional District ID'] = \
                stats_df['Cluster/Congressional District ID'].astype(
                str)
            # We need to get rid of ZZ because:
            # "In Connecticut, Illinois, and Michigan the state participant did
            # not assign the current (113th) congressional districts to cover
            # all of the state or equivalent area. The code "ZZ" has been
            # assigned to areas with no congressional district defined (usually
            # large water bodies). These unassigned areas are treated within
            # state as a single congressional district for purposes of data
            # presentation." from:
            # https://www.census.gov/rdo/data/113th_congressional_and_2012_state_legislative_district_plans.htm
            if stats_df[stats_df['Cluster/Congressional District ID'] ==
                        'ZZ'].empty:
                None
            else:
                stats_df.drop(stats_df.index[
                    stats_df[stats_df['Cluster/Congressional District ID']
                             == 'ZZ'].index], inplace=True)
            # End of tidying.
            pop_sum = stats_df.groupby(['Is Cluster'])[
                'Population'].sum().as_matrix()
            assert np.fabs(pop_sum[0] - pop_sum[1]) <= 0.00001

            # Get the mean pairwise distance for both clusters and districts:
            stats_df['Weighted'] = stats_df['Population'] * \
                stats_df['Mean Pairwise Distance']
            temp_mean_stats_df = \
                stats_df.groupby(['Is Cluster'])['Weighted'].sum(
                ) / stats_df.groupby(['Is Cluster'])['Population'].sum()
            temp_mean_stats_df = temp_mean_stats_df.to_frame()
            temp_mean_stats_df.rename(
                columns={0: 'Mean Pairwise Distance'}, inplace=True)

            # Include a column for the state since we are merging the results
            # from all the states into a single dataframe:
            temp_mean_stats_df['State'] = state_abbr
            assert \
                stats_df[stats_df['Is Cluster']]['Is Cluster'].count()\
                == \
                stats_df[not stats_df['Is Cluster']]['Is Cluster'].count()
            temp_mean_stats_df['Number of Congressional Districts'] = \
                stats_df[not stats_df['Is Cluster']]['Is Cluster'].count()

            try:
                # Try appending to previous population_df:
                mean_stats_df = mean_stats_df.append(temp_mean_stats_df)
            except NameError:
                # Otherwise create a new df:
                mean_stats_df = temp_mean_stats_df
        except IOError:
            pass
    mean_stats_df = mean_stats_df.reset_index()
    n_cong_dist_df = mean_stats_df.pivot(
        index='State', columns='Is Cluster',
        values='Number of Congressional Districts')
    del n_cong_dist_df[False]
    n_cong_dist_df = n_cong_dist_df.rename(columns={
        True: 'Number of Congressional Districts'})
    mean_stats_df = mean_stats_df.pivot(
        index='State', columns='Is Cluster', values='Mean Pairwise Distance')
    mean_stats_df = mean_stats_df.rename(columns={
        True: 'Cluster Mean Pairwise Distance',
        False: 'Congressional District Mean Pairwise Distance'})
    mean_stats_df['Number of Congressional Districts'] = n_cong_dist_df[
        'Number of Congressional Districts']
    del n_cong_dist_df
    mean_stats_df = mean_stats_df.rename_axis(None)
    mean_stats_df = mean_stats_df.rename_axis(None, axis=1)
    return mean_stats_df


def update_best_stats_dir(best_stats_dir=None, population=None,
                          clustered_means_dir=None, clustered_stats_dir=None):
    """Update the directory with the stats for the best solutions."""

    if best_stats_dir is None:
        from utils.settings import best_stats_dir
    if population is None:
        from utils.settings import population
    if clustered_means_dir is None:
        from utils.settings import clustered_means_dir
    if clustered_stats_dir is None:
        from utils.settings import clustered_stats_dir

    best_result_per_state = {str(state.abbr): None for state in us.STATES}
    best_result = {str(state.abbr): None for state in us.STATES}
    for filename in os.listdir(clustered_stats_dir):
        if filename.endswith(".csv") and not filename.startswith('.'):
            unique_filename_for_state = os.path.splitext(filename)[0]
            stats_filename = (clustered_stats_dir, unique_filename_for_state,
                              '.csv')
            means_filename = (clustered_means_dir, unique_filename_for_state,
                              '.csv')
            run_stats(unique_filename_for_state, population,
                      stats_filename, means_filename)
    for filename in os.listdir(clustered_means_dir):
        if filename.endswith(".csv") and not filename.startswith('.'):
            state = filename[0:2]
            state_df = pd.read_csv(clustered_means_dir + filename)
            cluster_result = state_df[state_df['Is Cluster']
                                      is True].as_matrix()[0][1]
            # district_result = state_df[state_df['Is Cluster']
            #                            is False].as_matrix()[0][1]
            if (best_result_per_state[state] is None
                    or cluster_result < best_result[state]):
                best_result_per_state[state] = filename
                best_result[state] = cluster_result

    if not os.path.exists(best_stats_dir):
        os.makedirs(best_stats_dir)

    # Clear out old results:
    directory = glob.glob(best_stats_dir + '*')
    for old_file in directory:
        os.remove(old_file)

    for state, filename in best_result_per_state.iteritems():
        if filename is not None:
            # Copy the file over:
            shutil.copy2(clustered_stats_dir + filename, best_stats_dir)


def run_stats(unique_filename_for_state, population, stats_filename,
              means_filename, df=None):
    """Calculate and save stats on the clusters and congressional districts."""
    try:
        print('Attempting to load stats for:', Style.BRIGHT,
              unique_filename_for_state, Style.RESET_ALL)
        # If we have it already from a previous run:
        stats_df = pd.read_csv(stats_filename)
    except IOError:
        if df is None:
            print('Error: You have to provide dataframe if stats have not been\
                   already calculated!')
            exit()
        print('Calculating stats for:', Style.BRIGHT,
              unique_filename_for_state, Style.RESET_ALL)
        subsample_step = 1  # meaning sample everything
        stats_df = get_stats(df, population, subsample_step)
        stats_df.to_csv(stats_filename)

    stats_df['Weighted'] = stats_df['Population'] * stats_df[
        'Mean Pairwise Distance']
    means_df = stats_df.groupby(['Is Cluster'])['Weighted'].sum()\
        / stats_df.groupby(['Is Cluster'])['Population'].sum()
    means_df = means_df.to_frame()
    means_df.rename(columns={0: 'Mean Pairwise Distance'}, inplace=True)
    means_df.to_csv(means_filename)
    print('Done!')


def get_stats(df, population, subsample_step=1, subsample_start=0):
    """Function that calculates the stats for districts and clusters.

    Required arguments:

    df          -- the dataframe containing the details of the congressional
                   districts and k-means clusters

    population  -- the string name of the column in the dataframe that contains
                   the population (estimated) for each congressional district
                   and k-means cluster.
    """
    # Calculate some stats with respect to congressional districts and the
    # clusters and save them to a new separate dataframe:
    # cluster_ids = df['Cluster'].unique()
    df[[population]] = df[[population]].apply(pd.to_numeric)
    # Stats for clusters:
    df1 = df.groupby(['Cluster'])[[population]].sum()
    df2 = df.groupby(['Cluster'])[['GEOID']].count()
    cluster_stats_df = pd.concat([df1, df2], axis=1)
    del df1, df2
    cluster_stats_df.columns = ['Population', 'Number of Blocks']
    is_cluster = [True for i in range(cluster_stats_df.shape[0])]
    cluster_stats_df['Is Cluster'] = is_cluster
    # Stats for congressional districts:
    df1 = df.groupby(['Congressional District'])[[population]].sum()
    df2 = df.groupby(['Congressional District'])[['GEOID']].count()
    # print df['Congressional District'].unique()

    cong_dist_stats_df = pd.concat([df1, df2], axis=1)
    del df1, df2
    cong_dist_stats_df.columns = ['Population', 'Number of Blocks']
    # We need to get rid of ZZ because:
    # "In Connecticut, Illinois, and Michigan the state participant did not
    # assign the current (113th) congressional districts to cover all of the
    # state or equivalent area. The code "ZZ" has been assigned to areas with
    # no congressional district defined (usually large water bodies). These
    # unassigned areas are treated within state as a single congressional
    # district for purposes of data presentation.
    # from:
    # https://www.census.gov/rdo/data/113th_congressional_and_2012_state_legislative_district_plans.htm
    try:
        cong_dist_stats_df.drop('ZZ', inplace=True)
    except ValueError:
        pass
    is_cluster = [False for i in range(cong_dist_stats_df.shape[0])]
    cong_dist_stats_df['Is Cluster'] = is_cluster

    def pairwise_distance(centroids):
        """Take the centroid and calculate the average pairwise distance."""
        # Centroids for each census block:

        X = np.stack(
            (np.asarray(centroids['Centroid Latitude'].apply(float)),
             np.asarray(centroids['Centroid Longitude'].apply(float))),
            axis=1)

        mean = \
            dist.c_mean_dist(X[subsample_start::subsample_step],
                             np.asarray(centroids[population].apply(float)
                                        [subsample_start::subsample_step]))

        return mean
    cluster_distance = df.groupby(['Cluster'])[
        ['Centroid Latitude', 'Centroid Longitude', population]].apply(
            pairwise_distance)
    cong_dist_distance = df.groupby(['Congressional District'])[
        ['Centroid Latitude',
         'Centroid Longitude', population]].apply(pairwise_distance)
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


def get_estimation(state, year, blocks_geojson_filename,
                   blocks_2010_shapefile_dir, block_assignments_dir,
                   acs5_shapefile_dir, acs5_population_dir,
                   usa_shapefile_path):
    """Return estimated population for each block given a state.

    Required arguments:
    state                     -- name, 2-letter abbreviation, or FIPS code,
                                 e.g., 'Rhode Island', 'RI', or 44.
    year                      -- year for the ACS5 data.
    blocks_geojson_filename   -- filename to save the output.
    blocks_2010_shapefile_dir -- directory for decennial shapefiles.
    block_assignments_dir     -- directory for decennial block assignments.
    acs5_shapefile_dir        -- directory for ACS5 shapefiles.
    acs5_population_dir       -- directory for ACS5 population data.
    """
    try:
        state = us.states.lookup(state).abbr
    except AttributeError:
        state = state.zfill(2)
        try:
            state = us.states.lookup(state).abbr
        except AttributeError:
            print('Warning: FIPS code not assigned to a state!')
            exit()
    state_fips = us.states.lookup(state).fips

    # Column name for the number of people in each block:
    population = 'Predicted ' + year + ' Population'
    try:
        print('Attempting to load predicted population by block for:',
              Style.BRIGHT, state, blocks_geojson_filename, Style.RESET_ALL)
        df = gpd.read_file(blocks_geojson_filename)
    except IOError:
        print('\tNot found on disk...')
        print('\tDownloading and processing blocks and block groups for:',
              state)
        # Open 2010 blocks for state (the population):
        print('\tOpening 2010 blocks shapefile for:', state)
        blocks_2010_shapefile_name = (blocks_2010_shapefile_dir,
                                      'tabblock2010_', state_fips,
                                      '_pophu.shp')
        blocks_df = gpd.read_file(blocks_2010_shapefile_name)
        blocks_df['BLOCKID10'] = blocks_df['BLOCKID10'].apply(str)

        def get_group_GEOID(blockGEOID):
            return blockGEOID[0:-3]
        blocks_df['Block Group GEOID'] = blocks_df['BLOCKID10'].apply(
            get_group_GEOID)
        assert (blocks_df['Block Group GEOID'].apply(len)).unique() == [12]
        # The filename for the whole state with block groups (without
        # population):
        print('\tOpening ACS5 block groups shapefile for:', state)
        block_groups_shapefile_name = (acs5_shapefile_dir + 'tl_' + year + '_'
                                       + state_fips + '_bg.shp')
        block_groups_df = gpd.read_file(block_groups_shapefile_name)
        # We will add to this the population per block group from the API.
        # Check GEOIDs are of length 12 because they have to be by definition,
        # see: https://www.census.gov/geo/reference/geoidentifiers.html
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
        #             assert block_row['Block Group GEOID'] == bg_row['GEOID']
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
            # Find the difference between the two lists of GEOIDs. These are
            # the block groups that have changed name:
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
        # NB: this is not the same code in the shapefiles for each block from
        # the 2010 census, which instead uses 'POP10'.

        with open("census_api_key", "r") as key_file:
            api_key = key_file.read().replace('\n', '')

        # Get all the counties' IDs so we can d/l each county's population per
        # block group:
        def block_group_to_county_geoid(geoid):
            return geoid[2:5]
        block_groups_df['County'] = block_groups_df['GEOID'].apply(
            block_group_to_county_geoid)
        block_groups_for_state = acs5_population_dir + state + '.json'
        try:
            print('\tAttempting to load ACS5 population ',
                  'for block groups by county for:', state)
            with open(block_groups_for_state, 'r') as infile:
                population_df = pd.read_json(infile)
        except IOError:
            print('\t\tNot found on disk...')
            counties = block_groups_df['County'].unique()
            for county in counties:
                print('\t\tDownloading block groups for county:', county)
                url = 'http://api.census.gov/data/' + year + '/acs5?get=NAME,'\
                    + api_population + '&for=block+group:*&in=state:' + \
                    state_fips + '+county:' + county + '&key=' + api_key
                # Make a get request to get the population of each county per
                # block group:
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
                columns = data.pop(0)  # columns as list and leaves data intact
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
        assert block_groups_df['GEOID'].shape == population_df['GEOID'].shape
        block_groups_df = block_groups_df.merge(population_df, on='GEOID')
        assert block_groups_df['GEOID'].shape == population_df['GEOID'].shape
        # Associate the appropriate population with the diff_block_groups too:
        try:
            diff_block_groups = diff_block_groups.merge(
                population_df, on='GEOID')
            # print('merging')
        except NameError:
            pass
        del population_df
        # Get rid of other columns, we are just interested in below for each
        # block group:
        block_groups_df = block_groups_df.loc[:, ['GEOID', api_population]]
        # We need to calculate the proportion of the populalion of 2010 block
        # group that a constituent 2010 block has:
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
                return row['POP10'] / row['Population for 2010 Block Group']
            except ZeroDivisionError:
                return row['Population for 2010 Block Group']
        blocks_df['Population Ratio'] = blocks_df.apply(
            calculate_ratio, axis=1)
        # c. get the population for the 2015 ACS5 block groups and associate
        # them with their respective blocks:
        blocks_df = pd.merge(blocks_df, block_groups_df,
                             left_on='Block Group GEOID',
                             right_on='GEOID')
        blocks_df.rename(columns={api_population:
                                  'Population for ' + year + ' Block Group'},
                         inplace=True)
        assert (blocks_df['GEOID'] == blocks_df['Block Group GEOID']).all()
        del blocks_df['GEOID']
        blocks_df['Population for ' + year + ' Block Group'] =\
            pd.to_numeric(blocks_df['Population for ' + year + ' Block Group'])
        # assert (block_groups_df[api_population].sum() >
        #         blocks_df['POP10'].sum())
        # assert blocks_df['Population for 2010 Block Group'].sum() <\
        # blocks_df['Population for ' + year + ' Block Group'].sum()
        del block_groups_df  # we do not need this anymore
        # d. calculate the predicted population for the block based on the
        # above:
        blocks_df[population] = blocks_df['Population Ratio'] *\
            blocks_df['Population for ' + year + ' Block Group']
        # Calculate the centroids for each block required for clustering:

        def get_x(p): return p.x

        def get_y(p): return p.y
        blocks_df['Centroid Longitude'] = blocks_df['geometry'].centroid.apply(
            get_x)
        blocks_df['Centroid Latitude'] = blocks_df['geometry'].centroid.apply(
            get_y)
        try:
            # Do the same for the block groups which changed name, if
            # applicable:
            diff_block_groups['Centroid Longitude'] =\
                diff_block_groups['geometry'].centroid.apply(get_x)
            diff_block_groups['Centroid Latitude'] =\
                diff_block_groups['geometry'].centroid.apply(get_y)
            diff_block_groups.rename(
                columns={'B01003_001E': 'Predicted 2015 Population'},
                inplace=True)
            diff_block_groups = diff_block_groups[[
                'GEOID', 'geometry', 'Predicted 2015 Population',
                'Centroid Longitude', 'Centroid Latitude']]
            # Also find which congressional district each of these new block
            # groups belong to:
            usa = gpd.read_file(usa_shapefile_path)
            cong_dist = usa[usa['STATEFP'].apply(
                int) == int(state_fips)]
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
        # belongs to. This is done using the block assignments files downloaded
        # from:
        # https://www.census.gov/geo/maps-data/data/baf.html
        # print '\tCalculating congressional districts for:', state
        block_assignments_df = pd.read_csv(block_assignments_dir +
                                           'National_CD115.txt',
                                           dtype={"BLOCKID": str,
                                                  "CD115": str})
        block_assignments_df.rename(columns={'BLOCKID': 'GEOID'}, inplace=True)
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
             'Population Ratio', 'Population for 2015 Block Group',
             'Predicted 2015 Population', 'Centroid Longitude',
             'Centroid Latitude', 'CD115']

        # Get rid of colums we do not need:
        blocks_df = blocks_df.loc[:, ['GEOID', 'CD115', 'geometry',
                                      'Predicted 2015 Population',
                                      'Centroid Longitude',
                                      'Centroid Latitude']]
        # Give columns better names:
        blocks_df.rename(columns={'CD115': 'Congressional District',
                                  'Centroid Longitude': 'Centroid Longitude',
                                  'Centroid Latitude': 'Centroid Latitude'},
                         inplace=True)

        # Now to add the missing ones, if applicable:
        try:
            blocks_df = blocks_df.append(diff_block_groups, ignore_index=True)
        except NameError:
            pass
        blocks_df['Congressional District'] = \
            blocks_df['Congressional District'].apply(str)
        blocks_df['Centroid Longitude'] = \
            blocks_df['Centroid Longitude'].apply(float)
        blocks_df['Centroid Latitude'] = \
            blocks_df['Centroid Latitude'].apply(float)
        blocks_df['Predicted 2015 Population'] = \
            blocks_df['Predicted 2015 Population'].apply(float)
        # We are finally done, save the file!
        print('Saving file...')
        blocks_df = gpd.GeoDataFrame(blocks_df)
        with open(blocks_geojson_filename, 'w') as outfile:
            outfile.write(blocks_df.to_json())
        # In the rest of this code the block-wise dataframe with predicted
        # populations is called just df:
        df = blocks_df
        del blocks_df
    print('Done!')
    return df
