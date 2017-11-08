"""Functions that are run on all files and/or do not depend on State class."""
from __future__ import division, print_function

import os
import us
import glob
import shutil

import pandas as pd
import numpy as np


def merge_pairwise_distances(clustered_stats_dir=None):
    """Return dataframe with mean pairwise distance per state in directory."""

    if clustered_stats_dir is None:
        from utils.settings import clustered_stats_dir

    # print clustered_stats_dir
    for state in us.states.STATES:
        state_abbr = state.abbr
        try:
            filename = glob.glob(clustered_stats_dir + state_abbr + '*.csv')
            if len(filename) > 0:
                filename = filename[0]
            stats_df = pd.read_csv(filename, index_col=False)
            # The following is to tidy up the df as they are not exactly
            # identical. This is now fixed, but some files are older:
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
                stats_df[stats_df['Is Cluster'] == False]['Is Cluster'].count()
            temp_mean_stats_df['Number of Congressional Districts'] = \
                stats_df[stats_df['Is Cluster'] == False]['Is Cluster'].count()

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


def update_best_stats_dir(best_stats_dir=None, clustered_means_dir=None,
                          clustered_stats_dir=None, best_plots_dir=None,
                          plots_dir=None, plots=False,
                          strip_unique_names=False):
    """Update the directory with the stats for the best solutions."""

    # if population is None:
    #     from settings import population
    if best_stats_dir is None:
        from settings import best_stats_dir
    if clustered_means_dir is None:
        from settings import clustered_means_dir
    if clustered_stats_dir is None:
        from settings import clustered_stats_dir
    if best_plots_dir is None:
        from settings import best_plots_dir
    if plots_dir is None:
        from settings import plots_dir

    best_result_per_state = {str(state.abbr): None for state in us.STATES}
    best_result = {str(state.abbr): None for state in us.STATES}
    # for filename in os.listdir(clustered_stats_dir):
    #     if filename.endswith(".csv") and not filename.startswith('.'):
    #         unique_filename_for_state = os.path.splitext(filename)[0]
    #         stats_filename = (clustered_stats_dir, unique_filename_for_state,
    #                           '.csv')
    #         means_filename = (clustered_means_dir, unique_filename_for_state,
    #                           '.csv')
    #         run_stats(unique_filename_for_state, population,
    #                   stats_filename, means_filename)
    for filename in os.listdir(clustered_means_dir):
        if filename.endswith(".csv") and not filename.startswith('.'):
            state = filename[0:2]
            state_df = pd.read_csv(clustered_means_dir + filename)
            cluster_result = \
                state_df[state_df['Is Cluster'] == True].as_matrix()[0][1]
            # district_result = state_df[state_df['Is Cluster']
            #                            is False].as_matrix()[0][1]
            if (best_result_per_state[state] is None
                    or cluster_result < best_result[state]):
                best_result_per_state[state] = filename
                best_result[state] = cluster_result

    # if not os.path.exists(best_stats_dir):
    #     os.makedirs(best_stats_dir)

    # Clear out old results:
    directory = glob.glob(best_stats_dir + '*')
    for old_file in directory:
        os.remove(old_file)

    directory = glob.glob(best_plots_dir + '*')
    for old_file in directory:
        os.remove(old_file)

    for state, filename in best_result_per_state.iteritems():
        if filename is not None:
            if strip_unique_names:
                save_filename = state
            else:
                save_filename = filename[:-4]
            # Copy the file over:
            shutil.copy2(clustered_stats_dir + filename, best_stats_dir +
                         save_filename + '.csv')
            # If we also want to put the plots into their own directory:
            if plots:
                shutil.copy2(plots_dir +
                             filename[:-4] + '.png', best_plots_dir +
                             save_filename + '.png')
                shutil.copy2(plots_dir +
                             filename[:-4] + '.pdf', best_plots_dir +
                             save_filename + '.pdf')
