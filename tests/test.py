"""Testing."""
import os
import random
import unittest

import numpy as np

from pdist.pdist import cdist
from pandas import DataFrame
from utils.state import State


class TestState(unittest.TestCase):

    def test_init(self):
        state = State('RI', '2015', 0, 0.5, 10, 0, 0.1, cdist, 10)
        self.assertTrue(state.state == 'RI')
        self.assertTrue(state.year == '2015')
        self.assertTrue(state.alpha == 0)
        self.assertTrue(state.alpha_increment == 0.5)
        self.assertTrue(state.alpha_max == 10)
        self.assertTrue(state.beta == 0)
        self.assertTrue(state.beta_increment == 0.1)
        self.assertTrue(state.dist == cdist)
        self.assertTrue(state.max_runs == 10)
        self.assertTrue(state.population == 'Predicted 2015 Population')
        self.assertIsInstance(state.cong_dist, DataFrame)

    def test_create_unique_filename_for_state(self):
        state = State('RI', '2015', 0, 0.5, 10, 0, 0.1, cdist, 10)
        # Check new filename is a unicode string:
        self.assertIsInstance(state.unique_filename_for_state, unicode)

        # Check it really is unique:
        self.assertFalse(os.path.isfile(state.clustered_geojson_results_dir +
                                        state.unique_filename_for_state))
        self.assertFalse(os.path.isfile(state.clustered_csv_results_dir +
                                        state.unique_filename_for_state))

    def test_cluster_population_is_out_of_bounds(self):
        # Single values of population:
        state = State('RI', '2015', 0, 0.5, 10, 0, 0.1, cdist, 10,
                      min_population=1, max_population=100)
        state.people_per_cluster = 0
        self.assertTrue(state.cluster_population_is_out_of_bounds())
        state.people_per_cluster = 101
        self.assertTrue(state.cluster_population_is_out_of_bounds())
        state.people_per_cluster = 1
        self.assertFalse(state.cluster_population_is_out_of_bounds())
        state.people_per_cluster = 100
        self.assertFalse(state.cluster_population_is_out_of_bounds())

        test = ((state.max_population -
                 state.min_population) * random.random() +
                state.min_population)
        state.people_per_cluster = test
        self.assertFalse(state.cluster_population_is_out_of_bounds())

        # Multiple values of population:
        state.people_per_cluster = [0, 100, 50]
        self.assertTrue(state.cluster_population_is_out_of_bounds())
        state.people_per_cluster = [0, 101, 50]
        self.assertTrue(state.cluster_population_is_out_of_bounds())
        state.people_per_cluster = [1, 101, 50]
        self.assertTrue(state.cluster_population_is_out_of_bounds())
        state.people_per_cluster = [1, 100, 50]
        self.assertFalse(state.cluster_population_is_out_of_bounds())
        state.people_per_cluster = [(state.max_population -
                                     state.min_population) * random.random() +
                                    state.min_population,
                                    (state.max_population -
                                     state.min_population) * random.random() +
                                    state.min_population]
        self.assertFalse(state.cluster_population_is_out_of_bounds())

    def test_get_estimation(self):
        state = State('RI', '2015', 0, 0.5, 10, 0, 0.1, cdist, 10)
        state.get_estimation()
        self.assertIsInstance(state.df, DataFrame)
        self.assertTrue('Centroid Latitude' in state.df.columns)
        self.assertTrue('Centroid Longitude' in state.df.columns)
        self.assertTrue('Congressional District' in state.df.columns)
        self.assertTrue('GEOID' in state.df.columns)
        self.assertTrue('Predicted 2015 Population' in state.df.columns)
        self.assertTrue('geometry' in state.df.columns)

    def test_run_stats(self):
        state = State('RI', '2015', 0, 0.5, 10, 0, 0.1, cdist, 10,
                      unique_filename_for_state='RI_2017_10_07_14_17_05')
        state.run_stats()
        self.assertTrue(os.path.isfile(state.stats_filename))
        self.assertTrue(os.path.isfile(state.means_filename))

    def test_get_stats(self):
        state = State('RI', '2015', 0, 0.5, 10, 0, 0.1, cdist, 10,
                      unique_filename_for_state='RI_2017_10_07_14_17_05')
        stats_df = state.get_stats()
        self.assertIsInstance(stats_df, DataFrame)
        self.assertTrue('Population' in stats_df.columns)
        self.assertTrue('Number of Blocks' in stats_df.columns)
        self.assertTrue('Is Cluster' in stats_df.columns)
        self.assertTrue('Mean Pairwise Distance' in stats_df.columns)
        self.assertTrue('Cluster/Congressional District ID' in
                        stats_df.columns)

    def plot(self):
        state = State('RI', '2015', 0, 0.5, 10, 0, 0.1, cdist, 10,
                      unique_filename_for_state='RI_2017_10_07_14_17_05')
        state.plot()
        self.assertTrue(os.path.isfile(state.cong_dist_plot_dir + state.state +
                                       '_cong_dist'))
        self.assertTrue(os.path.isfile(state.clusters_plot_filename + '.png'))
        self.assertTrue(os.path.isfile(state.clusters_plot_filename + '.pdf'))

    def test_create_run_and_return_kmeans(self):
        state = State('RI', '2015', 0, 0.5, 10, 0, 0.1, cdist, 10)
        state.load_estimation()
        state.K = 2
        state.X = np.stack((np.asarray(state.df['Centroid Latitude'].
                                       apply(float)),
                            np.asarray(state.df['Centroid Longitude'].
                                       apply(float))),
                           axis=1)
        state.counts = list(state.df[state.population].apply(float))
        state.create_run_and_return_kmeans()

    def test_cluster(self):
        # Below are known values for Alabama which allow convergence:
        state = State('AL', '2015', 2, 0.5, 30, 0.5, 0.1, cdist, 200)
        state.cluster()
        self.assertTrue(os.path.isfile(state.clustered_geojson_filename))
        self.assertTrue(os.path.isfile(state.clustered_csv_filename))


if __name__ == '__main__':
    unittest.main()
