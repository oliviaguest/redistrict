"""
Redistrict a state.

Download and process relevant data for a state from citySDK API.
Run the clustering on the state.
Graph the state and the results of the clustering.
"""
from __future__ import division
from __future__ import print_function

import argparse

from pdist import dist
from utils.state import State
from colorama import Fore, Style
# This file contains default directories and filenames:
from utils.settings import (alpha,
                            beta,
                            alpha_increment,
                            beta_increment,
                            max_runs,
                            alpha_max,
                            year)

# This is to handle the passed CLI arguments:
parser = argparse.ArgumentParser(
    description='Run weighted k-means clustering on a state.')
parser.add_argument('state', metavar='state', type=unicode,
                    help='A state to perform clustering on. It can be the full \
                    state name (use quotes if two words, e.g., \'New York\'), \
                    the FIPS code (e.g., 36), or the two-letter abbreviation \
                    (e.g., NY).')
parser.add_argument('-a', '--alpha', nargs='?',
                    type=float, default=alpha,
                    help='The initial alpha or weight parameter for clustering \
                    (default: ' + str(alpha) + ').')
parser.add_argument('-b', '--beta', nargs='?',
                    type=float, default=beta,
                    help='The beta or stickiness parameter for clustering \
                    (default: ' + str(beta) + ').')
parser.add_argument('-i', '--alpha-increment', nargs='?',
                    type=float, default=alpha_increment,
                    help='By how much to increment alpha per iteration \
                    (default: ' + str(alpha_increment) + ').')
parser.add_argument('-j', '--beta-increment', nargs='?',
                    type=float, default=beta_increment,
                    help='By how much to increment beta per run (default: ' +
                    str(beta_increment) + ').')
parser.add_argument('-r', '--max-runs', nargs='?',
                    type=float, default=max_runs,
                    help='After how many time-steps to stop individual \
                    clustering (default: ' + str(max_runs) + ').')
parser.add_argument('-f', '--filename', nargs='?',
                    help='Specify a filename to save to/load from.')
parser.add_argument('-v', '--verbose', action='store_true', default=True,
                    help='How much information to print.')
# NOTE: change above to False/remove default for this to work!

args = parser.parse_args()

# Get CLI alpha and beta, if applicable:
if args.beta is not None:
    if args.beta > 1.0 or args.beta < 0.0:
        print(Fore.RED + Style.BRIGHT + 'Invalid beta value: '
              + str(args.beta) + '. Beta must be in range [0.0, 1.0].')
        exit()
    beta = args.beta
if args.alpha is not None:
    if args.alpha < 0.0:
        print(Fore.RED + Style.BRIGHT + 'Invalid alpha value: '
              + str(args.alpha) + '. Alpha must be positive.')
        exit()
    alpha = args.alpha

# Also get the alpha increment:
if args.alpha_increment is not None:
    if args.alpha_increment < 0.0:
        print(Fore.RED + Style.BRIGHT + 'Invalid alpha increment value: ' +
              str(args.alpha_increment) +
              '. Alpha increment must be positive.')
        exit()
    alpha_increment = args.alpha_increment

# And the beta increment:
if args.beta_increment is not None:
    if args.beta_increment < 0.0:
        print(Fore.RED + Style.BRIGHT + 'Invalid beta increment value: ' +
              str(args.beta_increment) + '. Beta increment must be positive.')
        exit()
    beta_increment = args.beta_increment

# The maximum runs for clustering:
if args.max_runs is not None:
    max_runs = args.max_runs


###############################################################################
# Create the State object, we want to cluster:
###############################################################################
state = State(args.state, year,
              alpha=alpha, alpha_increment=alpha_increment,
              alpha_max=alpha_max,
              beta=beta, beta_increment=beta_increment,
              dist=dist.cdist,
              max_runs=max_runs,
              unique_filename_for_state=args.filename,
              verbose=args.verbose)

###############################################################################

##########################################################################
# Weighted k-means #######################################################
##########################################################################
state.cluster()

##########################################################################
# Statistics #############################################################
##########################################################################
state.run_stats()

##########################################################################
# Plotting ###############################################################
##########################################################################
state.plot()
