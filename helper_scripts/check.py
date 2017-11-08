"""Check you have run clustering on every state with 2 or more districts."""

from __future__ import print_function

import os
import us
import pandas as pd

from utils.settings import (clustered_geojson_results_dir, usa_csv_path)

# All states in case we need them (DC included so total is 51):
result_per_state = {str(state.abbr): False for state in us.STATES}

for filename in os.listdir(clustered_geojson_results_dir):
    if filename.endswith(".geojson") and not filename.startswith('.'):
        result_per_state[filename[0:2]] = True

# Load up the congressional districts for the whole of the USA:
usa = pd.read_csv(usa_csv_path)

forgot = 0
for state, value in result_per_state.iteritems():
    fips = int(us.states.lookup(state).fips)
    # print(state, value, fips)
    if not value:
        if 'at Large' in str(usa[usa['STATEFP'] == fips]['NAMELSAD']):
            None
        else:
            print('You forgot to run clustering on state:', state)
            forgot += 1
if forgot == 0:
    print('You have run clustering on all states!')
