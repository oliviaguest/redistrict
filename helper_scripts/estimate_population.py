"""Take the decennial and ACS5 census and estimate the population in 2015."""

from __future__ import division, print_function
from utils.utils import get_estimation

if __name__ == '__main__':
    import us
    import sys
    from utils.settings import (year, blocks_geojson_filename,
                                blocks_2010_shapefile_dir,
                                block_assignments_dir, acs5_shapefile_dir,
                                acs5_population_dir, usa_shapefile_path)
    # The state we want to run this on (taken from CLI):
    if len(sys.argv) > 1:
        state_input = unicode(sys.argv[1], "utf-8")
    else:
        state_input = u'RI'  # default to Rhode Island
    # States can be input in 3 different ways from CLI: FIPS code (e.g., 44),
    # 2-letter abbreviation (e.g., RI), or full name with quotes if required
    # (e.g., 'Rhode Island').
    # If FIPS code is without a leading 0, add one:
    try:
        state = us.states.lookup(state_input).abbr
    except AttributeError:
        state_input = state_input.zfill(2)
        try:
            state = us.states.lookup(state_input).abbr
        except AttributeError:
            print('FIPS code not assigned to a state!')
            exit()
    # state_fips = us.states.lookup(state_input).fips
    df = get_estimation(state, year, blocks_geojson_filename,
                        blocks_2010_shapefile_dir, block_assignments_dir,
                        acs5_shapefile_dir, acs5_population_dir,
                        usa_shapefile_path)
