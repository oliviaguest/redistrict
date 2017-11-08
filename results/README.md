# Results from the _k_-means clustering:

```best_stats```: the statistics of the best solutions which is a strict subset of the ```csv_stats``` directory. Populated by running ```update_best_stats_dir()```.

```cluster_plots```: maps for the cluster results per state per run.

```congressional_district_plots```: maps for the congressional districts per state.

```csv_means```: csv files containing the mean pairwise distances for the real districts and our solution per state per run.

```csv_results```: contains CSV files for each and every state per run with every possible detail for each block _except_ the geometry. This version of the results exists because if we do not need  the geometry these files are dramatically smaller.

```csv_stats```: detailled statistics for each district (both real and our clustering) per state per run.

```geojson_results```: contains GeoJSON files for each and every state per run with every possible detail for each block _including_ the geometry. 
