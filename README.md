# Gerrymandering and Computational Redistricting

For more information, see: Guest, O., Kanayet, F. J., & Love, B. C. (2019). [Gerrymandering and Computational Redistricting](https://doi.org/10.1007/s42001-019-00053-9). *Journal of Computational Social Science*.

## Results
If you are interested in looking at the maps check out the website: http://redistrict.science.

## Redistrict a State
```run.py``` is the main script to run.
It downloads (if required) and processes relevant data for a state.
Afterwards it runs the clustering on the state.
And finally, it graphs the state as it is and the results of the clustering.
For more information type:
```bash
python run.py --help
```
## Examples
For a simple and easy state to run (not super taxing on RAM, etc.), you may try Rhode Island.
Type the following and hit return:
```
python run.py RI
```

The ```run.py``` will now print information to do with the files it will create and the settings it is being run with.

## Acknowledgements
Thanks to Logan T. Powell ([@logantpowell](https://github.com/logantpowell)) who works at the United States Census Bureau for all his patience, help, and support. Also thanks to David Ellis ([@Ducksual](https://twitter.com/Ducksual)) and Michael Sumner ([@mdsumner](https://twitter.com/mdsumner)) for showing me useful documentation for GIS — and thanks to Grant R. Vousden-Dishington ([@usethespacebar](https://twitter.com/usethespacebar)) and  Dmitrii V. Pasechnik ([@dimpase](https://twitter.com/dimpase)) for pointing me to Cython.
