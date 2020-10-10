## Explanation of tests

test_franke_analysis.ipynb: This is a notebook which reproduces our whole analysis for the Franke function, with the addition of looking at the behavior for different lambda values for Ridge and Lasso. Can be run in one fell swoop to look at a fairly interesting test case. The user is encouraged to experiment with the commented control variables for number of points, max_degree, lambda values, etc. These control variables appear in the second code block, and the third to last code block. They can be changed independently.

test_terrain_analysis.ipynb: This notebook is practically identical to test_franke_analysis.ipynb. Once again, the user is encouraged to experiment with the control variables. Do note that the default settings may take some time to run, speed-up can be achieved by turning off do_boot, do_subset, or reducing max_degrees. Do also note that the default predictions are for too high a polynomial degree, 18. This is to show what happens when OLS hits a variance wall. A more sensible degree for the default (100) spacing would be 14.

bias_variance_test.py: This is a script to be run from terminal. Really just a unit test for our bias-variance calculations validated against code from the lecture notes.

unit_tests.py: This is a collection of functions performing unit tests. Not intended for use by user.
