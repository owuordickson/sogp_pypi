
RELEASE HISTORY
***************


0.6.7 (2024-11-28)
---------------------------

**Updates**

* Added GradPFS algorithm for univariate and multivariate regression tasks.



0.6.6 (2024-11-26)
---------------------------

**Updates**

* Multivariate feature selection through elimination of irrelevant and redundant features.



0.6.5 (2024-11-13)
---------------------------

**Updates**

* Added target-col to GRAANK. Changed licence to GNU GPL v3




0.6.4 (2024-11-12)
---------------------------

**Updates**

* Documentation additions.



0.6.3 (2024-11-09)
---------------------------

**Bug fixes**

* Gradual correlation method 'index' problem.




0.6.2 (2024-11-09)
---------------------------

**Bug fixes**

* Import errors fixed.



0.6.1 (2024-11-09)
---------------------------

**Bug fixes**

* Import errors fixed.



0.6.0 (2024-11-08)
---------------------------

**Updates**

* Added a method that computes gradual correlation for feature selection use case.



0.5.9 (2024-11-08)
---------------------------

**Updates**

* Constructors update and method for timeseries analysis and feature selection.



0.5.8 (2024-11-07)
---------------------------

**Bug fix**

* TGradAMI algorithm time-delay approximation error.



0.5.7 (2024-11-07)
---------------------------

**Bug fix**

* TGradAMI algorithm MI computation error.



0.5.6 (2024-11-07)
---------------------------

**Updates**

* Optimization and added documentation for TGradAMI algorithm.




0.5.5 (2024-11-05)
---------------------------

**Updates**

* Fixed error on get_time_delay method.



0.5.4 (2024-11-04)
---------------------------

**Updates**

* Added clustering option for TGradAMI.



0.5.3 (2024-11-04)
---------------------------

**Updates**

* Added class TGradAMI and method for decomposing gradual components.




0.5.2 (2024-10-30)
---------------------------

**Bug fix**

* Bug fix on computing support twice.



0.5.1 (2024-10-29)
---------------------------

**Bug fix**

* Bug fix on get_fuzzy_time_lag - wrong parameter type



0.5.0 (2024-10-29)
---------------------------

**Updates**

* Added TGrad algorithm




0.4.9 (2024-10-29)
---------------------------

**Updates**

* Added documentation to TimeDelay class

* Added TGP class



0.4.8 (2024-10-28)
---------------------------

**Bug fixes**

* problems with dateutil library


0.4.7 (2024-10-26)
---------------------------

**Updates**

* downgraded to Python/3.9


0.4.6 (2024-07-31)
---------------------------

**Updates**

* renamed TimeLag to TimeDelay



0.4.5 (2024-07-29)
---------------------------

**Bug fixes**

* apriori candidates correct row count

**Updates**

* requirements to pyproject.toml



0.4.4 (2024-07-22)
---------------------------

**Updates**

* Added function for removing subsets from GP list



0.4.3 (2024-07-22)
---------------------------
**Bug fixes**

* Test GP candidates to ensure no repeated attributes

**Updates**

* Renamed reference-col to target-col



0.4.2 (2024-07-17)
---------------------------

**Updates**

* Generate GP candidates w.r.t reference-colum



0.4.1 (2024-07-04)
---------------------------
**Bug fixes**

* cleared execution warnings

**Updates**

* Python library update



0.4.0 (2024-07-04)
---------------------------
**Updates**

* Restructured and library update



0.3.9 (2023-07-12)
---------------------------
**Updates**

* GRAANK uses ExtGP() class



0.3.8 (2022-10-27)
---------------------------
**Updates**

* retrieve nodes_matrix from CluDataGP


0.3.7 (2022-10-19)
---------------------------
**Bug fixes**

* removed class DfsDataGP


0.3.6 (2022-10-19)
---------------------------
**Bug fixes**

* fixed fit_bitmap method to show non-transposed bitmaps

**Updates**

* Added Doctests with a dummy dataframe source

* Added test results

* Added method fit_tids in class DataGP


0.3.5 (2022-10-17)
---------------------------
**Bug fixes**

* made method infer_gps private (_infer_gps)

**Updates**

* Updated docs to include Python code sample


0.3.4 (2022-10-14)
---------------------------
**Bug fixes**

* aco_graank returns DataGP object

**Updates**

* made get_attr_cols, get_time_cols private

* made construct_matrices, estimate_score_vector, estimate_support private

* renamed init_bitmap method to fit_bitmap

* renamed CluDataGP to ClusterGP

* added discover method to ClusterGP

* converted graank method to class

* converted aco_graank method to class

* converted ga_graank method to class

* converted pso_graank method to class

* converted rs_graank method to class

* converted hc_graank method to class

* updated usage documentation


0.3.3 (2022-10-13)
---------------------------
**Bug fixes**

* renamed CluDataGP attribute from all to no_prob

* renamed variables in gen_apriori_candidates

* corrected typos in docs

**Updates**

* added attribute gradual_patterns to DataGP class

* modified graank, acograd, psograd, gagrad, lsgrad, prgrad, clugrad to return DataGP object

* renamed acogps to aco_graank

* renamed gagps to ga_graank

* renamed psogps to pso_graank

* renamed hcgps to hc_graank

* renamed rsgps to rs_graank

* renamed clugps to clu_bfs



0.3.2 (2022-10-06)
---------------------------
**Updates**

* renamed method compare_gps to analyze_gps

* analyze_gps computes error, std, and returns tabulated results

* added docs

* added class DfsDataGP

* added method inv_gi


0.3.1 (2022-10-04)
---------------------------
**Bug fixes**

* option to fetch all matrices



0.3.0 (2022-10-04)
---------------------------
**Updates**

* added method construct_all_matrices



0.2.9 (2022-09-16)
----------------------------

**Updates**

* added method add_items_from_list()

**Bug fixes**
* generate all object pairs when e_prob is 0



0.2.8 (2022-09-08)
----------------------------

**Updates**

* added attribute freq_count to class ExtGP


0.2.7 (2022-09-08)
----------------------------

**Updates**

* renamed class GP4sw to ExtGP (stands for Extended GP)

* renamed class DataGP4clu to CluDataGP (stands for Clustering DataGP)

* added description statements to functions


0.2.6 (2022-09-01)
----------------------------

**Bug fixes**

* clustering attributes missing


0.2.5 (2022-08-31)
----------------------------

**Updates**

* added clugps function

* added class DataGP4clu

* updated README

* added compare_gps function


0.2.4 (2022-07-08)
----------------------------

**Updates**

* renamed functions

* added class GP4sw

* added class NumericSS

* count invalid GPs



0.2.3 (2022-06-15)
----------------------------

**Updates**

* count invalid GPs in GRAANK and ACO-GRAD



0.2.2 (2022-04-23)
-----------------------------

**Bug fixes**

* fixed import error on plot_curve


0.2.1 (2022-04-23)
-----------------------------

**Bug fixes**

* problem with import (removed matplotlib package)


0.2.0 (2022-04-22)
-----------------------------

**Updates**

* removed Profile class

* converted bitmap method into a class method

* added 4 methods for getting cpus, writing results, plotting evaluations


0.1.9 (2022-04-20)
-----------------------------

**Bug fixes**

* problem with import (class Profile not Found)


0.1.8 (2022-04-20)
-----------------------------

**Updates**

* added Profile class for profiling performance


0.1.7 (2022-03-17)
-------------------

**Updates**

* removed method for computing net-wins matrix


0.1.6 (2022-03-02)
-------------------

**Updates**

* added method for computing net-wins matrix


0.1.5 (2022-03-01)
-------------------

**Updates**

* added ability to return GPs as objects (using parameter 'return_gps=True')

* added docstrings to describe functionality



0.1.4 (2022-01-11)
-------------------

**Bug fixes**

* problem with import (Module not Found)


0.1.3 (2022-01-11)
------------------

* Renamed methods to simpler words



0.1.2 (2022-01-11)
------------------

* Updated documentation

**Bug fixes**

- removed so4gp_pkg package so that import is direct



0.1.1 (2022-01-10)
------------------

**Bug fixes**

- function for generating GP bitmap returns a binary array



0.1.0 (2022-01-06)
------------------

* Added graank algorithm

* Added function for generating binary matrix for gradual items

**Bug fixes**

- fixed an error that converted time columns to Strings and deleted them




0.0.7 (2022-01-06)
-------------------

* Added readthedocs url


v0.0.6 (2022-01-06)
-------------------

* Renamed to 'some optimizations for gradual patterns'
* Added function for generating binary matrix for gradual items



v0.0.5 (2021-09-15)
-------------------

* Added local search optimization algorithm.
* Added random search optimization algorithm.
* Added configuration file.



v0.0.4 (2021-09-15)
--------------------

**Bug fixes**

- Replaced class methods with plain methods to fix import issues.



v0.0.3 (2021-07-22)
-------------------

**Bug fixes**

- Upgraded to using Numpy in order to improve efficiency