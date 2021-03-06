
RELEASE HISTORY
***************



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