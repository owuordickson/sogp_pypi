*****
USAGE
*****

Ant Colony Optimization for GPs (ACO-GRAD)
------------------------------------------

Executing ACO for mining GPs:

.. code-block:: python

    import so4gp as so
    gps = so.acogps(data_src, min_sup)
    print(gps)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **evaporation_factor** - *[optional]* evaporation factor :code:`default = 0.5`

Genetic Algorithm for GPs (GA-GRAD)
--------------------------------------

Executing GA for mining GPs:

.. code-block:: python

    import so4gp as so
    gps = so.gagps(data_src, min_sup)
    print(gps)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **n_pop** - *[optional]* initial population :code:`default = 5`
* **pc** - *[optional]* offspring population multiple :code:`default = 0.5`
* **gamma** - *[optional]* crossover rate :code:`default = 1`
* **mu** - *[optional]* mutation rate :code:`default = 0.9`
* **sigma** - *[optional]* mutation rate :code:`default = 0.9`

Particle Swarm Optimization for GPs (PSO-GRAD)
-------------------------------------------------

Executing PSO for mining GPs:

.. code-block:: python

    import so4gp as so
    gps = so.run_particle_swarm(data_src, min_sup)
    print(gps)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format:code:` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **n_particles** - *[optional]* initial particle population :code:`default = 5`
* **velocity** - *[optional]* particle velocity :code:`default = 0.9`
* **coeff_p** - *[optional]* personal coefficient rate :code:`default = 0.01`
* **coeff_g** - *[optional]* global coefficient :code:`default = 0.9`

Local Search for GPs (LS-GRAD)
---------------------------------

Executing LS for mining GPs:

.. code-block:: python

    import so4gp as so
    gps = so.run_hill_climbing(data_src, min_sup)
    print(gps)

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **step_size** - *[optional]* step size :code:`default = 0.5`


Random Search for GPs (RS-GRAD)
----------------------------------

Executing RS for mining GPs:

.. code-block:: python

    import so4gp as so
    gps = so.run_random_search(data_src, min_sup)
    print(gps)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`


Sample Output
''''''''''''''

.. code-block:: JSON

    {
	"Algorithm": "RS-GRAD",
	"Best Patterns": [
            [["Age+", "Salary+"], 0.6],
            [["Expenses-", "Age+", "Salary+"], 0.6]
	],
	"Iterations": 20
    }