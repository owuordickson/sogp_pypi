*****
USAGE
*****

In order to run each algorithm for the purpose of extracting GPs, follow the instructions that follow.

First and foremost, import the **so4gp** python package via:

.. code-block:: python

    import so4gp as sgp


GRAdual rANKing Algorithm for GPs (GRAANK)
------------------------------------------

This is the classical approach (initially proposed by Anne Laurent) for mining gradual patterns. All the remaining algorithms are variants of this algorithm.

.. code-block:: python

    gp_json = sgp.graank(data_src, min_sup, eq, return_gps=False)
    print(gp_json)

    # OR

    gp_json, gp_list = sgp.graank(data_src, min_sup, eq, return_gps=True)
    print(gp_json)

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **eq** - *[optional]* encode equal values as gradual ```default = False```
* **return_gps** - *[optional]* additionally return object GPs ```default = False```




Ant Colony Optimization for GPs (ACO-GRAD)
------------------------------------------

Executing ACO for mining GPs:

.. code-block:: python

    gp_json = sgp.acogps(data_src, min_sup)
    print(gp_json)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **evaporation_factor** - *[optional]* evaporation factor :code:`default = 0.5`
* **return_gps** - *[optional]* additionally return object GPs ```default = False```

Genetic Algorithm for GPs (GA-GRAD)
--------------------------------------

Executing GA for mining GPs:

.. code-block:: python

    gp_json = sgp.gagps(data_src, min_sup)
    print(gp_json)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **n_pop** - *[optional]* initial population :code:`default = 5`
* **pc** - *[optional]* offspring population multiple :code:`default = 0.5`
* **gamma** - *[optional]* crossover rate :code:`default = 1`
* **mu** - *[optional]* mutation rate :code:`default = 0.9`
* **sigma** - *[optional]* mutation rate :code:`default = 0.9`
* **return_gps** - *[optional]* additionally return object GPs ```default = False```

Particle Swarm Optimization for GPs (PSO-GRAD)
-------------------------------------------------

Executing PSO for mining GPs:

.. code-block:: python

    gp_json = sgp.psogps(data_src, min_sup)
    print(gp_json)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format:code:` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **n_particles** - *[optional]* initial particle population :code:`default = 5`
* **velocity** - *[optional]* particle velocity :code:`default = 0.9`
* **coeff_p** - *[optional]* personal coefficient rate :code:`default = 0.01`
* **coeff_g** - *[optional]* global coefficient :code:`default = 0.9`
* **return_gps** - *[optional]* additionally return object GPs ```default = False```

Local Search for GPs (LS-GRAD)
---------------------------------

Executing LS for mining GPs:

.. code-block:: python

    gp_json = sgp.hcgps(data_src, min_sup)
    print(gp_json)

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **step_size** - *[optional]* step size :code:`default = 0.5`
* **return_gps** - *[optional]* additionally return object GPs ```default = False```

Random Search for GPs (RS-GRAD)
----------------------------------

Executing RS for mining GPs:

.. code-block:: python

    gp_json = sgp.rsgps(data_src, min_sup)
    print(gp_json)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **return_gps** - *[optional]* additionally return object GPs ```default = False```


Sample Output
''''''''''''''
The default output is the format of JSON:

.. code-block:: JSON

    {
	"Algorithm": "RS-GRAD",
	"Best Patterns": [
            [["Age+", "Salary+"], 0.6],
            [["Expenses-", "Age+", "Salary+"], 0.6]
	],
	"Iterations": 20
    }