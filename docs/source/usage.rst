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
In this approach, it is assumed that every column can be converted into gradual item (GI). If the GI is valid (i.e. its computed support is greater than the minimum support threshold) then it is either increasing or decreasing (+ or -), otherwise it is irrelevant (x). Therefore, a pheromone matrix is built using the number of columns and the possible variations (increasing, decreasing, irrelevant) or (+, -, x). The algorithm starts by randomly generating GP candidates using the pheromone matrix, each candidate is validated by confirming that its computed support is greater or equal to the minimum support threshold. The valid GPs are used to update the pheromone levels and better candidates are generated.

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
In this approach, it is assumed that every GP candidate may be represented as a binary gene (or individual) that has a unique position and cost. The cost is derived from the computed support of that candidate, the higher the support value the lower the cost. The aim of the algorithm is search through a population of individuals (or candidates) and find those with the lowest cost as efficiently as possible.

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
In this approach, it is assumed that every GP candidate may be represented as a particle that has a unique position and fitness. The fitness is derived from the computed support of that candidate, the higher the support value the higher the fitness. The aim of the algorithm is search through a population of particles (or candidates) and find those with the highest fitness as efficiently as possible.

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
In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value associated with it. The cost is derived from the computed support of that candidate, the higher the support value the lower the cost. The aim of the algorithm is search through group of positions and find those with the lowest cost as efficiently as possible.

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
In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value associated with it. The cost is derived from the computed support of that candidate, the higher the support value the lower the cost. The aim of the algorithm is search through group of positions and find those with the lowest cost as efficiently as possible.

Executing RS for mining GPs:

.. code-block:: python

    gp_json = sgp.rsgps(data_src, min_sup)
    print(gp_json)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **max_iterations** - *[optional]* maximum iterations :code:`default = 1`
* **return_gps** - *[optional]* additionally return object GPs ```default = False```


Clustering algorithm for GPs (Clu-GRAD)
----------------------------------
We borrow the net-win concept used in the work 'Clustering Using Pairwise Comparisons' proposed by R. Srikant to the problem of extracting gradual patterns (GPs). In order to mine for GPs, each feature yields 2 gradual items which we use to construct a bitmap matrix comparing each row to each other (i.e., (r1,r2), (r1,r3), (r1,r4), (r2,r3), (r2,r4), (r3,r4)).

In this approach, we convert the bitmap matrices into 'net-win vectors'. Finally, we apply spectral clustering to determine which gradual items belong to the same group based on the similarity of net-win vectors. Gradual items in the same cluster should have almost similar score vector.

Executing Clustering algorithm for mining GPs:

.. code-block:: python

    gp_json = sgp.clugps(data_src, min_sup)
    print(gp_json)


where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a :code:`file in csv format` or a :code:`Pandas DataFrame`}
* **min_sup** - *[optional]* minimum support :code:`default = 0.5`
* **e_probability** - *[optional]* erasure probability ```default = 0.5```
* **max_iteration** - *[optional]* maximum iterations for estimating score vectors ```default = 10```
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