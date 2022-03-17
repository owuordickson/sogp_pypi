
**SO4GP** stands for: "Some Optimizations for Gradual Patterns". SO4GP applies optimizations such as swarm intelligence, HDF5 chunks, SVD and many others in order to improve the efficiency of extracting gradual patterns. It provides Python algorithm implementations for these optimization techniques. The algorithm implementations include:

* (Classical) GRAANK algorithm for extracting GPs
* Ant Colony Optimization algorithm for extracting GPs
* Genetic Algorithm for extracting GPs
* Particle Swarm Optimization algorithm for extracting GPs
* Random Search algorithm for extracting GPs
* Local Search algorithm for extracting GPs

## Install Requirements
Before running **so4gp**, make sure you install the following ```Python Packages```:

```shell
pip3 install numpy~=1.21.2 pandas~=1.3.3 python-dateutil~=2.8.2 ypstruct~=0.0.2
```

## Usage
In order to run each algorithm for the purpose of extracting GPs, follow the instructions that follow.

First and foremost, import the **so4gp** python package via:

```python
import so4gp as sgp
```

### 1.  GRAdual rANKing Algorithm for GPs (GRAANK)

This is the classical approach (initially proposed by Anne Laurent) for mining gradual patterns. All the remaining algorithms are variants of this algorithm.

```python

gp_json = sgp.graank(data_src, min_sup, eq, return_gps=False)
print(gp_json)

# OR

gp_json, gp_list = sgp.graank(data_src, min_sup, eq, return_gps=True)
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **eq** - *[optional]* encode equal values as gradual ```default = False```
* **return_gps** - *[optional]* additionally return object GPs ```default = False```


### 2. Ant Colony Optimization for GPs (ACO-GRAD)

```python

gp_json = sgp.acogps(data_src, min_sup)
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iterations** - *[optional]* maximum iterations ```default = 1```
* **evaporation_factor** - *[optional]* evaporation factor ```default = 0.5```
* **return_gps** - *[optional]* additionally return object GPs ```default = False```


### 3. Genetic Algorithm for GPs (GA-GRAD)

```python

gp_json = sgp.gagps(data_src, min_sup)
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iterations** - *[optional]* maximum iterations ```default = 1```
* **n_pop** - *[optional]* initial population ```default = 5```
* **pc** - *[optional]* offspring population multiple ```default = 0.5```
* **gamma** - *[optional]* crossover rate ```default = 1```
* **mu** - *[optional]* mutation rate ```default = 0.9```
* **sigma** - *[optional]* mutation rate ```default = 0.9```
* **return_gps** - *[optional]* additionally return object GPs ```default = False```

### 4. Particle Swarm Optimization for GPs (PSO-GRAD)

```python

gp_json = sgp.psogps(data_src, min_sup)
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iterations** - *[optional]* maximum iterations ```default = 1```
* **n_particles** - *[optional]* initial particle population ```default = 5```
* **velocity** - *[optional]* particle velocity ```default = 0.9```
* **coeff_p** - *[optional]* personal coefficient rate ```default = 0.01```
* **coeff_g** - *[optional]* global coefficient ```default = 0.9```
* **return_gps** - *[optional]* additionally return object GPs ```default = False```

### 5. Local Search for GPs (LS-GRAD)

```python

gp_json = sgp.hcgps(data_src, min_sup)
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iterations** - *[optional]* maximum iterations ```default = 1```
* **step_size** - *[optional]* step size ```default = 0.5```
* **return_gps** - *[optional]* additionally return object GPs ```default = False```


### 6. Random Search for GPs (RS-GRAD)

```python
import so4gp as sgp

gp_json = sgp.rsgps(data_src, min_sup)
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iterations** - *[optional]* maximum iterations ```default = 1```
* **return_gps** - *[optional]* additionally return object GPs ```default = False```


## Sample Output
The default output is the format of JSON:

```json
{
	"Algorithm": "RS-GRAD",
	"Best Patterns": [
            [["Age+", "Salary+"], 0.6], 
            [["Expenses-", "Age+", "Salary+"], 0.6]
	],
	"Iterations": 20
}
```

### References
* Owuor, D., Runkler T., Laurent A., Menya E., Orero J (2021), Ant Colony Optimization for Mining Gradual Patterns. International Journal of Machine Learning and Cybernetics. https://doi.org/10.1007/s13042-021-01390-w
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. https://doi.org/10.1109/FUZZ-IEEE.2019.8858883.
* Laurent A., Lesot MJ., Rifqi M. (2009) GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In: Andreasen T., Yager R.R., Bulskov H., Christiansen H., Larsen H.L. (eds) Flexible Query Answering Systems. FQAS 2009. Lecture Notes in Computer Science, vol 5822. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-04957-6_33
