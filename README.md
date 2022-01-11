
**SO4GP** stands for: "Some Optimizations for Gradual Patterns". SO4GP applies optimizations such as swarm intelligence, HDF5 chunks, SVD and many others in order to improve the efficiency of extracting gradual patterns. It provides Python algorithm implementations for these optimization techniques. The algorithm implementations include:

* Ant Colony Optimization
* Genetic Algorithm
* Particle Swarm Optimization
* Random Search
* Local Search

## Install Requirements
Before running **so4gp**, make sure you install the following ```Python Packages```:

```shell
pip3 install numpy~=1.21.2 pandas~=1.3.3 python-dateutil~=2.8.2 ypstruct~=0.0.2
```

## Usage
Write the following code:

### 1. Ant Colony Optimization for GPs (ACO-GRAD)

```python
import so4gp as sgp

gps = sgp.acogps(data_src, min_sup)
print(gps)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iterations** - *[optional]* maximum iterations ```default = 1```
* **evaporation_factor** - *[optional]* evaporation factor ```default = 0.5```

### 2. Genetic Algorithm for GPs (GA-GRAD)

```python
import so4gp as sgp

gps = sgp.gagps(data_src, min_sup)
print(gps)
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

### 3. Particle Swarm Optimization for GPs (PSO-GRAD)

```python
import so4gp as sgp

gps = sgp.psogps(data_src, min_sup)
print(gps)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iterations** - *[optional]* maximum iterations ```default = 1```
* **n_particles** - *[optional]* initial particle population ```default = 5```
* **velocity** - *[optional]* particle velocity ```default = 0.9```
* **coeff_p** - *[optional]* personal coefficient rate ```default = 0.01```
* **coeff_g** - *[optional]* global coefficient ```default = 0.9```

### 4. Local Search for GPs (LS-GRAD)

```python
import so4gp as sgp

gps = sgp.hcgps(data_src, min_sup)
print(gps)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iterations** - *[optional]* maximum iterations ```default = 1```
* **step_size** - *[optional]* step size ```default = 0.5```


### 5. Random Search for GPs (RS-GRAD)

```python
import so4gp as sgp

gps = sgp.rsgps(data_src, min_sup)
print(gps)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iterations** - *[optional]* maximum iterations ```default = 1```


## Sample Output
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
