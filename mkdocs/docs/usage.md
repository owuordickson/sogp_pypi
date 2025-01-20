# Usage
In order to run each algorithm for the purpose of extracting GPs, follow the instructions that follow.


## 1.  GRAANK

GRAdual rANKing Algorithm for GPs (GRAANK) is the classical approach (initially proposed by Anne Laurent) for mining gradual patterns. All the remaining algorithms are variants of this algorithm.

```python
import so4gp as sgp

mine_obj = sgp.GRAANK(data_source=f_path, min_sup=0.5, eq=False)
gp_json = mine_obj.discover()
print(gp_json)

```

where you specify the parameters as follows:

* **data_source** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **eq** - *[optional]* encode equal values as gradual ```default = False```


## 2. ACO-GRAANK
ACO-GRAANK stands for Ant Colony Optimization for GPs. In this approach, it is assumed that every column can be converted into gradual item (GI). If the GI is valid (i.e. its computed support is greater than the minimum support threshold) then it is either increasing or decreasing (+ or -), otherwise it is irrelevant (x). Therefore, a pheromone matrix is built using the number of columns and the possible variations (increasing, decreasing, irrelevant) or (+, -, x). The algorithm starts by randomly generating GP candidates using the pheromone matrix, each candidate is validated by confirming that its computed support is greater or equal to the minimum support threshold. The valid GPs are used to update the pheromone levels and better candidates are generated.

```python
import so4gp as sgp

mine_obj = sgp.AntGRAANK(data_src)
gp_json = mine_obj.discover()
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iteration** - *[optional]* maximum number of iterations ```default = 1```
* **evaporation_factor** - *[optional]* evaporation factor ```default = 0.5```


## 3. GA-GRAANK
GA-GRAANK stands for Genetic Algorithm for GPs. In this approach, it is assumed that every GP candidate may be represented as a binary gene (or individual) that has a unique position and cost. The cost is derived from the computed support of that candidate, the higher the support value the lower the cost. The aim of the algorithm is search through a population of individuals (or candidates) and find those with the lowest cost as efficiently as possible.

```python
import so4gp as sgp

mine_obj = sgp.GeneticGRAANK(data_src)
gp_json = mine_obj.discover()
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iteration** - *[optional]* maximum number of algorithm iterations ```default = 1```
* **n_pop** - *[optional]* initial population ```default = 5```
* **pc** - *[optional]* offspring population multiple ```default = 0.5```
* **gamma** - *[optional]* crossover rate ```default = 1```
* **mu** - *[optional]* mutation rate ```default = 0.9```
* **sigma** - *[optional]* mutation rate ```default = 0.9```

## 4. PSO-GRAANK 
PSO-GRAANK stands for Particle Swarm Optimization for GPs. In this approach, it is assumed that every GP candidate may be represented as a particle that has a unique position and fitness. The fitness is derived from the computed support of that candidate, the higher the support value the higher the fitness. The aim of the algorithm is search through a population of particles (or candidates) and find those with the highest fitness as efficiently as possible.

```python
import so4gp as sgp

mine_obj = sgp.ParticleGRAANK(data_src)
gp_json = mine_obj.discover()
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iteration** - *[optional]* maximum number of algorithm iterations ```default = 1```
* **n_particles** - *[optional]* initial particle population ```default = 5```
* **velocity** - *[optional]* particle velocity ```default = 0.9```
* **coeff_p** - *[optional]* personal coefficient rate ```default = 0.01```
* **coeff_g** - *[optional]* global coefficient ```default = 0.9```

## 5. LS-GRAANK 
LS-GRAANK stands for Local Search for GPs. In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value associated with it. The cost is derived from the computed support of that candidate, the higher the support value the lower the cost. The aim of the algorithm is search through group of positions and find those with the lowest cost as efficiently as possible.

```python
import so4gp as sgp

mine_obj = sgp.HillClimbingGRAANK(data_src, min_sup)
gp_json = mine_obj.discover()
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iteration** - *[optional]* maximum number of algorithm iterations ```default = 1```
* **step_size** - *[optional]* step size ```default = 0.5```


## 6. RS-GRAANK
RS-GRAANK stands for Random Search for GPs. In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value associated with it. The cost is derived from the computed support of that candidate, the higher the support value the lower the cost. The aim of the algorithm is search through group of positions and find those with the lowest cost as efficiently as possible.

```python
import so4gp as sgp

mine_obj = sgp.RandomGRAANK(data_src, min_sup)
gp_json = mine_obj.discover()
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **max_iteration** - *[optional]* maximum number of algorithm iterations ```default = 1```


## 7. ClusterGP
Clustering algorithm for GPs (Clu-GRAD).  We borrow the net-win concept used in the work 'Clustering Using Pairwise Comparisons' proposed by R. Srikant to the problem of extracting gradual patterns (GPs). In order to mine for GPs, each feature yields 2 gradual items which we use to construct a bitmap matrix comparing each row to each other (i.e., (r1,r2), (r1,r3), (r1,r4), (r2,r3), (r2,r4), (r3,r4)).

In this approach, we convert the bitmap matrices into 'net-win vectors'. Finally, we apply spectral clustering to determine which gradual items belong to the same group based on the similarity of net-win vectors. Gradual items in the same cluster should have almost similar score vector.

```python
import so4gp as sgp

mine_obj = sgp.ClusterGP(data_source=data_src, min_sup=0.5, e_prob=0.1)
gp_json = mine_obj.discover()
print(gp_json)
```

where you specify the parameters as follows:

* **data_src** - *[required]* data source {either a ```file in csv format``` or a ```Pandas DataFrame```}
* **min_sup** - *[optional]* minimum support ```default = 0.5```
* **e_probability** - *[optional]* erasure probability ```default = 0.5```
* **max_iteration** - *[optional]* maximum iterations for estimating score vectors ```default = 10```


## Sample Output
The default output is the format of JSON:

```json
{
	"Algorithm": "RS-GRAANK",
	"Best Patterns": [
            [["Age+", "Salary+"], 0.6], 
            [["Expenses-", "Age+", "Salary+"], 0.6]
	],
	"Iterations": 20
}
```