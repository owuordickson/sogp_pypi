
**SO4GP** stands for: "Swarm Optimization for Gradual Patterns". SO4GP applies swarm intelligence to extraction of gradual patterns. It provides Python algorithm implementations of swarm-based optimization algorithms for mining gradual patterns. The algorithm implementations include:

* Ant Colony Optimization
<!--- * Genetic Algorithm
* Particle Swarm Optimization
* Wasp Swarm Optimization
* Pure Random Search
* Pure Local Search --->

### Usage
Write the following code:

```python
from so4gp_pkg import so4gp as so
gps = so.run_ant_colony('filename.csv', min_sup)
print(gps)
```

where you specify the parameters as follows:

* **filename.csv** - *[required]* a file in csv format
* **min_sup** - *[optional]* minimum support ```default = 0.5```


```json
Sample Output

{
  'Best Patterns': [
    [['Expenses-', 'Age+'], 1.0], 
    [['Expenses-', 'Age+', 'Salary+'], 0.6]
  ], 
  'Iterations': 100
}
```

### References
* Owuor, D., Runkler T., Laurent A., Menya E., Orero J (2021), Ant Colony Optimization for Mining Gradual Patterns. International Journal of Machine Learning and Cybernetics.
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. https://doi.org/10.1109/FUZZ-IEEE.2019.8858883.
