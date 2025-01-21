---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/logo_su.png
:alt: SU Logo
```

```{project-heading}
A collection of gradual pattern mining algoithms and tools
```

```{figure} _static/img/gp-mining.png
   :alt: GP Mining
   :width: 500
```

**SO4GP** stands for: "Some Optimizations for Gradual Patterns". SO4GP applies optimizations such as swarm intelligence, HDF5 chunks, cluster analysis and many others in order to improve the efficiency of extracting gradual patterns. 

A GP (Gradual Pattern) is a set of gradual items (GI) and its quality is measured by its computed support value. For example given a data set with 3 features (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the values of **age** *'increasing'* and **salary** *'decreasing'*.

```{code-block} python
import so4gp as sgp

mine_obj = sgp.GRAANK(data_source=f_path, min_sup=0.5, eq=False)
gp_json = mine_obj.discover()
print(gp_json)
```

```{toctree}
:hidden:
:caption: Introduction

introduction/basic_usage
introduction/custom_data
```

```{toctree}
:hidden:
:caption: API

api/algorithms
api/functions
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/feature_selection
tutorials/timeseries_analysis
```


```{toctree}
:hidden:
:caption: Development

Github <https://github.com/owuordickson/sogp_pypi>
Paper <https://ieeexplore.ieee.org/abstract/document/8858883>
release_notes/index
```