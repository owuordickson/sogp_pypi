# CausalRiver Benchmark Instructions

## Evaluating a Model on the CausalRiver Dataset

Follow the steps below to evaluate your model using the CausalRiver benchmark.

### 1. Generate the Benchmark Dataset

1. Download the **product** dataset from the CausalRiver benchmark release:

   https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip

2. Extract the downloaded archive.

3. Run the dataset generation script:

   ```
   generate_datasets.py
   ```

This will generate the datasets required for benchmarking.

---

### 2. Register Your Model

To benchmark a new model, complete the following steps.

#### a. Create a model configuration

Create a YAML configuration file for your model in:

```
config/method/
```

For example:

```
config/method/my_model.yaml
```

Replace `my_model` with an appropriate name for your method.

#### b. Update the benchmark configuration

Open:

```
config/benchmark.yaml
```

Set the `model` field to the name of your configuration file **without** the `.yaml` extension.

Example:

```yaml
model: my_model
```

#### c. Register the benchmark implementation

1. Add a function that executes your model in:

```
tools/baseline_methods.py
```

2. Update the model selection logic in:

```
benchmark.py
```

by adding your method to the `if`/`elif` dispatch inside the `main()` function.

---

### 3. Run the Benchmark

Once your model has been registered and configured, execute:

```
python benchmark.py
```

The benchmark will load your model configuration, execute your implementation, and evaluate the results on the generated CausalRiver datasets.
