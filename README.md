# Harissa Benchmark
Benchmark for Harissa's inference and simulation methods.

## Installation

`harissa-benchmark` depends on the version 4.1.0 of [harissa](https://github.com/harissa-framework/harissa). 
This version is not released yet, so to be able to use this package you need to
clone the `harissa` repository, switch branch to `add-cardamom` 
and install it in local (with the extra dependencies).
You can use a virtual environment if you have another version of harissa installed.
Now that you have the correct version of `harissa`
you can install `harissa-benchmark`.

```console
git clone https://github.com/harissa-framework/harissa.git
cd harissa
git checkout add-cardamom
pip install .[extra]
pip install harissa-benchmark
```

## Usage

By default this package contains some networks and inferences methods.
You can display them with the functions `available_networks` and `available_inferences`.

```python
from harissa_benchmark import available_networks, available_inferences
print(available_networks())
print(available_inferences())
```

To run a benchmark on those and generate scores, you only need to import the
`Benchmark` class and to call its method `generate`. 
The generated scores will be accessible inside the attribute `items` or the property
`scores`.

```python
from harissa_benchmark import Benchmark
benchmark = Benchmark()
benchmark.generate()
print(benchmark.scores.keys())
```

It is a dictionary that contains another dictionary for each network which contains
the results of the inference per inference method.
You can create reports or statistic with them.
For example you can display [roc](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html) curves for every inference method for a given network.
`harrisa-benchmark` provides helper class and functions to do it:

```python
from harissa_benchmark.plotters import UnDirectedPlotter
network_name = 'BN8'
plotter = UnDirectedPlotter(benchmark.networks[network_name])
plotter.plot_roc_curves(benchmark.scores[network_name])
```
