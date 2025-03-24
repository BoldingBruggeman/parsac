# parsac

parsac is a Python-based tool for sensitivity analysis and auto-calibration in parallel.
It is designed for analysis of models that take significant time to run.
For that reason, it focuses on storing and exploiting every single model result,
and performing model runs in parallel on either a single machine or
on computer clusters. It works with models that are run by calling one binary,
that use YAML-based configuration files and that write their output to NetCDF.

## Installation

```
conda env create -f environment.yml
conda activate parsac
pip install .
```

## Usage

Examples are included in the `examples` subdirectory.
