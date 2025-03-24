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

To view optimization results during or after optimization, you currently can use the following:

```
python -m parsac.optimize.plot <DBFILE>
```

Here, `<DBFILE>` is the result database created by the optimization.
By default, it has the name of the run scirpt (minus `.py`) with `.results.db` appended.
