# parsac

parsac (formerly acpy) is a Python-based tool for sensitivity analysis and auto-calibration in parallel.
It is designed for analysis of models that take significant time to run.
For that reason, it focuses on storing and exploiting every single model result,
and performing model runs in parallel on either a single machine or
on computer clusters. It works with models that are run by calling one binary,
that use text-based configuration files based on YAML or Fortran namelists,
and that write their output to NetCDF.

[![DOI](https://zenodo.org/badge/206791023.svg)](https://zenodo.org/badge/latestdoi/206791023) [![Build Status](https://travis-ci.com/BoldingBruggeman/parsac.svg?branch=master)](https://travis-ci.com/BoldingBruggeman/parsac)

## Installation

`pip install parsac --user`

Remove `--user` to install in the system's shared Python directory (not recommended).
Some systems have multiple versions of pip, e.g., pip for Python 2, pip3 for Python 3.
Make sure you use the command that corresponds to the Python version you want to install into.

### Dependencies

parsac supports parallel simulations through [Parallel Python](https://www.parallelpython.com).
This package supports Python 2 out of the box (`pip install pp --user`), but its Python 3 version
is currently in beta. To install pp in Python 3, [download the zip file with the Python 3 port of Parallel Python](https://www.parallelpython.com/content/view/18/32), extract its contents, go to the contained directory and open a command prompt there, then run `python setup.py install`.

parsac uses [SALib](https://github.com/SALib/SALib) for sensitivity analysis. Typically, this can be installed with `pip install SALib --user`. If you are using [the Anaconda Python distribution](https://www.anaconda.com), you can instead do `conda install SALib` (you may need to add `-c conda-forge`).

## Known issues

* On Windows, parallel runs may finish with several "ERROR: The process "xxx" not found." messages. These are harmless and can be ignored - the analysis has completed successfully and all results have been correctly processed.
