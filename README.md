# Parallel Sensitivity Analysis and Calibration

Python-based tool for sensitvity analysis and auto-calibration in Parallel.
This tool is designed for analysis of models that take significant time to run.
For that reason, it focuses on storing and exploiting every single model result,
and performing model runs in parallel on either a single machine or
on computer clusters. It works with models that are run by calling one binary,
that use text-based configuration files based on YAML or Fortran namelists,
and that write their output to NetCDF.

### To install

pip install <--upgrade> --user parsac

Remove --user to install in the system's shared Python directory.
Some systems have multiple versions of pip, e.g., pip for Python 2, pip3 for Python 3.
Make sure you use the command that corresponds to the Python version you want to install into.

### To work with a wheel in edit mode

pip install -e . --user --upgrade
