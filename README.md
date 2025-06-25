# parsac

parsac is a Python-based tool for sensitivity analysis and auto-calibration in parallel.
It is designed for analysis of models that take significant time to run.
For that reason, it focuses on storing and exploiting every single model result,
and performing model runs in parallel on either a single machine or
on computer clusters. It works with models that are run by calling one binary,
that use YAML-based configuration files and that write their output to NetCDF.

## Installation

The latest release using [conda](https://docs.conda.io/):

```
conda install conda-forge::parsac
```

The latest release using [pip](https://docs.python.org/3/installing/index.html):

```
python -m pip install parsac
```

The current development version from this repository:

```
conda env create -f environment.yml
conda activate parsac
python -m pip install .
```

If you install the development version, you can update later with:

```
git pull
python -m pip install .
```

## Usage

Reference documentation is available at https://parsac.readthedocs.io/

Examples are included in the `examples` subdirectory.
Those using the General Ocean Turbulence Model (GOTM) are designed
to work with [its latest stable release, v6.0](https://github.com/gotm-model/code/tree/v6.0).

To view optimization results during or after optimization, you currently can use the following:

```
python -m parsac.optimize.plot <DBFILE>
```

Here, `<DBFILE>` is the result database created by the optimization.
By default, it has the name of the run script, with `.results.db` appended instead of `.py`.
Run with `-h` to see all supported command line options.

To write the full set of results to a tab-separated text file, you can use the following:

```
python -m parsac.record <DBFILE> --dump <OUTFILE>
```

Examples of advanced use are available from [the wiki](https://github.com/BoldingBruggeman/parsac/wiki/Recipes-for-v2).

### On HPC clusters

Parsac uses MPI for communication, which it accesses via [mpi4py](https://mpi4py.readthedocs.io/en/stable/). Therefore, mpi4py needs to be installed in your conda environment. If your system has MPI libraries installed,
then you'll want to install mpi4py with:

```
conda activate parsac
pip install mpi4py
```

This builds mpi4py against your existing MPI libraries. More information about installing mpi4py can be found [here](https://mpi4py.readthedocs.io/en/stable/install.html).

Alternatively, if you do not have any MPI libraries on your system yet (`mpiexec` and `mpirun` commands are not available), you can install both MPI and mpi4py with:

```
conda install -c conda-forge mpi4py
```

To run a parsac experiment in parallel, you need to specify how many workers (CPU cores) you will use via environment variable `MPI4PY_FUTURES_MAX_WORKERS`.

You now start the calibration with a single MPI process, like this:

```
export MPI4PY_FUTURES_MAX_WORKERS=<NWORKERS>
mpiexec -n 1 python <RUNSCRIPT>
```

If you are using Intel MPI, you may need to allow MPI spawning beforehand with:

```
export I_MPI_SPAWN=on
```

Parsac uses your MPI implementation's dynamic process management to spawn the workers. If you experience problems with this, you can
instead use [an MPI-1 compatible mechanism](https://mpi4py.github.io/mpi4py/stable/html/mpi4py.futures.html#command-line) to start all processes (a single controller and all workers). Note that the total number of processes then is equivalent to `MPI4PY_FUTURES_MAX_WORKERS + 1`. As long as your MPI settings
allow oversubscription, you only need to request a total of `MPI4PY_FUTURES_MAX_WORKERS` cores from the queue manager, as the controller process uses little to no CPU.

This typically goes into your job submission script.

## GOTM - file formats

Observations for GOTM must be provided as a whitespace-separated (e.g., tab-separated)
text file with one line per observation. Each observed variable must use a separate file.
Each line must contain:
* the date + time as `YYYY-mm-dd HH:MM:SS`. For instance, a value of `2000-05-08 06:20:00` for 6:20 am on 8 May 2000.
* _only if you are providing depth-dependent observations:_ the depth (m) at which the
   observation was taken. It decreases downwards from the water surface, e.g., a
   value of `-5.4` indicates a depth of 5.4 meter below the water surface.
* the observed value
* _optional:_ the standard deviation of the observation. This will be interpreted as the combination of measurement and model error.

The format of the observations is specified when you call `parsac.job.gotm.Simulation.request_comparison`
in your run script:
`parsac.util.TextFormat.DEPTH_INDEPENDENT` specifies depth-independent observations,
`parsac.util.TextFormat.DEPTH_DEPENDENT` specifies depth-independent observations.