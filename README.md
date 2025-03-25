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
By default, it has the name of the run script, `.results.db` appended insetad of `.py`.

## GOTM - file formats

Observations for GOTM must be provided as a whitespace-separated (e.g., tab-separated)
text file with one line per observation. Each observed variable must use a separate file.
Each line must contain:
* the date + time as `YYYY-mm-dd HH:MM:SS`. For instance, a value of `2000-05-08 06:20:00` for 6:20 am on 8 May 2000.
* _only if you are providing depth-dependent observations:_ the depth (m) at which the
   observation was taken. It decreases downwards from the water surface, e.g., a
   value of `5.4` indicates a depth of 5.4 meter below the water surface.
* the observed value
* _optional:_ the standard deviation of the observation. This will be interpreted as the combination of measurement and model error.

The format of the observations is specified when you call `parsac.job.gotm.Simulation.request_comparison`
in your run script:
`parsac.util.TextFormat.DEPTH_INDEPENDENT` specifies depth-independent observations,
`parsac.util.TextFormat.DEPTH_DEPENDENT` specifies depth-independent observations.