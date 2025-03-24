from typing import Optional, Union, Iterable, Any, Mapping, NamedTuple
from pathlib import Path
import os
import logging
import datetime
import tempfile
import shutil
import atexit

import numpy as np
import netCDF4

from .. import core
from .. import util


class Output(NamedTuple):
    name: str
    file: Path
    expression: str
    times: Optional[list[datetime.datetime]]
    zs: Optional[np.ndarray]


class OutputExtractor:
    def __init__(self, output: Output, wrappednc: util.NcDict, logger: logging.Logger):
        self.compiled_expression = compile(output.expression, "<string>", "eval")
        self.numtimes: Optional[np.ndarray] = None
        self.ileft: Optional[np.ndarray] = None
        self.iright: Optional[np.ndarray] = None
        self.ileft_weights: Optional[np.ndarray] = None
        self.iright_weights: Optional[np.ndarray] = None
        self.depth_dimension: Optional[str] = None
        self.zs = output.zs
        self.name = output.name
        self.logger = logger

        if output.times is not None:
            logger.info(
                f"{output.name}: calculating weights for linear interpolation to observations..."
            )
            time_units: str = wrappednc.nc.variables["time"].units
            time_vals = wrappednc["time"]
            numtimes = netCDF4.date2num(output.times, time_units)
            assert isinstance(numtimes, np.ndarray)
            assert (
                np.diff(numtimes).min() >= 0
            ), "Observation times are not strictly increasing."
            iright = time_vals.searchsorted(numtimes)
            ileft = np.maximum(iright - 1, 0)
            iright = np.minimum(iright, time_vals.size - 1)
            w = np.ones((len(numtimes),))
            np.divide(
                numtimes - time_vals[ileft],
                time_vals[iright] - time_vals[ileft],
                out=w,
                where=ileft != iright,
            )
            self.numtimes = numtimes
            self.ileft = ileft
            self.iright = iright
            self.iright_weights = w
            self.ileft_weights = 1.0 - w

            # Retrieve data and check dimensions
            wrappednc.dimensions = set()
            data = wrappednc.eval(self.compiled_expression)
            if data.shape[0] != time_vals.size:
                raise Exception(
                    f"{output.expression}: first dimension should have length {time_vals.size}"
                    f" (= number of time points), but has length {data.shape[0]}."
                )
            if output.zs is not None:
                if data.ndim != 2:
                    raise Exception(
                        f"{output.expression}: Expected 2 dimensions (time, depth),"
                        f" but found {data.ndim} dimensions."
                    )
                depth_dimensions = wrappednc.dimensions.intersection(("z", "z1", "zi"))
                if len(depth_dimensions) == 0:
                    raise Exception(
                        f"{output.expression}: No depth dimension (z, zi or z1) used"
                        f" Only dimensions found: {', '.join(wrappednc.dimensions)}."
                    )
                elif len(depth_dimensions) > 1:
                    raise Exception(
                        f"{output.expression}: More than one depth dimension"
                        f" ({', '.join(depth_dimensions)}) used."
                    )
                self.depth_dimension = depth_dimensions.pop()
            elif data.ndim != 1:
                raise Exception(
                    f"{output.expression}: Expected one dimension (time),"
                    f" but found {data.ndim} dimensions."
                )

    def __call__(self, wrappednc: util.NcDict) -> np.ndarray:
        values = wrappednc.eval(self.compiled_expression)

        if self.numtimes is not None:
            values = self._interpolate(values, wrappednc)

        return values

    def _interpolate(self, values: np.ndarray, wrappednc: util.NcDict) -> np.ndarray:
        assert self.numtimes is not None
        assert self.ileft is not None
        assert self.iright is not None
        assert self.ileft_weights is not None
        assert self.iright_weights is not None

        if self.zs is None:
            # Only interpolate in time
            return (
                values[self.ileft] * self.ileft_weights
                + values[self.iright] * self.iright_weights
            )

        # Get model depth coordinates
        # (currently expressed as elevation relative to water surface)
        h = wrappednc["h"]
        h = h.reshape(h.shape[:2])
        h_cumsum = h.cumsum(axis=1)
        if self.depth_dimension == "z":
            # Centres (all)
            zs_model = h_cumsum - h_cumsum[:, -1:] - 0.5 * h
        else:
            # Interfaces (all except bottom)
            zs_model = h_cumsum - h_cumsum[:, -1:]

        # Interpolate in depth
        model_vals = np.empty(self.numtimes.shape)
        previous_numtime = None
        for i, (numtime, il, ir, wl, wr, z) in enumerate(
            zip(
                self.numtimes,
                self.ileft,
                self.iright,
                self.ileft_weights,
                self.iright_weights,
                self.zs,
            )
        ):
            if previous_numtime != numtime:
                # Interpolate depths and values in time
                zprofile = wl * zs_model[il, :] + wr * zs_model[ir, :]
                profile = wl * values[il, :] + wr * values[ir, :]
                previous_numtime = numtime

            # Interpolate in depth - clamp at top and bottom
            jright = zprofile.searchsorted(z)
            jleft = max(0, jright - 1)
            jright = min(jright, len(zprofile) - 1)
            if jleft == jright:
                model_vals[i] = profile[jright]
            else:
                w = (z - zprofile[jleft]) / (zprofile[jright] - zprofile[jleft])
                model_vals[i] = (1.0 - w) * profile[jleft] + w * profile[jright]
        return model_vals


class Simulation(core.Runner):
    def __init__(
        self,
        setup_dir: Union[os.PathLike[str], str],
        executable: Union[os.PathLike[str], str] = "gotm",
        use_shell: bool = False,
        exclude_files: Iterable[str] = ("*.nc", "*.cache"),
        exclude_dirs: Iterable[str] = ("*",),
        work_dir: Optional[Union[os.PathLike[str], str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Set up a GOTM simulation.

        Args:
            setup_dir: path to the GOTM setup directory
            executable: path to the GOTM executable
            use_shell: whether to run the executable using the shell.
                This argument is passed to subprocess.Popen.
            exclude_files: patterns to exclude files from being copied to the
                working directory
            exclude_dirs: patterns to exclude directories from being copied to
                the working directory
            work_dir: working directory for the simulation.
                If None, a temporary directory is created.
            logger: logger to use for diagnostic messages
        """
        logging.basicConfig(level=logging.INFO)        
        setup_dir = Path(setup_dir)
        if not setup_dir.is_dir():
            raise Exception(f"GOTM setup directory {setup_dir} does not exist.")
        self.setup_dir = setup_dir
        super().__init__(f"gotm({setup_dir})")
        if not use_shell:
            executable = (setup_dir / Path(executable)).resolve()
            if not executable.is_file():
                raise Exception(f"Executable {executable} not found.")
        self.executable = executable
        if work_dir is not None:
            work_dir = Path(work_dir)
        self.work_dir = work_dir
        self.use_shell = use_shell
        self.tempdir = None
        self.symlink = False
        self.exclude_files = exclude_files
        self.exclude_dirs = exclude_dirs
        self.parameters: list[tuple[str, Path, str]] = []
        self.outputs: list[Output] = []
        self.output2extractor: dict[str, OutputExtractor] = {}
        self.logger = logger or logging.getLogger(name=self.name)

        gotmyaml = util.YAMLFile(setup_dir / "gotm.yaml")
        self.start_time: datetime.datetime = gotmyaml["time/start"]
        self.stop_time: datetime.datetime = gotmyaml["time/stop"]

        self.parsed_yaml: dict[Path, util.YAMLFile] = {}

    def get_parameter(
        self, file: Union[os.PathLike[str], str], variable_path: str
    ) -> core.InitializedParameter:
        """
        Get a parameter from a YAML file.

        Args:
            file: path to the YAML file (relative to the setup directory)
            variable_path: path to the variable in the YAML file
        """
        file = Path(file)
        original = self.setup_dir / file
        if not original.is_file():
            raise Exception(f"File {file} not found.")
        try:
            initial_value = util.YAMLFile(original)[variable_path]
        except KeyError:
            raise Exception(f"Parameter {variable_path} not found in {file}.")
        parameter = core.InitializedParameter(f"{self.name}:{file}:{variable_path}", initial_value)
        self.parameters.append((parameter.name, file, variable_path))
        return parameter

    def request_comparison(
        self,
        output_file: Union[os.PathLike[str], str],
        output_expression: str,
        obs_file: Union[os.PathLike[str], str],
        obs_variable: Optional[str] = None,
        obs_depth_variable: Optional[str] = None,
        obs_file_format: util.TextFormat = util.TextFormat.DEPTH_EXPLICIT,
        spinupyears: int = 0,
        mindepth: float = -np.inf,
        maxdepth: float = np.inf,
        cache: bool = True,
    ) -> Optional[core.Comparison]:
        """
        Request comparison of model results and observations for a particular variable.

        Args:
            output_file: path to the output file (relative to the setup directory)
            output_expression: expression to evaluate in the output file.
                It should produce the model equivalent of the provided observations.
                It can reference variables from the output file as well as numpy functions.
            obs_file: path to the file with observations (relative to the current working directory)
            obs_variable: variable name in the NetCDF file with observations
            obs_depth_variable: name of depth variable in the NetCDF file with observations
            obs_file_format: format of the observations file
            spinupyears: number of years to skip from the beginning of the simulation
                when comparing model results to the observations
            mindepth: minimum depth of observations; shallowed observations are ignored
            maxdepth: maximum depth of observations; deeper observations are ignored
            cache: whether to cache the parsed observations for faster access
        """
        output_file = Path(output_file)
        obs_file = Path(obs_file)
        name = (
            f"{self.name}:{output_file}:{output_expression}={obs_file.relative_to(self.setup_dir)}"
        )

        blob = None
        if cache:
            postfix = "" if obs_variable is None else f".{obs_variable}"
            cached_copy = util.CachedFile(obs_file, self.logger, postfix)
            blob = cached_copy.load()

        if isinstance(blob, tuple) and len(blob) == 4:
            times, zs, values, sds = blob
        else:
            self.logger.info(
                f"Reading observations for variable {name} from {obs_file}."
            )
            if not obs_file.is_file():
                raise Exception(f"{obs_file} is not a file.")
            if obs_file.suffix == ".nc":
                if obs_variable is None:
                    raise Exception(
                        "variable argument must be provided since {obs_file} is a NetCDF file."
                    )
                times, zs, values = util.readVariableFromNcFile(
                    obs_file,
                    obs_variable,
                    depth_expression=obs_depth_variable,
                    logger=self.logger,
                    mindepth=mindepth,
                    maxdepth=maxdepth,
                )
                sds = None
            else:
                times, zs, values, sds = util.readVariableFromTextFile(
                    obs_file,
                    format=obs_file_format,
                    logger=self.logger,
                    mindepth=mindepth,
                    maxdepth=maxdepth,
                )

            if cache:
                cached_copy.save((times, zs, values, sds))

        # Filter out observations that lie outide simulated period (excluding spin-up)
        obs_start = self.start_time.replace(year=self.start_time.year + spinupyears)
        valid = np.array([t >= obs_start and t <= self.stop_time for t in times])
        if not valid.any():
            self.logger.warning(
                f"{name}: skipping because {obs_file} has no observations in "
                f" simulated interval {obs_start} - {self.stop_time}."
            )
            return None
        times = [t for t, v in zip(times, valid) if v]
        if zs is not None:
            zs = zs[valid]
        values = values[valid]
        if sds is not None:
            sds = sds[valid]

        self.outputs.append(Output(name, output_file, output_expression, times, zs))
        return core.Comparison(name, self, values, sd=sds)

    def on_start(self) -> None:
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.work_dir = Path(tempfile.mkdtemp(prefix="gotmopt", dir=self.tempdir))
            atexit.register(shutil.rmtree, self.work_dir, True)

        self.logger.info(f"Copying model setup to {self.work_dir}...")
        util.copy_directory(
            self.setup_dir,
            self.work_dir,
            exclude_files=self.exclude_files,
            exclude_dirs=self.exclude_dirs,
            symlink=self.symlink,
            logger=self.logger,
        )

        for _, file, _ in self.parameters:
            if file not in self.parsed_yaml:
                local_file = self.work_dir / file
                bck = util.backup_file(local_file)
                self.logger.info(f"Backed up {file} to {bck.name}.")
                self.parsed_yaml[file] = util.YAMLFile(local_file)

        self.output2extractor.clear()

    def record_output(
        self, output_file: Union[os.PathLike[str], str], output_expression: str
    ) -> core.Metric:
        """
        Request an expression of output variables to be recorded.

        Args:
            output_file: path to the output file (relative to the setup directory)
            output_expression: expression to evaluate in the output file.
                It should return a scalar value.

        Returns:
            A metric object that can be used to refer the output when setting
            up an optimization or sensitivity analysis.
        """
        output_file = Path(output_file)
        name = f"{self.name}:{output_file}:{output_expression}"
        self.outputs.append(Output(name, output_file, output_expression, None, None))
        return core.Metric(name, self)

    def __call__(
        self, name2value: Mapping[str, float], show_output: bool = False
    ) -> Mapping[str, Any]:
        assert self.work_dir is not None

        update_yaml: set[util.YAMLFile] = set()
        for name, file, path in self.parameters:
            if name in name2value:
                yaml = self.parsed_yaml[file]
                value = name2value[name]
                if isinstance(value, (np.ndarray, np.generic)):
                    value = value.item()
                yaml[path] = value
                update_yaml.add(yaml)
        for yaml in update_yaml:
            yaml.save()

        returncode = util.run_program(
            self.executable,
            self.work_dir,
            logger=self.logger,
            use_shell=self.use_shell,
            show_output=show_output,
        )

        if returncode != 0:
            raise Exception(f"Simulation failed with return code {returncode}.")

        outputpath2nc: dict[Path, util.NcDict] = {}
        for output in self.outputs:
            ncpath = self.work_dir / output.file
            if not ncpath.is_file():
                raise Exception(f"Output file {ncpath} was not created.")
            outputpath2nc[output.file] = util.NcDict(ncpath)

        results: dict[str, Optional[Union[float, np.ndarray]]] = {}
        for output in self.outputs:
            wrappednc = outputpath2nc[output.file]

            if output.name not in self.output2extractor:
                self.output2extractor[output.name] = OutputExtractor(
                    output, wrappednc, self.logger
                )
            extractor = self.output2extractor[output.name]
            results[output.name] = extractor(wrappednc)

        for wrappednc in outputpath2nc.values():
            wrappednc.finalize()

        for transform in self.transforms:
            transform(self.logger, results)

        simple_results = {}
        for k, v in results.items():
            if isinstance(v, np.generic) or (isinstance(v, np.ndarray) and v.size == 1):
                # Unpack numpy scalars to simple Python data types (float, int, etc.)
                v = v.item()
            simple_results[k] = v

        return simple_results
