from typing import Optional, Union, Mapping, Iterator, Sequence, Any, cast, Iterable
from pathlib import Path
import os
import logging
import datetime
import timeit
import sys
import subprocess
import types
import enum
import shutil

import numpy as np
import netCDF4
import yaml


# YAML: no special treatment of "on" and "off"
# In PyYAML (YAML spec 1.1), they are Booleans
del yaml.loader.Loader.yaml_implicit_resolvers["o"]
del yaml.loader.Loader.yaml_implicit_resolvers["O"]


# YAML: when dumping, represent None by an empty string
def none_representer(self: yaml.SafeDumper, _: None) -> Any:
    return self.represent_scalar("tag:yaml.org,2002:null", "")


yaml.SafeDumper.add_representer(type(None), none_representer)


def backup_file(file: Path) -> Path:
    """Create a backup of a file by copying it to a new file with a .bck suffix.

    Args:
        file: Path to the file to backup.

    Returns:
        Path to the backup file.
    """
    i = 0
    while 1:
        bck = file.with_suffix(f"{file.suffix}.bck{i}")
        if not bck.is_file():
            break
        i += 1
    shutil.copy2(file, bck)
    return bck


def copy_directory(
    src_dir: Path,
    dst_dir: Path,
    *,
    exclude_files: Iterable[str] = (),
    exclude_dirs: Iterable[str] = (),
    symlink: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Copy files from src_dir to dst_dir, excluding files and directories
    that match the patterns in exclude_files and exclude_dirs, respectively.

    Args:
        src_dir: Source directory.
        dst_dir: Destination directory.
        exclude_files: Patterns for files to exclude.
        exclude_dirs: Patterns for directories to exclude.
        symlink: Whether to create symlinks instead of copying files.
        logger: Logger to use for diagnostic messages.
    """
    logger = logger or logging.getLogger()
    copy_function = shutil.copy2 if not symlink else os.symlink
    for src in src_dir.iterdir():
        is_dir = src.is_dir()
        local_name = src.relative_to(src_dir)
        exclude = exclude_dirs if is_dir else exclude_files
        for pattern in exclude:
            if src.match(pattern):
                logger.info(
                    f"   {local_name}: skipped because it matches pattern {pattern}"
                )
                break
        else:
            dst = dst_dir / local_name
            if is_dir:
                shutil.copytree(src, dst, copy_function=copy_function)
            else:
                copy_function(src, dst)


class TextFormat(enum.Enum):
    DEPTH_EXPLICIT = 0
    DEPTH_INDEPENDENT = 1
    GOTM_PROFILES = 2


def readVariableFromTextFile(
    file: Path,
    *,
    format: TextFormat,
    logger: logging.Logger,
    mindepth: float = -np.inf,
    maxdepth: float = np.inf,
) -> tuple[
    Sequence[datetime.datetime], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]
]:
    """Read a variable from a text file.

    Args:
        file: Path to the text file.
        format: Format of the text file.
        logger: Logger to use for diagnostic messages.
        mindepth: Minimum depth to include.
        maxdepth: Maximum depth to include.

    Returns:
        Tuple of time, depth, values, and standard deviations
    """
    times: list[datetime.datetime] = []
    zs: list[float] = []
    values: list[float] = []
    sds: list[float] = []

    def parse_line(line: str) -> None:
        dt = datetime.datetime.fromisoformat(line[:19])
        data = line[20:].split()
        if format == TextFormat.DEPTH_EXPLICIT:
            # We now expect the z coordinate (<0 below surface)
            item = data.pop(0)
            z = float(item)
            if not np.isfinite(z):
                raise Exception(f"Depth {item} is not a valid number")
            if -z < mindepth or -z > maxdepth:
                return
            zs.append(z)
        times.append(dt)  # only append now that we know depth is in range
        item = data.pop(0)
        value = float(item)
        if not np.isfinite(value):
            raise Exception(f"Observed value {item} is not a valid number.")
        values.append(value)
        if data:
            item = data.pop(0)
            sd = float(item)
            if not np.isfinite(sd):
                raise Exception(f"Standard deviation {item} is not a valid number.")
            sds.append(sd)

    with file.open() as f:
        for iline, line in enumerate(f):
            if (iline + 1) % 20000 == 0:
                logger.info(f"Read {file} upto line {iline}.")
            line = line.split("#", 1)[0].strip()
            if line:
                try:
                    parse_line(line)
                except Exception as e:
                    raise Exception(f"Error on line {iline+1} of {file}: {e}\n{line}")

    sds_ = np.array(sds) if sds else None
    zs_ = np.array(zs) if format == TextFormat.DEPTH_EXPLICIT else None
    return times, zs_, np.array(values), sds_


class NcDict(Mapping[str, np.ndarray]):
    def __init__(self, file: Union[os.PathLike[str], str]):
        """Create a dictionary-like interface to a NetCDF file. The file is
        opened  when the object is created and closed when it is finalized."""
        self.file = Path(file)
        self.nc = netCDF4.Dataset(self.file)
        self.nc.set_auto_mask(False)
        self.cache: dict[str, np.ndarray] = {}
        self.dimensions: Optional[set[str]] = None

    def finalize(self) -> None:
        self.nc.close()

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.cache or self.dimensions is not None:
            ncvar = self.nc.variables[key]
            if self.dimensions is not None:
                self.dimensions.update(ncvar.dimensions)
            self.cache[key] = ncvar[...]
        return self.cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.nc.variables)

    def __len__(self) -> int:
        return len(self.nc.variables)

    def __contains__(self, key: object) -> bool:
        return key in self.nc.variables

    def __enter__(self) -> "NcDict":
        return self

    def __exit__(
        self,
        type: Optional[type],
        value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        self.finalize()

    def eval(
        self,
        expression: Union[str, types.CodeType],
        no_trailing_singletons: bool = True,
    ) -> np.ndarray:
        namespace = {
            "filter_by_time": lambda values, months: filter_by_time(
                values, self["time"], self.nc.variables["time"].units, months
            )
        }
        for n in dir(np):
            namespace[n] = getattr(np, n)
        try:
            data: np.ndarray = eval(expression, namespace, self)
        except NameError as e:
            raise NameError(f"{e.args[0]} in {self.file}") from None
        data = np.asarray(data)
        if no_trailing_singletons:
            while data.ndim > 0 and data.shape[-1] == 1:
                data = data[..., 0]
        return data


def filter_by_time(
    values: np.ndarray, time: np.ndarray, time_units: str, months: Iterable[int] = ()
) -> np.ndarray:
    dts = netCDF4.num2date(time, time_units)
    assert isinstance(dts, np.ndarray)
    current_months = np.array([dt.month for dt in dts], dtype=int)
    valid = np.zeros(current_months.shape, dtype=bool)
    for month in months:
        valid |= current_months == month
    return values[valid, ...]


def readVariableFromNcFile(
    file: Path,
    expression: str,
    *,
    depth_expression: Optional[str],
    logger: logging.Logger,
    mindepth: float = -np.inf,
    maxdepth: float = np.inf,
    time_name: str = "time",
) -> tuple[Sequence[datetime.datetime], Optional[np.ndarray], np.ndarray]:
    """Read a variable from a netCDF file.

    Args:
        file: Path to the netCDF file.
        expression: Expression to evaluate.
        depth_expression: Expression to evaluate for depth.
        logger: Logger to use for diagnostic messages.
        mindepth: Minimum depth to include.
        maxdepth: Maximum depth to include.
        time_name: Name of the time variable.

    Returns:
        Tuple of time, depth, and values.
    """
    with NcDict(file) as wrapped_nc:
        nctime = wrapped_nc.nc.variables[time_name]
        numtimes = nctime[:]
        time_unit = nctime.units
        values: np.ndarray = wrapped_nc.eval(expression)
        assert values.ndim == 1
        zs: Optional[np.ndarray] = None
        if depth_expression is not None:
            zs = wrapped_nc.eval(depth_expression)

    mask = np.ma.getmask(values)
    if mask is not np.ma.nomask:
        valid = ~mask
        numtimes, values = numtimes[valid], values[valid]
        if zs is not None:
            zs = zs[valid]
    times = netCDF4.num2date(numtimes, time_unit, only_use_cftime_datetimes=False)
    return cast(Sequence[datetime.datetime], times), zs, values


def run_program(
    executable: Union[os.PathLike[str], str],
    rundir: Path,
    *,
    logger: logging.Logger,
    use_shell: bool = False,
    show_output: bool = True,
    args: Iterable[str] = (),
):
    """Run a program in a subprocess.

    Args:
        executable: Path to the executable.
        rundir: Directory in which to run the program.
        logger: Logger to use for diagnostic messages.
        use_shell: Whether to use a shell to run the program.
        show_output: Whether to show the output of the program.

    Returns:
        The return code of the program.
    """
    time_start = timeit.default_timer()
    logger.debug("Starting model run...")

    exe = str(executable)
    args = [exe] + list(args)
    if exe.endswith(".py"):
        args = [sys.executable] + args
        use_shell = False
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.IDLE_PRIORITY_CLASS
    proc = subprocess.Popen(
        args,
        cwd=rundir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=use_shell,
        universal_newlines=True,
        **kwargs,
    )
    assert proc.stdout is not None

    if show_output:
        while 1:
            line = proc.stdout.readline()
            if line == "":
                break
            logger.debug(line.rstrip("\n"))
    stdout_data, _ = proc.communicate()

    elapsed = timeit.default_timer() - time_start
    logger.debug(f"Model run took {elapsed:.1f} s.")
    if proc.returncode != 0:
        last_lines = stdout_data.rsplit("\n", 10)[-10:]
        last_output = "\n".join([f"> {line}" for line in last_lines])
        raise Exception(f"{exe} returned non-zero code {proc.returncode}.", last_output)


class YAMLFile:
    def __init__(self, file: Union[os.PathLike[str], str]):
        """Create a dictionary-like interface to a YAML file. The file is
        parsed when the object is created.

        Args:
            file: Path to the YAML file.
        """
        self.file = Path(file)
        with self.file.open(encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def save(self) -> None:
        with self.file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.data, f, default_flow_style=False)

    def __getitem__(self, key: str) -> Any:
        path_comps = key.split("/")
        target_dict = self.data
        for comp in path_comps:
            target_dict = target_dict[comp]
        return target_dict

    def __setitem__(self, key: str, value: Any) -> None:
        path_comps = key.split("/")
        target_dict = self.data
        for i, comp in enumerate(path_comps[:-1]):
            if comp not in target_dict:
                target_dict[comp] = {}
            target_dict = target_dict[comp]
        target_dict[path_comps[-1]] = value
