from typing import (
    Optional,
    Union,
    Callable,
    Any,
    Iterable,
    TypeVar,
    Mapping,
    NamedTuple,
    TYPE_CHECKING,
    Sequence,
)
import logging
import functools
import asyncio
import sys
import concurrent.futures
from pathlib import Path
import os

import numpy as np
import numpy.typing as npt
import scipy.stats

try:
    import mpi4py.futures
except ImportError:
    mpi4py = None

from . import record

if TYPE_CHECKING:
    import matplotlib.axes


class Runner:
    def __init__(self, name: str, work_dir: Union[os.PathLike[str], str, None] = None):
        self.name = name
        self.transforms: list[Callable[[Mapping[str, float], dict[str, Any]], None]] = (
            []
        )
        if work_dir is not None:
            work_dir = Path(work_dir)
        self.work_dir = work_dir

    def __call__(self, name2value: Mapping[str, float]) -> Mapping[str, Any]:
        raise NotImplementedError

    def on_start(self) -> None:
        pass


class RunnerPool:
    active: dict[str, Runner] = {}

    def __init__(
        self,
        runners: Mapping[str, Runner],
        *,
        distributed: Optional[bool] = None,
        max_workers: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        logging.basicConfig(level=logging.INFO)
        logger = logger or logging.getLogger(self.__class__.__name__)
        if distributed is None:
            distributed = "MPI4PY_FUTURES_MAX_WORKERS" in os.environ
            if distributed:
                logger.info(
                    "Running in distributed mode as MPI4PY_FUTURES_MAX_WORKERS is set."
                )
            else:
                logger.info("Running locally as MPI4PY_FUTURES_MAX_WORKERS is not set.")
        if distributed and mpi4py is None:
            raise ImportError(
                "mpi4py is not installed. Please install it to use distributed mode."
            )

        self.runners = runners
        kwargs = dict(
            max_workers=max_workers,
            initializer=self._init,
            initargs=(tuple(runners.values()),),
        )
        self.executor: concurrent.futures.Executor
        if distributed:
            self.executor = mpi4py.futures.MPIPoolExecutor(**kwargs)
            workers = self.executor.starmap(
                self._get_worker_name, [()] * self.executor.num_workers
            )
            work2count = {}
            for worker in workers:
                work2count[worker] = work2count.get(worker, 0) + 1
            logger.info(
                f"Using {self.executor.num_workers} workers"
                f" on {len(work2count)} machines:"
            )
            for name in sorted(work2count.keys()):
                logger.info(f"  {name}: {work2count[name]} workers")
        else:
            self.executor = concurrent.futures.ProcessPoolExecutor(**kwargs)
            logger.info(f"Using {self.executor._max_workers} workers.")

    async def __call__(
        self, name2value: Mapping[str, float], **kwargs
    ) -> dict[str, Any]:
        futures = [
            self.executor.submit(self._run, r, name2value, **kwargs)
            for r in self.runners
        ]
        results = await asyncio.gather(*map(asyncio.wrap_future, futures))
        name2output: dict[str, Any] = {}
        for result in results:
            name2output.update(result)
        return name2output

    @staticmethod
    def _init(runners: Iterable[Runner]) -> None:
        """Start the given runners on this worker and add them to the runner's
        active dictionary.

        Args:
            runners: The runners to start.
        """
        logging.getLogger().setLevel(logging.WARNING)
        for r in runners:
            r.on_start()
            RunnerPool.active[r.name] = r

    @staticmethod
    def _get_worker_name() -> None:
        comm = mpi4py.futures.get_comm_workers()
        comm.Barrier()
        return mpi4py.MPI.Get_processor_name()

    @staticmethod
    def _run(name: str, name2value: Mapping[str, float], **kwargs) -> Mapping[str, Any]:
        """Run the runner with the given name and parameter values.
        All runners must be started on this worker with _init before calling
        this function.

        Args:
            name: The name of the runner to run.
            name2value: The parameter values to use.

        Returns:
            The output of the runner.
        """
        return RunnerPool.active[name](name2value, **kwargs)


class Metric:
    def __init__(self, name: str, runner: Runner):
        self.name = name
        self.runner = runner

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class Comparison(Metric):
    def __init__(
        self,
        name: str,
        runner: Runner,
        obs_vals: np.ndarray,
        sd: Optional[Union[float, np.ndarray]] = None,
    ):
        self.obs_vals = obs_vals
        self.sd = sd
        super().__init__(name, runner)


class Calculator:
    def __init__(
        self,
        returns: Iterable[str],
        fn: Callable[..., tuple[float]],
        *args: Any,
        **kwargs: Any,
    ):
        self.returns = returns
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.ready = False

    def reset(self) -> None:
        self.ready = False

    def update(self, name2value: dict[str, float]) -> None:
        if self.ready:
            return
        final_args = []
        for arg in self.args:
            if isinstance(arg, Parameter):
                arg = name2value[arg.name]
            final_args.append(arg)
        final_kwargs = {}
        for k, v in self.kwargs.items():
            if isinstance(v, Parameter):
                v = name2value[v.name]
            final_kwargs[k] = v
        result = self.fn(*final_args, **final_kwargs)
        for name, value in zip(self.returns, result):
            name2value[name] = value
        self.ready = True


F = TypeVar("F")


def _wrap_result_in_tuple(fn: Callable[..., F], *args: Any, **kwargs: Any) -> tuple[F]:
    return (fn(*args, **kwargs),)


class Parameter:
    def __init__(self, name: str):
        self.name = name
        self.status: int = 0
        self.calculator: Optional[Calculator] = None
        self.dependencies: list[Parameter] = []
        self.dependents: list[Parameter] = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def reset(self) -> None:
        """Register that this parameter has changed.This ensures that any
        dependent parameters are recalculated."""
        self.status = 0
        if self.calculator is not None:
            self.calculator.reset()
        for d in self.dependents:
            d.reset()

    def update(self, name2value: dict[str, float]) -> None:
        if self.status == 1:
            return
        if self.calculator is not None:
            assert self.status != -1, "Circular dependency detected"
            self.status = -1
            for dep in self.dependencies:
                if dep.name not in name2value:
                    dep.update(name2value)
            self.calculator.update(name2value)
            assert self.name in name2value, "Calculator did not set {self.name}."
        self.status = 1

        for d in self.dependents:
            d.update(name2value)

    def infer(self, fn: Callable[..., float], *args: Any, **kwargs: Any) -> None:
        """Specify that this parameter will be calculated dynamically by the
        given function.

        Args:
            fn: The function that will return the parameter value.
            *args: Arguments to the function. Any parameter objects in the
                list will be replaced by their values. If these parameters
                are themselves calculated dynamically, they will be updated
                before the function is called.
            **kwargs: The named parameters to pass to the function. Any
                parameter objects will be replaced by their values.
        """
        wrapped_fn = functools.partial(_wrap_result_in_tuple, fn)
        self.link((self,), wrapped_fn, *args, **kwargs)

    @staticmethod
    def link(
        outputs: tuple["Parameter"],
        fn: Callable[..., tuple[float]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Specify that the specified parameters will be calculated
        dynamically by the given function.

        Args:
            outputs: The parameters to calculate.
            fn: The function that will return a tuple with values for the
                specified parameters.
            *args: Arguments to the function. Any parameter objects in the
                list will be replaced by their values. If these parameters
                are themselves calculated dynamically, they will be updated
                before the function is called.
            **kwargs: The named parameters to pass to the function. Any
                parameter objects will be replaced by their values.
        """
        calculator = Calculator([o.name for o in outputs], fn, *args, **kwargs)
        inputs = [a for a in args + tuple(kwargs.values()) if isinstance(a, Parameter)]
        for p in outputs:
            assert p.calculator is None, f"Parameter {p.name} already has a calculator."
            p.calculator = calculator
            p.dependencies.extend(inputs)
        for p in inputs:
            p.dependents.extend(outputs)


class InitializedParameter(Parameter):
    def __init__(self, name: str, initial_value: float):
        super().__init__(name)
        self.initial_value = initial_value


def _null_transform(x: float) -> float:
    return x


class _TargetedParameter(NamedTuple):
    parameter: Parameter
    dist: scipy.stats.rv_continuous
    fwt: Callable[[float], float] = _null_transform
    bwt: Callable[[float], float] = _null_transform


class Prior:
    def logpdf(self, name2value: Mapping[str, float]) -> float:
        raise NotImplementedError


class UnivariatePrior(Prior):
    def __init__(self, parameter: _TargetedParameter) -> None:
        self.name = parameter.parameter.name
        self.dist = parameter.dist
        self.logscale = parameter.fwt == np.log

    def logpdf(self, name2value: Mapping[str, float]) -> float:
        value = name2value[self.name]
        lnl = self.dist.logpdf(value)
        if self.logscale:
            lnl += np.log(value)
        return lnl


class Experiment:
    def __init__(
        self,
        *,
        db_file: Optional[Union[str, Path]] = None,
        distributed: Optional[bool] = None,
        max_workers: Optional[int] = None,
        seed: Optional[Union[int, Sequence[int]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        logging.basicConfig(level=logging.INFO)
        self.parameters: list[_TargetedParameter] = []
        self.runners: dict[str, Runner] = {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if db_file is None:
            db_file = Path(sys.argv[0]).with_suffix(".results.db")
        self.recorder = record.Recorder(db_file)
        self.distributed = distributed
        self.max_workers = max_workers
        self.pool: Optional[RunnerPool] = None
        self.row_metadata: dict[str, Any] = {}
        ss = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(ss)
        self.priors: list[Prior] = []
        self.global_transforms: list[
            Callable[[Mapping[str, float], dict[str, Any]], None]
        ] = []
        self._config = dict(seed=ss.entropy, global_transforms=self.global_transforms)

    def get_parameter_bounds(
        self, transform: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """The minimum and maximum value of the parameters after applying any transforms"""
        bounds = np.empty((2, len(self.parameters)))
        for i, target in enumerate(self.parameters):
            bounds[:, i] = target.dist.support()
            if transform:
                bounds[:, i] = target.fwt(bounds[:, i])
        return bounds[0], bounds[1]

    def sample_parameters(
        self, n: int
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Sample n parameter sets from the parameter distributions."""
        sample = np.empty((n, len(self.parameters)), dtype=float)
        for i, p in enumerate(self.parameters):
            sample[:, i] = p.fwt(p.dist.rvs(10 * len(self.parameters)))
        return sample

    def add_parameter(
        self,
        parameter: Union[str, Parameter],
        minimum: Union[float, scipy.stats.rv_continuous],
        maximum: Optional[float] = None,
        logscale: bool = False,
    ) -> Parameter:
        """Mark a parameter as target for optimization or sensitivity analysis.

        Args:
            parameter: The parameter to select.
                This can be a name of a new parameter or an existing parameter
                from a runner.
            minimum: The minimum value of the parameter.
            maximum: The maximum value of the parameter.
            logscale: Whether to vary the parameter value on a logarithmic scale.
        """
        if isinstance(parameter, str):
            parameter = Parameter(parameter)
        if maximum is None:
            assert isinstance(
                minimum, scipy.stats.distributions.rv_frozen
            ), f"{type(minimum)} is not a distribution."
            dist = minimum
            logscale = dist.dist.name in ("lognorm", "loguniform")
        else:
            assert minimum < maximum, "Minimum must be less than maximum."
            if logscale:
                dist = scipy.stats.loguniform(minimum, maximum)
            else:
                dist = scipy.stats.uniform(minimum, maximum - minimum)
            assert dist.support() == (minimum, maximum)
        minimum, maximum = dist.support()
        kwargs: dict[str, Any] = {}
        if logscale:
            assert (
                minimum >= 0.0
            ), f"{parameter.name}: minimum must be non-negative for logscale, but is {minimum}."
            kwargs.update(fwt=np.log, bwt=np.exp)
        target = _TargetedParameter(parameter, dist, **kwargs)
        self.parameters.append(target)
        self.priors.append(UnivariatePrior(target))
        return parameter

    async def start(self, record: bool) -> None:
        """Start the optimization or sensitivity analysis.

        This starts all worker processes, performs a single evaluation with the
        median parameter values, and initializes the recorder. The evaluation
        serves as check that the runners (model configurations) are working
        correctly, and also to determine which outputs are available.
        """
        self.pool = RunnerPool(
            self.runners, distributed=self.distributed, max_workers=self.max_workers
        )

        parmin, parmax = self.get_parameter_bounds()
        par_info = dict(
            names=[p.parameter.name for p in self.parameters],
            minimum=[float(v) if np.isfinite(v) else None for v in parmin],
            maximum=[float(v) if np.isfinite(v) else None for v in parmax],
            logscale=[p.fwt == np.log for p in self.parameters],
        )
        median = [p.fwt(p.dist.median()) for p in self.parameters]
        name2value = self.unpack_parameters(median)
        self.logger.info(
            "Running single initial evaluation with median parameter set (in serial)."
        )
        name2output = await self.async_eval(name2value)
        config = self._config | dict(parameters=par_info, runners=self.runners)
        self.recorder.start(config, self.row_metadata | name2value, name2output)
        if record:
            self.recorder.record(
                exception=None, **name2value, **name2output, **self.row_metadata
            )

    def unpack_parameters(self, values: npt.NDArray[np.float64]) -> dict[str, float]:
        """Unpack the vector with parameter values into a dictionary
        and add any dynamically calculated parameters.
        """
        assert len(values) == len(self.parameters)
        name2value: dict[str, float] = {}
        for target, value in zip(self.parameters, values):
            value = target.bwt(value)
            if isinstance(value, (np.ndarray, np.generic)):
                value = value.item()
            name2value[target.parameter.name] = value
            target.parameter.reset()
        for target in self.parameters:
            target.parameter.update(name2value)
        return name2value

    async def async_eval(
        self, values: Union[Mapping[str, float], np.ndarray]
    ) -> Mapping[str, Any]:
        """Evaluate the runners with the given parameter values

        Args:
            values: The parameter values to evaluate.

        Returns:
            A dictionary with the combind output of all runners.
        """
        name2value = (
            values if isinstance(values, Mapping) else self.unpack_parameters(values)
        )
        assert self.pool is not None
        self.logger.info(f"Running parameter set {name2value}.")
        exception: Optional[Exception] = None
        try:
            name2output = await self.pool(name2value)
            for transform in self.global_transforms:
                transform(name2value, name2output)
        except Exception as e:
            exception = e
            name2output = {}
            raise
        finally:
            self.recorder.record(
                exception=exception, **name2value, **name2output, **self.row_metadata
            )
        return name2output

    def eval(self, values: Union[Mapping[str, float], np.ndarray]) -> Mapping[str, Any]:
        return asyncio.run(self.async_eval(values))


class Plotter:
    POSTFIX = ":plotter"

    def __init__(
        self, sharex: Optional["Plotter"] = None, sharey: Optional["Plotter"] = None
    ):
        self.obs_values: Optional[np.ndarray] = None
        self.scale_factor = 1.0
        self.sd: Optional[np.ndarray] = None
        self.sharex = sharex
        self.sharey = sharey

    def plot(self, ax: "matplotlib.axes.Axes") -> None:
        raise NotImplementedError
