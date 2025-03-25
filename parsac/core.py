from typing import Optional, Union, Callable, Any, Iterable, TypeVar, Mapping, NamedTuple
import logging
import functools
import asyncio
import concurrent.futures
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

from . import record


class Metric:
    def __init__(self, name: str, runner: "Runner"):
        self.name = name
        self.runner = runner

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class Comparison(Metric):
    def __init__(
        self,
        name: str,
        runner: "Runner",
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
    minimum: float
    maximum: float
    fwt: Callable[[float], float] = _null_transform
    bwt: Callable[[float], float] = _null_transform


class Experiment:
    def __init__(
        self,
        *,
        db_file: Optional[Union[str, Path]] = None,
        max_workers: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        logging.basicConfig(level=logging.INFO)        
        self.parameters: list[_TargetedParameter] = []
        self.runners: set[Runner] = set()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.last_result: Optional[dict[str, Any]] = None
        if db_file is None:
            db_file = Path(sys.argv[0]).with_suffix(".results.db")
        self.recorder = record.Recorder(db_file)
        self.max_workers = max_workers
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

    @property
    def minbounds(self) -> np.ndarray:
        """The minimum value of the parameters after applying any transforms"""
        return np.array([target.fwt(target.minimum) for target in self.parameters])

    @property
    def maxbounds(self) -> np.ndarray:
        """The maximum value of the parameters after applying any transforms"""
        return np.array([target.fwt(target.maximum) for target in self.parameters])

    def add_parameter(
        self,
        parameter: Union[str, Parameter],
        minimum: float,
        maximum: float,
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
        assert minimum < maximum, "Minimum must be less than maximum."
        kwargs: dict[str, Any] = {}
        if logscale:
            assert minimum > 0, "Minimum must be positive for logscale."
            kwargs.update(fwt=np.log10, bwt=lambda x: 10**x)
        self.parameters.append(
            _TargetedParameter(parameter, minimum, maximum, **kwargs)
        )
        return parameter

    def start(self) -> None:
        """Start the optimization or sensitivity analysis.
        
        This starts all worker processes, performs a single evaluation with the
        median parameter values, and initializes the recorder. The evaluation
        serves as check that the runners (model configurations) are working
        correctly, and also to determine which outputs are available.
        """
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=Runner.start_all,
            initargs=tuple(self.runners),
        )
        par_info = dict(
            names=[p.parameter.name for p in self.parameters],
            minimum=[p.minimum for p in self.parameters],
            maximum=[p.maximum for p in self.parameters],
            logscale=[p.fwt == np.log10 for p in self.parameters],
        )
        p = self.unpack_parameters(0.5 * (self.minbounds + self.maxbounds))
        result = self.eval({})
        self.recorder.start(p, result, {"parameters": par_info})

    def unpack_parameters(self, values: npt.NDArray[np.float64]) -> Mapping[str, float]:
        """Unpack the vector with parameter values into a dictionary
        and add any dynamically calculated parameters.
        """
        assert len(values) == len(self.parameters)
        name2value: dict[str, float] = {}
        for target, value in zip(self.parameters, values):
            name2value[target.parameter.name] = target.bwt(value)
            target.parameter.reset()
        for target in self.parameters:
            target.parameter.update(name2value)
        return name2value

    def add_global_metrics(
        self, name2value: Mapping[str, float], name2output: dict[str, float]
    ) -> None:
        """Allow inheriting classes to add global metrics to the output.
        For example, the sum of all likelihood contributions (the target
        metric), in the case of optimization.
        """
        pass

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
        self.logger.info(f"Running parameter set {name2value}.")
        assert self.executor is not None
        cor = [
            self.executor.submit(Runner.run, r.name, name2value) for r in self.runners
        ]
        results = await asyncio.gather(*map(asyncio.wrap_future, cor))
        name2output: dict[str, Any] = {}
        for result in results:
            name2output.update(result)
        self.add_global_metrics(name2value, name2output)
        self.recorder.record(name2value, name2output)
        return name2output

    def eval(self, values: Union[Mapping[str, float], np.ndarray]) -> Mapping[str, Any]:
        return asyncio.run(self.async_eval(values))


class Runner:
    active: dict[str, "Runner"] = {}

    @staticmethod
    def start_all(*runners: "Runner") -> None:
        """Start the given runners and add them to the global active dictionary."""
        logging.getLogger().setLevel(logging.ERROR)
        for r in runners:
            r.on_start()
            Runner.active[r.name] = r

    @staticmethod
    def run(name: str, name2value: Mapping[str, float]) -> Mapping[str, Any]:
        """Run the runner with the given name and parameter values.
        The runner must be started with start_all before calling this method.
        """
        return Runner.active[name](name2value)

    def __init__(self, name: str):
        self.name = name
        self.transforms: list[Callable[[logging.Logger, dict[str, Any]], None]] = []

    def __call__(self, name2value: Mapping[str, float]) -> Mapping[str, Any]:
        raise NotImplementedError

    def on_start(self) -> None:
        pass
