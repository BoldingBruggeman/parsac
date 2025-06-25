import asyncio
from typing import Optional, Any, Iterable, Union
import os
from pathlib import Path

import numpy as np
import numpy.linalg
import scipy.stats

import SALib.analyze.sobol
import SALib.sample.sobol
import SALib.analyze.morris
import SALib.sample.morris

from .. import core


class SensitivityAnalysis(core.Experiment):
    """A sensitivity analysis.

    To configure the analysis, add parameters to include
    by calling :meth:`~parsac.core.Experiment.add_parameter`,
    and add jobs that calculate target metrics by calling
    :meth:`add_job`. These target metrics should be
    scalar outputs of the job. Some jobs may require you to specify
    explicitly which outputs to record by calling methods such as
    :meth:`parsac.job.gotm.Simulation.record_output`.
    """

    def __init__(self, **kwargs) -> None:
        """
        Args:
            kwargs: Additional keyword arguments to passed to
                :class:`~parsac.core.Experiment`.
        """
        super().__init__(**kwargs)
        self.targets: list[str] = []

    def add_job(self, runner: core.Runner) -> None:
        """Add a job that takes parameter as input and produces scalar
        outputs, suitable as target for sensitivity analysis."""
        if runner.name in self.runners:
            assert runner is self.runners[runner.name]
        self.runners[runner.name] = runner

    def run(self, **kwargs) -> Union[np.ndarray, dict[str, Any]]:
        """Run the sensitivity analysis.

        Args:
            **kwargs: Additional keyword arguments to pass to :func:`run_async`.
        """
        return asyncio.run(self.run_async(**kwargs))

    async def run_async(
        self,
        work_dirs: Union[Iterable[Union[os.PathLike[Any], str]], str, None] = None,
        return_details: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, dict[str, Any]]:
        """Run the sensitivity analysis asynchronously.

        Args:
            work_dirs: A list of directories to use to store setups and results per
                parameter set, or a format string with a single placeholder that incorporates
                the parameter set index ``i``, for instance, ``workdirs="{i:03}"`` to place results
                in directories ``000``, ``001``, ... If this argument is not provided, temporary
                directories will be used to store results while evaluating the parameter sets.
            return_details: If ``True``, return a dictionary with all results from the analysis.
            **kwargs: Additional keyword arguments to pass to the analysis method.

        Returns:
            If ``return_details`` is ``False``, an array that specifies the sensitivity
            of each target (second dimension) to each parameter (first dimension).
            If ``return_details`` is ``True``, a dictionary with raw results of the analysis
            (values) for each target (keys).
        """
        if not self.parameters:
            raise Exception("No parameters have been added")
        self.targets.extend(await self.start(record=False))
        if not self.targets:
            raise Exception("No targets have been added to any job")

        # Sample
        X = self._sample()

        # Run
        self.logger.info(f"Evaluating targets for {X.shape[0]} parameter sets")
        if isinstance(work_dirs, str):
            work_dirs = [work_dirs.format(i=i) for i in range(X.shape[0])]
            if len(set(work_dirs)) != len(work_dirs):
                raise Exception(
                    "The work_dirs format string must produce unique directories"
                    " for each member i, for instance with '{i:03}'."
                )
        results = await self.batch_eval(X, work_dirs)
        Y = np.empty((X.shape[0], len(self.targets)))
        for i, r in enumerate(results):
            assert not isinstance(r, BaseException)
            Y[i, :] = [r[n] for n in self.targets]
        self.stop()

        # Analyze
        analyses: dict[str, Any] = {}
        sensitivities = np.full((len(self.parameters), len(self.targets)), np.nan)
        self.logger.info(
            "Sensitivities per target and parameter (higher means more sensitive):"
        )
        for j, n in enumerate(self.targets):
            if Y[:, j].min() == Y[:, j].max():
                self.logger.warning(
                    f"{n}: skipping because no variation in target detected."
                )
                continue
            self.logger.info(n)
            analysis = self._analyze(X, Y[:, j], **kwargs)
            sens = self._extract_sensitivity_metric(analysis)
            assert len(sens) == len(self.parameters)
            for p, s in sorted(zip(self.parameters, sens), key=lambda x: -abs(x[1])):
                self.logger.info(f"  {p.parameter.name}: {s:.5g}")
            analyses[n] = analysis
            sensitivities[:, j] = sens
        return analyses if return_details else sensitivities

    def _sample(self) -> np.ndarray:
        raise NotImplementedError

    def _analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def _extract_sensitivity_metric(self, analysis: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class MVR(SensitivityAnalysis):
    """Sensitivity analysis based on Monte Carlo sampling and linear
    regression, as described by `Saltelli et al.
    <https://doi.org/10.1002/9780470725184>`__ (section 1.2.5)

    The sensitivity metric that is returned by default by
    :meth:`~SensitivityAnalysis.run` and :meth:`~SensitivityAnalysis.run_async`
    is the value of the standardized regression coefficients.
    These can take negative values, indicating that the parameter
    has an inverse relationship with the target metric.
    When ranking parameters by their sensitivity, the absolute value
    of the coefficients is used.
    """

    def __init__(self, n: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def _sample(self) -> np.ndarray:
        return self.sample_parameters(self.n or 10 * len(self.parameters))

    def _analyze(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        comp = _Compressor(X)
        X = comp.X

        n, k = X.shape
        if n < k + 2:
            raise Exception(
                f"This sample has only {n} members, but {k} free parameters."
                ' Analysis method "mvr" requires sample_size >= number_of_free_parameters + 2.'
            )

        # z-score transform the data
        def z_score(X: np.ndarray):
            sd = np.std(X, axis=0)
            return np.divide(X - X.mean(axis=0), sd, where=sd > 0)

        X = z_score(X)
        y = z_score(y)

        beta, SS_residuals, rank, s = numpy.linalg.lstsq(X, y, rcond=None)

        # Equivalent expressions for sum of squares.
        # y_hat = np.dot(X, beta)
        # SS_residuals = ((y-y_hat)**2).sum(axis=0)
        # SS_residuals = np.dot(y.T, y - y_hat)
        # SS_total = np.dot(y.T, y -  y.mean(axis=0))
        # SS_explained = np.dot(y.T, y_hat - y.mean(axis=0))

        # total sum of squares
        # (this equals n if observations are z-score transformed)
        SS_total = (y**2).sum()

        # Fraction of variance explained by the model
        R2 = 1.0 - SS_residuals / SS_total

        # Compute F statistic to describe significance of overall model
        SS_explained = SS_total - SS_residuals
        MS_explained = SS_explained / k
        MS_residuals = SS_residuals / (n - k - 1)
        F = MS_explained / MS_residuals

        # t test on slopes of individual parameters (testing whether each is different from 0)
        se_scaled = np.sqrt(np.diag(numpy.linalg.inv(X.T.dot(X))))
        se_beta = se_scaled[:] * np.sqrt(MS_residuals)
        t = beta / se_beta
        P = scipy.stats.t.cdf(abs(t), n - k - 1)
        p = 2 * (1 - P)

        return {
            "beta": comp.expand(beta, 0.0),
            "se_beta": comp.expand(se_beta, 0.0),
            "t": comp.expand(t, 0.0),
            "p": comp.expand(p, 1.0),
            "R2": R2,
            "F": F,
        }

    def _extract_sensitivity_metric(self, analysis: dict[str, Any]) -> np.ndarray:
        return np.abs(analysis["beta"])


class _Compressor:
    def __init__(self, X: np.ndarray):
        self.keep = X.min(axis=0) != X.max(axis=0)
        self.X = X[:, self.keep]

    def expand(self, x: np.ndarray, fill_value=np.nan) -> np.ndarray:
        """Expand a vector to the full parameter set."""
        res = np.full_like(x, fill_value, shape=self.keep.shape)
        res[self.keep] = x
        return res


class CV(SensitivityAnalysis):
    """Ratio of coefficients of variation of each target metric and input parameter,
    based on Monte Carlo sampling.

    This analysis is only meaningful when performed for one parameter at a time.
    """

    def __init__(self, n: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def _sample(self) -> np.ndarray:
        if len(self.parameters) > 1:
            raise Exception("CV analysis is only meaningful for a single parameter.")
        return self.sample_parameters(self.n or 10 * len(self.parameters))

    def _analyze(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        comp = _Compressor(X)
        X = comp.X

        X_cv = np.std(X, axis=0) / np.mean(X, axis=0)
        Y_cv = np.std(y, axis=0) / np.mean(y, axis=0)
        return {"cv_ratio": comp.expand(Y_cv / X_cv, 0.0)}

    def _extract_sensitivity_metric(self, analysis: dict[str, Any]) -> np.ndarray:
        return np.abs(analysis["cv_ratio"])


class _SALibAnalysis(SensitivityAnalysis):
    """Sensitivity analysis based on SALib."""

    def __init__(
        self,
        *sampler_args,
        db_file: Optional[Union[str, Path]] = None,
        distributed: Optional[bool] = None,
        max_workers: Optional[int] = None,
        **sampler_kwargs,
    ):
        """
        Args:
            sampler_args: Positional arguments to pass to the SALib sampler.
            db_file: The file to store the results in. If ``None``, a file with
                the same name as the script will be created with the suffix
                ".results.db".
            distributed: Whether to run the experiment in distributed mode
                using MPI. If ``None``, distributed mode is activated if variable
                ``MPI4PY_FUTURES_MAX_WORKERS`` is present in the environment.
            max_workers: The maximum number of workers to use. If ``None``, it will
                be set to the number of available CPUs.
            sampler_kwargs: Keyword arguments to pass to the SALib sampler.
        """
        super().__init__(
            db_file=db_file, distributed=distributed, max_workers=max_workers
        )
        self.sampler_args = sampler_args
        self.sampler_kwargs = sampler_kwargs
        self.problem = {}

    def _sample(self):
        bounds = np.array(self.get_parameter_bounds(transform=True)).T
        parnames = [p.parameter.name for p in self.parameters]
        self.problem.update(num_vars=len(parnames), names=parnames, bounds=bounds)
        return self._sampler(self.problem, *self.sampler_args, **self.sampler_kwargs)

    def _analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, Any]:
        return self._analyzer(self.problem, y, **kwargs)

    def _extract_sensitivity_metric(self, analysis):
        return analysis[self._metric_name]


class Sobol(_SALibAnalysis):
    """Sobol’ Sensitivity Analysis, using :func:`SALib.sample.sobol.sample`
    and :func:`SALib.analyze.sobol.analyze`.

    The sensitivity metric that is returned by default by
    :meth:`~SensitivityAnalysis.run` and :meth:`~SensitivityAnalysis.run_async`
    is the total sensitivity (``ST``).

    References:
    `Sobol’ (2001) <https://doi.org/10.1016/S0378-4754(00)00270-6>`__;
    `Saltelli (2002) <https://doi.org/10.1016/S0010-4655(02)00280-1>`__;
    `Saltelli et al. (2010) <https://doi.org/10.1016/j.cpc.2009.09.018>`__
    """

    _sampler = staticmethod(SALib.sample.sobol.sample)
    _analyzer = staticmethod(SALib.analyze.sobol.analyze)
    _metric_name = "ST"


class Morris(_SALibAnalysis):
    """Method of Morris, using :func:`SALib.sample.morris.sample`
    and :func:`SALib.analyze.morris.analyze`.

    The sensitivity metric that is returned by default by
    :meth:`~SensitivityAnalysis.run` and :meth:`~SensitivityAnalysis.run_async`
    is the mean of the distribution of the absolute values of the elementary
    effects (``mu_star``).

    References:
    `Morris (1991) <https://doi.org/10.1080/00401706.1991.10484804>`__;
    `Campolongo et al. (2007) <https://doi.org/10.1016/j.envsoft.2006.10.004>`__
    """

    _sampler = staticmethod(SALib.sample.morris.sample)
    _metric_name = "mu_star"

    def _analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, Any]:
        return SALib.analyze.morris.analyze(self.problem, X, y, **kwargs)
