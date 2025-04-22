import asyncio
from typing import Optional, Any

import numpy as np
import numpy.linalg
import scipy.stats

import SALib.analyze.sobol
import SALib.sample.sobol
import SALib.analyze.morris
import SALib.sample.morris

from .. import core


class SensitivityAnalysis(core.Experiment):
    def __init__(self, **kwargs) -> None:
        """Set up a sensitivity analysis.

        To configure the analysis, add parameters to include
        by calling `add_parameter`, and add target metrics to
        determine the sensitivity of by calling `add_target`.

        Args:
            kwargs: Additional keyword arguments to passed to `core.Experiment`.
        """
        super().__init__(**kwargs)
        self.targets: list[str] = []

    def add_job(self, runner: core.Runner) -> None:
        """Add a metric to assess sensitivity of"""
        if runner.name in self.runners:
            assert runner is self.runners[runner.name]
        self.runners[runner.name] = runner

    def run(self, **kwargs):
        """Run the sensitivity analysis"""
        return asyncio.run(self.run_async())

    async def run_async(self, **kwargs):
        if not self.parameters:
            raise Exception("No parameters have been added")
        self.targets.extend(await self.start(record=False))
        if not self.targets:
            raise Exception("No targets have been added to any job")

        # Sample
        X = self._sample()

        # Run
        self.logger.info(f"Evaluating targets for {X.shape[0]} parameter sets")
        tasks = [self.async_eval(p) for p in X]
        results = await asyncio.gather(*tasks)
        Y = np.empty((X.shape[0], len(self.targets)))
        for i, r in enumerate(results):
            Y[i, :] = [r[n] for n in self.targets]
        self.stop()

        # Analyze
        sensitivities = np.full((len(self.parameters), len(self.targets)), -1.0)
        keep = np.std(X, axis=0) > 0
        X = X[:, keep]
        for j, n in enumerate(self.targets):
            self.logger.info(n)
            analyis = self._analyze(X, Y[:, j], **kwargs)
            sensitivities[keep, j] = self._extract_sensitivity_metric(analyis)
        return sensitivities

    def _sample(self) -> np.ndarray:
        raise NotImplementedError

    def _analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def _extract_sensitivity_metric(self, analysis: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class MVR(SensitivityAnalysis):
    """Sensitivity analysis based on Monte Carlo sampling and linear
    regression, as described in by Saltelli et al.
    (https://dx.doi.org/10.1002/9780470725184, section 1.2.5)
    """

    def __init__(self, n: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def _sample(self) -> np.ndarray:
        return self.sample_parameters(self.n or 10 * len(self.parameters))

    def _analyze(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        parnames = [p.parameter.name for p in self.parameters]
        assert X.shape == (y.shape[0], len(parnames))
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

        self.logger.info(f"  Model fit: R2 = {R2[0]:.5f}, F = {F[0]:.5g}")
        self.logger.info("  Regression coefficients:")
        for curname, curbeta, curse_beta, curt, curp in sorted(
            zip(parnames, beta, se_beta, t, p), key=lambda x: -abs(x[1])
        ):
            self.logger.info(
                f"    {curname}: beta = {curbeta:.5g} (s.e. {curse_beta:.5g}),"
                f" p={curp:.5f} for beta=0"
            )

        return {"beta": beta, "se_beta": se_beta, "t": t, "p": p, "R2": R2, "F": F}

    def _extract_sensitivity_metric(self, analysis: dict[str, Any]) -> np.ndarray:
        return np.abs(analysis["beta"])


class SALibAnalysis(SensitivityAnalysis):
    def __init__(self, *sampler_args, **sampler_kwargs):
        super().__init__()
        self.sampler_args = sampler_args
        self.sampler_kwargs = sampler_kwargs
        self.problem = {}

    def _sample(self):
        bounds = np.array(self.get_parameter_bounds(transform=True)).T
        parnames = [p.parameter.name for p in self.parameters]
        self.problem.update(num_vars=len(parnames), names=parnames, bounds=bounds)
        return self._sampler(self.problem, *self.sampler_args, **self.sampler_kwargs)

    def _analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, Any]:
        analyis = self._analyzer(self.problem, y, **kwargs)
        self.logger.info(analyis)
        return analyis

    def _extract_sensitivity_metric(self, analysis):
        return analysis[self._metric_name]


class Sobol(SALibAnalysis):
    """Sobol’ Sensitivity Analysis

    https://dx.doi.org/10.1016/S0378-4754(00)00270-6
    https://dx.doi.org/10.1016/S0010-4655(02)00280-1
    https://dx.doi.org/10.1016/j.cpc.2009.09.018
    """

    _sampler = staticmethod(SALib.sample.sobol.sample)
    _analyzer = staticmethod(SALib.analyze.sobol.analyze)
    _metric_name = "ST"


class Morris(SALibAnalysis):
    """Method of Morris

    https://dx.doi.org/10.1080/00401706.1991.10484804
    https://dx.doi.org/10.1016/j.envsoft.2006.10.004
    """

    _sampler = staticmethod(SALib.sample.morris.sample)
    _metric_name = "mu_star"

    def _analyze(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, Any]:
        analyis = SALib.analyze.morris.analyze(self.problem, X, y, **kwargs)
        self.logger.info(analyis)
        return analyis
