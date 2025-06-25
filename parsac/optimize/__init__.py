from typing import Optional, Any, Mapping, TYPE_CHECKING
import logging
import asyncio

import numpy as np
import numpy.typing as npt

from .. import core
from . import desolver

if TYPE_CHECKING:
    import matplotlib.axes


class Optimization(core.Experiment):
    """An optimization experiment.

    To configure the experiment, add parameters to calibrate
    by calling :meth:`~parsac.core.Experiment.add_parameter`, and add
    contributions to the objective function (log likelihood) by calling
    :meth:`add_target`.
    """

    def __init__(self, **kwargs) -> None:
        """
        Args:
            kwargs: Additional keyword arguments to passed to :class:`~parsac.core.Experiment`.
        """
        super().__init__(**kwargs)
        self.total_lnl = TotalLikelihood(self.priors)
        self.global_transforms.append(self.total_lnl)
        self.row_metadata["generation"] = -1

    def add_target(self, metric: core.Comparison, **kwargs: Any) -> None:
        """Add a contribution to the fitness (log-likelihood) function.

        Args:
            metric: The metric to add to the fitness function.
                It is typically produced by a job from :mod:`parsac.job`.
            kwargs: Additional keyword arguments to pass to the likelihood function
                :class:`GaussianLikelihood`.
        """
        if metric.runner.name in self.runners:
            assert metric.runner is self.runners[metric.runner.name]
        self.runners[metric.runner.name] = metric.runner
        if metric.sd is not None:
            kwargs.setdefault("sd", metric.sd)
        model_vals2lnl = GaussianLikelihood(metric.name, metric.obs_vals, **kwargs)
        self.total_lnl.components.append(model_vals2lnl.name)
        metric.runner.transforms.append(model_vals2lnl)

    def run(self, **kwargs: Any) -> Mapping[str, float]:
        """Run the optimization

        Args:
            kwargs: Additional keyword arguments to pass to :func:`desolver.solve`.
        """
        return asyncio.run(self.run_async(**kwargs))

    async def run_async(self, **kwargs) -> Mapping[str, float]:
        def cb(igen_finished: int) -> None:
            self.row_metadata["generation"] = igen_finished + 1

        if not self.total_lnl.components:
            raise Exception("No optimization targets defined.")
        await self.start(record=True)
        self.row_metadata["generation"] = 0
        pop = self.sample_parameters(10 * len(self.parameters))
        minbounds, maxbounds = self.get_parameter_bounds(transform=True)
        result = await desolver.solve(
            self.get_lnls,
            minbounds,
            maxbounds,
            initial_population=pop,
            callback=cb,
            logger=self.logger.getChild("DE"),
            **kwargs,
        )
        self.stop()
        return self.unpack_parameters(result)

    async def get_lnls(
        self, values: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray:
        """Calculate the log-likelihood of a parameter set."""
        results = await self.batch_eval(values, return_exceptions=True)
        lnls = np.full(len(values), -np.inf, dtype=float)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                self.logger.debug(f"Error evaluating parameters: {r}")
            elif not isinstance(r, BaseException):
                lnls[i] = r["lnl"]
        return lnls


class TotalLikelihood(core.Transform):
    """Sum of all likelihood contributions."""

    def __init__(self, priors: list[core.Prior]) -> None:
        self.components: list[str] = []
        self.priors = priors

    def __call__(
        self, name2value: Mapping[str, float], name2output: dict[str, Any]
    ) -> None:
        lnl = 0.0
        for component in self.components:
            lnl += name2output[component]
        for prior in self.priors:
            lnl += prior.logpdf(name2value)
        name2output["lnl"] = lnl if np.isfinite(lnl) else -np.inf


class GaussianLikelihood(core.Transform):
    def __init__(
        self,
        source: str,
        obs_vals: npt.ArrayLike,
        sd: Optional[npt.ArrayLike] = None,
        logscale: bool = False,
        minimum: Optional[float] = None,
        estimate_scale_factor: bool = False,
        min_scale_factor: Optional[float] = None,
        max_scale_factor: Optional[float] = None,
        scale_factor: float = 1.0,
        estimate_sd: Optional[bool] = None,
    ):
        """Set up a Gaussian likelihood function
        This typically contributes to the objective function in an optimization
        experiment.

        This function assumes that observations are normally distributed around
        modelled values, with a standard deviation that can be either
        provided or estimated from the model-observation differences.

        It can also be used to represent observations that are *log*-normally
        distributed around modelled values by setting argument ``logscale`` to
        ``True``. In that case both modelled and observed values are
        log-transformed.

        Observations are provided upon initialization, and modelled values
        are provided when the likelihood is called.

        Args:
            source: The name of the output that will contain the model values.
            obs_vals: The observed values.
            sd: The standard deviation of the observed values.
                If not provided, it will be estimated from the differences
                betweeen modelled and observed values.
            logscale: Whether to log-transform modelled and observed values
                before calculating the likelihood. This implies the distribution
                of observations around the modelled values is log-normal.
            minimum: The minimum allowed value of the modelled and observed
                values. Lower values will be clipped to this value. This argument
                must be provided if ``logscale`` is ``True``. It must then be
                greater than 0.0.
            estimate_scale_factor: Whether to estimate the scale factor with which
                model values are multiplied before comparing to observations.
            min_scale_factor: Lower bound of the estimated scale factor.
            max_scale_factor: Upper bound of the estimated scale factor.
            scale_factor: Fixed scale factor, active only if ``estimate_scale_factor``
                is ``False``.
            estimate_sd: Whether to estimate the standard deviation of the observed values.
                If ``None`` (the default), the standard deviation is estimated if ``sd``
                is not provided. If the observations do come with standard deviations,
                but you want to ignore these and estimate the standard deviation from
                model-observation differences instead, set this argument to ``True``.
        """
        if logscale and (minimum is None or minimum <= 0.0):
            raise Exception(
                "For log scale fitting, a minimum value > 0 must be specified."
            )
        self.source = source
        self.name = source + ":lnl"
        self.obs_vals = np.asarray(obs_vals)
        if self.obs_vals.size == 0:
            raise Exception(f"{self.source}: no observations provided.")
        if not np.isfinite(self.obs_vals).all():
            raise Exception(f"{self.source}: observations contain non-finite values.")
        if estimate_sd is None:
            estimate_sd = sd is None
        self.sd = None
        if not estimate_sd:
            assert sd is not None
            self.sd = np.asarray(sd, dtype=float)
            if not np.isfinite(self.sd).all():
                raise Exception(
                    f"{self.source}: standard deviations contain non-finite values."
                )
            if (self.sd <= 0).any():
                raise Exception(f"{self.source}: standard deviation must be positive.")
            np.broadcast_shapes(self.sd.shape, self.obs_vals.shape)
        self.logscale = logscale
        self.minimum = minimum
        self.estimate_scale_factor = estimate_scale_factor
        if min_scale_factor is not None and max_scale_factor is not None:
            assert min_scale_factor < max_scale_factor
        if estimate_sd:
            assert (
                self.obs_vals.size > 1
            ), "> 1 observations are required to estimate the standard deviation."
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_factor = scale_factor
        self.logger = logging.getLogger(self.name)

    def __call__(self, name2value: Mapping[str, float], name2output: dict[str, Any]):
        model_vals = name2output.pop(self.source)
        obs_vals = self.obs_vals
        assert model_vals.shape == obs_vals.shape
        if not np.isfinite(model_vals).all():
            raise Exception(f"{self.source} contains non-finite values.")
        if self.minimum is not None:
            model_vals = np.maximum(model_vals, self.minimum)
            obs_vals = np.maximum(obs_vals, self.minimum)
        obs_vals_nontf = obs_vals
        if self.logscale:
            model_vals = np.log10(model_vals)
            obs_vals = np.log10(obs_vals)

        # If the model fit is relative, calculate the optimal model to observation scaling factor.
        scale = self.scale_factor
        if self.estimate_scale_factor:
            w = np.ones(model_vals.shape) / (1.0 if self.sd is None else self.sd**2)
            if self.logscale:
                # Optimal scale factor is calculated from optimal offset on a log scale.
                scale = 10.0 ** ((obs_vals - model_vals).dot(w) / w.sum())
            elif model_vals.any():
                # Calculate optimal scale factor
                # (= covariance divided by variance of model values, with means of 0)
                # NB if model values are all zero, the optimal scale factor is undefined,
                # but scaling the model values then has no effect anyway.
                scale = (obs_vals * model_vals).dot(w) / (model_vals**2).dot(w)

            # Report and check optimal scale factor.
            self.logger.info(
                f"{self.name}: optimal model-to-observation scale factor = {scale:.6g}."
            )
            if self.min_scale_factor is not None and scale < self.min_scale_factor:
                self.logger.info(
                    f"{self.name}: clipping scale factor to minimum = {self.min_scale_factor:.6g}."
                )
                scale = self.min_scale_factor
            elif self.max_scale_factor is not None and scale > self.max_scale_factor:
                self.logger.info(
                    f"{self.name}: clipping scale factor to maximum = {self.max_scale_factor:.6g}."
                )
                scale = self.max_scale_factor
            name2output[self.name + ":scale"] = scale

        # Apply scale factor if set
        if scale != 1.0:
            if self.logscale:
                model_vals += np.log10(scale)
            else:
                model_vals *= scale

        # Calculate squared difference between model outcome and observations
        diff2 = (model_vals - obs_vals) ** 2

        sd = self.sd
        if sd is None:
            # No standard deviation specified: calculate the optimal s.d.
            sd = np.sqrt(diff2.sum() / (diff2.size - 1))
            self.logger.info(f"{self.name}: optimal s.d. = {sd:.6g}")
            name2output[self.name + ":sd"] = sd

        # Calculate likelihood contribution
        # Differences are weighted according to standard deviation of current data.
        # Note: this assumes normally distributed errors
        # Omitting constant terms in the log likelihood = -n*ln(2*pi)/2
        name2output[self.name] = (-np.log(sd) - diff2 / (2 * sd * sd)).sum()

        cmp_plot = name2output.get(self.source + ComparisonPlotter.POSTFIX)
        if cmp_plot is not None:
            assert isinstance(cmp_plot, ComparisonPlotter)
            cmp_plot.obs_values = obs_vals_nontf
            cmp_plot.scale_factor = scale
            cmp_plot.logscale = self.logscale
            cmp_plot.sd = np.broadcast_to(sd, obs_vals.shape)

            name2output[self.source + ":Correlations"] = ModelVsObservationPlotter(
                model_vals,
                obs_vals,
            )
            name2output[self.source + ":Error histograms"] = HistogramPlotter(
                self.source,
                model_vals,
                obs_vals,
            )


class ComparisonPlotter(core.Plotter):
    POSTFIX = ":Comparisons"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.obs_values: Optional[np.ndarray] = None
        self.scale_factor = 1.0
        self.sd: Optional[np.ndarray] = None
        self.logscale = False


class ModelVsObservationPlotter(core.Plotter):
    """Correlation plot showing model (y) versus observations (x)."""

    def __init__(self, model_vals: np.ndarray, obs_vals: np.ndarray):
        super().__init__()
        self.model_vals = model_vals
        self.obs_vals = obs_vals

    def plot(self, ax: "matplotlib.axes.Axes", logger: logging.Logger) -> None:
        ax.plot(self.obs_vals, self.model_vals, ".")
        ax.axis("tight")
        ax.grid(True)
        ax.axline((0.0, 0.0), slope=1.0, color="k")
        ax.set_xlabel("observation")
        ax.set_ylabel("model")


class HistogramPlotter(core.Plotter):
    """Histogram of model-observation differences."""

    def __init__(self, name: str, model_vals: np.ndarray, obs_vals: np.ndarray):
        super().__init__()
        self.name = name
        self.model_vals = model_vals
        self.obs_vals = obs_vals

    def plot(self, ax: "matplotlib.axes.Axes", logger: logging.Logger) -> None:
        diff = self.model_vals - self.obs_vals
        cov = np.cov(self.model_vals, self.obs_vals, ddof=0)
        sd_mod = np.sqrt(cov[0, 0])
        sd_obs = np.sqrt(cov[1, 1])
        cor = cov[0, 1] / (sd_obs * sd_mod)
        bias = diff.mean()
        mae = np.abs(diff).mean()
        rmse = np.sqrt((diff**2).mean())
        n, bins, patches = ax.hist(diff, 100, density=True)
        ax.set_xlabel("model - observation")
        logger.info(f"{self.name}")
        logger.info(f"  model bias: {bias:.4g}")
        logger.info(f"  mean absolute error: {mae:.4g}")
        logger.info(f"  root mean square error: {rmse:.4g}")
        logger.info(f"  correlation: {cor:.4g}")
        logger.info(f"  s.d. mod: {sd_mod:.4g}")
        logger.info(f"  s.d. obs: {sd_obs:.4g}")
        y = (1.0 / np.sqrt(2 * np.pi) / rmse) * np.exp((-0.5 / rmse**2) * bins**2)
        ax.plot(bins, y, "r--", linewidth=2)
