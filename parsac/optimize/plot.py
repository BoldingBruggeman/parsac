from typing import Optional, Any, Union, Mapping, Iterable
from pathlib import Path
import os
import asyncio
import enum
import logging

from matplotlib import pyplot as plt
import matplotlib.artist
import matplotlib.axes
import matplotlib.figure
from matplotlib import animation
import numpy as np

from .. import record
from .. import core


class PlotType(enum.Enum):
    MARGINAL = enum.auto()
    GENERATIONS = enum.auto()


def _get_marginal(
    par: np.ndarray, lnl: np.ndarray, logscale: bool = False, bincount: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate marginal by estimating upper contour of cloud."""
    assert par.ndim == 1
    assert lnl.ndim == 1
    assert par.shape[0] == lnl.shape[0]
    valid = np.isfinite(lnl)
    par = par[valid]
    lnl = lnl[valid]
    order = par.argsort()
    par = par[order]
    lnl = lnl[order]
    if logscale:
        par = np.log10(par)

    def _next_i(i: int = 0) -> int:
        i += 1
        while i < len(par) and par[i] == xlist[-1]:
            i += 1
        return i

    xlist, ylist = [par[0]], [lnl[0]]
    i = _next_i()

    step = max(int(round(2 * par.size / bincount)), 1)
    # step = max(bincount, par.size // 1000)
    while i < len(par):
        ilast = i + step
        slope = (lnl[i:ilast] - ylist[-1]) / (par[i:ilast] - xlist[-1])
        i += slope.argmax()
        xlist.append(par[i])
        ylist.append(lnl[i])
        i = _next_i(i)
    xs, ys = np.array(xlist), np.array(ylist)
    if logscale:
        xs = 10.0**xs
    return xs, ys


# Likelihood ratio critical value for 95% confidence interval
LNL_CRIT = 1.920729410347062  # 0.5 * scipy.stats.chi2.ppf(0.95, 1)


class Result:
    def __init__(
        self,
        db_file: Union[str, os.PathLike[Any]],
        skip_lnl: bool = True,
        skip_inferred=False,
    ):
        db_path = Path(db_file)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file {db_file} not found.")
        self.rec = record.Recorder(db_path, read_only=True)

        par_info = self.rec.get_config("parameters")
        self.parmin = dict(zip(par_info["names"], par_info["minimum"]))
        self.parmax = dict(zip(par_info["names"], par_info["maximum"]))
        self.parlog = dict(zip(par_info["names"], par_info["logscale"]))

        self.iselect: list[int] = []
        self.parnames: list[str] = []
        SKIP_COLS = {"id", "run_id", "exception", "generation", "lnl"}
        if skip_lnl:
            SKIP_COLS.update(n for n in self.rec.headers if n.endswith(":lnl"))
        if skip_inferred:
            SKIP_COLS.update(n for n in self.rec.headers if n not in par_info["names"])
        for i, parname in enumerate(self.rec.headers):
            if parname not in SKIP_COLS:
                self.iselect.append(i)
                self.parnames.append(parname)

        nprefix = len(os.path.commonprefix(self.parnames))
        self.prettyparnames = [n[nprefix:] for n in self.parnames]
        self.npar = len(self.parnames)

        self._lastcount = 0
        self.generations: Optional[np.ndarray] = None

        self.update()

    @property
    def rowcount(self) -> int:
        """Number of rows in the results table."""
        return self.values.shape[0]

    def get_errors(self) -> list[str]:
        iex = self.rec.headers.index("exception")
        exceptions = []
        for r in self.rec.rows(where="WHERE exception IS NOT NULL"):
            exceptions.append(r[iex])
        return exceptions

    def update(self) -> int:
        """Update the results from the database."""
        where = ""
        if "exception" in self.rec.headers:
            where += " WHERE exception IS NULL"
        res = self.rec.to_ndarray(where=where + " ORDER BY lnl")

        newcount = res.shape[0] - self._lastcount
        if newcount == 0:
            return newcount

        self.lnls = res[:, -1]
        self.run_ids = res[:, 1].astype(int)
        if "generation" in self.rec.headers:
            icol = self.rec.headers.index("generation")
            self.generations = res[:, icol].astype(int)
        else:
            self.generations = None
        self.values = res[:, self.iselect]

        # Show best parameter set
        self.maxlnl = self.lnls[-1]
        self.best = self.values[-1]
        self.ibest = int(res[-1, 0])

        self._lastcount = res.shape[0]
        return newcount

    def get_confidence_interval(
        self, lnl_crit=LNL_CRIT
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the confidence interval for all parameters."""
        iinc = self.lnls.searchsorted(self.maxlnl - lnl_crit)
        lbounds = self.values[iinc:].min(axis=0)
        rbounds = self.values[iinc:].max(axis=0)
        all_out = self.values[:iinc, :][np.isfinite(self.lnls[:iinc]), :]
        for i in range(lbounds.size):
            # Get conservative confidence interval by extending it to the first point
            # from the boundary that has a likelihood value outside the allowed range.
            out = all_out[:, i]
            lower = out < lbounds[i]
            if lower.any():
                lbounds[i] = out.max(where=lower, initial=-np.inf)
            higher = out > rbounds[i]
            if higher.any():
                rbounds[i] = out.min(where=higher, initial=np.inf)
        return lbounds, rbounds

    def save_best(self, file: Union[str, os.PathLike[Any]], *, sep: str = "\t") -> None:
        """Save the best parameter set to a file."""
        with open(file, "w") as f:
            for parname, value in zip(self.parnames, self.best):
                f.write(f"{parname}{sep}{value}\n")

    def plot(
        self,
        lnl_range: Optional[float] = None,
        bincount: int = 25,
        keep_updating: bool = False,
        save: Optional[Union[str, os.PathLike[Any]]] = None,
        plot_type: PlotType = PlotType.MARGINAL,
        logger: Optional[logging.Logger] = None,
    ) -> matplotlib.figure.Figure:
        logger = logger or logging.getLogger(__name__)

        if lnl_range is not None:
            lnl_range = abs(lnl_range)

        artists: list[matplotlib.artist.Artist] = []

        def update(frame: Optional[int] = None):
            n_new = self.update()
            first_time = not artists
            if n_new == 0 and not first_time:
                return

            if not first_time:
                print(f" {n_new} found.")

            # Clear previous points and lines and reset color cycle
            # We do preserve axes limits and titles.
            for a in artists:
                a.remove()
            artists.clear()
            for ax in axes:
                ax.set_prop_cycle(None)

            lci, uci = self.get_confidence_interval()
            logger.info(
                f"Best parameter set is # {self.ibest} with ln likelihood = {self.maxlnl:.6g}:"
            )
            for parname, value, l, u in zip(self.prettyparnames, self.best, lci, uci):
                logger.info(f"  {parname}: {value:.6g} ({l:.6g} - {u:.6g})")

            if save is not None:
                logger.info(f"Writing best parameter set to {save}...")
                self.save_best(save)

            # For each run, print max lnl and add points to each parameter plot
            logger.info("Points per run:")
            for run_id in sorted(set(self.run_ids)):
                match = self.run_ids == run_id
                curres = self.values[match, :]
                lnl = self.lnls[match]
                if plot_type != plot_type.MARGINAL:
                    assert self.generations is not None
                    gen = self.generations[match]
                logger.info(
                    f"  {run_id}: {match.sum()} points, best lnl = {lnl.max():.8g}."
                )
                for ipar, ax in enumerate(axes):
                    if plot_type == plot_type.MARGINAL:
                        (points,) = ax.plot(curres[:, ipar], lnl, ".", label=run_id)
                    else:
                        (points,) = ax.plot(
                            gen, curres[:, ipar], ".", alpha=0.5, label=run_id
                        )

                    artists.append(points)

            iref: Optional[int] = None
            if self.generations is not None:
                iref = (self.generations == -1).nonzero()[0][0]
            for ipar, (name, lbound, rbound, ax) in enumerate(
                zip(self.parnames, lci, uci, axes)
            ):
                if plot_type == PlotType.MARGINAL:
                    # Plot marginal
                    logscale = self.parlog.get(name, False)
                    margx, margy = _get_marginal(
                        self.values[:, ipar], self.lnls, logscale, bincount
                    )
                    (line_marg,) = ax.plot(margx, margy, "-k", label="_nolegend_")
                    artists.append(line_marg)

                plotci = ax.axhline if plot_type == PlotType.GENERATIONS else ax.axvline
                line_cil = plotci(lbound, color="k", linestyle="--")
                line_cir = plotci(rbound, color="k", linestyle="--")
                artists.extend([line_cil, line_cir])

                if iref is not None:
                    artists.append(plotci(self.values[iref, ipar], color="r"))

            if first_time:
                if lnl_range is None:
                    lnl_min = self.lnls.min(
                        where=np.isfinite(self.lnls), initial=self.maxlnl
                    )
                    cur_lnl_range = self.maxlnl - lnl_min
                else:
                    cur_lnl_range = lnl_range
                for name, title, ax in zip(self.parnames, self.prettyparnames, axes):
                    ax.set_title(title, fontsize="medium")
                    mi, ma = self.parmin.get(name), self.parmax.get(name)
                    if self.parlog.get(name, False) and mi == 0.0:
                        mi = None
                    if plot_type == PlotType.GENERATIONS:
                        if self.parlog.get(name, False):
                            ax.set_yscale("log")
                        ax.set_ylim(mi, ma)
                    else:
                        if self.parlog.get(name, False):
                            ax.set_xscale("log")
                        ax.set_xlim(mi, ma)
                        ax.set_ylim(
                            self.maxlnl - cur_lnl_range,
                            ymax=self.maxlnl + 0.1 * cur_lnl_range,
                        )

            if keep_updating:
                print("Waiting for new results...", end="", flush=True)

        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.05, hspace=0.3)
        nrow = int(np.ceil(np.sqrt(0.5 * self.npar)))
        ncol = int(np.ceil(float(self.npar) / nrow))

        # Create subplots
        ax: Optional[matplotlib.axes.Axes] = None
        axes: list[matplotlib.axes.Axes] = []
        share = "x" if plot_type == PlotType.GENERATIONS else "y"
        for ipar in range(self.npar):
            kwargs = {f"share{share}": ax}
            ax = fig.add_subplot(nrow, ncol, ipar + 1, **kwargs)
            axes.append(ax)

        update()
        if keep_updating:
            self._anim = animation.FuncAnimation(
                fig, update, interval=5000, cache_frame_data=False
            )
        return fig

    def plot_best(
        self,
        target_dir: Union[str, os.PathLike[Any], None] = None,
        logger: Optional[logging.Logger] = None,
    ) -> list[matplotlib.figure.Figure]:
        logger = logger or logging.getLogger(__name__)

        logger.info("Evaluating best parameter set...")
        best = {n: float(v) for n, v in zip(self.parnames, self.best)}
        for n, v in best.items():
            logger.info(f"  {n}: {v:.6g}")

        runners: Mapping[str, core.Runner] = self.rec.get_config("runners")
        if target_dir is not None:
            logger.info(
                "Worker configurations for best parameter set will be"
                f" created in {target_dir}"
            )
        pool = core.RunnerPool(runners, distributed=False, logger=logger)
        name2output = asyncio.run(pool(best, work_dir=target_dir, plot=True))
        transforms: Iterable[core.Transform] = []
        try:
            transforms = self.rec.get_config("global_transforms")
        except Exception as e:
            logger.warning(f"Unable to get global transforms from database. {e}")
        for transform in transforms:
            transform(best, name2output)

        logger.info("Building plots...")
        figs: list[matplotlib.figure.Figure] = []
        for category, name2plotter in _collect_plotters(name2output).items():
            nprefix = len(os.path.commonprefix(list(name2plotter)))
            nplots = len(name2plotter)
            nrows = int(np.ceil(np.sqrt(nplots)))
            ncols = int(np.ceil(float(nplots) / nrows))
            fig = plt.figure(num=category)
            id2ax: dict[int, matplotlib.axes.Axes] = {}
            for i, (name, plotter) in enumerate(name2plotter.items()):
                ax = fig.add_subplot(
                    nrows,
                    ncols,
                    i + 1,
                    sharex=id2ax.get(id(plotter.sharex)),
                    sharey=id2ax.get(id(plotter.sharey)),
                )
                plotter.plot(ax, logger)
                ax.set_title(name[nprefix:], fontsize="medium")
                id2ax[id(plotter)] = ax
            figs.append(fig)
        return figs


def _collect_plotters(
    name2output: Mapping[str, Any],
) -> dict[str, dict[str, core.Plotter]]:
    """Collect plotters from the output."""
    category2name2plotter: dict[str, dict[str, core.Plotter]] = {}
    for name, output in name2output.items():
        if isinstance(output, core.Plotter):
            name, category = name.rsplit(":", 1)
            if category not in category2name2plotter:
                category2name2plotter[category] = {}
            category2name2plotter[category][name] = output
    return category2name2plotter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--best", action="store_true", help="Show current best result")
    parser.add_argument(
        "--target_dir",
        help=(
            "Target directory to create setups (worker configurations) of best parameter"
            " set in. If not provided a temporary directory will be used."
            " This can only be used in combination with --best."
        ),
    )
    parser.add_argument(
        "--marg",
        action="store_true",
        help=("Plot marginal likelihood (x: parameter value, y: ln likelihood) "
         " rather than generations (x: generation, y: parameter value)."),
    )
    parser.add_argument(
        "--range", type=float, default=None, help="Likelihood range for marginal plot"
    )
    parser.add_argument(
        "--primary_only",
        action="store_true",
        help="Only plot primary parameters (no inferred parameters, no extra scalar outputs)",
    )
    parser.add_argument(
        "db_file", help="SQLite database file with optimization results"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    result = Result(args.db_file, skip_inferred=args.primary_only)
    args.marg |= result.generations is None
    plot_type = PlotType.MARGINAL if args.marg else PlotType.GENERATIONS
    if args.best:
        result.plot_best(target_dir=args.target_dir, logger=logger)
    else:
        result.plot(
            keep_updating=True, plot_type=plot_type, lnl_range=args.range, logger=logger
        )

    # Show figure and wait until the user closes it.
    plt.show()
