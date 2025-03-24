from typing import Optional, Any, Union
from pathlib import Path
import os

from matplotlib import pyplot as plt
import matplotlib.artist
import matplotlib.axes
from matplotlib import animation
import numpy as np

from .. import record


def _get_marginal(
    par: np.ndarray, lnl: np.ndarray, logscale: bool = False, bincount: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate marginal by estimating upper contour of cloud."""
    assert par.ndim == 1
    assert lnl.ndim == 1
    assert par.shape[0] == lnl.shape[0]
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
    def __init__(self, db_file: Union[str, os.PathLike[Any]]):
        db_path = Path(db_file)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file {db_file} not found.")
        self.rec = record.Recorder(db_path)
        self.parnames = self.prettyparnames = self.rec.headers[2:-1]
        prefixes = [parnames.split(":", 1)[0] for parnames in self.parnames]
        if all(prefix == prefixes[0] for prefix in prefixes):
            self.prettyparnames = [
                parnames.split(":", 1)[1] for parnames in self.parnames
            ]
        self.npar = len(self.parnames)

        par_info = list(self.rec.run_info.values())[0]["info"]["parameters"]
        self.parmin = dict(zip(par_info["names"], par_info["minimum"]))
        self.parmax = dict(zip(par_info["names"], par_info["maximum"]))
        self.parlog = dict(zip(par_info["names"], par_info["logscale"]))

        self._lastcount = 0

        self.update()

    @property
    def rowcount(self) -> int:
        """Number of rows in the results table."""
        return self.values.shape[0]

    def update(self) -> int:
        """Update the results from the database."""
        res = self.rec.to_ndarray()

        newcount = res.shape[0] - self._lastcount
        if newcount == 0:
            return newcount

        # Sort by likelihood
        self.order = res[:, -1].argsort()
        res = res[self.order, :]
        self.lnls = res[:, -1]
        self.run_ids = res[:, 1]
        self.values = res[:, 2:-1]

        # Show best parameter set
        self.maxlnl = self.lnls[-1]
        self.best = self.values[-1]

        self._lastcount = res.shape[0]
        return newcount

    def get_confidence_interval(
        self, lnl_crit=LNL_CRIT
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the confidence interval for all parameters."""
        iinc = self.lnls.searchsorted(self.maxlnl - lnl_crit)
        lbounds = self.values[iinc:].min(axis=0)
        rbounds = self.values[iinc:].max(axis=0)
        outside = self.values[:iinc]
        for ipar in range(self.values.shape[-1]):
            # Get conservative confidence interval by extending it to the first point
            # from the boundary that has a likelihood value outside the allowed range.
            lvalid = outside[:, ipar] < lbounds[ipar]
            rvalid = outside[:, ipar] > rbounds[ipar]
            if lvalid.any():
                lbounds[ipar] = outside[lvalid, ipar].max()
            if rvalid.any():
                rbounds[ipar] = outside[rvalid, ipar].min()
        return lbounds, rbounds

    def save_best(self, file: Union[str, os.PathLike[Any]], *, sep: str = "\t") -> None:
        """Save the best parameter set to a file."""
        with open(file, "w") as f:
            for parname, value in zip(self.parnames, self.best):
                f.write(f"{parname}{sep}{value}\n")


def plot(
    db_file: Union[str, os.PathLike[Any]],
    lnl_range: Optional[float] = None,
    bincount: int = 25,
    keep_updating: bool = False,
    save: Optional[Union[str, os.PathLike[Any]]] = None,
) -> None:
    res = Result(db_file)

    if lnl_range is not None:
        lnl_range = abs(lnl_range)

    artists: list[matplotlib.artist.Artist] = []

    def update(frame: Optional[int] = None):
        n_new = res.update()
        first_time = not artists
        if n_new == 0 and not first_time:
            return

        if not first_time:
            print(f"  {n_new} found.")

        # Clear previous points and lines and reset color cycle
        # We do preserve axes limits and titles.
        for a in artists:
            a.remove()
        artists.clear()
        for ax in axes:
            ax.set_prop_cycle(None)

        lci, uci = res.get_confidence_interval()
        print(
            f"Best parameter set is # {res.order[-1]} with ln likelihood = {res.maxlnl:.6g}:"
        )
        for parname, value, l, u in zip(res.prettyparnames, res.best, lci, uci):
            print(f"  {parname}: {value:.6g} ({l:.6g} - {u:.6g})")

        if save is not None:
            print(f"Writing best parameter set to {save}...")
            res.save_best(save)

        # For each run, print max lnl and add points to each parameter plot
        print("Points per run:")
        for run_id in sorted(set(res.run_ids)):
            match = res.run_ids == run_id
            curres = res.values[match, :]
            lnl = res.lnls[match]
            print(f"  {run_id}: {match.sum()} points, best lnl = {lnl.max():.8g}.")
            for ipar, ax in enumerate(axes):
                (points,) = ax.plot(curres[:, ipar], lnl, ".", label=run_id)
                artists.append(points)

        for ipar, (name, lbound, rbound, ax) in enumerate(
            zip(res.parnames, lci, uci, axes)
        ):
            # Plot marginal
            logscale = res.parlog.get(name, False)
            margx, margy = _get_marginal(
                res.values[:, ipar], res.lnls, logscale, bincount
            )
            (line_marg,) = ax.plot(margx, margy, "-k", label="_nolegend_")

            # Show confidence interval
            line_cil = ax.axvline(lbound, color="k", linestyle="--")
            line_cir = ax.axvline(rbound, color="k", linestyle="--")

            artists.extend([line_marg, line_cil, line_cir])

        if first_time:
            # First time we are plotting - put finishing touches on subplots
            cur_lnl_range = res.maxlnl - res.lnls[0] if lnl_range is None else lnl_range
            for name, title, ax in zip(res.parnames, res.prettyparnames, axes):
                ax.set_title(title)
                ax.set_xlim(res.parmin.get(name), res.parmax.get(name))
                ax.set_ylim(
                    res.maxlnl - cur_lnl_range, ymax=res.maxlnl + 0.1 * cur_lnl_range
                )
                if res.parlog.get(name, False):
                    ax.set_xscale("log")

        if keep_updating:
            print("Waiting for new results...")

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.05, hspace=0.3)
    nrow = int(np.ceil(np.sqrt(0.5 * res.npar)))
    ncol = int(np.ceil(float(res.npar) / nrow))

    # Create subplots
    ax: Optional[matplotlib.axes.Axes] = None
    axes: list[matplotlib.axes.Axes] = []
    for ipar in range(res.npar):
        ax = fig.add_subplot(nrow, ncol, ipar + 1, sharey=ax)
        axes.append(ax)

    update()
    if keep_updating:
        anim = animation.FuncAnimation(
            fig, update, interval=5000, cache_frame_data=False
        )
    else:
        fig.savefig("estimates.png", dpi=300)

    # Show figure and wait until the user closes it.
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "db_file", help="SQLite database file with optimization results"
    )
    args = parser.parse_args()
    plot(args.db_file, keep_updating=True)
