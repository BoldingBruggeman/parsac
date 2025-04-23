import logging
from typing import (
    Optional,
    Callable,
    Awaitable,
    Iterable,
    Iterator,
    overload,
    Union,
    Sequence,
)
import asyncio

import numpy as np
import numpy.typing as npt


async def solve(
    fn: Callable[
        [np.ndarray[tuple[int, int], np.dtype[np.float64]]],
        Awaitable[np.ndarray[tuple[int], np.dtype[np.float64]]],
    ],
    minbounds: npt.ArrayLike,
    maxbounds: npt.ArrayLike,
    *,
    F: float = 0.5,
    CR: float = 0.9,
    npop: Optional[int] = None,
    initial_population: Optional[npt.ArrayLike] = None,
    maxgen: int = 500,
    reltol: Union[float, np.ndarray[tuple[int], np.dtype[np.float64]]] = 0.01,
    abstol: Union[float, np.ndarray[tuple[int], np.dtype[np.float64]]] = 1e-8,
    ftol: float = np.inf,
    random_ancestor: bool = True,
    ndiffvector: int = 1,
    logger: Optional[logging.Logger] = None,
    enforce_bounds: bool = True,
    rng: Optional[np.random.Generator] = None,
    callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    Solve an optimization problem using the Differential Evolution algorithm.

    Args:
        fn: function that  returns the fitness of a parameter vector
        minbounds: lower bounds for each parameter
        maxbounds: upper bounds for each parameter
        F: scale
        CR: cross-over probability
        npop: population size, defaults to 10 times the number of parameters
        maxgen: maximum number of generations
        reltol: relative tolerance for parameter convergence
        abstol: absolute tolerance for parameter convergence
        ftol: tolerance for fitness convergence
        random_ancestor: whether to pick a random ancestor for each trial vector
        ndiffvector: number of difference vectors to add to the base vector
        logger: logger to use
        enforce_bounds: whether to force parameter vectors to stay within
            prescribed initial range [minbounds, maxbounds]
        rng: random number generator
        callback: function to call after each generation.
            It should take the generation number as an argument.

    Returns:
        The optimal parameter vector.
    """
    # Set up logger
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    if rng is None:
        rng = np.random.default_rng()

    # Set up initial population
    minbounds, maxbounds = np.broadcast_arrays(minbounds, maxbounds)
    if initial_population is not None:
        population = np.array(initial_population, dtype=np.float64)
        assert population.ndim == 2, "Initial population must be a 2D array."
        assert population.shape[1:] == minbounds.shape, (
            f"Initial population must have a shape ending with {minbounds.shape}."
            f" Its current shape is {population.shape}."
        )
        npop = population.shape[0]
    else:
        if npop is None:
            npop = 10 * minbounds.size
        population = rng.uniform(minbounds, maxbounds, size=(npop,) + minbounds.shape)

    logger.info(f"Population size: {npop}")
    logger.info("Evaluating initial population")

    # Evaluate fitness for initial population
    fitness = await fn(population)
    if not np.isfinite(fitness).any():
        raise Exception(
            "Fitness function returned non-finite values"
            " for all members of the initial population."
        )
    ibest = int(fitness.argmax())
    ancestor = population[ibest]

    if fitness[ibest] == fitness.min():
        logger.warning(
            "All members of the initial population have the same fitness."
            " This indicates that the fitness function is not sensitive"
            " to any of the parameters."
        )

    def draw(
        exclude: Iterable[int] = (),
    ) -> Iterator[np.ndarray[tuple[int], np.dtype[np.float64]]]:
        """Draws n vectors at random from the population, ensuring they
        do not overlap.
        """
        excluded = list(exclude)
        while True:
            i = rng.integers(population.shape[0] - len(excluded))
            excluded.sort()
            for oldind in excluded:
                if i >= oldind:
                    i += 1
            yield population[i]
            excluded.append(i)

    def check_ready(igen: int, nbad: int) -> bool:
        curminpar = population.min(axis=0)
        curmaxpar = population.max(axis=0)
        currange = curmaxpar - curminpar
        curcent = 0.5 * (curmaxpar + curminpar)
        tol = np.maximum(abstol, np.abs(curcent) * reltol)
        frange = fitness[ibest] - fitness.min()

        logger.info(f"  Range:     {', '.join([f'{v:.2e}' for v in currange])}")
        logger.info(f"  Tolerance: {', '.join([f'{v:.2e}' for v in tol])}")
        logger.info(f"  Fitness range: {frange}")
        if nbad:
            logger.warning(
                f"Non-finite fitness for {nbad} of {npop} candidate population members."
            )

        if callback is not None:
            callback(igen)

        ready = (currange <= tol).all() and frange <= ftol
        if ready:
            logger.info(
                "Optimization complete as parameter and fitness ranges"
                " within specified tolerance"
            )
        return ready

    if check_ready(0, npop - np.isfinite(fitness).sum()):
        return population[ibest]

    for igeneration in range(maxgen):
        logger.info(f"Starting generation {igeneration + 1}")
        trials = population.copy()
        for itarget, trial in enumerate(trials):
            # Draw random vectors
            if random_ancestor:
                # Randomly picked ancestor
                vectors = draw(exclude=(itarget,))
                ancestor = next(vectors)
            else:
                vectors = draw(exclude=(itarget, ibest))

            # Mutate base vector
            delta = np.zeros_like(ancestor)
            for _ in range(ndiffvector):
                delta += next(vectors) - next(vectors)
            mutant = ancestor + F * delta

            # Cross-over
            cross = rng.random(mutant.shape) < CR
            if not np.any(cross):
                # Ensure at least one parameter will have a value different
                # from that of the target vector. Otherwise there is no point
                # evaluating the trial vector
                cross[rng.integers(cross.size)] = True
            np.putmask(trial, cross, mutant)

            # Reflect parameter values if they have digressed beyond the specified
            # boundaries. This may need to be done multiple times, if the allowed
            # range is small and the parameter deviation large.
            while enforce_bounds:
                too_small = trial < minbounds
                too_large = trial > maxbounds
                if not (too_small.any() or too_large.any()):
                    break
                trial[too_small] += 2 * (minbounds - trial)[too_small]
                trial[too_large] -= 2 * (trial - maxbounds)[too_large]

        trial_fitnesses = await fn(trials)
        for itarget, (trial, trial_fitness) in enumerate(zip(trials, trial_fitnesses)):
            # Determine whether trial vector is better than target vector.
            # If so, replace target with trial.
            if trial_fitness >= fitness[itarget]:
                population[itarget, ...] = trial
                fitness[itarget] = trial_fitness
                if trial_fitness > fitness[ibest]:
                    ibest = itarget

        # Next default ancestor is current best
        ancestor = population[ibest]

        if check_ready(igeneration + 1, npop - np.isfinite(trial_fitnesses).sum()):
            break
    else:
        logger.warning(f"No convergence within the maximum {maxgen} generations.")

    return population[ibest]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    allx = []

    async def fn(X: np.ndarray) -> np.ndarray:
        allx.append(X)
        return fitness(X[:, 0], X[:, 1])

    def fitness(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 5 * (-np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2))

    minbounds = np.array([-5, -5])
    maxbounds = np.array([5, 5])
    result = asyncio.run(solve(fn, minbounds, maxbounds))
    print("Optimal value:", result)

    import matplotlib.pyplot as plt

    x = np.linspace(minbounds[0], maxbounds[0], 100)
    y = np.linspace(minbounds[1], maxbounds[1], 100)
    xs, ys = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    v = fitness(xs, ys)
    ax.contourf(xs, ys, v, levels=100)
    X = np.vstack(allx)
    ax.plot(X[:, 0], X[:, 1], ".k", alpha=0.25)
    ax.plot(result[0], result[1], "ok", mfc="w")
    plt.show()
