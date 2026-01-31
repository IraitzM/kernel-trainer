import time
import random
import numpy as np
from tqdm import tqdm

from loguru import logger
from itertools import product
from deap import base, creator, tools, algorithms
from multiprocessing import Pool

from kernel_trainer.kernels import evaluation_function

# Erase any previous creator configuration
try:
    del creator.Individual
    del creator.FitnessMin
    del creator.Strategy
except Exception as e:
    logger.error(e)

creator.create("FitnessMin", base.Fitness, weights=(1.0,))  # Minimization
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", np.ndarray)


def initRD(icls, scls, num_params):
    """
    Initialize a real-valued individual with strategy parameters.

    Parameters
    ----------
    icls : callable
        Individual class constructor.
    scls : callable
        Strategy class constructor.
    num_params : int
        Number of parameters for the individual/strategy vectors.

    Returns
    -------
    Individual
        Instantiated individual with attached strategy.
    """
    ind = icls(np.random.uniform(low=-np.pi, high=np.pi, size=num_params))
    ind.strategy = scls(np.random.uniform(low=-np.pi, high=np.pi, size=num_params))
    return ind


def initES(icls, scls, size):
    """
    Initialize a discrete individual using encoded integers.

    Parameters
    ----------
    icls : callable
        Individual class constructor.
    scls : callable
        Strategy class constructor.
    size : int
        Length of the individual.

    Returns
    -------
    Individual
        Instantiated individual with attached strategy.
    """
    ind = icls(random.choice(range(4)) for _ in range(size))
    ind.strategy = scls(random.choice(range(4)) for _ in range(size))
    return ind


def mutate(individual):
    """
    Discrete mutation operator: increment a random gene modulo 4.

    Parameters
    ----------
    individual : array-like
        Encoded individual to mutate.

    Returns
    -------
    tuple
        Single-element tuple containing the mutated individual.
    """
    idx = random.choice(range(len(individual)))
    individual[idx] = (individual[idx] + 1) % 4

    return (individual,)


def mutate_rnd(individual):
    """
    Gaussian mutation operator: add normal noise to a random gene.

    Parameters
    ----------
    individual : array-like
        Individual to mutate.

    Returns
    -------
    tuple
        Single-element tuple containing the mutated individual.
    """
    idx = random.choice(range(len(individual)))
    individual[idx] += np.random.normal(0, np.pi / 2)  # Modified mutation

    return (individual,)


def kernel_generator(
    X,
    y,
    backend: str = "qiskit",
    metric: str = "KTA",
    num_pop: int = 1000,
    chain_size: int = 10,
    ngen: int = 50,
    cxpb: float = 0.2,
    mutpb: float = 0.1,
    processes: int = 1,
    penalize_complexity: bool = False,
    tournament_size: int = 10,
):
    """
    Evolutionary search to find promising feature-map individuals.

    Parameters
    ----------
    X : array-like
        Training features used for fitness evaluation.
    y : array-like
        Training labels.
    backend : {'qiskit', 'pennylane'}, optional
        Backend used to build kernels (default 'qiskit').
    metric : {'KTA', 'CKA'}, optional
        Proxy metric used for fitness evaluation.
    num_pop : int, optional
        Population size (default 1000).
    chain_size : int, optional
        Length of encoded feature map individuals (default 10).
    ngen : int, optional
        Number of generations (default 50).
    cxpb, mutpb : float, optional
        Crossover and mutation probabilities.
    processes : int, optional
        Number of parallel processes to use.
    penalize_complexity : bool, optional
        Penalize complex circuits in fitness scoring.
    tournament_size : int, optional
        Tournament selection size.

    Returns
    -------
    tuple
        ``(population, logbook)`` with the final population and recorded statistics.
    """

    # Init toolbox
    toolbox = base.Toolbox()
    toolbox.register(
        "individual", initES, creator.Individual, creator.Strategy, chain_size
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Distributed
    pool = Pool(processes=processes, maxtasksperchild=1)
    toolbox.register("map", pool.map)
    toolbox.register(
        "evaluate",
        evaluation_function,
        X=X,
        y=y,
        backend=backend,
        metric=metric,
        penalize_complexity=penalize_complexity,
    )
    # Population
    population = toolbox.population(n=num_pop)

    # Set stats and logs
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "ts"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    valid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, valid_ind)
    for ind, fit in zip(valid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(valid_ind), ts=0, **record)
    logger.info(logbook.stream)

    # Begin the generational process
    for gen in tqdm(range(1, ngen + 1)):
        start_time = time.time()

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        valid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, valid_ind)
        for ind, fit in zip(valid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        timediff = time.time() - start_time
        logbook.record(gen=gen, nevals=len(valid_ind), ts=timediff, **record)
        logger.info(logbook.stream)

        # Early stop
        last_iterations = [elem["max"] for elem in logbook[-10:]]
        max_std = np.std(last_iterations)
        if len(last_iterations) >= 10 and max_std < 1e-6:
            break

    # Order
    population.sort(key=lambda e: e.fitness, reverse=True)

    return population, logbook


def create_all(size: int):
    """
    Return all non-trivial combinations for discrete individuals.

    Parameters
    ----------
    size : int
        Length of the combinations to generate.

    Returns
    -------
    list
        List of tuples enumerating all combinations in ``{0,1,2,3}^size``
        excluding the all-zero tuple.
    """
    return_list = list(product(range(4), repeat=size))
    return_list.remove((0,) * size)

    return return_list


def brute_force(
    X,
    y,
    backend: str = "qiskit",
    metric: str = "KTA",
    chain_size: int = 10,
    processes: int = 1,
    penalize_complexity: bool = False,
):
    """
    Brute-force evaluation over the full discrete search space.

    Parameters
    ----------
    X, y : array-like
        Training data and labels used for evaluation.
    backend : {'qiskit', 'pennylane'}, optional
        Backend for kernel construction (default 'qiskit').
    metric : {'KTA', 'CKA'}, optional
        Proxy metric to evaluate candidate kernels.
    chain_size : int, optional
        Length of the feature-map encoding.
    processes : int, optional
        Number of parallel processes to use.
    penalize_complexity : bool, optional
        If True, penalise complex circuits in the score.

    Returns
    -------
    tuple
        ``(output, logbook)`` sorted by descending score.
    """

    toolbox = base.Toolbox()
    toolbox.register("population", create_all)

    # Distributed
    pool = Pool(processes=processes)
    toolbox.register("map", pool.map)
    toolbox.register(
        "evaluate",
        evaluation_function,
        X=X,
        y=y,
        backend=backend,
        metric=metric,
        penalize_complexity=penalize_complexity,
    )

    # One all-combinations call
    population = toolbox.population(size=chain_size)

    # Evaluate the individuals with an invalid fitness
    output = {}
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        pauli_string = [str(code) for code in ind]
        output["".join(pauli_string)] = fit[0]

    # Statistics to show
    stats = tools.Statistics(lambda ind: ind)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = stats.fields if stats else []
    record = stats.compile(output.values()) if stats else {}
    logbook.record(**record)
    # stats
    logger.info(logbook.stream)

    # Sort output
    output = sorted(output.items(), key=lambda x: x[1], reverse=True)

    return output, logbook
