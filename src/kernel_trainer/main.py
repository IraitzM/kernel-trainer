import time
import random
import numpy as np

import multiprocessing
from loguru import logger
from itertools import product
from deap import base, creator, tools, algorithms

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
    Initialize the population

    Args:
        icls (_type_): Individuals
        scls (_type_): Strategy
        num_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    ind = icls(np.random.uniform(low=-np.pi, high=np.pi, size=num_params))
    ind.strategy = scls(np.random.uniform(low=-np.pi, high=np.pi, size=num_params))
    return ind


def initES(icls, scls, size):
    """ """
    ind = icls(random.choice(range(4)) for _ in range(size))
    ind.strategy = scls(random.choice(range(4)) for _ in range(size))
    return ind


def mutate(individual):
    # Do some hard computing on the individual
    idx = random.choice(range(len(individual)))
    individual[idx] = (individual[idx] + 1) % 4

    return (individual,)


def mutate_rnd(individual):
    # Do some hard computing on the individual
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
    cache: dict = None,
):
    """
    Iterated over a population of potential kernels checking
    their fitness so that best individuals are obtained.

    Args:
        X (_type_): Sample data
        y (_type_): Target label
        backend (str) : Choice between qiskit and pennylane
        metric (str) : Choice between KTA and CKA (only available for qiskit)
        population (int, optional): _description_. Defaults to 1000.
        chain_size (int, optional): _description_. Defaults to 10.
        ngen (int, optional): _description_. Defaults to 50.
        cxpb (float, optional): _description_. Defaults to 0.2.
        mutpb (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
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
    pool = multiprocessing.Pool(processes=processes)
    toolbox.register("map", pool.map)
    toolbox.register(
        "evaluate",
        evaluation_function,
        X=X,
        y=y,
        backend=backend,
        metric=metric,
        penalize_complexity=penalize_complexity,
        cache=cache,
    )
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
    for gen in range(1, ngen + 1):
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
    Create all combiantions
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
    """Iterate over all possible options

    Args:
        X (_type_): _description_
        y (_type_): _description_
        backend (str, optional): _description_. Defaults to "qiskit".
        metric (str, optional): _description_. Defaults to 'KTA'.
        chain_size (int, optional): _description_. Defaults to 10.
        processes (int, optional): _description_. Defaults to 1.
        penalize_complexity (bool, optional): _description_. Defaults to False.
    """

    toolbox = base.Toolbox()
    toolbox.register("population", create_all)

    # Distributed
    pool = multiprocessing.Pool(processes=processes)
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
