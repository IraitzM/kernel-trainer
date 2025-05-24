import time
import random
import numpy as np

import multiprocessing
from loguru import logger
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
    population: int = 1000,
    chain_size: int = 10,
    ngen: int = 50,
    cxpb: float = 0.2,
    mutpb: float = 0.1,
    processes: int = 1,
):
    """
    Iterated over a population of potential kernels checking
    their fitness so that best individuals are obtained.

    Args:
        X (_type_): Sample data
        y (_type_): Target label
        population (int, optional): _description_. Defaults to 1000.
        chain_size (int, optional): _description_. Defaults to 10.
        ngen (int, optional): _description_. Defaults to 50.
        cxpb (float, optional): _description_. Defaults to 0.2.
        mutpb (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    toolbox = base.Toolbox()
    toolbox.register(
        "individual", initES, creator.Individual, creator.Strategy, chain_size
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Distributed
    pool = multiprocessing.Pool(processes=processes)
    toolbox.register("map", pool.map)
    toolbox.register("evaluate", evaluation_function, X=X, y=y, backend=backend)

    population = toolbox.population(n=population)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Algorithm
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "ts"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pops = [population]
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), ts=0, **record)
    logger.info(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        start_time = time.time()

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the current population by the offspring
        population[:] = offspring
        pops.append(offspring)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        timediff = time.time() - start_time
        logbook.record(gen=gen, nevals=len(invalid_ind), ts=timediff, **record)
        logger.info(logbook.stream)

        # Early stop
        last_iterations = [elem["max"] for elem in logbook[-10:]]
        max_std = np.std(last_iterations)
        if len(last_iterations) >= 10 and max_std < 1e-6:
            break

    # Order
    population.sort(key=lambda e: e.fitness, reverse=True)

    return population, logbook, pops
