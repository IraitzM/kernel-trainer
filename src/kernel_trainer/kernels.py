"""
Lists predefined kernel structures
"""

import numpy as np
import pennylane as qml

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel,
    TrainableFidelityQuantumKernel,
)

import multiprocessing
from joblib import Parallel, delayed
from deap import base, creator, tools, algorithms

pool = multiprocessing.Pool()

creator.create("FitnessMin", base.Fitness, weights=(1.0,))  # Minimization
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", np.ndarray)


def pennylane_pauli_kernel(
    reps: int = 1, paulis: list[str] = None, entanglement: str = "linear"
):
    """
    Creates the kernel based on a feature embedding represented by the Pauli list.

    Args:
        reps (int, optional): Number of repetitions. Defaults to 1.
        paulis (list[str], optional): List of Pauli terms encoding the
    feature map. Defaults to None.
        entanglement (str, optional): Entanglement to be considered creating the
    feature map. Defaults to "linear".

    Returns:
        function: Function to be used as a quantum circuit evaluation
    """
    if not paulis:
        paulis = ["Z"]

    def inner_func(x1, x2):
        n_qubits = len(x1)

        projector = np.zeros((2**n_qubits, 2**n_qubits))
        projector[0, 0] = 1

        # ComputeUncompute pattern
        for i in range(n_qubits):
            qml.Hadamard(wires=[i])

        # Compute
        for _ in range(reps):
            for w in paulis:
                if len(w) == 1:
                    for i in range(n_qubits):
                        qml.PauliRot(2 * x1[i], w, wires=i)
                elif len(w) == 2 and entanglement == "linear":
                    for i in range(n_qubits - 1):
                        qml.PauliRot(
                            2 * (np.pi - x1[i]) * (np.pi - x1[i + 1]),
                            w,
                            wires=[i, i + 1],
                        )
                elif entanglement == "pauli":
                    indexes = [i for i, val in enumerate(w) if val != "I"]
                    word = w.replace("I", "")
                    qml.PauliRot(
                        2 * np.prod(np.pi - x1[indexes], axis=0), word, wires=indexes
                    )

        # Uncompute
        for _ in range(reps):
            for w in reversed(paulis):
                if len(w) == 1:
                    for i in reversed(range(n_qubits)):
                        qml.PauliRot(-(2 * x2[i]), w, wires=i)
                elif len(w) == 2 and entanglement == "linear":
                    for i in range(n_qubits - 1):
                        qml.PauliRot(
                            -(2 * (np.pi - x2[i]) * (np.pi - x2[i + 1])),
                            w,
                            wires=[i, i + 1],
                        )
                elif entanglement == "pauli":
                    indexes = [i for i, val in enumerate(w) if val != "I"]
                    word = w.replace("I", "")
                    qml.PauliRot(
                        -2 * np.prod(np.pi - x2[indexes], axis=0), word, wires=indexes
                    )

        for i in range(n_qubits):
            qml.Hadamard(wires=[i])

        return qml.probs(wires=range(n_qubits))

    return inner_func


def qiskit_target_alignment(kernel: QuantumCircuit, X: np.array, y: np.array):
    """
    Target alignment loss function.
    The definition of the function is taken from Equation (27,28) of [1].
    The log-likelihood function is defined as:

    .. math::

        TA(K_{θ}) =
        \\frac{\\sum_{i,j} K_{θ}(x_i, x_j) y_i y_j}
        {\\sqrt{\\sum_{i,j} K_{θ}(x_i, x_j)^2 \\sum_{i,j} y_i^2 y_j^2}}

    Refs:

    [1]: T. Hubregtsen et al.,
    "Training Quantum Embedding Kernels on Near-Term Quantum Computers",
    `arXiv:2105.02276v1 (2021) <https://arxiv.org/abs/2105.02276>`_.

    Args:
        kernel (quantumcircuit): Qiskit quantum circuit
        X (np.array): Samples
        y (np.array): Target

    Returns:
        _type_: _description_
    """

    # Get estimated kernel matrix
    kmatrix = kernel.evaluate(X)

    # Rescale
    nplus = np.count_nonzero(np.array(y) == 1)
    nminus = len(y) - nplus
    _Y = np.array([y / nplus if y == 1 else y / nminus for y in y])

    # Target matrix
    T = np.outer(_Y, _Y)
    inner_product = np.sum(kmatrix * T)
    norm = np.sqrt(np.sum(kmatrix * kmatrix) * np.sum(T * T))
    alignment = inner_product / norm

    return alignment


def qiskit_pauli_kernel(
    dims: int, reps: int = 1, paulis: list[str] = None, trainable_block: list = None
):
    """
    Creates the kernel based on a feature embedding represented by the Pauli list.

    Args:
        dims (int): Dimensions of the entry data
        reps (int, optional): Number of repetitions. Defaults to 1.
        paulis (list[str], optional): List of Pauli terms encoding the
    feature map. Defaults to None.

    Returns:
        function: Function to be used as a quantum circuit evaluation
    """
    if not paulis:
        feature_map = ZFeatureMap(feature_dimension=dims, reps=reps)
    else:
        feature_map = QuantumCircuit(dims)
        x = [Parameter(f"x{i}") for i in range(dims)]

        if trainable_block:
            trainable_params = [
                Parameter(f"p{i}") for i in range(dims * reps * len(trainable_block))
            ]

        for r_iter in range(reps):
            for word in paulis:
                w = word.replace("I", "")
                if len(w) == 1:
                    idx = word.index(w)
                    if w == "X":
                        feature_map.rx(theta=2 * (x[idx]), qubit=idx)
                    elif w == "Y":
                        feature_map.ry(theta=2 * (x[idx]), qubit=idx)
                    else:
                        feature_map.rz(phi=2 * (x[idx]), qubit=idx)
                else:
                    # Basis change block
                    indexes = []
                    for qubit, op_label in enumerate(word):
                        if op_label == "Z":
                            indexes.append(qubit)
                        elif op_label == "X":
                            feature_map.h(qubit)
                            indexes.append(qubit)
                        elif op_label == "Y":
                            feature_map.sdg(qubit)
                            feature_map.h(qubit)
                            indexes.append(qubit)

                    # Interactions (CNOTs) block
                    if len(indexes) > 0:
                        two_body_it = [
                            (indexes[i], indexes[i + 1])
                            for i in range(len(indexes) - 1)
                        ]
                        for pair in two_body_it:
                            feature_map.cx(pair[0], pair[1])

                        # Rotation
                        angle = 2
                        for i in indexes:
                            angle *= np.pi - x[i]
                        feature_map.rz(angle, qubit=indexes[-1])

                        for pair in list(reversed(two_body_it)):
                            feature_map.cx(pair[0], pair[1])

                    # Basis change block
                    for qubit, op_label in enumerate(word):
                        if op_label == "X":
                            feature_map.h(qubit)
                        elif op_label == "Y":
                            feature_map.h(qubit)
                            feature_map.s(qubit)

            if trainable_block:
                for qi in range(dims):
                    for i, block in enumerate(trainable_block):
                        if block == "X":
                            feature_map.rx(
                                theta=trainable_params[(r_iter * dims + qi + i)],
                                qubit=qi,
                            )
                        elif block == "Y":
                            feature_map.ry(
                                theta=trainable_params[(r_iter * dims + qi + i)],
                                qubit=qi,
                            )
                        else:  # Z
                            feature_map.rz(
                                phi=trainable_params[(r_iter * dims + qi + i)], qubit=qi
                            )

    # Aligned with pennylanes ordering
    feature_map = feature_map.reverse_bits()

    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)

    # Instantiate quantum kernel
    if trainable_block:
        kernel = TrainableFidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=feature_map,
            training_parameters=trainable_params,
        )
    else:
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    return kernel


def compute_kernel_matrix(kernel_func):
    """
    Compute the matrix whose entries are the kernel
    evaluated on pairwise data from sets A and B.
    """

    def kernel_matrix(A, B):
        """
        Returns the array of a NxN matrix
        filled with results from the Kernel
        function.
        """

        rows = Parallel(n_jobs=-1, verbose=10)(
            delayed(kernel_func)(a, np.transpose(B)) for a in A
        )

        return np.array(rows)

    return kernel_matrix


def circuit_evals_kernel(n_data, split):
    """
    Compute how many circuit evaluations one needs for kernel-based
    training and prediction.
    """

    M = int(np.ceil(split * n_data))
    Mpred = n_data - M

    n_training = M * M
    n_prediction = M * Mpred

    return n_training + n_prediction


def ind_to_pennylane_kernel(individual: np.ndarray, dev: qml.devices.LegacyDevice):
    """Crates a kernel out of an individual

    Args:
        individual (np.ndarray): Numpy array describing the individual
        dev (qml.devices.LegacyDevice): Pennylane device

    Returns:
        QNode: Pennylane qnode to execute the quantum kernel function
    """

    replacements = {0: "I", 1: "X", 2: "Z", 3: "Y"}
    replacer = replacements.get  # For faster gets.

    # Substitute numbers by paulis
    pauli_strings = []
    for i in np.arange(0, len(individual), len(dev.wires)):
        section = individual[i : i + len(dev.wires)]
        if sum(section) > 0:
            pauli_strings.append("".join([replacer(n, n) for n in section]))

    # Kernel
    kernel_func = pennylane_pauli_kernel(paulis=pauli_strings, entanglement="pauli")
    kernel = qml.QNode(kernel_func, dev)

    return kernel


def ind_to_qiskit_kernel(
    individual: np.ndarray, qc: QuantumCircuit, trainable_block: list = None
):
    """Crates a kernel out of an individual

    Args:
        individual (np.ndarray): Numpy array describing the individual
        qc (QuantumCircuit) : Qiskit Quantum circuit

    Returns:
        Quantum Kernel: Pennylane qnode to execute the quantum kernel function
    """

    replacements = {0: "I", 1: "X", 2: "Z", 3: "Y"}
    replacer = replacements.get  # For faster gets.

    # Substitute numbers by paulis
    pauli_strings = []
    for i in np.arange(0, len(individual), qc.num_qubits):
        section = individual[i : i + qc.num_qubits]
        if sum(section) > 0:
            pauli_strings.append("".join([replacer(n, n) for n in section]))

    # Kernel
    kernel = qiskit_pauli_kernel(
        dims=qc.num_qubits, paulis=pauli_strings, trainable_block=trainable_block
    )

    return kernel


def evaluation_function(
    individual, X: np.ndarray, y: np.ndarray, backend: str = "pennylane"
):
    """Creates the evaluator function

    Args:
        individual (np.ndarray): Selected feature map individual
        X (np.ndarray): _description_
        y (np.ndarray): _description_
        per_class (int, optional): _description_. Defaults to 10.
    """
    if backend == "pennylane":
        device = qml.device("qulacs.simulator", wires=X.shape[1])  # lightning.gpu
        kernel = ind_to_pennylane_kernel(individual, device)

        return (qml.kernels.target_alignment(X, y, lambda x1, x2: kernel(x1, x2)[0]),)

    elif backend == "qiskit":
        qc = QuantumCircuit(X.shape[1])

        # Do some hard computing on the individual
        kernel = ind_to_qiskit_kernel(individual, qc)

        try:
            return (qiskit_target_alignment(kernel, X, y),)
        except Exception:
            # Some individuals raise issues when building the target alignment
            return (0,)


def kernel_generator(
    X,
    y,
    backend: str = "qiskit",
    population: int = 1000,
    chain_size: int = 10,
    ngen: int = 50,
    cxpb: float = 0.2,
    mutpb: float = 0.1,
    logger=None,
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
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    logger.info(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
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

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        logger.info(logbook.stream)

    # Order
    population.sort(key=lambda e: e.fitness, reverse=True)

    return population, logbook
