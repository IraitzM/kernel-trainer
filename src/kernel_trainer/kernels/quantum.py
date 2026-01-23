"""
Lists predefined kernel structures
"""
import time
import numpy as np
import pennylane as qml

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel,
    TrainableFidelityQuantumKernel,
)
from kernel_trainer.metrics import (
    qiskit_target_alignment,
    qiskit_centered_target_alignment,
    EntanglingCapacity,
    Expressivity,
)

import os
import psutil
from loguru import logger
from joblib import Parallel, delayed
from deap import base, creator

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
    hadamard = False
    if not paulis:
        hadamard = True
        paulis = ["Z"]

    def inner_func(x1, x2):
        n_qubits = len(x1)

        projector = np.zeros((2**n_qubits, 2**n_qubits))
        projector[0, 0] = 1

        # Compute
        for _ in range(reps):
            for w in paulis:
                if len(w) == 1:
                    for i in range(n_qubits):
                        if hadamard:
                            qml.Hadamard(i)
                        qml.PauliRot(2 * (np.pi - x1[i]), w, wires=i)
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
                        qml.PauliRot(-(2 * (np.pi - x2[i])), w, wires=i)
                        if hadamard:
                            qml.Hadamard(i)
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

        return qml.probs(wires=range(n_qubits))

    return inner_func


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
                if len(word) == 1:
                    idx = word.index(w)
                    if w == "X":
                        feature_map.rx(theta=2 * (np.pi - x[idx]), qubit=idx)
                    elif w == "Y":
                        feature_map.ry(theta=2 * (np.pi - x[idx]), qubit=idx)
                    else:
                        feature_map.rz(phi=2 * (np.pi - x[idx]), qubit=idx)
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

    # Aligned with pennylane's ordering
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


def get_stats(individual, num_qubits):
    """
    Gets some key statistics for a feature map individual
    """
    qc = QuantumCircuit(num_qubits)

    kernel_auto = ind_to_qiskit_kernel(individual, qc)
    depth = kernel_auto.feature_map.depth()

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.compose(kernel_auto.feature_map, inplace=True)
    qc.measure([x for x in range(num_qubits)], [x for x in range(num_qubits)])

    expr = Expressivity(dims=120)
    expr_m = expr.calculate(qc, nshots=10_000, samples=4_000)

    ent_cap = EntanglingCapacity(qc)
    entang = ent_cap.calculate(samples=4_000)

    return depth, expr_m, entang


def evaluation_function(
    individual,
    X: np.ndarray,
    y: np.ndarray,
    backend: str = "pennylane",
    metric: str = "KTA",
    penalize_complexity: bool = False,
):
    """Creates the evaluator function

    Args:
        individual (np.ndarray): Selected feature map individual
        X (np.ndarray): _description_
        y (np.ndarray): _description_
        per_class (int, optional): _description_. Defaults to 10.
    """
    # Fitness score, default 0
    fit_score = 0

    # No operation
    if sum(individual) == 0:
        return (fit_score,)

    if backend == "pennylane":
        start_time = time.time()
        proc = psutil.Process(os.getpid())
        ram_used = proc.memory_info().rss / (1024 * 1024)
        logger.debug(f"Creating individual in process {proc.pid} (MEM: {round(ram_used, 2)} MB)")
        device = qml.device("qulacs.simulator", wires=X.shape[1])  # lightning.gpu
        kernel = ind_to_pennylane_kernel(individual, device)

        logger.debug(f"Going for KTA proxy metric {proc.pid} (MEM: {round(ram_used, 2)} MB)")
        fit_score = qml.kernels.target_alignment(X, y, lambda x1, x2: kernel(x1, x2)[0])

        if penalize_complexity:
            specs = qml.specs(qnode=kernel)(X[0], X[0])
            resources = specs["resources"]

            depth = resources.depth
            non_local_gates = resources.gate_types["PauliRot"]

            fit_score = fit_score * non_local_gates * depth

        ram_used = proc.memory_info().rss / (1024 * 1024)
        logger.debug(f"Finishing {proc.pid} (MEM: {round(ram_used, 2)} MB)")
        timediff = time.time() - start_time
        logger.debug(f"Proxy calculation on {proc.pid} took {timediff} (proc. time {time.process_time()})")

    elif backend == "qiskit":
        start_time = time.time()
        qc = QuantumCircuit(X.shape[1])

        # Do some hard computing on the individual
        proc = psutil.Process(os.getpid())
        ram_used = proc.memory_info().rss / (1024 * 1024)
        logger.debug(f"Creating individual in process {proc.pid} (MEM: {round(ram_used, 2)} MB)")
        kernel = ind_to_qiskit_kernel(individual, qc)

        # Mask that fixes qiskit's idle qubit removal
        qubit_alloc = np.array(individual).reshape(-1, X.shape[1]).sum(axis=0)
        mask = (qubit_alloc > 0).tolist()

        try:
            ram_used = proc.memory_info().rss / (1024 * 1024)
            logger.debug(f"Going for the proxy metric {proc.pid} (MEM: {round(ram_used, 2)} MB)")
            if metric == "KTA":
                fit_score = qiskit_target_alignment(kernel, X[:, mask], y, logger)
            else:
                fit_score = qiskit_centered_target_alignment(kernel, X[:, mask], y, logger)

            # Penalization if circuit is deep
            if penalize_complexity:
                non_local_gates = kernel.feature_map.num_nonlocal_gates()
                depth = kernel.feature_map.depth()
                fit_score = fit_score * non_local_gates * depth

            ram_used = proc.memory_info().rss / (1024 * 1024)
            logger.debug(f"Finishing {proc.pid} (MEM: {round(ram_used, 2)} MB)")
            timediff = time.time() - start_time
            logger.debug(f"Proxy calculation on {proc.pid} took {timediff} (proc. time {time.process_time()})")
        except Exception as e:
            # Some individuals raise issues when building the target alignment
            logger.error(f"{individual}: {e}")

    return (fit_score,)


def get_matrices(X_train, X_test, y_train, fm: str = "Z"):
    num_dim = X_train.shape[1]

    if fm == "Z":
        kernel = qiskit_pauli_kernel(dims=num_dim, paulis=None)
    elif fm == "ZZ":
        from qiskit.circuit.library import ZZFeatureMap

        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)

        # Instantiate quantum kernel
        feature_map = ZZFeatureMap(num_dim, reps=1)
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    cka = qiskit_centered_target_alignment(kernel, X_train, y_train)

    matrix_train = kernel.evaluate(x_vec=X_train)
    matrix_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)

    return matrix_train, matrix_test, cka


def get_matrices_ind(X_train, X_test, y_train, ind: list):
    num_dim = X_train.shape[1]
    qc = QuantumCircuit(num_dim)

    kernel = ind_to_qiskit_kernel(individual=ind, qc=qc)

    cka = qiskit_centered_target_alignment(kernel, X_train, y_train)

    matrix_train = kernel.evaluate(x_vec=X_train)
    matrix_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)

    return matrix_train, matrix_test, cka
