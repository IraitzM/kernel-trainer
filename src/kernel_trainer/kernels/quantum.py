"""
Lists predefined kernel structures
"""

import time
import numpy as np
import pennylane as qml

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score

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
    pennylane_centered_kernel_alignment,
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
    Create a Pennylane Pauli-based feature-map kernel.

    Parameters
    ----------
    reps : int, optional
        Number of repetitions (default is 1).
    paulis : list of str, optional
        Pauli strings that define the feature map (e.g., 'Z', 'XX', 'ZY'). If
        ``None`` the default is ``['Z']`` which applies single-qubit Z rotations.
    entanglement : {'linear', 'pauli'}, optional
        Entanglement pattern used for multi-qubit Pauli terms. ``'linear'``
        applies two-body terms between adjacent qubits; ``'pauli'`` uses the
        explicit Pauli words provided in ``paulis``.

    Returns
    -------
    callable
        A kernel function ``f(x1, x2)`` that can be used inside a Pennylane
        ``QNode`` and returns a probability vector or scalar depending on the
        circuit measurement.

    Notes
    -----
    This implements a Pauli-rotation feature embedding aligned with common
    quantum-kernel constructions (see Havlíček et al., 2019).

    References
    ----------
    Havlíček, V., et al., "Supervised learning with quantum-enhanced feature spaces", Nature (2019).
    """
    hadamard = False
    if not paulis:
        hadamard = True
        paulis = ["Z"]

    def inner_func(x1, x2):
        """Kernel evaluation function for two input vectors.

        Parameters
        ----------
        x1, x2 : array-like
            Input vectors representing features to be embedded into the
            quantum circuit.

        Returns
        -------
        numpy.ndarray
            Probabilities or measurement outcomes resulting from the QNode.
        """
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
    Build a Qiskit quantum kernel using a Pauli feature map.

    Parameters
    ----------
    dims : int
        Input dimensionality (number of classical features to encode).
    reps : int, optional
        Number of repeated layers in the feature map (default 1).
    paulis : list of str, optional
        List of Pauli words used to construct the feature map. If ``None`` a
        standard ``ZFeatureMap`` is used.
    trainable_block : list, optional
        If provided, a list of trainable gate labels to insert as additional
        parameterised layers.

    Returns
    -------
    qiskit_machine_learning.kernels.FidelityQuantumKernel or TrainableFidelityQuantumKernel
        Instantiated Qiskit kernel object ready to evaluate kernel matrices.

    Notes
    -----
    The returned kernel is compatible with Qiskit's kernel evaluation methods
    (``evaluate``) and can be used in kernel-based classifiers.
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
    Return a function that computes the pairwise kernel matrix using ``kernel_func``.

    Parameters
    ----------
    kernel_func : callable
        Kernel function of two inputs ``f(x, y)`` that returns a scalar
        similarity value or array-like result.

    Returns
    -------
    callable
        ``kernel_matrix(A, B)`` which computes ``f(a, b)`` for all ``a`` in ``A``
        and ``b`` in ``B`` and returns a :class:`numpy.ndarray` with shape
        ``(len(A), len(B))``.
    """

    def kernel_matrix(A, B):
        """
        Compute the kernel Gram matrix between sets A and B.

        Parameters
        ----------
        A, B : array-like
            Collections of input vectors.

        Returns
        -------
        numpy.ndarray
            Kernel Gram matrix of shape ``(len(A), len(B))``.
        """
        rows = Parallel(n_jobs=-1, verbose=10)(
            delayed(kernel_func)(a, np.transpose(B)) for a in A
        )

        return np.array(rows)

    return kernel_matrix


def circuit_evals_kernel(n_data, split):
    """
    Compute the number of quantum circuit evaluations required for training
    and prediction given a dataset split.

    Parameters
    ----------
    n_data : int
        Total number of samples in the dataset.
    split : float
        Fraction of data used for training (0 < split < 1).

    Returns
    -------
    int
        Total number of circuit evaluations required (training + prediction).
    """
    M = int(np.ceil(split * n_data))
    Mpred = n_data - M

    n_training = M * M
    n_prediction = M * Mpred

    return n_training + n_prediction


def ind_to_pennylane_kernel(individual: np.ndarray, dev: qml.devices.LegacyDevice):
    """
    Convert an encoded individual into a Pennylane QNode kernel.

    Parameters
    ----------
    individual : numpy.ndarray
        Integer-encoded description of the feature-map individual.
    dev : pennylane.devices.LegacyDevice
        PennyLane device on which the QNode will be executed.

    Returns
    -------
    qml.QNode
        A PennyLane QNode that evaluates the kernel for two inputs ``x1`` and
        ``x2`` and returns the measured probabilities or fidelity-related value.
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
    """
    Convert an encoded individual into a Qiskit quantum kernel object.

    Parameters
    ----------
    individual : numpy.ndarray
        Integer-encoded description of the feature-map individual.
    qc : qiskit.QuantumCircuit
        QuantumCircuit instance used as a template for the feature map.
    trainable_block : list, optional
        Optional list of trainable gate labels to insert in the feature map.

    Returns
    -------
    qiskit_machine_learning.kernels.FidelityQuantumKernel
        Instantiated Qiskit kernel ready to evaluate precomputed kernel matrices.
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
    Compute quick diagnostics for a given feature-map individual.

    Parameters
    ----------
    individual : array-like
        Encoded individual describing the feature map.
    num_qubits : int
        Number of qubits (i.e., input dimensionality) to build the circuit.

    Returns
    -------
    tuple
        ``(depth, expressivity, entangling_capacity)`` where ``depth`` is the
        circuit depth, ``expressivity`` is a numeric score estimated by the
        ``Expressivity`` metric, and ``entangling_capacity`` is the
        entanglement capacity estimate.
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
    """
    Evaluate a feature-map individual using a proxy metric.

    Parameters
    ----------
    individual : array-like
        Encoded feature map individual (integer array). A zero-coded individual
        is considered a no-op and returns a zero fitness tuple.
    X : numpy.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Target labels corresponding to ``X``.
    backend : {'pennylane', 'qiskit'}, optional
        Backend used to build and evaluate the kernel (default 'pennylane').
    metric : {'KTA', 'CKA'}, optional
        Proxy metric to use for scoring: Kernel Target Alignment (KTA) or
        Centered Kernel Alignment (CKA).
    penalize_complexity : bool, optional
        If True, penalize solutions with larger circuit depth or more
        non-local gates.

    Returns
    -------
    tuple
        A 1-tuple containing the (minimization) fitness score: ``(fitness,)``.

    Notes
    -----
    The function computes the proxy metric using the selected backend and
    optionally penalises complex circuits by multiplying by depth and number
    of non-local gates.
    """  # Fitness score, default 0
    fit_score = 0

    # No operation
    if sum(individual) == 0:
        return (fit_score,)

    if backend == "pennylane":
        start_time = time.time()
        proc = psutil.Process(os.getpid())
        ram_used = proc.memory_info().rss / (1024 * 1024)
        logger.debug(
            f"Creating individual in process {proc.pid} (MEM: {round(ram_used, 2)} MB)"
        )
        device = qml.device("qulacs.simulator", wires=X.shape[1])  # lightning.gpu
        kernel = ind_to_pennylane_kernel(individual, device)

        if metric == "KTA":
            logger.debug(
                f"Going for KTA proxy metric {proc.pid} (MEM: {round(ram_used, 2)} MB)"
            )
            fit_score = qml.kernels.target_alignment(
                X, y, lambda x1, x2: kernel(x1, x2)[0]
            )
        else:
            logger.debug(
                f"Going for CKA proxy metric {proc.pid} (MEM: {round(ram_used, 2)} MB)"
            )
            fit_score = pennylane_centered_kernel_alignment(
                X, y, lambda x1, x2: kernel(x1, x2)[0], logger=logger
            )

        if penalize_complexity:
            specs = qml.specs(qnode=kernel)(X[0], X[0])
            resources = specs["resources"]

            depth = resources.depth
            non_local_gates = resources.gate_types["PauliRot"]

            fit_score = fit_score * non_local_gates * depth

        ram_used = proc.memory_info().rss / (1024 * 1024)
        logger.debug(f"Finishing {proc.pid} (MEM: {round(ram_used, 2)} MB)")
        timediff = time.time() - start_time
        logger.debug(
            f"Proxy calculation on {proc.pid} took {timediff} (proc. time {time.process_time()})"
        )

    elif backend == "qiskit":
        start_time = time.time()
        qc = QuantumCircuit(X.shape[1])

        # Do some hard computing on the individual
        proc = psutil.Process(os.getpid())
        ram_used = proc.memory_info().rss / (1024 * 1024)
        logger.debug(
            f"Creating individual in process {proc.pid} (MEM: {round(ram_used, 2)} MB)"
        )
        kernel = ind_to_qiskit_kernel(individual, qc)

        # Mask that fixes qiskit's idle qubit removal
        qubit_alloc = np.array(individual).reshape(-1, X.shape[1]).sum(axis=0)
        mask = (qubit_alloc > 0).tolist()

        try:
            ram_used = proc.memory_info().rss / (1024 * 1024)
            logger.debug(
                f"Going for the proxy metric {proc.pid} (MEM: {round(ram_used, 2)} MB)"
            )
            if metric == "KTA":
                fit_score = qiskit_target_alignment(kernel, X[:, mask], y, logger)
            else:
                fit_score = qiskit_centered_target_alignment(
                    kernel, X[:, mask], y, logger
                )

            # Penalization if circuit is deep
            if penalize_complexity:
                non_local_gates = kernel.feature_map.num_nonlocal_gates()
                depth = kernel.feature_map.depth()
                fit_score = fit_score * non_local_gates * depth

            ram_used = proc.memory_info().rss / (1024 * 1024)
            logger.debug(f"Finishing {proc.pid} (MEM: {round(ram_used, 2)} MB)")
            timediff = time.time() - start_time
            logger.debug(
                f"Proxy calculation on {proc.pid} took {timediff} (proc. time {time.process_time()})"
            )
        except Exception as e:
            # Some individuals raise issues when building the target alignment
            logger.error(f"{individual}: {e}")

    return (fit_score,)


def get_matrices(X_train, X_test, y_train, fm: str = "Z"):
    """
    Build kernel matrices for QSVC evaluation using the selected feature map.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training features of shape ``(n_train, n_features)``.
    X_test : numpy.ndarray
        Test features of shape ``(n_test, n_features)``.
    y_train : numpy.ndarray
        Training labels used to compute alignment metrics.
    fm : str, optional
        Feature-map identifier string (e.g., ``'Z'``, ``'ZZ-linear'``, or a
        Pauli string), by default ``'Z'``.

    Returns
    -------
    tuple
        ``(matrix_train, matrix_test, cka)`` where matrices are precomputed
        kernel Gram matrices and ``cka`` is the centered kernel alignment
        score on the training set.
    """  # Num features
    num_dim = X_train.shape[1]
    kernel = None

    if fm == "Z":
        kernel = qiskit_pauli_kernel(dims=num_dim, paulis=None)
    elif fm.startswith("ZZ"):
        from qiskit.circuit.library import ZZFeatureMap

        _, entanglement = fm.split("-")

        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)

        # Instantiate quantum kernel
        feature_map = ZZFeatureMap(num_dim, reps=1, entanglement=entanglement)
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    else:
        from qiskit.circuit.library import PauliFeatureMap

        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)

        if "-" in fm:
            pauli, entanglement = fm.split("-")
            feature_map = PauliFeatureMap(
                num_dim, reps=1, paulis=[pauli], entanglement=entanglement
            )
        else:
            feature_map = PauliFeatureMap(num_dim, reps=1, paulis=[fm])
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    cka = qiskit_centered_target_alignment(kernel, X_train, y_train)

    matrix_train = kernel.evaluate(x_vec=X_train)
    matrix_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)

    return matrix_train, matrix_test, cka


def get_scores_ind(
    X_train, X_test, y_train, y_test, ind: list, backend: str = "qiskit", seed: int = 42
):
    """
    Evaluate a given individual by computing train/test kernel matrices and
    training a precomputed-kernel SVM.

    Parameters
    ----------
    X_train, X_test : numpy.ndarray
        Training and test feature matrices.
    y_train, y_test : numpy.ndarray
        Training and test labels.
    ind : list
        Encoded individual that defines the feature map.
    backend : {'qiskit', 'pennylane'}, optional
        Backend to use for kernel evaluation (default 'qiskit').
    seed : int, optional
        Random seed used when fitting the classifier (default 42).

    Returns
    -------
    tuple
        ``(roc_auc, f1score, cka)`` evaluated on the test set and training CKA.
    """
    num_dim = X_train.shape[1]

    if backend == "qiskit":
        # Create kernel
        qc = QuantumCircuit(num_dim)
        kernel = ind_to_qiskit_kernel(individual=ind, qc=qc)

        # Eval circuit
        cka = qiskit_centered_target_alignment(kernel, X_train, y_train)

        # Matrices
        m_train = kernel.evaluate(x_vec=X_train)
        m_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
    else:
        device = qml.device("qulacs.simulator", wires=X_train.shape[1])  # lightning.gpu
        kernel = ind_to_pennylane_kernel(ind, device)

        def pennylane_matrix_compute(A, B):
            """
            Compute kernel evaluations using a PennyLane QNode and return
            the |0> probability for each pair.

            Parameters
            ----------
            A, B : array-like
                Input sets for which to compute the kernel matrix.

            Returns
            -------
            numpy.ndarray
                Precomputed kernel matrix with |0> probabilities.
            """
            if len(A.shape) == 2:
                # Single vector
                return np.array([[kernel(a, b)[0] for b in B] for a in A])
            else:
                # Matrix
                return kernel(A, B)[0]

        cka = pennylane_centered_kernel_alignment(
            X_train, y_train, pennylane_matrix_compute
        )

        m_train = pennylane_matrix_compute(X_train, X_train)
        m_test = pennylane_matrix_compute(X_test, X_train)

    # Compute the model
    model = SVC(kernel="precomputed", probability=True, random_state=seed)
    model.fit(m_train, y_train)

    # Get the metrics
    y_pred = model.predict_proba(m_test)[:, 1]
    roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred)
    f1score = f1_score(y_test, model.predict(m_test))

    return roc_auc, f1score, cka
