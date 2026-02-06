"""
Metrics to be used for both evolutionary search and
kernel evaluation (expresivity and entanglement capacity)
"""

import gc
import numpy as np
from math import pi
from random import random
import matplotlib.pyplot as plt
from scipy.special import rel_entr  # kl_div

from pennylane.kernels.utils import square_kernel_matrix
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
from qiskit_aer import AerSimulator

import os
import psutil
from tqdm import trange


def qiskit_target_alignment(
    kernel: QuantumCircuit, X: np.array, y: np.array, logger=None
):
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

    Parameters
    ----------
    kernel : qiskit.QuantumCircuit
        Qiskit kernel object providing an ``evaluate`` method.
    X : numpy.ndarray
        Feature matrix of samples.
    y : numpy.ndarray
        Labels corresponding to samples in ``X``.
    logger : logging.Logger or None, optional
        Optional logger for debug messages.

    Returns
    -------
    float
        Target alignment (alignment score).

    References
    ----------
    Hubregtsen et al., "Training Quantum Embedding Kernels on Near-Term Quantum Computers", arXiv:2105.02276 (2021).
    """
    # Get estimated kernel matrix
    kmatrix = kernel.evaluate(X)

    if logger:
        proc = psutil.Process(os.getpid())
        ram_used = proc.memory_info().rss / (1024 * 1024)
        logger.debug(f"Kernel evaluated at {proc.pid} (MEM: {round(ram_used, 2)})")

    # Rescale
    nplus = np.count_nonzero(np.array(y) == 1)
    nminus = len(y) - nplus
    _Y = np.array([yi / nplus if yi == 1 else yi / nminus for yi in y])

    # Target matrix
    T = np.outer(_Y, _Y)
    inner_product = np.sum(kmatrix * T)
    norm = np.sqrt(np.sum(kmatrix * kmatrix) * np.sum(T * T))
    alignment = inner_product / norm

    # Free resources
    del kmatrix
    gc.collect()

    return alignment


def qiskit_centered_target_alignment(
    kernel: QuantumCircuit, X: np.array, y: np.array, logger=None
):
    """
    Compute Centered Kernel Alignment (CKA) between a kernel and the target.

    Parameters
    ----------
    kernel : qiskit.QuantumCircuit
        Qiskit kernel object with an ``evaluate`` method.
    X : numpy.ndarray
        Feature matrix used to evaluate the kernel.
    y : numpy.ndarray
        Target labels vector.
    logger : logging.Logger or None, optional
        Optional logger for debug messages.

    Returns
    -------
    float
        The CKA score (0..1) comparing the centered kernel matrix and the label kernel.

    References
    ----------
    Cortes et al., "Algorithms for Learning Kernels Based on Centered Alignment".
    """
    # Get estimated kernel matrix
    kmatrix = kernel.evaluate(X)

    if logger:
        proc = psutil.Process(os.getpid())
        ram_used = proc.memory_info().rss / (1024 * 1024)
        logger.debug(f"Kernel evaluated at {proc.pid} (MEM: {round(ram_used, 2)} MB)")

    n = len(y)
    # Rescale
    nplus = np.count_nonzero(np.array(y) == 1)
    nminus = len(y) - nplus
    _Y = np.array([yi / nplus if yi == 1 else yi / nminus for yi in y])

    # Create centering matrix H = I - (1/n) * 1 * 1^T
    H = np.eye(n) - (1 / n) * np.ones((n, n))

    # Center the kernel matrix: HKH
    centered_kmatrix = H @ kmatrix @ H

    # Target matrix (outer product of y)
    T = np.outer(_Y, _Y)

    # Center the target matrix: HTH
    centered_T = H @ T @ H

    # Compute CKA using the Frobenius inner product
    inner_product = np.sum(centered_kmatrix * centered_T)
    norm_k = np.sqrt(np.sum(centered_kmatrix * centered_kmatrix))
    norm_t = np.sqrt(np.sum(centered_T * centered_T))

    # Free resources
    del kmatrix
    gc.collect()

    # Handle edge case where norms are zero
    if norm_k == 0 or norm_t == 0:
        return 0.0

    return inner_product / (norm_k * norm_t)


def schmidt_decomp(qc: QuantumCircuit):
    """
    Compute Schmidt decomposition statistics for a given circuit.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        Quantum circuit to simulate.

    Returns
    -------
    tuple
        ``(schmidt_number, schmidt_coefficients)`` where ``schmidt_coefficients``
        are the singular values of the reshaped statevector and ``schmidt_number``
        is the number of non-zero singular values (with numerical tolerance).
    """

    # Simulate the state
    state = Statevector.from_instruction(qc)

    # Convert statevector to a numpy array
    state_array = state.data

    # Reshape the state vector into a matrix for the SVD
    # For bipartite split, divide qubits roughly in half
    n = qc.num_qubits
    n_a = n // 2
    n_b = n - n_a
    matrix = state_array.reshape(2**n_a, 2**n_b)

    # Perform SVD
    _, s, _ = np.linalg.svd(matrix)

    # The Schmidt coefficients are the singular values (s)
    schmidt_coefficients = s

    # The Schmidt number is the number of non-zero singular values
    # (accounting for numerical precision)
    schmidt_number = np.sum(s > 1e-10)

    return schmidt_number, schmidt_coefficients


def pennylane_centered_kernel_alignment(
    X, Y, kernel, assume_normalized_kernel=False, logger=None
):
    r"""Centered kernel alignment of a given kernel function.

    Centered kernel alignment (CKA) is a similarity index that measures the
    relationship between representational similarity matrices. Unlike regular
    kernel-target alignment, CKA centers the kernel matrices before computing
    alignment, making it invariant to isotropic scaling.

    For a dataset with feature vectors :math:`\{x_i\}` and associated labels
    :math:`\{y_i\}`, the centered kernel alignment is given by:

    .. math ::

        \operatorname{CKA}(K, L) = \frac{\operatorname{HSIC}(K, L)}
        {\sqrt{\operatorname{HSIC}(K, K)\operatorname{HSIC}(L, L)}}

    where :math:`\operatorname{HSIC}` is the Hilbert-Schmidt Independence Criterion:

    .. math ::

        \operatorname{HSIC}(K, L) = \frac{1}{(n-1)^2}\operatorname{tr}(KHLH)

    Here, :math:`H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T` is the centering matrix,
    :math:`K` is the kernel matrix for features :math:`X`, and :math:`L` is the
    kernel matrix for labels :math:`Y`.

    For binary classification with labels :math:`y_i \in \{-1, 1\}`, the label
    kernel is typically :math:`L_{ij} = y_i y_j`.

    Parameters
    ----------
    X : list[datapoint]
        List of datapoints used to evaluate the kernel.
    Y : list[float]
        Class labels (typically -1 or 1) associated with each datapoint.
    kernel : callable
        Kernel function mapping two datapoints to a scalar similarity.
    assume_normalized_kernel : bool, optional
        If True, assume the kernel is normalized (k(x,x)=1).

    Returns
    -------
    float
        The centered kernel alignment (CKA) score.

    **Example:**

    Consider a simple kernel function based on :class:`~.templates.embeddings.AngleEmbedding`:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev)
        def circuit(x1, x2):
            qml.templates.AngleEmbedding(x1, wires=dev.wires)
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=dev.wires)
            return qml.probs(wires=dev.wires)

        kernel = lambda x1, x2: circuit(x1, x2)[0]

    We can then compute the centered kernel alignment on a set of 4 (random)
    feature vectors ``X`` with labels ``Y`` via

    >>> rng = np.random.default_rng(seed=1234)
    >>> X = rng.random((4, 2))
    >>> Y = np.array([-1, -1, 1, 1])
    >>> qml.kernels.centered_kernel_alignment(X, Y, kernel)
    np.float64(0.0582...)

    **References:**

    [1] Kornblith et al., "Similarity of Neural Network Representations Revisited"
        https://arxiv.org/abs/1905.00414
    """
    n = len(X)

    # Compute kernel matrices
    K = square_kernel_matrix(
        X, kernel, assume_normalized_kernel=assume_normalized_kernel
    )

    if logger:
        proc = psutil.Process(os.getpid())
        ram_used = proc.memory_info().rss / (1024 * 1024)
        logger.debug(f"Kernel evaluated at {proc.pid} (MEM: {round(ram_used, 2)} MB)")

    # Compute target kernel matrix (outer product of labels)
    _Y = np.array(Y)
    L = np.outer(_Y, _Y)

    # Construct centering matrix H = I - (1/n) * 1 * 1^T
    H = np.eye(n) - np.ones((n, n)) / n

    # Center the kernel matrices: K_centered = H @ K @ H
    K_centered = H @ K @ H
    L_centered = H @ L @ H

    # Compute HSIC values
    # HSIC(K, L) = (1/(n-1)^2) * tr(K_centered @ L_centered)
    hsic_KL = np.trace(K_centered @ L_centered) / ((n - 1) ** 2)
    hsic_KK = np.trace(K_centered @ K_centered) / ((n - 1) ** 2)
    hsic_LL = np.trace(L_centered @ L_centered) / ((n - 1) ** 2)

    # Compute CKA
    if hsic_KK == 0 or hsic_LL == 0:
        return 0.0
    cka = hsic_KL / np.sqrt(hsic_KK * hsic_LL)

    return cka


class Expressivity:
    """Expressivity measures as the capacity of a circuit
    to cover all possible states for a given Hilbert space.
    """

    def __init__(self, dims: int = 75):
        """
        Initialize expressivity metric with a discretization resolution.

        Parameters
        ----------
        dims : int, optional
            Number of bins used to estimate probability histograms (default 75).
        """
        self.dims = dims

        # Possible Bin
        self.bins_list = []
        for i in range(dims):
            self.bins_list.append((i) / (dims - 1))

        # Center of the Bin
        self.bins_x = []
        for i in range(dims - 1):
            self.bins_x.append(self.bins_list[1] + self.bins_list[i])

        # Haar histogram
        self.p_haar_hist = []

        # Fidelity
        self.fidelity = []
        self.weights = []

    def _set_p_haar(self, num: int):
        """
        Compute the Haar distribution histogram for a circuit with ``num`` qubits.

        Parameters
        ----------
        num : int
            Number of qubits used to compute the Haar expectation histogram.
        """
        self.p_haar_hist = []
        for i in range(self.dims - 1):
            self.p_haar_hist.append(
                (1 - self.bins_list[i]) ** (2**num - 1)
                - (1 - self.bins_list[i + 1]) ** (2**num - 1)
            )

    def plot(self):
        """
        Plot the fidelity histogram and Haar reference distribution.

        This function creates a quick visual diagnostic of the circuit fidelity
        distribution against the Haar-uniform histogram.
        """
        # Plot
        plt.hist(
            self.fidelity,
            bins=self.bins_list,
            weights=self.weights,
            label="Circuit",
            range=[0, 1],
        )
        plt.plot(self.bins_x, self.p_haar_hist, label="Haar")
        plt.legend(loc="upper right")
        plt.show()

    def calculate(
        self, circuit: QuantumCircuit, nshots: int = 10_000, samples: int = 4_000
    ) -> float:
        """
        Estimate expressivity by sampling circuit fidelities and comparing to
        Haar-random expectations.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            Circuit used as the basis for randomised parameter sampling.
        nshots : int, optional
            Number of shots for simulator sampling (default 10_000).
        samples : int, optional
            Number of random parameter samples to use (default 4_000).

        Returns
        -------
        float
            Expressivity metric (KL divergence of observed fidelity histogram
            against the Haar distribution).
        """  # Init Haar
        nqubits = circuit.num_qubits
        self._set_p_haar(nqubits)

        # Select the AerSimulator from the Aer provider
        simulator = AerSimulator(method="matrix_product_state")

        # Copy circuit to avoid mutating the input
        circuit = circuit.copy()

        # Remove measurements
        zero_state = "0" * nqubits
        circuit.remove_final_measurements(inplace=True)

        self.fidelity = []
        for _ in trange(samples):
            # Restart the circuit
            qc = QuantumCircuit(nqubits, nqubits)

            # Add random parameters
            # U(x)
            params = {}
            for p in circuit.parameters:
                params[p] = 2 * pi * random()
            qc.compose(circuit.assign_parameters(params), inplace=True)

            # U^{\dagger}(y)
            params = {}
            for p in circuit.parameters:
                params[p] = 2 * pi * random()
            qc_dagger = circuit.assign_parameters(params).inverse()
            qc.compose(qc_dagger, inplace=True)

            for i in range(nqubits):
                qc.measure(i, i)

            job = simulator.run([qc], shots=nshots)
            result = job.result()
            count = result.get_counts()

            # Checkout overlap with |0> state
            if zero_state in count:
                ratio = count[zero_state] / nshots
            else:
                ratio = 0
            self.fidelity.append(ratio)

        # Weight of the results
        self.weights = np.ones_like(self.fidelity) / float(len(self.fidelity))

        # Example of calculating the KL divergence (relative entropy) with scipy
        pi_hist = np.histogram(
            self.fidelity, bins=self.bins_list, weights=self.weights, range=[0, 1]
        )[0]

        # Add small epsilon to avoid inf in KL divergence
        eps = 1e-10
        p_haar_smooth = np.array(self.p_haar_hist) + eps
        p_haar_smooth /= p_haar_smooth.sum()
        pi_hist_smooth = pi_hist + eps
        pi_hist_smooth /= pi_hist_smooth.sum()

        return sum(rel_entr(pi_hist_smooth, p_haar_smooth))


class EntanglingCapacity:
    """Computes the entanglement capacity of the circuit."""

    def __init__(self, circuit: QuantumCircuit):
        """
        Initialize entangling capacity calculator for a circuit.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            Circuit whose entangling capacity will be estimated.
        """
        self.circuit = circuit
        self.N = circuit.num_qubits

    def calculate(self, nshots: int = 10_000, samples: int = 4_000):
        """
        Estimate Meyer–Wallach entanglement measure by sampling parameterised circuits.

        Parameters
        ----------
        nshots : int, optional
            Number of simulator shots per sample (default 10_000).
        samples : int, optional
            Number of random parameter samples to average over (default 4_000).

        Returns
        -------
        float
            Average Meyer–Wallach entanglement measure across samples.
        """
        # Select the AerSimulator from the Aer provider
        simulator = AerSimulator(method="matrix_product_state")

        res = np.zeros(samples, dtype=complex)
        for i in trange(samples):
            # Add random parameters
            params = {}
            for p in self.circuit.parameters:
                params[p] = 2 * pi * random()
            qc = self.circuit.assign_parameters(params)
            qc.save_statevector(label="statevector")
            job = simulator.run([qc], shots=nshots)
            result = job.result()
            data = result.data()

            # Reduce the full state vector to density matrix for the entire system
            # reduce_statevector(state, indices=self.dev.wires)
            rho = DensityMatrix(data["statevector"])

            entropy = self._calculate_entropy(rho)

            # Meyer-Wallach measure for the current state
            res[i] = 1 - (entropy / self.N)

        # Average over the samples and return
        return float(2 * np.mean(res).real)

    def _calculate_entropy(self, rho):
        """
        Calculate average subsystem purity-derived entropy across qubits.

        Parameters
        ----------
        rho : qiskit.quantum_info.DensityMatrix
            Density matrix of the full system.

        Returns
        -------
        float
            Average purity-based entropy used for the Meyer–Wallach measure.
        """
        entropy = 0
        qb_indices = list(range(self.N))

        # Loop over each qubit, calculate the partial trace and its entropy
        for j in range(self.N):
            # Partial trace over all qubits except the j-th qubit
            pt = partial_trace(rho, qb_indices[:j] + qb_indices[j + 1 :])
            entropy += pt.purity()

        return entropy
