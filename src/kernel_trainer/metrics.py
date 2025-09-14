"""
Metrics to be used for both evolutionary search and
kernel evaluation (expresivity and entanglement capacity)
"""

import numpy as np
from math import pi
from random import random
import matplotlib.pyplot as plt
from scipy.special import rel_entr  # kl_div

from tqdm import trange

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
from qiskit_aer import AerSimulator


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


def qiskit_centered_target_alignment(kernel: QuantumCircuit, X: np.array, y: np.array):
    """
    Compute Centered Kernel Alignment (CKA) between kernel matrix and target.

    Refs:

    [1]: Cortes et al.,
    "Algorithms for Learning Kernels Based on Centered Alignment",
    `https://arxiv.org/pdf/1203.0550`_.

    Args:
        kmatrix: Kernel matrix K (n x n)
        y: Target vector (n,)

    Returns:
        float: CKA score
    """
    # Get estimated kernel matrix
    kmatrix = kernel.evaluate(X)

    n = len(y)
    # Rescale
    nplus = np.count_nonzero(np.array(y) == 1)
    nminus = len(y) - nplus
    _Y = np.array([y / nplus if y == 1 else y / nminus for y in y])

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

    # Handle edge case where norms are zero
    if norm_k == 0 or norm_t == 0:
        return 0.0

    return inner_product / (norm_k * norm_t)


def schmidt_decomp(qc: QuantumCircuit):
    """
    Schmidt decomposition
    """

    # Simulate the state
    state = Statevector.from_instruction(qc)

    # Convert statevector to a numpy array
    state_array = state.data

    # Reshape the state vector into a matrix for the SVD
    matrix = state_array.reshape(
        qc.num_qubits, qc.num_qubits
    )  # Reshape for a 2x2 matrix

    # Perform SVD
    _, s, _ = np.linalg.svd(matrix)

    # The Schmidt coefficients are the singular values (s)
    schmidt_coefficients = s

    # The Schmidt number is the number of non-zero singular values
    # (accounting for numerical precision)
    schmidt_number = np.sum(s > 1e-10)

    return schmidt_number, schmidt_coefficients


class Expressivity:
    """Expressivity measures as the capacity of a circuit
    to cover all possible states for a given Hilbert space.
    """

    def __init__(self, dims: int = 75):
        """Inits the metrics considering a resolution based on provided
        bins.

        Args:
            dims (int, optional): _description_. Defaults to 75.
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
        """Set the Haar probability for a given circuit

        Args:
            num (int): _description_
        """
        self.p_haar_hist = []
        for i in range(self.dims - 1):
            self.p_haar_hist.append(
                (1 - self.bins_list[i]) ** (2**num - 1)
                - (1 - self.bins_list[i + 1]) ** (2**num - 1)
            )

    def plot(self):
        """Simple plot to check the overlap between the two
        functions.
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
        """Computes the expressivity.

        Args:
            circuit (QuantumCircuit): _description_
            nshots (int, optional): _description_. Defaults to 10_000.
            samples (int, optional): _description_. Defaults to 4_000.

        Returns:
            float: _description_
        """
        # Init Haar
        nqubits = circuit.num_qubits
        self._set_p_haar(nqubits)

        # Select the AerSimulator from the Aer provider
        simulator = AerSimulator(method="matrix_product_state")

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

        return sum(rel_entr(pi_hist, self.p_haar_hist))


class EntanglingCapacity:
    """Computes the entanglement capacity of the circuit."""

    def __init__(self, circuit: QuantumCircuit):
        """
        Initializes the EntanglingCapacity class with the given
        circuit.

        Args:
            circuit: the QuantumCircuit
        """
        self.circuit = circuit
        self.N = circuit.num_qubits

    def calculate(self, nshots: int = 10_000, samples: int = 4_000):
        """
        Computes the Meyer-Wallach entanglement measure for the quantum circuit.

        Args:
            nshots (int, optional): _description_. Defaults to 10_000.
            samples (int, optional): _description_. Defaults to 4_000.

        Returns:
            float: Meyer-Wallach entanglement measure averaged over multiple samples.
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
        Helper function to calculate the average entropy over all qubits.

        Args:
            rho: the reduced state of the quantum system (density matrix).

        Returns:
            float: Entropy
        """
        entropy = 0
        qb_indices = list(range(self.N))

        # Loop over each qubit, calculate the partial trace and its entropy
        for j in range(self.N):
            # Partial trace over all qubits except the j-th qubit
            pt = partial_trace(rho, qb_indices[:j] + qb_indices[j + 1 :])
            entropy += pt.purity()

        return entropy
