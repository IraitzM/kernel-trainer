"""
Expressivity and entanglement capacity.
"""

import numpy as np
from math import pi
from random import random
import matplotlib.pyplot as plt
from scipy.special import rel_entr  # kl_div

from tqdm import trange

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit_aer import AerSimulator


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

        # Possible Bin
        self.bins_list = []
        for i in range(dims):
            self.bins_list.append((i) / (dims - 1))

        # Center of the Bin
        self.bins_x = []
        for i in range(dims - 1):
            self.bins_x.append(self.bins_list[1] + self.bins_list[i])

        # Harr histogram
        self.p_harr_hist = []
        for i in range(dims - 1):
            self.p_harr_hist.append(
                self._p_harr(self.bins_list[i], self.bins_list[i + 1], 2)
            )

        # Fidelity
        self.fidelity = []
        self.weights = []

    def _p_harr(self, low: float, up: float, num: float) -> float:
        """Harr-random states probability function.

        Args:
            low (float): Lower bound
            up (float): Upper bound
            num (float): Number of samples

        Returns:
            float: Harr probability
        """
        return (1 - low) ** (num - 1) - (1 - up) ** (num - 1)

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
        plt.plot(self.bins_x, self.p_harr_hist, label="Harr")
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

        # Select the AerSimulator from the Aer provider
        simulator = AerSimulator(method="matrix_product_state")

        # Remove measurements
        nqubits = circuit.num_clbits
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

        return sum(rel_entr(pi_hist, self.p_harr_hist))


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

    def calculate(self, n_samples):
        """
        Computes the Meyer-Wallach entanglement measure for the quantum circuit.

        Args:
            n_samples (int): Number of samples to calculate the entanglement measure

        Returns:
            float: Meyer-Wallach entanglement measure averaged over multiple samples.
        """
        # Select the AerSimulator from the Aer provider
        simulator = AerSimulator(method="matrix_product_state")

        res = np.zeros(n_samples, dtype=complex)
        for i in trange(n_samples):
            # Add random parameters
            params = {}
            for p in self.circuit.parameters:
                params[p] = 2 * pi * random()
            qc = self.circuit.assign_parameters(params)
            qc.save_statevector(label="statevector")
            job = simulator.run([qc], shots=10_000)
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
