"""
Lists predefined classical kernel structures
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.metrics import accuracy_score, roc_auc_score

# Sin kernel
def sin_kernel(X, Y):
    """
    Simple sine kernel: K(x,y) = amplitude * sin(frequency * ||x-y||)
    """

    amplitude = 0.25
    frequency = 0.2

    if Y is None:
        Y = X

    # Compute pairwise distances
    dists = np.sqrt(np.sum((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=2))

    # Apply sine transformation
    return amplitude * np.sin(frequency * dists)

# ExpSineSquared
def expsine2_kernel(X, Y):
    """
    https://scikit-learn.org/stable/modules/gaussian_process.html#gp-kernels

    Parameters:
    * length: controls kernel function decay, low more sensitive, large smoother patterns
    * periodicity: controles the period of the data
    """
    kernel = ExpSineSquared(length_scale=0.25, periodicity=0.1)
    return kernel(X, Y)

class QuantumInspiredKernels:
    """
    Collection of quantum-inspired classical kernels to compete with RZ kernel
    """

    @staticmethod
    def rz_inspired_kernel(X, Y=None, rotation_scale=1.0, phase_shift=0.0):
        """
        RZ-inspired kernel: K(x,y) = Re[exp(i * rotation_scale * <x,y> + phase_shift)]
        Mimics the RZ gate's phase rotation behavior
        """
        if Y is None:
            Y = X

        # Compute dot products (inner products)
        dot_products = np.dot(X, Y.T)

        # Apply phase rotation transformation
        phases = rotation_scale * dot_products + phase_shift

        # Take real part of complex exponential (like RZ gate)
        return np.real(np.exp(1j * phases))

    @staticmethod
    def complex_exponential_kernel(X, Y=None, alpha=1.0, beta=1.0):
        """
        Complex exponential kernel: K(x,y) = Re[exp(alpha * <x,y>) * exp(i * beta * ||x-y||²)]
        Combines amplitude and phase modulation
        """
        if Y is None:
            Y = X

        dot_products = np.dot(X, Y.T)
        sq_distances = np.sum((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=2)

        # Complex exponential with both real and imaginary parts
        amplitude_part = np.exp(alpha * dot_products)
        phase_part = np.exp(1j * beta * sq_distances)

        return np.real(amplitude_part * phase_part)

    @staticmethod
    def trigonometric_feature_kernel(X, Y=None, freq_scale=1.0, phase_shifts=None):
        """
        Trigonometric feature kernel: Explicit feature mapping with sin/cos
        Maps each feature to [cos(freq*x), sin(freq*x)] then uses inner product
        """
        if Y is None:
            Y = X

        if phase_shifts is None:
            phase_shifts = np.zeros(X.shape[1])

        # Transform features to trigonometric space
        X_trig = np.hstack([
            np.cos(freq_scale * X + phase_shifts),
            np.sin(freq_scale * X + phase_shifts)
        ])

        Y_trig = np.hstack([
            np.cos(freq_scale * Y + phase_shifts),
            np.sin(freq_scale * Y + phase_shifts)
        ])

        # Standard inner product in transformed space
        return np.dot(X_trig, Y_trig.T)

    @staticmethod
    def bloch_sphere_kernel(X, Y=None, theta_scale=1.0, phi_scale=1.0):
        """
        Bloch sphere inspired kernel: Maps features to sphere coordinates
        K(x,y) = cos(θ_x - θ_y) * cos(φ_x - φ_y)
        """
        if Y is None:
            Y = X

        # Map features to spherical coordinates
        theta_X = theta_scale * np.arctan2(X[:, 1], X[:, 0]) if X.shape[1] >= 2 else theta_scale * X[:, 0]
        phi_X = phi_scale * np.sum(X, axis=1) if X.shape[1] > 1 else phi_scale * X[:, 0]

        theta_Y = theta_scale * np.arctan2(Y[:, 1], Y[:, 0]) if Y.shape[1] >= 2 else theta_scale * Y[:, 0]
        phi_Y = phi_scale * np.sum(Y, axis=1) if Y.shape[1] > 1 else phi_scale * Y[:, 0]

        # Compute kernel as product of cosines of angle differences
        theta_diff = theta_X[:, np.newaxis] - theta_Y[np.newaxis, :]
        phi_diff = phi_X[:, np.newaxis] - phi_Y[np.newaxis, :]

        return np.cos(theta_diff) * np.cos(phi_diff)

    @staticmethod
    def quantum_fourier_kernel(X, Y=None, n_frequencies=5, scale=1.0):
        """
        Quantum Fourier Transform inspired kernel
        Uses multiple frequency components like QFT
        """
        if Y is None:
            Y = X

        kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))

        for k in range(1, n_frequencies + 1):
            freq = 2 * np.pi * k / n_frequencies

            # Create frequency components for each dimension
            cos_X = np.cos(scale * freq * X)
            sin_X = np.sin(scale * freq * X)
            cos_Y = np.cos(scale * freq * Y)
            sin_Y = np.sin(scale * freq * Y)

            # Add contribution from this frequency
            kernel_matrix += (np.dot(cos_X, cos_Y.T) + np.dot(sin_X, sin_Y.T)) / n_frequencies

        return kernel_matrix

    @staticmethod
    def pauli_inspired_kernel(X, Y=None, pauli_scale=1.0):
        """
        Pauli matrices inspired kernel
        Mimics Pauli-X, Y, Z transformations
        """
        if Y is None:
            Y = X

        # Pauli-X like: flip operation
        X_flip = -X
        Y_flip = -Y

        # Pauli-Y like: rotation with phase
        X_rot = np.column_stack([-X[:, 1], X[:, 0]]) if X.shape[1] >= 2 else X
        Y_rot = np.column_stack([-Y[:, 1], Y[:, 0]]) if Y.shape[1] >= 2 else Y

        # Pauli-Z like: phase modification
        phases_X = pauli_scale * np.sum(X, axis=1)
        phases_Y = pauli_scale * np.sum(Y, axis=1)

        # Combine all Pauli-like operations
        kernel = (np.dot(X, Y.T) +
                 np.dot(X_flip, Y_flip.T) +
                 np.dot(X_rot, Y_rot.T))

        # Add phase-like contribution
        phase_contribution = np.cos(phases_X[:, np.newaxis] - phases_Y[np.newaxis, :])

        return kernel + phase_contribution

class HadamardRZKernels:
    """
    Quantum-inspired kernels combining Hadamard and RZ gate operations
    
    Hadamard: H = (1/√2) * [[1, 1], [1, -1]] - creates superposition
    RZ(φ): [[e^(-iφ/2), 0], [0, e^(iφ/2)]] - adds phase rotation
    
    Combined operation: RZ(φ) * H creates superposition then phase rotation
    """

    @staticmethod
    def hadamard_rz_kernel(X, Y=None, rotation_scale=1.0, superposition_weight=0.5):
        """
        Direct Hadamard+RZ inspired kernel
        
        Hadamard creates superposition: (x + x_orthogonal) / √2
        RZ adds phase: exp(i * rotation_scale * features)
        """
        if Y is None:
            Y = X

        # Hadamard-like transformation: create superposition states
        # For each point, create orthogonal component
        X_hadamard = HadamardRZKernels._apply_hadamard_transform(X, superposition_weight)
        Y_hadamard = HadamardRZKernels._apply_hadamard_transform(Y, superposition_weight)

        # RZ-like phase rotation on transformed features
        dot_products = np.dot(X_hadamard, Y_hadamard.T)
        phases = rotation_scale * dot_products

        # Take real part of complex exponential
        return np.real(np.exp(1j * phases))

    @staticmethod
    def _apply_hadamard_transform(X, superposition_weight=0.5):
        """Apply Hadamard-like transformation to create superposition"""
        # Original features
        original = X

        # Create orthogonal/complementary features (Hadamard mixing)
        if X.shape[1] >= 2:
            # For 2D+, create rotated version
            orthogonal = np.column_stack([
                -X[:, 1], X[:, 0]  # 90-degree rotation
            ])
            if X.shape[1] > 2:
                # Add remaining dimensions with sign flip
                orthogonal = np.hstack([orthogonal, -X[:, 2:]])
        else:
            # For 1D, create complementary feature
            orthogonal = -X

        # Hadamard superposition: (|0⟩ + |1⟩)/√2 analog
        return (original + superposition_weight * orthogonal) / np.sqrt(1 + superposition_weight**2)

    @staticmethod
    def sequential_hadamard_rz_kernel(X, Y=None, rotation_scales=None, n_layers=3):
        """
        Sequential application of Hadamard+RZ layers (like quantum circuit)
        
        Each layer: H → RZ(φ_i) → measurement contribution
        Final kernel is sum of all layer contributions
        """
        if Y is None:
            Y = X

        if rotation_scales is None:
            rotation_scales = np.linspace(0.5, 2.0, n_layers)

        total_kernel = np.zeros((X.shape[0], Y.shape[0]))

        X_current = X.copy()
        Y_current = Y.copy()

        for _, rot_scale in enumerate(rotation_scales):
            # Apply Hadamard transformation
            X_h = HadamardRZKernels._apply_hadamard_transform(X_current, 0.7)
            Y_h = HadamardRZKernels._apply_hadamard_transform(Y_current, 0.7)

            # Apply RZ rotation
            phases = rot_scale * np.dot(X_h, Y_h.T)
            layer_contribution = np.real(np.exp(1j * phases))

            # Add to total kernel
            total_kernel += layer_contribution / n_layers

            # Prepare for next layer (simulate quantum evolution)
            X_current = X_h
            Y_current = Y_h

        return total_kernel

    @staticmethod
    def interferometric_hadamard_rz_kernel(X, Y=None, rotation_scale=1.0, interference_param=0.5):
        """
        Interferometric kernel: Hadamard creates interference patterns, RZ modulates phases
        
        Simulates quantum interference by combining multiple paths
        """
        if Y is None:
            Y = X

        # Path 1: Direct transformation
        X_path1 = HadamardRZKernels._apply_hadamard_transform(X, 0.5)
        Y_path1 = HadamardRZKernels._apply_hadamard_transform(Y, 0.5)

        # Path 2: Alternative transformation (different superposition)
        X_path2 = HadamardRZKernels._apply_hadamard_transform(X, -0.5)
        Y_path2 = HadamardRZKernels._apply_hadamard_transform(Y, -0.5)

        # Apply different RZ rotations to each path
        phases1 = rotation_scale * np.dot(X_path1, Y_path1.T)
        phases2 = rotation_scale * np.dot(X_path2, Y_path2.T) + np.pi * interference_param

        # Quantum interference: add amplitudes then take magnitude squared
        amplitude1 = np.exp(1j * phases1)
        amplitude2 = np.exp(1j * phases2)

        # Interference pattern
        total_amplitude = (amplitude1 + amplitude2) / np.sqrt(2)
        return np.real(total_amplitude * np.conj(total_amplitude))

    @staticmethod
    def parameterized_hadamard_rz_kernel(X, Y=None, theta_params=None, phi_params=None):
        """
        Parameterized quantum circuit inspired kernel
        
        Uses parameterized rotations: RY(θ) H RZ(φ)
        More flexible than fixed Hadamard+RZ
        """
        if Y is None:
            Y = X

        n_features = X.shape[1]

        if theta_params is None:
            theta_params = np.ones(n_features)
        if phi_params is None:
            phi_params = np.ones(n_features)

        # Apply parameterized transformations feature-wise
        X_transformed = np.zeros_like(X)
        Y_transformed = np.zeros_like(Y)

        for i in range(n_features):
            # RY rotation (parameterized Hadamard-like)
            X_ry = X[:, i] * np.cos(theta_params[i]) + np.ones_like(X[:, i]) * np.sin(theta_params[i])
            Y_ry = Y[:, i] * np.cos(theta_params[i]) + np.ones_like(Y[:, i]) * np.sin(theta_params[i])

            # Hadamard mixing
            X_h = (X_ry + X[:, i]) / np.sqrt(2)
            Y_h = (Y_ry + Y[:, i]) / np.sqrt(2)

            # RZ rotation
            X_transformed[:, i] = X_h
            Y_transformed[:, i] = Y_h

        # Final kernel with phase rotations
        phases = np.sum(phi_params) * np.dot(X_transformed, Y_transformed.T)
        return np.real(np.exp(1j * phases))

    @staticmethod
    def quantum_fourier_hadamard_kernel(X, Y=None, n_qubits=None, rotation_strength=1.0):
        """
        Quantum Fourier Transform + Hadamard inspired kernel
        
        Combines QFT-like multi-frequency analysis with Hadamard superposition
        """
        if Y is None:
            Y = X

        if n_qubits is None:
            n_qubits = min(8, X.shape[1] + 2)  # Reasonable default

        kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))

        # Apply Hadamard first (creates superposition)
        X_super = HadamardRZKernels._apply_hadamard_transform(X, 0.7)
        Y_super = HadamardRZKernels._apply_hadamard_transform(Y, 0.7)

        # QFT-like frequency analysis
        for k in range(n_qubits):
            freq = 2 * np.pi * k / (2**n_qubits)

            # Create frequency components
            for dim in range(X.shape[1]):
                cos_X = np.cos(rotation_strength * freq * X_super[:, dim])
                sin_X = np.sin(rotation_strength * freq * X_super[:, dim])
                cos_Y = np.cos(rotation_strength * freq * Y_super[:, dim])
                sin_Y = np.sin(rotation_strength * freq * Y_super[:, dim])

                # Add contribution from this frequency and dimension
                kernel_matrix += (np.outer(cos_X, cos_Y) + np.outer(sin_X, sin_Y)) / (n_qubits * X.shape[1])

        return kernel_matrix

    @staticmethod
    def entangled_hadamard_rz_kernel(X, Y=None, entanglement_strength=0.5, rotation_scale=1.0):
        """
        Simulates entangled qubits with Hadamard + RZ
        
        Creates correlations between features (entanglement-like)
        """
        if Y is None:
            Y = X

        n_features = X.shape[1]

        # Create entangled feature pairs
        kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))

        for i in range(n_features):
            for j in range(i, n_features):
                # Apply Hadamard to feature pair
                X_pair = np.column_stack([X[:, i], X[:, j] if j < n_features else X[:, i]])
                Y_pair = np.column_stack([Y[:, i], Y[:, j] if j < n_features else Y[:, i]])

                X_h = HadamardRZKernels._apply_hadamard_transform(X_pair, 0.7)
                Y_h = HadamardRZKernels._apply_hadamard_transform(Y_pair, 0.7)

                # Entanglement: correlate the features
                entangled_X = X_h[:, 0] + entanglement_strength * X_h[:, 1]
                entangled_Y = Y_h[:, 0] + entanglement_strength * Y_h[:, 1]

                # Apply RZ rotation
                phases = rotation_scale * np.outer(entangled_X, entangled_Y)
                pair_kernel = np.real(np.exp(1j * phases))

                # Add to total kernel
                weight = 2.0 / (n_features * (n_features + 1))  # Normalize
                kernel_matrix += weight * pair_kernel

        return kernel_matrix

class QuantumInspiredSVM:
    """Wrapper class for easy kernel experimentation"""

    def __init__(self, kernel_func, kernel_params=None, C=1.0):
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params or {}
        self.C = C
        self.svm = None
        self.X_train = None

    def fit(self, X, y):
        # Scaled data
        self.X_train = X

        # Create kernel matrix
        K_train = self.kernel_func(self.X_train, **self.kernel_params)

        # Ensure kernel matrix is positive semi-definite
        K_train = self._ensure_psd(K_train)

        # Train SVM
        self.svm = SVC(kernel='precomputed', C=self.C, probability=True)
        self.svm.fit(K_train, y)
        return self

    def predict(self, X):
        if self.svm is None:
            raise ValueError("Model must be fitted first")

        K_test = self.kernel_func(X, self.X_train, **self.kernel_params)
        K_test = self._ensure_psd(K_test)

        return self.svm.predict(K_test)

    def score(self, X, y):
        """
        ROC AUC Score
        """
        predictions = self.predict(X)
        return roc_auc_score(y, predictions)

    def _ensure_psd(self, K):
        """Ensure kernel matrix is positive semi-definite"""
        # Add small regularization to diagonal
        if K.shape[0] == K.shape[1]:
            K += 1e-8 * np.eye(K.shape[0])
        return K

class HadamardRZSVM:
    """Wrapper for Hadamard+RZ inspired SVM"""

    def __init__(self, kernel_func, kernel_params=None, C=1.0):
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params or {}
        self.C = C
        self.svm = None
        self.X_train = None

    def fit(self, X, y):
        # Scale the data if needed
        X_scaled = X.copy()

        self.X_train = X_scaled.copy()

        # Create kernel matrix
        K_train = self.kernel_func(X_scaled, **self.kernel_params)

        # Ensure positive semi-definite
        K_train = self._regularize_kernel(K_train)

        # Train SVM
        self.svm = SVC(kernel='precomputed', C=self.C, probability=True)
        self.svm.fit(K_train, y)
        return self

    def predict(self, X):
        if self.svm is None:
            raise ValueError("Model must be fitted first")

        X_scaled = X.copy()

        K_test = self.kernel_func(X_scaled, self.X_train, **self.kernel_params)
        K_test = self._regularize_kernel(K_test)

        return self.svm.predict(K_test)

    def score(self, X, y):
        """
        ROC AUC Score
        """
        predictions = self.predict(X)
        return roc_auc_score(y, predictions)

    def _regularize_kernel(self, K):
        """Add regularization to ensure numerical stability"""
        if K.shape[0] == K.shape[1]:
            K += 1e-8 * np.eye(K.shape[0])

        # Clip extreme values
        K = np.clip(K, -100, 100)
        return K
