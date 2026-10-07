import numpy as np


def test_get_matrices_backends():
    """Ensure get_matrices returns valid matrices for both qiskit and pennylane."""

    from kernel_trainer.kernels.quantum import get_matrices

    # create a trivial dataset (2 samples, 2 features, binary labels)
    X_train = np.array([[0.1, 0.2], [0.4, 0.3]])
    X_test = np.array([[0.5, 0.6]])
    y_train = np.array([0, 1])

    m_train, m_test, cka = get_matrices(
        X_train, X_test, y_train, fm="Z", backend="qiskit"
    )
    # Gram matrices should have compatible shapes
    assert m_train.shape == (2, 2)
    assert m_test.shape == (1, 2)
    # alignment score must be numeric and finite
    assert isinstance(cka, float)
    assert np.isfinite(cka)

    m_train, m_test, cka = get_matrices(
        X_train, X_test, y_train, fm="Z", backend="pennylane"
    )
    # Gram matrices should have compatible shapes
    assert m_train.shape == (2, 2)
    assert m_test.shape == (1, 2)
    # alignment score must be numeric and finite
    assert isinstance(cka, float)
    assert np.isfinite(cka)
