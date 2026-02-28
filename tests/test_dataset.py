import numpy as np
import pandas as pd

from kernel_trainer.dataset import DataGenerator


def test_checkerboard_centres_overlap():
    """Verify that the helper produces sensible centre coordinates."""
    gen = DataGenerator(samples=100, imbalance_ratio=0.5)

    # overlap=0 should reproduce the hard‑coded 2c centres used
    # in the original implementation.  verify both coordinates and label
    # pattern.
    radius = 0.15
    c0, lbl0 = gen._checkerboard_centres(4, radius=radius, overlap=0.0)
    assert c0.shape == (16, 3)
    assert lbl0.shape == (16,)

    # the helper uses linspace to compute the grid
    base = np.linspace(radius, 1 - radius, 4)
    expected = []
    for y in base:
        for x in base:
            expected.append([x, y, 1 - x])
    expected = np.array(expected)
    assert np.allclose(c0, expected)

    # the supplied labels should form a checkerboard starting with 1
    assert lbl0[0] == 1
    assert lbl0[1] == 0
    assert lbl0[4] == 0
    assert lbl0[5] == 1

    # overlap=1 collapses all centres to the midpoint of the cube
    c1, _ = gen._checkerboard_centres(4, radius=radius, overlap=1.0)
    assert np.allclose(c1, 0.5)

    # full overlap collapses all centres to the midpoint
    c1, _ = gen._checkerboard_centres(4, radius=radius, overlap=1.0)
    assert np.allclose(c1, 0.5)

    # the checkerboard pattern should start with a '1' in the top-left corner
    assert lbl0[0] == 1
    assert lbl0[1] == 0
    assert lbl0[4] == 0
    assert lbl0[5] == 1


def test_generate_dataset_2c_counts():
    """Dataset produced for ``2c`` has the expected number of points per label."""
    samples = 160
    imratio = 0.3
    gen = DataGenerator(samples=samples, imbalance_ratio=imratio, seed=123)

    df = gen.generate_dataset("2c", overlap=0.5)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (samples, 4)  # three features plus 'y'

    ones = int(df["y"].sum())
    zeros = samples - ones

    expected_n1 = int((samples * imratio) / 8)
    expected_n2 = int((samples - expected_n1 * 8) / 8)
    assert ones == 8 * expected_n1
    assert zeros == 8 * expected_n2
