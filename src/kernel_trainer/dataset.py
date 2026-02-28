"""
Dataset generation functions
"""

import random
import numpy as np
import pandas as pd


class DataGenerator:
    """
    Synthetic 3D dataset generator used for benchmarks and examples.

    The generator supports several pre-defined dataset templates (IDs such as
    ``'1a'``, ``'2c'``, etc.) returned as a :class:`pandas.DataFrame` with three
    feature columns and a ``'y'`` target column.
    """

    def __init__(
        self, samples: int = 100, imbalance_ratio: float = 0.5, seed: int = 4321
    ):
        """
        Parameters
        ----------
        samples : int, optional
            Number of total samples to generate (default 100).
        imbalance_ratio : float, optional
            Ratio of class proportions for binary datasets (default 0.5).
        seed : int, optional
            RNG seed for reproducibility (default 4321).
        """
        random.seed(seed)

        self.seed = seed
        self.samples = samples
        self.imbalance_ratio = imbalance_ratio

    def _ellipsoid(self, center, rx, ry, rz, nmax):
        """
        Generate random points uniformly inside a 3D ellipsoid.

        Parameters
        ----------
        center : sequence of float
            Center coordinates ``[x, y, z]`` of the ellipsoid.
        rx, ry, rz : float
            Radii along the x, y, and z axes.
        nmax : int
            Number of points to generate.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(nmax, 3)`` containing generated points in Cartesian coordinates.
        """
        x3 = []
        y3 = []
        z3 = []

        for _ in range(nmax):
            # Generate random spherical coordinates
            r3 = np.cbrt(random.random())  # Cube root for uniform volume distribution
            theta3 = np.arccos(1 - 2 * random.random())  # Polar angle
            phi3 = 2 * np.pi * random.random()  # Azimuthal angle

            # Convert to Cartesian coordinates with ellipsoid scaling
            x3.append(rx * r3 * np.sin(theta3) * np.cos(phi3) + center[0])
            y3.append(ry * r3 * np.sin(theta3) * np.sin(phi3) + center[1])
            z3.append(rz * r3 * np.cos(theta3) + center[2])

        return np.transpose([x3, y3, z3])

    def _checkerboard_centres(self, grid_size: int, radius: float, overlap: float):
        """
        Compute evenly-spaced centres for a square checkerboard layout.

        Parameters
        ----------
        grid_size : int
            Number of points along each axis (4 for a 4x4 grid).
        radius : float
            Ellipsoid radius used when constructing the linear spacing; the
            returned coordinates will always lie in ``[radius, 1-radius]``.
        overlap : float
            Value in ``[0,1]``.  When ``0`` the centres are placed at
            ``linspace(radius, 1-radius)``; when ``1`` all values collapse to
            ``0.5``.  Intermediate values slide the points toward the cube
            centre to increase mutual overlap.

        Returns
        -------
        centres : numpy.ndarray, shape (grid_size**2, 3)
            Cartesian coordinates of the centres in row-major order.
        labels : numpy.ndarray, shape (grid_size**2,)
            Checkerboard pattern of 0/1 labels; ``(i+j)%2 == 0`` yields a
            ``1``.
        """
        # clamp overlap and guard against misuse
        if overlap < 0:
            overlap = 0.0
        elif overlap > 1:
            overlap = 1.0

        base = np.linspace(radius, 1 - radius, grid_size)
        centre_point = 0.5
        adjusted = centre_point + (base - centre_point) * (1 - overlap)

        centres = []
        labels = []
        for iy, y in enumerate(adjusted):
            for ix, x in enumerate(adjusted):
                z = 1 - x
                centres.append([x, y, z])
                labels.append(1 if (ix + iy) % 2 == 0 else 0)

        return np.array(centres), np.array(labels)

    def generate_dataset(self, dataset_id: str, overlap: float = 0.0):
        """
        Create a synthetic dataset according to the provided template ID.

        Some templates subdivide the unit cube into a number of ellipsoidal
        clusters.  ``dataset_id=='2c'`` is a 4x4 checkerboard of 16 ellipsoids;
        the ``overlap`` argument lets the caller compress the grid towards the
        centre so that the spheres range from just-touching (``overlap=0``) to
        completely overlapping ``(overlap=1``).

        Parameters
        ----------
        dataset_id : str
            Identifier of the template dataset (e.g., '1a', '2c', etc.).
        overlap : float, optional
            Amount of overlap between ellipsoids when the template defines a
            grid.  Values are clamped to ``[0, 1]``; only ``'2c'`` currently
            supports this argument and it is ignored for other identifiers.
            ``0`` produces evenly-spaced (distant) clusters, ``1`` collapses
            all centres to the midpoint.

        Returns
        -------
        pandas.DataFrame
            DataFrame with three feature columns and a target column ``'y'``.
        """
        # make sure external callers haven't passed garbage
        if not 0.0 <= overlap <= 1.0:
            raise ValueError("overlap must be between 0 and 1")

        match dataset_id:
            case "0":
                n1 = int(self.samples * self.imbalance_ratio)
                n2 = self.samples - n1

                X1 = self._ellipsoid([0.24, 0.5, 0.6], 0.2, 0.4, 0.8, n1)
                X2 = self._ellipsoid([0.76, 0.5, 0.7], 0.2, 0.4, 0.8, n2)

                X_1 = np.concatenate((X1, X2))
                y_1 = np.concatenate(([0] * n1, [1] * n2))

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])

                return pd.concat([features, target], axis=1)
            case "1a":
                n1 = int(self.samples * self.imbalance_ratio)
                n2 = self.samples - n1

                X1 = self._ellipsoid([0.5, 0.5, 0.6], 0.2, 0.4, 0.8, n1)
                X2 = self._ellipsoid([0.7, 0.5, 0.7], 0.2, 0.4, 0.8, n2)

                X_1 = np.concatenate((X1, X2))
                y_1 = np.concatenate(([0] * n1, [1] * n2))

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])

                return pd.concat([features, target], axis=1)

            case "1b":
                n1 = int(self.samples * self.imbalance_ratio)
                n2 = int((self.samples - n1) / 2)

                X1 = self._ellipsoid([0.25, 0.25, 0.25], 0.25, 0.15, 0.15, n2)
                X2 = self._ellipsoid([0.5, 0.5, 0.5], 0.25, 0.15, 0.15, n1)
                X3 = self._ellipsoid([0.75, 0.75, 0.75], 0.25, 0.15, 0.15, n2)

                X_1 = np.concatenate((X1, X2, X3))
                y_1 = np.concatenate(([0] * n2, [1] * n1, [0] * n2))

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])
                return pd.concat([features, target], axis=1)
            case "1c":
                n1 = int((self.samples * self.imbalance_ratio) / 2)
                n2 = int((self.samples - n1 * 2) / 2)

                X0 = self._ellipsoid([0.3, 0.2, 0.8], 0.25, 0.15, 0.15, n1)
                X1 = self._ellipsoid([0.7, 0.4, 0.6], 0.25, 0.15, 0.15, n2)
                X2 = self._ellipsoid([0.3, 0.6, 0.4], 0.25, 0.15, 0.15, n1)
                X3 = self._ellipsoid([0.7, 0.8, 0.2], 0.25, 0.15, 0.15, n2)

                X_1 = np.concatenate((X0, X1, X2, X3))
                y_1 = np.concatenate(([1] * n1, [0] * n2, [1] * n1, [0] * n2))

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])
                return pd.concat([features, target], axis=1)
            case "2a":
                n1 = int((self.samples * self.imbalance_ratio) / 2)
                n2 = int((self.samples - n1 * 2) / 2)

                X0 = self._ellipsoid([0.3, 0.2, 0.8], 0.25, 0.15, 0.15, n1)
                X1 = self._ellipsoid([0.7, 0.4, 0.6], 0.25, 0.15, 0.15, n2)
                X2 = self._ellipsoid([0.3, 0.6, 0.4], 0.25, 0.15, 0.15, n1)
                X3 = self._ellipsoid([0.7, 0.8, 0.2], 0.25, 0.15, 0.15, n2)

                X_1 = np.concatenate((X0, X1, X2, X3))
                y_1 = np.concatenate(([0] * n1, [1] * n2, [1] * n2, [0] * n1))

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])
                return pd.concat([features, target], axis=1)
            case "2b":
                n1 = int((self.samples * self.imbalance_ratio) / 5)
                n2 = int((self.samples - n1 * 5) / 4)

                X0 = self._ellipsoid([0.2, 0.2, 0.8], 0.15, 0.15, 0.15, n1)
                X1 = self._ellipsoid([0.5, 0.2, 0.5], 0.15, 0.15, 0.15, n2)
                X2 = self._ellipsoid([0.8, 0.2, 0.2], 0.15, 0.15, 0.15, n1)
                X3 = self._ellipsoid([0.2, 0.5, 0.8], 0.15, 0.15, 0.15, n2)
                X4 = self._ellipsoid([0.5, 0.5, 0.5], 0.15, 0.15, 0.15, n1)
                X5 = self._ellipsoid([0.8, 0.5, 0.2], 0.15, 0.15, 0.15, n2)
                X6 = self._ellipsoid([0.2, 0.8, 0.8], 0.15, 0.15, 0.15, n1)
                X7 = self._ellipsoid([0.5, 0.8, 0.5], 0.15, 0.15, 0.15, n2)
                X8 = self._ellipsoid([0.8, 0.8, 0.2], 0.15, 0.15, 0.15, n1)

                X_1 = np.concatenate((X0, X1, X2, X3, X4, X5, X6, X7, X8))
                y_1 = np.concatenate(
                    (
                        [1] * n1,
                        [0] * n2,
                        [1] * n1,
                        [0] * n2,
                        [1] * n1,
                        [0] * n2,
                        [1] * n1,
                        [0] * n2,
                        [1] * n1,
                    )
                )

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])
                return pd.concat([features, target], axis=1)
            case "2c":
                # eight ellipsoids for each class; counts are balanced according
                # to ``imbalance_ratio``.  this is the only template that
                # currently makes use of the ``overlap`` argument.
                n1 = int((self.samples * self.imbalance_ratio) / 8)
                n2 = int((self.samples - n1 * 8) / 8)

                # build a 4x4 checkerboard of centres; ``overlap`` slides them
                # towards 0.5 so that 0==no overlap (evenly spaced) and 1==all
                # coincide.
                radius = 0.15
                centres, labels = self._checkerboard_centres(
                    grid_size=4, radius=radius, overlap=overlap
                )

                X_list = []
                y_list = []
                for lbl, centre in zip(labels, centres):
                    count = n1 if lbl == 1 else n2
                    X_list.append(
                        self._ellipsoid(centre, radius, radius, radius, count)
                    )
                    y_list.append(np.full(count, lbl))

                X_1 = np.concatenate(X_list)
                y_1 = np.concatenate(y_list)

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])
                return pd.concat([features, target], axis=1)
            case "3a":
                n1 = int(self.samples * self.imbalance_ratio)
                n2 = int(self.samples - n1)

                X1 = []
                radius = 0.25

                index = 0
                while index < n1:
                    x = random.random()
                    y = random.random()
                    z = random.random()
                    if (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 > radius**2:
                        X1 = X1 + [[x, y, z]]
                        index = index + 1

                X2 = self._ellipsoid([0.5, 0.5, 0.5], radius, radius, radius, n2)

                X_1 = np.concatenate((X1, X2))
                y_1 = np.concatenate(([1] * n1, [0] * n2))

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])
                return pd.concat([features, target], axis=1)
            case "3b":
                n1 = int((self.samples * self.imbalance_ratio))
                n2 = int((self.samples - n1) / 2)

                X1 = []
                radius = 0.25

                index = 0
                while index < n1:
                    x = random.random()
                    y = random.random()
                    z = random.random()
                    if (x - 0.25) ** 2 + (y - 0.25) ** 2 + (
                        z - 0.25
                    ) ** 2 > radius**2 and (x - 0.75) ** 2 + (y - 0.75) ** 2 + (
                        z - 0.75
                    ) ** 2 > radius**2:
                        X1 = X1 + [[x, y, z]]
                        index = index + 1

                X2 = self._ellipsoid([0.25, 0.25, 0.25], radius, radius, radius, n2)
                X3 = self._ellipsoid([0.75, 0.75, 0.75], radius, radius, radius, n2)

                X_1 = np.concatenate((X1, X2, X3))
                y_1 = np.concatenate(([1] * n1, [0] * n2 * 2))

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])
                return pd.concat([features, target], axis=1)
            case "3c":
                n1 = int((self.samples * self.imbalance_ratio))
                n2 = int((self.samples - n1) / 4)

                X1 = []
                radius = 0.25

                index = 0
                while index < n1:
                    x = random.random()
                    y = random.random()
                    z = random.random()
                    if (
                        (x - 0.25) ** 2 + (y - 0.25) ** 2 + (z - 0.25) ** 2 > radius**2
                        and (x - 0.75) ** 2 + (y - 0.75) ** 2 + (z - 0.75) ** 2
                        > radius**2
                        and (x - 0.25) ** 2 + (y - 0.75) ** 2 + (z - 0.25) ** 2
                        > radius**2
                        and (x - 0.75) ** 2 + (y - 0.25) ** 2 + (z - 0.75) ** 2
                        > radius**2
                    ):
                        X1 = X1 + [[x, y, z]]
                        index = index + 1

                X2 = self._ellipsoid([0.25, 0.25, 0.25], radius, radius, radius, n2)
                X3 = self._ellipsoid([0.75, 0.75, 0.75], radius, radius, radius, n2)
                X4 = self._ellipsoid([0.75, 0.25, 0.75], radius, radius, radius, n2)
                X5 = self._ellipsoid([0.25, 0.75, 0.25], radius, radius, radius, n2)

                X_1 = np.concatenate((X1, X2, X3, X4, X5))
                y_1 = np.concatenate(([1] * n1, [0] * n2 * 4))

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])
                return pd.concat([features, target], axis=1)
            case _:
                raise NotImplementedError(f"Unsupported dataset_id: {dataset_id}")
