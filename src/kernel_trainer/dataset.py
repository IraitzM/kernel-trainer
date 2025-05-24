"""
Dataset generation functions
"""

import random
import numpy as np
import pandas as pd


class DataGenerator:
    """_summary_"""

    def __init__(
        self, samples: int = 100, imbalance_ratio: float = 0.5, seed: int = 4321
    ):
        random.seed(seed)

        self.seed = seed
        self.samples = samples
        self.imbalance_ratio = imbalance_ratio

    def _ellipsoid(self, center, rx, ry, rz, nmax):
        """
        Generate random points inside a 3D ellipsoid.

        Args:
            center (list): Center location of the ellipsoid [x,y,z]
            rx (float): Radius of the ellipsoid in x-axis
            ry (float): Radius of the ellipsoid in y-axis
            rz (float): Radius of the ellipsoid in z-axis
            nmax (int): Total number of data points to generate

        Returns:
            numpy.ndarray: Array of generated points [[x1,y1,z1],[x2,y2,z2]....[xn,yn,zn]]
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

    def generate_dataset(self, dataset_id: str):
        """
        Creates the dataset according to provided ID.

        Args:
            dataset_id (str): Dataset ID
        """
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
                n2 = int((self.samples - n1) / 2)

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
                n2 = int((self.samples - n1) / 2)

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
                n2 = int((self.samples - n1) / 4)

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
                n1 = int((self.samples * self.imbalance_ratio) / 8)
                n2 = int((self.samples - n1) / 8)

                X0 = self._ellipsoid([0.15, 0.15, 0.85], 0.15, 0.15, 0.15, n1)
                X1 = self._ellipsoid([0.38, 0.15, 0.62], 0.15, 0.15, 0.15, n2)
                X2 = self._ellipsoid([0.62, 0.15, 0.38], 0.15, 0.15, 0.15, n1)
                X3 = self._ellipsoid([0.85, 0.15, 0.15], 0.15, 0.15, 0.15, n2)
                X4 = self._ellipsoid([0.15, 0.38, 0.85], 0.15, 0.15, 0.15, n1)
                X5 = self._ellipsoid([0.38, 0.38, 0.62], 0.15, 0.15, 0.15, n2)
                X6 = self._ellipsoid([0.62, 0.38, 0.38], 0.15, 0.15, 0.15, n1)
                X7 = self._ellipsoid([0.85, 0.38, 0.15], 0.15, 0.15, 0.15, n2)
                X8 = self._ellipsoid([0.15, 0.62, 0.85], 0.15, 0.15, 0.15, n1)
                X9 = self._ellipsoid([0.38, 0.62, 0.62], 0.15, 0.15, 0.15, n2)
                X10 = self._ellipsoid([0.62, 0.62, 0.38], 0.15, 0.15, 0.15, n1)
                X11 = self._ellipsoid([0.85, 0.62, 0.15], 0.15, 0.15, 0.15, n2)
                X12 = self._ellipsoid([0.15, 0.85, 0.85], 0.15, 0.15, 0.15, n1)
                X13 = self._ellipsoid([0.38, 0.85, 0.62], 0.15, 0.15, 0.15, n2)
                X14 = self._ellipsoid([0.62, 0.85, 0.38], 0.15, 0.15, 0.15, n1)
                X15 = self._ellipsoid([0.85, 0.85, 0.15], 0.15, 0.15, 0.15, n2)

                X_1 = np.concatenate(
                    (
                        X0,
                        X1,
                        X2,
                        X3,
                        X4,
                        X5,
                        X6,
                        X7,
                        X8,
                        X9,
                        X10,
                        X11,
                        X12,
                        X13,
                        X14,
                        X15,
                    )
                )
                y_1 = np.concatenate(
                    (
                        [1] * n1,
                        [0] * n2,
                        [1] * n1,
                        [0] * n2,
                        [0] * n1,
                        [1] * n2,
                        [0] * n1,
                        [1] * n2,
                        [1] * n1,
                        [0] * n2,
                        [1] * n1,
                        [0] * n2,
                        [0] * n1,
                        [1] * n2,
                        [0] * n1,
                        [1] * n2,
                    )
                )

                _, c = X_1.shape
                features = pd.DataFrame(X_1, columns=[f"x{i}" for i in range(c)])
                target = pd.DataFrame(y_1, columns=["y"])
                return pd.concat([features, target], axis=1)
            case "3a":
                n1 = int((self.samples * self.imbalance_ratio) / 2)
                n2 = int((self.samples - n1) / 2)

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

                r, c = X_1.shape
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
                return NotImplementedError
