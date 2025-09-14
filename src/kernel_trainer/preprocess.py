"""
Preprocessing module
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class Preprocessor:
    """
    Implements preprocessing steps
    """

    def __init__(self, dimensions: int, mode: str = "tsne", scale: bool = False):
        """
        Initialize preprocessing class

        Args:
            dimensions (int): Number of dimensions original
        dataset should be reduced to.
            mode (str): Accepts different modes with with the dataset should be
        reduced in the number of features [raw, pca, lda, tsne]
        """
        self.ndims = dimensions
        self.mode = mode
        self.scale = scale

        if scale:
            self.scaler = MinMaxScaler((0, 2 * np.pi))

        if self.mode == "lda":
            self.kmeans = KMeans(n_clusters=self.ndims, random_state=0, n_init=10)
            self.lda = [None] * dimensions
            for i in range(dimensions):
                self.lda[i] = LDA(n_components=1)
        elif self.mode == "pca":
            self.model = PCA(n_components=self.ndims)
        elif self.mode == "tsne":
            self.model = TSNE(
                n_components=self.ndims, perplexity=20, learning_rate=0.01
            )

    def fit_transform(self, features: pd.DataFrame):
        """
        Trains the preprocessor so that the outcome can be
        transformed afterwards.

        Args:
            features (DataFrame): Features DataFrame
        """
        output = features.copy()
        if self.scale:
            output = self.scaler.fit_transform(output)

        if self.mode == "pca" or self.mode == "tsne":
            return self.model.fit_transform(output)
        elif self.mode == "raw":
            return output

        raise NotImplementedError

    def fit(self, features: pd.DataFrame, target: pd.DataFrame):
        """
        Trains the preprocessor so that the outcome can be
        transformed afterwards.

        Args:
            features (DataFrame): Features DataFrame
            target (DataFrame): Target to balance the features
        """
        output = features.copy()
        if self.scale:
            output = self.scaler.fit_transform(output)

        if self.mode == "lda":
            # Calculate the correlation of each feature with the target variable
            correlations = output.corrwith(target)
            correlations.fillna(0, inplace=True)

            # Reshape the correlations into a 2D array
            correlations_reshaped = np.reshape(correlations.values, (-1, 1))

            # Fit the k-means algorithm to the reshaped correlations
            self.kmeans.fit(correlations_reshaped)
            clusters = self.kmeans.labels_

            # Split the features into groups based on the cluster assignments
            groups = [np.where(clusters == i)[0] for i in range(self.ndims)]

            # Apply LDA to each group of features to create a new feature
            for i, group in enumerate(groups):
                self.lda[i].fit(output.iloc[:, group], target)
        elif self.mode == "pca":
            self.model.fit(output)

    def transform(self, features: pd.DataFrame, target: pd.DataFrame = None):
        """
        Transforms the original features into the LDA based dimensions

        Args:
            features (pd.DataFrame): Original feature set to be transformed
            target (pd.DataFrame): Target distribution, required to compute the correlation

        Returns:
            np.array: Numpy array that contains the reduced dataset
        """
        output = features.copy()
        if self.scale:
            output = self.scaler.transform(output)

        if self.mode == "lda":
            if not target:
                raise Exception("You must provide a target when selecting LDA")

            # Calculate the correlation of each feature with the target variable
            correlations = output.corrwith(target)
            correlations.fillna(0, inplace=True)

            # Reshape the correlations into a 2D array
            correlations_reshaped = np.reshape(correlations.values, (-1, 1))

            # Use existing KMeans
            self.kmeans.predict(correlations_reshaped)
            clusters = self.kmeans.labels_

            # Split the features into groups based on the cluster assignments
            groups = [np.where(clusters == i)[0] for i in range(self.ndims)]

            # Apply LDA to each group of features to create a new feature
            features_lda = np.empty((output.shape[0], self.ndims))
            for i, group in enumerate(groups):
                features_lda[:, i] = (
                    self.lda[i].transform(output.iloc[:, group]).ravel()
                )

            return features_lda
        elif self.mode == "pca":
            return self.model.transform(output)
        else:
            return output
