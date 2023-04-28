""" 
This module contains utilities for classifying events with the Blob method.
"""
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from typing import Tuple
import logging
from deeplar import PACKAGE
from typing import Union
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(PACKAGE + ".blob")


def distance_matrix(point_cloud: np.ndarray, setup: dict, euclid: bool) -> np.ndarray:
    """Computing the distances between each pair of hits in a sample.

    Parameters
    ----------
    point_cloud: np.ndarray
        Collection of hits [Xpos, Ypos, Zpos, Energy] corresponding to a single track.
    setup: dict
        Settings dictionary.
    euclid: bool:
        Whether to return euclidean distance or a normalized one.

    Returns:
    --------
    distance: np.ndarray
        The distance matrix.
    """
    if not euclid:
        binwidths = np.array(setup["detector"]["resolution"])
        point_cloud[:, :3] = point_cloud[:, :3] / binwidths
        distance = (cdist(point_cloud, point_cloud) <= np.sqrt(3)) * 1
        distance = distance - np.eye(distance.shape[0]) * np.diag(distance)
    else:
        distance = cdist(point_cloud, point_cloud)
    return distance


def find_main_trajectory(graph: nx.classes.graph.Graph) -> Tuple[int, int]:
    """Computes the shortest path length between each pair of hits and returns the farthest pair indexes (the blobs index).

    Parameters:
    -----------
    graph: nx.classes.graph.Graph
        Graph representation of a single track.

    Returns:
    --------
    blob_1_id: int
        First blob index.
    blob_2_id: int
        Second blob index.
    """
    n_nodes = graph.number_of_nodes()
    max_path_length = 0
    blob_1_id = -1
    blob_2_id = -1
    for i in range(n_nodes):
        for j in range(n_nodes):
            try:
                path_length = nx.shortest_path_length(graph, source=i, target=j)
            except:
                path_length = -1
            if path_length > max_path_length:
                max_path_length = path_length
                blob_1_id = i
                blob_2_id = j
    return blob_1_id, blob_2_id


def get_blob_energies(
    point_clouds: np.ndarray, setup: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Identifying track-endpoints as blobs and computing their energy.

    Parameters:
    -----------
    point_clouds: np.ndarray
        Collection of hits [Xpos, Ypos, Zpos, Energy] gouped into tracks.
    setup: dict
        Settings dictionary.

    Returns:
    --------
    en_1: np.ndarray
        Highest-energy blob array.
    en_2: np.ndarray
        Lowest-energy blob array.
    """
    n = point_clouds.shape[0]
    en_1 = np.zeros(n)
    en_2 = np.zeros(n)
    radius = 2  # blob radius [mm]
    for i in range(n):
        sample = point_clouds[i]
        edges = distance_matrix(sample, setup, euclid=False)
        distances = distance_matrix(sample, setup, euclid=True)
        graph = nx.from_numpy_matrix(
            edges
        )  # Graphs can be visualized with nx.draw(G, with_labels = True)
        blob1, blob2 = find_main_trajectory(graph)
        en_blob_1 = np.sum(sample[distances[blob1] < radius, -1])
        en_blob_2 = np.sum(sample[distances[blob2] < radius, -1])
        if en_blob_1 > en_blob_2:
            en_1[i] = en_blob_1
            en_2[i] = en_blob_2
        else:
            en_1[i] = en_blob_2
            en_2[i] = en_blob_1
    return en_1, en_2


class NPModel:
    """Class defining a classification model based on Neyman-Pearson's lemma."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """Parameters:
        --------------
        features: np.ndarray
            Dataset for estimating the underlying pdf.
        labels: np.ndarray
            Truth values.
        """
        self.bins = (75, 75)
        self.range = [[0, 2.5], [0, 2.5]]
        hist1 = np.histogram2d(
            features[labels == 0, 0],
            features[labels == 0, 1],
            bins=self.bins,
            range=self.range,
        )
        hist2 = np.histogram2d(
            features[labels == 1, 0],
            features[labels == 1, 1],
            bins=self.bins,
            range=self.range,
        )
        self.pdf1 = hist1[0] / hist1[0].sum() + 1e-13
        self.pdf2 = hist2[0] / hist2[0].sum() + 1e-13
        # np.save("features.npy", features)
        # np.save("labels.npy", labels)

    def q_evaluate(self, features: np.ndarray) -> np.ndarray:
        """Assigning bins to input features and returning a normalized q-distribution.

        Parameters:
        -----------
        features: np.ndarray
            Dataset for evaluating the q-statistics.

        Returns:
        --------
        q_dist: np.ndarray
            q-distribution.
        """
        idx_1 = (
            ((features[:, 0] - self.range[0][0]) * self.bins[0] / self.range[0][1])
        ).astype(int)
        idx_2 = (
            ((features[:, 1] - self.range[1][0]) * self.bins[1] / self.range[1][1])
        ).astype(int)
        q_dist = np.log(self.pdf1[idx_1, idx_2]) / np.log(self.pdf2[idx_1, idx_2])
        q_dist = q_dist / q_dist.max()
        return q_dist

    def q_train(self, features: np.ndarray, labels: np.ndarray):
        """Getting the best threshold in the q-space.

        Parameters:
        -----------
        features: np.ndarray
            Training distribution.
        labels: np.ndarray
            Truth labels.
        """
        q_distr = self.q_evaluate(features)
        scan_density = 100
        span = np.linspace(0, 1, scan_density)
        accuracy = np.zeros(scan_density)
        for i, threshold in enumerate(span):
            prediction = q_distr > threshold
            accuracy[i] = np.sum(labels == prediction) / len(q_distr)
        self.accuracy_train = np.max(accuracy)
        self.threshold = np.argmax(accuracy) / scan_density

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Classifying data.

        Parameters:
        -----------
        features: np.ndarrray
            Dataset to predict.

        Returns:
        --------
        predictions: np.ndarray
            Class predictions:
        """
        if not hasattr(self, "threshold"):
            logger.info("Cannot predict without previous training.")
        else:
            q_dist = self.q_evaluate(features)
            predictions = (q_dist > self.threshold) * 1
            return predictions

    def score(self, features: np.ndarray, labels: np.ndarray) -> np.float64:
        """Returns prediction accuracy.

        Paramters:
        ----------
        features: np.ndarray
            Dataset to predict.
        labels: np.ndarray
            Truth labels.

        Returns:
        --------
        accuracy: np.float64
            Prediction accuracy.
        """
        predictions = self.predict(features)
        accuracy = np.sum(predictions == labels) / labels.shape[0]
        return accuracy


def svm_model(train_features: np.ndarray, train_labels: np.ndarray) -> SVC:
    """Returning a SVM model for the blob method.

    Parameters:
    -----------
    train_features: np.ndarray
        The training dataset.
    train_labels: np.ndarray
        The truth labels.

    Returns:
    --------
    model: SVC
        The classification model.

    """
    hyp_search = {"C": [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(SVC(probability=True), hyp_search, refit=True, verbose=2, cv=3)
    grid.fit(train_features, train_labels)
    logger.info(grid.best_params_)
    model = SVC(probability=True, **grid.best_params_)
    model.fit(train_features, train_labels)
    return model


def neyman_pearson_model(
    train_features: np.ndarray, train_labels: np.ndarray
) -> NPModel:
    """Returning a classification model for the blob method based on Neyman-Pearson's lemma.

    Parameters:
    -----------
    train_features: np.ndarray
        The training dataset.
    train_labels: np.ndarray
        The truth labels.

    Returns:
    --------
    model: NPModel
        The classification model.
    """
    model = NPModel(train_features, train_labels)
    model.q_train(train_features, train_labels)
    return model


def train_blobs(
    train_features: np.ndarray, train_labels: np.ndarray, inference_model: str
) -> Union[NPModel, SVC]:
    """Returning a clasification model for the blob method.

    Parameters:
    -----------
    train_features: np.ndarray
        The training dataset.
    train_labels: np.ndarray
        The truth labels.
    inference_model: str
        Whether to train a Neyman Pearson model or a SVM

    Returns:
    --------
    model: Union[NPModel, SVC]
        The classification model.
    """
    if inference_model == "neyman_pearson":
        logger.info("Predicting classes with Neyman-Pearson lemma")
        model = neyman_pearson_model(train_features, train_labels)
    elif inference_model == "svm":
        logger.info("Predicting classes with SVM")
        model = svm_model(train_features, train_labels)
    else:
        logger.error("Avaliable inference models for blobs are neyman-pearson or SVM.")
    return model
