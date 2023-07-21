# Some distance measures
import numpy as np


def jaccard_distance(x, y):
    """
    Jaccard distance between two sets
    :param x: set
    :param y: set
    :return: Jaccard distance
    """
    x = set(x)
    y = set(y)
    return 1 - len(x.intersection(y)) / len(x.union(y))

def euclid_distance(x, y):
    """
    Euclidean distance between two vectors
    :param x: vector
    :param y: vector
    :return: Euclidean distance
    """
    return np.linalg.norm(np.array(x) - np.array(y))


def cosine_distance(x, y):
    """
    Cosine distance between two vectors
    :param x: vector
    :param y: vector
    :return: Cosine distance
    """
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def manhattan_distance(x, y):
    """
    Manhattan distance between two vectors
    :param x: vector
    :param y: vector
    :return: Manhattan distance
    """
    return np.sum(np.abs(np.array(x) - np.array(y)))