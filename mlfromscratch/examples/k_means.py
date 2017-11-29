from __future__ import division, print_function
#from sklearn import datasets
from mlxtend.data import three_blobs_data
import numpy as np

from mlfromscratch.unsupervised_learning import KMeans
from mlfromscratch.utils import Plot


if __name__ == "__main__":
    # Load the dataset
    X, y = three_blobs_data()

    # Cluster the data using K-Means
    clf = KMeans(k=3)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    p = Plot()
    p.plot_in_2d(X, y_pred, title="K-Means Clustering")
    p.plot_in_2d(X, y, title="Actual Clustering")




