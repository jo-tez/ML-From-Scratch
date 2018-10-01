from __future__ import print_function
#from sklearn import datasets
import numpy as np

from mlfromscratch.supervised_learning import MultiClassLDA
from mlfromscratch.utils import normalize
from mlxtend.data import iris_data

def main():
    # Load the dataset
    #data = datasets.load_iris()
    X, y = iris_data()
    X = normalize(X)

    # Project the data onto the 2 primary components
    multi_class_lda = MultiClassLDA()
    multi_class_lda.plot_in_2d(X, y, title="LDA")

if __name__ == "__main__":
    main()
