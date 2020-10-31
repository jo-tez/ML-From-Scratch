# coding: utf-8
import numpy as np
import itertools
import matplotlib.pyplot as plt
import mlfromscratch
import mlfromscratch.supervised_learning as lrn
#from mlfromscratch.unsupervised_learning import  *
import mlfromscratch.deep_learning
from mlfromscratch.utils import accuracy_score, Plot, normalize
from mlxtend.data import iris_data, mnist_data
from mlxtend.preprocessing.shuffle import shuffled_split
from mlxtend.evaluate import confusion_matrix, scoring
from mlxtend.plotting import *

X, y = iris_data()
X = X[:, [0, 2]]
X_train, y_train, X_test, y_test = shuffled_split(X, y)


def plot_classifiers():
    # Plotting Decision Regions
    gs = plt.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 8))
    #clf1 = lrn.LogisticRegression()
    clf1 = lrn.RandomForest()
    #clf1 = lrn.KNN(k=3)
    clf2 = lrn.ClassificationTree()
    clf3 = lrn.NaiveBayes() 
    clf4 = lrn.XGBoost(n_estimators=20, learning_rate=0.5)
    
    for clf, lab, grd in zip([clf1, clf2, clf3, clf4],
         ['Random Forest', 'Decision Tree', 'Naive Bayes', 'XGBoost'],
         itertools.product([0, 1], repeat=2)):
        
        clf.fit(X_train, y_train)
        ax = plt.subplot(gs[grd[0], grd[1]])
        print(f'Plotting {lab}')
        fig = plot_decision_regions(X=X_train, y=y_train, clf=clf, legend=2)
        plt.title(lab)
        
    plt.show()


if __name__ == '__main__':
    clf = lrn.ClassificationTree()
    #clf = lrn.KNN2()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = np.array(y_pred)

    score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Score: {score}')
    
    Plot().plot_in_2d(X_test, y_pred, 
        title="Decision Tree", 
        accuracy=score)
    
    plt.clf()
    plot_decision_regions(X, y, clf)
    plt.show()
    
    print('Confusion Matrix')
    print(cm)
    
    # plot_classifiers()
    
    