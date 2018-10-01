from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
# Import helper functions
from mlfromscratch.supervised_learning import ElasticNet
from mlfromscratch.utils import k_fold_cross_validation_sets, normalize, mean_squared_error
from mlfromscratch.utils import train_test_split, polynomial_features, Plot
from mlxtend.preprocessing.shuffle import shuffled_split


if __name__ == "__main__":

    # Load temperature data
    DF = '../data/TempLinkoping2016.txt'
    data = np.genfromtxt(DF, delimiter='\t', names=True)

    time = np.atleast_2d(data['time']).T
    temp = data['temp']
    #temp = np.atleast_2d(data['temp'])

    X = time # fraction of the year [0, 1]
    y = temp

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_train, y_train, X_test, y_test = shuffled_split(X, y)

    poly_degree = 13

    model = ElasticNet(degree=15, 
                        reg_factor=0.01,
                        l1_ratio=0.7,
                        learning_rate=0.001,
                        n_iterations=4000)
    model.fit(X_train, y_train)

    # Training error plot
    n = len(model.training_errors)
    training, = plt.plot(range(n), model.training_errors, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean squared error: %s (given by reg. factor: %s)" % (mse, 0.05))

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap()

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Elastic Net")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()


