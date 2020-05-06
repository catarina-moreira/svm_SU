import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import datasets
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn import svm

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from base import Kernel
from svm_vis import *
from abc import abstractmethod, ABCMeta
from cmath import *

import time
import re

import warnings
warnings.filterwarnings('ignore')

def plot_dataset( X, y, class_val, dataset_name ):
  red  = X[ y == class_val[0]]
  blue = X[ y == class_val[1]]

  # plot figure
  fig=plt.figure(dpi=150, figsize=(5,3))

  plt.scatter(red[:,0], red[:,1], c='r', marker='x', s=10, label=' class ' + str(class_val[0]), cmap='RdBu')
  plt.scatter(blue[:,0], blue[:,1], c='b', marker='o', s=10, label=' class ' + str(class_val[1]), cmap='RdBu')
  plt.ylabel('feature 2', fontsize=8)
  plt.xlabel('feature 1', fontsize=8)
  plt.title(dataset_name, fontsize=12)
  plt.legend()
  plt.show()

def plot_iris_data( X, y, clf ):
  # create a mesh to plot in  
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

  fig=plt.figure(dpi=150, figsize=(5,3))

  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

  # Plot also the training points
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
  plt.xlabel('Sepal length')
  plt.ylabel('Sepal width')
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.xticks(())
  plt.yticks(())
  plt.show()

def plot_decision_boundary( X_train, y_train, clf ):
  xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                      np.linspace(-3, 3, 500))

  # plot the decision function for each datapoint on the grid
  Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # plot the decision function for each datapoint on the grid
  Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  fig=plt.figure(dpi=150, figsize=(5,3))
  
  plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
            origin='lower', cmap=plt.cm.PuOr_r)

  contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='dashed')
  plt.scatter(X_train[:, 0], X_train[:, 1], s=30, c=y_train, cmap=plt.cm.Paired, edgecolors='k')
  plt.xticks(())
  plt.yticks(())
  plt.axis([-5, 5, -5, 5])
  plt.show()


# some useful auxiliary functions for the analysis
def plot_learning_curve(svclassifier, X_train, y_train ):

  fig=plt.figure(dpi=150, figsize=(5,3))

  # plot learning curve
  train_sizes, train_scores, test_scores = learning_curve(svclassifier, 
                                                        X_train, y_train, 
                                                        scoring="neg_mean_squared_error", cv=5,
                                                        train_sizes=np.linspace(0.1, 1, 30))
  plt.plot(train_sizes, -test_scores.mean(1), 'o-', color="g", label="test")
  plt.plot(train_sizes, -train_scores.mean(1), 'o-', color="r", label="train")
  plt.xlabel("Train size")
  plt.ylabel("Mean Squared Error")
  plt.title('Learning curves')
  plt.legend(loc="best")

