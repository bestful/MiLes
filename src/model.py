#!/usr/bin/env python
# coding: utf-8

# In[618]:


from sklearn import datasets # iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from random import randint

import random
import os 

from copy import deepcopy
from math import ceil

import numpy as np
from matplotlib import pyplot as plt


# # Data loading and its structure

# In[13]:


iris = datasets.load_iris()
print("Dataset loaded into variable:\t", "iris")
print("\nIris keys:\t\t\t", list(iris.keys()) )


# In[734]:


# Data info structure

print("Iris data example:\n\n", iris['data'][0:4], " \n ...")
print("\nIris target example:\t\t", iris['target'][0:4], "...")
print("\nIris target_names example:\t", iris['target_names'][0:4])

 
# # Environment definition

# In[116]: 


print('Our environment:\t\t X, Y, l')
def start_env():
    global X, Y, l
    X = iris['data']
    Y = iris['target']
    l = len(iris['target_names'])
    
def start_test_env(n):
    global X, Y, l
    l = n
    X, Y = X[:l], Y[:l]
    print('Starting test environment:\t\t X, Y,', n)
    
def end_test_env():
    start_env()
    print('Ending test environment...')
    
def testify(test, params, what, exp=''):
    print(test, ':\t\t\t\t', params, '\t', what, '\t', '->', exp)
    
start_env()


# # List of models generation

# In[15]:


knn_metrics = ['euclidean', 'manhattan', 'chebyshev']
knn_neightbors = range(1, 10)

models = [KNeighborsClassifier(n_neighbors=n, metric=m) for n in knn_neightbors for m in knn_metrics]
models_info = [['KNeighborsClassifier', n, m] for n in knn_neightbors for m in knn_metrics]


# ### Table of our models list

# In[16]:


print('\tclassifier   neightbors  metric\n')
models_info


# ### Example of learning and prediction

# In[17]:


x = [[5.1, 3.5, 1.4, 0.2]]
print('a( %s ) = ...' % x[0])
models[0].fit(X, Y).predict(x)


# ### For these purpose we need write fit-predict strean

# In[18]:


def fit_predict(X, Y, x):
    global models
    return [model.fit(X, Y).predict(x) for model in models]


# ### And we need validate models 
# It is ability to learn

# In[19]:


# Model validation
x = [[5.1, 3.5, 1.4, 0.2]]
print('Result: we tested %d models of %d' % (len(fit_predict(X, Y, x*3)),  len(models)) )


# # Making list of criterias

# ### For validation spliting into train:test define function

# In[132]: