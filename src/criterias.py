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


def three_d_split(l, r):
    """
    Split to tuple of 3 lists : [0, l) [l,r) [r,n)
    """
    a = {'X': X[0:l],  'Y': Y[0:l]}
    b = {'X': X[l:r],  'Y': Y[l:r]}
    c = {'X': X[r:],     'Y': Y[r:]}
    return a, b, c

def combine(a, c):
    ax, cx = a['X'], c['X']
    ay, cy = a['Y'], c['Y']
    
    combined = {'X': np.concatenate((ax, cx), axis=0), 'Y': np.concatenate((ay, cy), axis=0)}
    return combined


# ### Testing of 3d split

# In[117]:


# Testing
start_test_env(10)
a, b, c = three_d_split(5, 7)
a, b, c = a['X'], b['X'], c['X']
print("three_d_split(5, 7):\t\t\t", len(a), len(b), len(c), (5, 2, 3))
end_test_env()


# ### Shuffle implementation

# In[501]:


# -- Pasted
def __shuffle_int(n):
    lst = [x for x in range(n)]
    for _it in range(2*n):
        i, j = randint(0, n-1), randint(0, n-1)
        lst[i], lst[j] = lst[j], lst[i]
    return lst

def my_shuffle(lst):
    size = len(lst)
    if size == 0 or size==1:
        return lst

    indexes = __shuffle_int(size)
    return [lst[i] for i in indexes]

    
def shuffle_XY():
    combined = list(zip(X, Y))
    shuffled = my_shuffle(combined)
    X[:], Y[:] = zip(*shuffled)
    return X, Y

def q_from_size(size):
    return ceil(l/size)


# ### Generator definition

# In[605]:


# Bosses
def generator_qfold(q):
    size = int(l/q)
    # q_it in [1, q]
    for q_it in range(1,q+1):
        a, b, c = three_d_split(q_it*size, max(q_it*(size+1), l) )
        yield combine(a, c), b

def generator_bootstrap(t):
    for _t in range(t):
        size = randint(1, l-1)
        index = randint(0, l-size)
        
        X, Y = shuffle_XY()
        a, b, c = three_d_split(index, index+size)
        yield combine(a, c), b
            
# Workes;)

def generator_ccv():
    for q in range(1, l):
        for generated in generator_qfold(q):
            yield generated
            
def generator_loo():
    return generator_qfold(l)
        
def generator_txq(t, q):
    for _t in range(t):
        X, Y = shuffle_XY()
        for generated in generator_qfold(q):
            yield generated
        
def generator_randomcv(t):
    for _t in range(t):        
        size = randint(1, l-1)
        q = int(l/size)
        for generated in generator_qfold(q):
            yield generated


# ### Testing

# In[579]:


# Testing
start_test_env(10)
q = 2
t = 4
a, b, c = three_d_split(5, 7)
a, b, c = a['X'], b['X'], c['X']
testify('three_d_split', (5, 7), (len(a), len(b), len(c)), (5, 2, 3) )
end_test_env()


# In[613]:


# X =np.arange(500).reshape(100, 5)
# print(X)
start_test_env(10)
t=3
q=4

_generator_qfold = generator_qfold(q)
_generator_ccv = generator_ccv()
_generator_bootstrap = generator_bootstrap(t)
_generator_loo = generator_loo()
_generator_txq = generator_txq(t, q)
_generator_randomcv = generator_randomcv(t)


def print_first(g, n):
    for _i in range(n):
        a = next(g)
        ax, ay = a[0]['X'], a[0]['Y']
        print(ax)
        ax, ay = a[1]['X'], a[1]['Y']
        print(ax, '\n\n')

# print_first(generator_txq(t, q), 12)

testify('gen_ccv', (), len([x for x in _generator_ccv]), 2**(l-2)-2*l)
testify('gen_qfold', (), len([x for x in _generator_qfold]), q)
testify('gen_bootstrap', (), len([x for x in _generator_bootstrap]), t)
testify('gen_loo', (), len([x for x in _generator_loo]), l)
testify('gen_txq', (), len([x for x in _generator_txq]), t*q)
testify('gen_randomcv', (), len([x for x in _generator_randomcv]), )

_generators = [generator_qfold(q), generator_ccv(), generator_bootstrap(t), generator_loo(), generator_txq(t, q), generator_randomcv(t)]
end_test_env()


# # Learning

# In[735]:


class Method:
    def __init__(self, model, generator=''):
        self.model = model
        if generator:
            self.generator = generator
        
    def fit_predict(self):
        for generated in self.generator:
            g = generated
            ax, ay = g[0]['X'], g[0]['Y']
            bx, by = g[1]['X'], g[1]['Y']
            yield self.__fit_predict(self.model, ax, ay, bx), by
    
    def fit_predict_count(self, generator):
        self.generator = generator
        res = 0.0
        times = 0
        for _y, y in self.fit_predict():
            sum = 0
            num = 0
            for _el, el in zip(_y, y):
                if(_el == el):
                    sum += 1
                num += 1
            sum /= num
            times += 1
            res += sum
        sum /= times
        return sum
    
    @staticmethod
    def __fit_predict(model, X, Y, x):
        _model = deepcopy(model)
        return _model.fit(X, Y).predict(x)


# In[737]:


start_env()
print(X)
method = Method(models[15], generator_qfold(q))
print(method.fit_predict_count(generator_qfold(q)))

end_test_env()
