#!/usr/bin/env pythonfd
# coding:fd utf-8fd

# In[3]:
fd

from sklearn import datasets # iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from random import randint

import random
import os

from copy import copy
from math import ceil

import numpy as np
from matplotlib import pyplot as plt


# # Data loading and its structure

# In[4]:


iris = datasets.load_iris()
print("Dataset loaded into variable:\t", "iris")
print("\nIris keys:\t\t\t", list(iris.keys()) )


# In[5]:


# Data info structure
print("Iris data example:\n\n", iris['data'][0:4], " \n ...")
print("\nIris target example:\t\t", iris['target'][0:4], "...")
print("\nIris target_names example:\t", iris['target_names'][0:4])


# # Environment definition

# In[6]:


print('Our environment:\t\t X, Y, l')
def start_env():
    global X, Y, l
    X = copy(iris['data'])
    Y = copy(iris['target'])
    l = len(iris['data'])
    
def start_test_env(n):
    global X, Y, l
    l = n
    X, Y = X[:l], Y[:l]
    print('Starting test environment:\t\t X, Y,', n)

def testify_fake(c):
    global X
    X =np.arange(l*c).reshape(l, c)
    
def end_test_env():
    start_env()
    print('Ending test environment...')
    
def testify(test, params, what, exp=''):
    print(test, ':\t\t\t\t', params, '\t', what, '\t', '->', exp)

def testify_assert(test, what, exp=''):
    print(test, ':\t\t\t\t', what==exp)
    
start_env()


# # List of models generation

# In[7]:


knn_metrics = ['euclidean', 'manhattan', 'chebyshev']
knn_neightbors = range(1, 10)

models = [KNeighborsClassifier(n_neighbors=n, metric=m) for n in knn_neightbors for m in knn_metrics]
models_info = [['KNeighborsClassifier', n, m] for n in knn_neightbors for m in knn_metrics]


# ### Table of our models list

# In[8]:


print('\tclassifier   neightbors  metric\n')
models_info


# ### Example of learning and prediction

# In[9]:


x = [[5.1, 3.5, 1.4, 0.2]]
print('a( %s ) = ...' % x[0])
models[0].fit(X, Y).predict(x)


# ### For these purpose we need write fit-predict strean

# In[10]:


def fit_predict(X, Y, x):
    global models
    return [model.fit(X, Y).predict(x) for model in models]


# ### And we need validate models 
# It is ability to learn

# In[11]:


# Model validation
x = [[5.1, 3.5, 1.4, 0.2]]
print('Result: we tested %d models of %d' % (len(fit_predict(X, Y, x*3)),  len(models)) )


# # Making list of criterias

# ### For validation spliting into train:test define function

# In[12]:


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

# In[13]:


# Testing
start_test_env(10)
a, b, c = three_d_split(5, 7)
a, b, c = a['X'], b['X'], c['X']
print("three_d_split(5, 7):\t\t\t", len(a), len(b), len(c), (5, 2, 3))
end_test_env()


# ### Shuffle implementation

# In[14]:


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

def combine_transpose(a, b):
    print(a, b)
    return np.array([X, Y]).T

def transpose(a):
    return np.array(a).T
    
def shuffle_XY():
    combined = combine_transpose(X, np.asmatrix(Y).T)
    shuffled = transpose(my_shuffle(combined))
    return tuple(shuffled)

def q_from_size(size):
    return ceil(l/size)

# In[]
# Shuffle test

start_test_env(10)
testify_fake(5)
n = 10
testify("__shuffle_int", n, __shuffle_int(n))
X, Y= shuffle_XY()
print(X)
testify("__shuffle_int", n, X)

end_test_env()

# ### Generator definition

# In[15]:


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
    for generated in generator_qfold(l):
         yield generated
        
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

# In[16]:


# Testing
n = 10
start_test_env(10)
q = 2
t = 4
a, b, c = three_d_split(5, 7)
d = combine(a, b)
a, b, c = a['X'], b['X'], c['X']
testify('three_d_split', (5, 7), (len(a), len(b), len(c)), (5, 2, 3) )

testify('combine', ('a', 'b'), len(d), 7 )

X_real = iris['data'][:n]

testify_assert("env_immutability", np.array_equal(X, X_real), True)
end_test_env()


# In[]

# Generators init

def generators_reinit():
    global generators, generators_info
    _generator_qfold = generator_qfold(q)
    _generator_ccv = generator_ccv()
    _generator_bootstrap = generator_bootstrap(t)
    _generator_loo = generator_loo()
    _generator_txq = generator_txq(t, q)
    _generator_randomcv = generator_randomcv(t)
    generators = [_generator_qfold , _generator_ccv,
                  _generator_bootstrap, _generator_loo,
                  _generator_txq, _generator_randomcv]
    generators_info = [['qfold'], ['CCV'], 
                       ['bootstrap'], ['LOO'], 
                       ['txqfold'], ['random CV'] ]

def print_first(g, n):
    for _i in range(n):
        a = next(g)
        ax, ay = a[0]['X'], a[0]['Y']
        print(ax)
        ax, ay = a[1]['X'], a[1]['Y']
        print(ax, '\n\n')
        
        print('X: ', X, '\n\n\n')
    
generators_reinit()

# In[17]:

# Generators test

X =np.arange(500).reshape(100, 5)
# print(X)

n = 10
start_test_env(10)
t=3
q=4

# Generators
_generator_qfold = generator_qfold(q)
_generator_ccv = generator_ccv()
_generator_bootstrap = generator_bootstrap(t)
_generator_loo = generator_loo()
_generator_txq = generator_txq(t, q)
_generator_randomcv = generator_randomcv(t)

print_first(_generator_txq, n)

testify('gen_ccv', (), len([x for x in _generator_ccv]), 2**(l-2)-2*l)
testify('gen_qfold', (), len([x for x in _generator_qfold]), q)
testify('gen_bootstrap', (), len([x for x in _generator_bootstrap]), t)
testify('gen_loo', (), len([x for x in _generator_loo]), l)
testify('gen_txq', (), len([x for x in _generator_txq]), t*q)
testify('gen_randomcv', (), len([x for x in _generator_randomcv]), )

end_test_env()


# # Learning

# In[ ]:


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
    
    def fit_predict_count(self, generator=''):
        if generator:
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


# In[ ]:


start_env()
method = Method(models[15], generator_qfold(q))

data = []
for m, mi in zip(models, range(len(models))):
    generators_reinit()
    for g, gi in zip(models, range(len(generators))):
        method = Method(m, g)
        data += models_info[mi] + generators_info[mi] + [method.fit_predict_count()]
        

end_test_env()

