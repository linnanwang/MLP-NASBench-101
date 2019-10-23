import numpy as np
# import matplotlib.pyplot as plt
import json
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from scipy.stats import norm
from scipy.optimize import minimize
import random
import time
import os
import itertools
import operator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
from torch import optim
import numpy as np



class LinearModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_( self.fc1.weight )
    
    def forward(self, x):
        y = self.fc1(x)
        return y
        
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)
        
        torch.nn.init.xavier_uniform_( self.fc1.weight )
        torch.nn.init.xavier_uniform_( self.fc2.weight )        
    
    def forward(self, x):
        x1 = self.fc1(x)
        y  = self.fc2(x1)
        return y


####preprocess data####
dataset = []
with open('nasbench_dataset', 'r') as infile:
    dataset = json.loads( infile.read() )

samples = {}
for data in dataset:
    samples[json.dumps(data["feature"])] = data["acc"]

BEST_ACC   = 0
BEST_ARCH  = None
CURT_BEST  = 0
BEST_TRACE = {}
for i in dataset:
    arch = i['feature']
    acc  = i['acc']
    if acc > BEST_ACC:
        BEST_ACC  = acc
        BEST_ARCH = json.dumps( arch )
print("##target acc:", BEST_ACC)
#######################

# bounds = np.array([[-1.0, 2.0]])
noise = 0.2
#
#
# def f(X, noise=noise):
#     return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)
#
# X_init = np.array([[-0.9], [1.1]])
# Y_init = f(X_init)
#
# X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
# Y = f(X,0)
#

def propose_location(predictor, X_samples, Y_samples, samples):
    ''' Proposes the next sampling point by optimizing the acquisition function. 
    Args: acquisition: Acquisition function. X_sample: Sample locations (n x d). 
    Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. 
    Returns: Location of the acquisition function maximum. '''
    dim = X_sample.shape[1]
    networks = []
    for network in samples.keys():
        networks.append( json.loads(network) )
    X    = np.array( networks )
    X    = torch.from_numpy( np.asarray(X, dtype=np.float32).reshape(X.shape[0], X.shape[1]) )
    Y    = predictor.forward( X )
    Y    = Y.data.numpy()
    Y    = Y.reshape(len(samples) )
    X    = X.data.numpy()
    proposed_networks = []
    n    = 50
    if Y.shape[0] < n:
        n = Y.shape[0]
    indices = np.argsort(Y)[-n:]
    print("indices:", indices.shape)
    #print( X[indices] )
    proposed_networks = X[indices] 
    
    # for i in range(0, n):
    #     idx       = np.argmax( Y )
    #     best_arch = X[idx]
    #     Y    = np.delete(Y, idx, axis = 0 )
    #     X    = np.delete(X, idx, axis = 0 )
    #     proposed_networks.append( best_arch )
    return proposed_networks


# Gaussian process with Matern kernel as surrogate model

init_samples = random.sample(samples.keys(), 100)
X_sample = None
Y_sample = None
for sample in init_samples:
    if X_sample is None or Y_sample is None:
        X_sample = np.array( json.loads(sample) )
        Y_sample = np.array( samples[ sample ]  )
    else:
        X_sample = np.vstack([X_sample, json.loads(sample) ] )
        Y_sample = np.vstack([Y_sample, samples[ sample ] ] )
    del samples[sample]

# Initialize samples
#
# Number of iterations
n_iter = 1000000000000
#
# plt.figure(figsize=(12, n_iter * 3))
# plt.subplots_adjust(hspace=0.4)
#
predictor  = LinearModel(49, 1)
optimiser  = optim.Adam(predictor.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)

window_size = 100
sample_counter = 0
for i in range(n_iter):
    print("####################iter:", i)
    print("x_sample:", X_sample.shape )
    print("y_sample:", Y_sample.shape )
    print("dataset:",  len(samples) )
    chunks = int( X_sample.shape[0] / window_size )
    if  X_sample.shape[0] % window_size > 0:
        chunks += 1
    
    optimiser.zero_grad()

    for epoch in range(0, 100):
        X_sample_split = np.array_split(X_sample, chunks)
        Y_sample_split = np.array_split(Y_sample, chunks)
        #print("epoch=", epoch)
        for i in range(0, chunks):
            inputs = torch.from_numpy( np.asarray(X_sample_split[i], dtype=np.float32).reshape(X_sample_split[i].shape[0], X_sample_split[i].shape[1]) )
            #print(inputs.shape, X_sample_split[i].shape, np.asarray(X_sample_split[i], dtype=np.float32).shape )
            outputs = predictor.forward( inputs )
            loss = nn.MSELoss()(outputs, torch.from_numpy( np.asarray(Y_sample_split[i], dtype=np.float32) ).reshape(-1, 1)  )
            loss.backward()# back props
            nn.utils.clip_grad_norm_(predictor.parameters(), 5)
            optimiser.step()# update the parameters

#     # Obtain next sampling point from the acquisition function (expected_improvement)
    X_next = propose_location(predictor, X_sample, Y_sample, samples)
#     # Obtain next noisy sample from the objective function
    for network in X_next:
        X_sample = np.vstack([X_sample, network] )
    for network in X_next:
        sample_counter += 1
        acc = samples[ json.dumps( network.tolist() ) ]
        if acc > CURT_BEST:
            BEST_TRACE[json.dumps( network.tolist() ) ] = [acc, sample_counter]
            CURT_BEST = acc
        if acc == BEST_ACC:
            sorted_best_traces = sorted(BEST_TRACE.items(), key=operator.itemgetter(1))
            for item in sorted_best_traces:
                print(item[0],"==>", item[1])
            final_results = []
            for item in sorted_best_traces:
                final_results.append( item[1] )
            final_results_str = json.dumps(final_results)
            with open("result.txt", "a") as f:
                f.write(final_results_str + '\n')
            print("$$$$$$$$$$$$$$$$$$$CONGRATUGLATIONS$$$$$$$$$$$$$$$$$$$")
            os._exit(1)

        print(network, acc)
        del samples[ json.dumps( network.tolist() ) ]
        Y_sample = np.vstack([Y_sample, acc] )
     
    
     
     
     
     
     
     
     