from __future__ import division

from scipy.stats import truncnorm
from scipy import integrate

import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns

np.set_printoptions(threshold=np.nan)

def sample(dists):
    params = []
    tnorm_l1 = truncnorm(a=dists[0][0], b=dists[0][1], loc=dists[0][2], scale=dists[0][3]**.5)
    tnorm_l2 = truncnorm(a=dists[1][0], b=dists[1][1], loc=dists[1][2], scale=dists[1][3]**.5)
    for i in range(len(dists)):
        distribution = dists[i]
        if len(distribution) == 2:
            left, right = distribution
            param = random.uniform(left, right)
        elif len(distribution) == 4:
            if i==0:
                param = tnorm_l1.rvs(size=1)[0]
            if i==1:
                param = tnorm_l2.rvs(size=1)[0]
        params.append(param)
    return params

def dX_dt(X, t, *params):
    l1, l2, xi1, xi2, mu3, k1, k2 = params
    x1 = l1*X[0]*(1.0-X[0]/k1) - xi1*X[0]
    x2 = l2*X[1]*(1.0-X[1]/k2) - xi2*X[1] + xi1*X[0]
    x3 = -mu3*X[2] + xi2*X[1]
    return np.array([x1, x2, x3])

def simulate_data(params, domain):
    X0 = np.array([100000.0, 0.0, 0.0])
    X, infodict = integrate.odeint(dX_dt, X0, domain, args=tuple(params), full_output=True)
    x1, x2, x3 = X.T
    return np.hstack((x1, x2, x3))

def distance(observed, simulated):
    accum = 0.0
    for i in range(len(observed)):
        diff = observed[i] - simulated[i]
        accum += (diff**2) / (observed[i]**2)
    return accum

def query_params(query, param_sets, i=0):
    if query == []:
        return param_sets
    
    filtered_param_sets = []
    left, right = query[0]
    for params in param_sets:
        if params[i] <= right and params[i] >= left:
            filtered_param_sets.append(params)
    return query_params(query[1:], filtered_param_sets, i+1)

def query_prob(query, param_sets):
    return len(query_params(query, param_sets)) / len(param_sets)

def inference(observed_data, dists, domain, threshold, num_samples):
    accepted_param_sets = []
    argmin = 0
    minval = threshold + 1
    for _ in range(num_samples):
        params = sample(dists)
        simulated_data = simulate_data(params, domain)
        closeness = distance(observed_data, simulated_data)
        if closeness < threshold:
            accepted_param_sets.append(params)
            if closeness < minval:
                argmin = params
                minval = closeness
    return accepted_param_sets, argmin

def vis_data(param_sets):
    data = np.array(param_sets)
    print data[:, [1, 2]]
    print len(data[:, [1, 2]])
    data = pd.DataFrame(data[:, [1, 2]], columns=["X", "Y"])
    sns.kdeplot(data.X, data.Y, shade=True)
    mpl.pyplot.show()

def infer_from_experiment():
    fname = "data/spleen_data_partial" #str(input("Please enter the complete file name of the observed data (please refer to readme for required format of input data): "))
    observed_data = []
    dists = []
    domain = []
    threshold = 30 #float(input("Please enter the error threshold (type 30 to use default): "))
    num_samples = 10000 #int(input("Please enter the number of samples (type 10000 to use default): "))

    param_names = ['lambda1', 'lambda2', 'xi1', 'xi2', 'mu3', 'kappa1', 'kappa2']
    """
    for name in param_names:
        left = input("Please enter the left boundary of parameter " + name + ": ")
        right = input("Please enter the right boundary of parameter " + name + ": ")
        #initial = float(input("Please enter an initial guess (or 0, if no guess) of parameter " + name + ": "))
        dists.append([left, right])
        #dists.append([left, right, initial])
    """
    dists = [[.0001, 14, .15, 1], [.0001, 14, .45, 1], [0, 4], [0, 4], [0, .1], [0, 10**7], [0, 10**7]]

    l1_l, l1_r = dists[0][0], dists[0][1]
    l2_l, l2_r = dists[1][0], dists[1][1]
    xi1_l, xi1_r = dists[2]
    xi2_l, x21_r = dists[3]

    if xi1_l >= l1_l:
        print "xi1's left boundary needs to be earlier than lambda1's left boundary"
    if xi2_l >= l2_l:
        print "xi2's left boundary needs to be earlier than lambda2's left boundary"

    fobj = open(fname)
    x1_s, x2_s, x3_s = [], [], []
    for line in iter(fobj.readline, ''):
        line = line.split(' ')
        t, x1, x2, x3 = map(float, line)
        domain.append(t)
        x1_s.append(x1)
        x2_s.append(x2)
        x3_s.append(x3)
    
    observed_data = x1_s + x2_s + x3_s
    param_sets, argmin = inference(observed_data, dists, domain, threshold, num_samples)

    vis_data(param_sets)

if __name__ == '__main__':
    infer_from_experiment()
