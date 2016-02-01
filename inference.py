from __future__ import division

from scipy.stats import truncnorm
from scipy import integrate

import random
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns

np.set_printoptions(threshold=np.nan)

def sample(dists):
    """
    Samples parameters from list of distributions specified by dists
    dists -- list of 2 or 4 element arrays
        2 elements signify left and right bounds of a uniform distribution
        4 elements signify left, right, mean, and variance of a truncated normal distribution
    Order of parameters: lambda1, lambda2, xi1, xi2, mu3, kappa1, kappa2
    """
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
    """
    Computes derivative of population over time given population quantities X and parameters params.
    Used by ode integrator.
    """
    l1, l2, xi1, delt, mu3, k1, k2, al, bet = params
    mem, intr, eff, ant = X

    dx1_dt = l1 * mem * (1.0 - mem / k1) - xi1 * mem
    dx2_dt = l2 * intr * (1.0 - intr / k2) - delt * ant * intr + xi1 * mem
    dx3_dt = -mu3 * eff + delt * ant * intr
    da_dt = al * ant - bet * ant * eff
    return np.array([dx1_dt, dx2_dt, dx3_dt, da_dt])

def simulate_data(params, domain):
    """
    Uses initial conditions and dX_dt method to numerically solve for x1, x2, x3, and a as functions of t.
    Times for which X is computed is specified by domain, which is a list of values.
    params is a list of each parameter's value.
    Returns a list of x1's quantities over time and then repeats for x2 and 3 (all in one row).
    """
    X0 = np.array([100000.0, 0.0, 0.0, 14622.412]) #initial conditions
    X, infodict = integrate.odeint(dX_dt, X0, domain, args=tuple(params), mxstep=1000, full_output=True)
    x1, x2, x3, a = X.T
    return np.hstack((x1, x2, x3))

def distance(observed, simulated):
    """
    Computes pairwise similarity between observed and simulated data sets.
    Observed and simulated are each one row with columns 0 to size(domain)-1 corresponding to x1's population over time. Row then repeats for x2 and x3.
    """
    accum = 0.0
    for i in range(len(observed)):
        diff = observed[i] - simulated[i]
        accum += (diff**2) / (observed[i]**2)
    return accum

def query_params(query, param_sets, i=0):
    """
    Returns all sampled instances of parameters in which each parameter falls within the corresponding range specified in query, given the observed data.
    query -- list of 2 element arrays
        Each 2 element array corresponds to a different parameter. First and second elements correspond to the query region for that parameter.
    param_sets -- List of lists. Inner list contains one sampled value from each parameter. Outer list contains every multi-parameter sample that was accepted.
    i -- don't pass in anything!
    """
    if query == []:
        return param_sets
    
    filtered_param_sets = []
    left, right = query[0]
    for params in param_sets:
        if params[i] <= right and params[i] >= left:
            filtered_param_sets.append(params)
    return query_params(query[1:], filtered_param_sets, i+1)

def query_prob(query, param_sets):
    """
    Computes probability that each parameter lambda1, lambda2, etc is within the corresponding range in query, given the observed data.
    query -- list of 2 element arrays
        Each 2 element array corresponds to a different parameter. First and second elements correspond to the query region for that parameter.
    param_sets -- List of lists. Inner list contains one sampled value from each parameter. Outer list contains every multi-parameter sample that was accepted.
    """
    return len(query_params(query, param_sets)) / len(param_sets)

def inference(observed_data, dists, domain, threshold, num_samples):
    """
    Takes in observed data, priors, a time domain, a similarity threshold, and number of times to sample and then
    returns the probability distribution in the form of a list of lists. Inner list contains one sampled value from each parameter.
    Outer list contains every multi-parameter sample that was accepted.

    In a machine learning framework, this is step one (training phase). The observed_data is the training data and the threshold is a hyperparameter.
    """
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
    """
    Visualizes probability distribution
    """
    data = np.array(param_sets)
    print len(data[:, [0, 1]])
    data = pd.DataFrame(data[:, [0, 1]], columns=["X", "Y"])
    sns.kdeplot(data.X, data.Y, shade=True)
    mpl.pyplot.show()

def posteriors_to_data(posts, domain, num_samples):
    """
    Samples from posterior distributions given evidence (trained parameter distributions) and averages over num_samples time courses of each cell's population.
    Results in an average population vs time course.
    """
    data = np.array([0.0 for _ in range(len(domain)*3)])
    for _ in range(num_samples):
        params = sample(posts)
        simulated_data = simulate_data(params, domain)
        data += simulated_data
    return data / num_samples

def recreate_posts():
    """
    Creates fake distributions and simulates a time course. Then, use this data as observed data and run the inference to infer the parameter distributions.
    The idea is to get the trained/inferred distributions to be similar to the ones we started with.

    In a machine learning framework, the first half of this method, which involves using a set of parameter distributions to predict a time course, can be used
    for validation and hyperparameter tuning. We can predict a time course from a set of initial conditions and sampled parameters (from a parameter distribution derived from training data)
    and then evaluate how well we predicted an experimental data set that had the same initial conditions. This experimental data then becomes our validation data set and our validation
    error is a similarity metric between the predicted time course and actual experiment. It might be worth exploring to use the validation and training accuracies
    to tune the threshold hyperparameter in the inference method.

    Currently, the code cannot recreate the posteriors. I think we need more information about the system. There is too much variance in the distributions.
    It might be a good idea to create very tight posterior distributions and then experiment with different priors. That way, the data generated has low error, which allows
    us to tweak the priors and see how our prediction capabilities change.
    """
    E = 525670.444
    rho = .152209
    beta = math.exp(math.log(rho)-math.log(E))
    posts = [[.0001, 14, .10, .5], [.0001, 14, .49, .5], [.5, 4], [0, .1], [1, 2], [175000, 180000], [175000, 180000], [.01, 3], [0, beta+1]]
    domain = np.linspace(0, 140, 140)
    num_samples = 600
    data = posteriors_to_data(posts, domain, num_samples)

    priors = [[0, 14, 0, 1], [0, 14, 0, 1], [0, 5], [0, 1], [0, 3], [0, 10**7], [0, 10**7], [0, 3], [0, 3]]
    threshold = 30.0
    num_samples_2 = 3000
    param_sets, argmin = inference(data, priors, domain, threshold, num_samples_2)

    vis_data(param_sets)

def infer_from_experiment():
    """
    Provides front to end, fully independent framework to infer parameters from experimentally observed data.

    In a machine learning framework, this is step one (training phase).
    """
    fname = "data/spleen_data" #str(input("Please enter the complete file name of the observed data (please refer to readme for required format of input data): "))
    observed_data = []
    dists = []
    domain = []
    threshold = 30.0 #float(input("Please enter the error threshold (type 30 to use default): "))
    num_samples = 3000 #int(input("Please enter the number of samples (type 10000 to use default): "))

    param_names = ['lambda1', 'lambda2', 'xi1', 'delta', 'mu3', 'kappa1', 'kappa2', 'alpha', 'beta']
    """
    for name in param_names:
        left = input("Please enter the left boundary of parameter " + name + ": ")
        right = input("Please enter the right boundary of parameter " + name + ": ")
        #initial = float(input("Please enter an initial guess (or 0, if no guess) of parameter " + name + ": "))
        dists.append([left, right])
        #dists.append([left, right, initial])
    """
    E = 5.25670444
    rho = .152209
    beta = math.exp(math.log(rho)-math.log(E))
    print beta
    dists = [[.0001, 14, .15, 1], [.0001, 14, .4, 1], [0, 4], [0, 1.5], [0, .1], [10**4, 10**7], [10**4, 10**7], [0.01, 3], [0, beta+1]]

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
    #recreate_posts()
