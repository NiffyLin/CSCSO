import multiprocessing as mp
from functions import *
import numpy as np
import time
from scipy.stats import norm
from scipy.stats import qmc



d = 2

lb = 0 * np.ones(d)
ub = 3 * np.ones(d)
q = 5
gamma = 0.3
cs = 3
co = 1
ratio = cs/(cs+co)
mu = np.linspace(0,0.4,q)

if d==2:
    stepsize = 0.8
if d==10:
    stepsize = 4
if d==50:
    stepsize = 25




'''allocation'''
n = 2**10
T = 100
Gamma = n*T

rep = 100

'''Test T'''
np.random.seed(2026)
n_test = 2**10
X = np.random.uniform(low = lb, high =ub, size = (n_test, d))

#True optimal solutions and optimal values (for check)
opt_solutions = optimal_solution_vec(X,mu,ratio,gamma)
opt_values = obj_value_vec(X,mu,opt_solutions,cs,co,gamma) 

opt_gaps = np.zeros(rep)

# PR-SGD solutions (Tuning T)
for l in range(rep):
    np.random.seed(2*l+1)
    theta_avg = pr_sgd_vec(X,mu,q,d,n,cs,co,gamma,T,stepsize)
    obj_values_prsgd = obj_value_vec(X,mu,theta_avg,cs,co,gamma) 
    opt_gaps[l] = np.mean((obj_values_prsgd-opt_values)/opt_values)
    
print('=====Tuning T========')
print('d',d)
print('T',T)
print('stepsize', stepsize)
print('average relative optimality gap',np.mean(opt_gaps))


