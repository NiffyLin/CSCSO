import numpy as np
from scipy.stats import norm


def obj_value_vec(X,mu,theta_vec,cs,co,gamma): # shape of x (n,d)
    e = np.sum(X,axis = 1).reshape(-1,1) + mu #(n,q)
    std = gamma * np.sqrt(np.sum(X**2, axis=1).reshape(-1,1) + mu**2) #(n,q)
    obj_vec = (cs+co)*std*norm.pdf((theta_vec-e)/std) + (cs+co)*(theta_vec-e)*norm.cdf((theta_vec-e)/std)-cs*(theta_vec-e)
    obj_value = np.sum(obj_vec, axis=1) #(n,1)
    return obj_value #(n,1)


def optimal_solution_vec(X,mu,ratio,gamma): # (n,d)
    opt_solution = np.sum(X, axis=1).reshape(-1,1) + mu + norm.ppf(ratio) * gamma * np.sqrt(np.sum(X**2,axis=1).reshape(-1,1)+mu**2)
    return opt_solution #(n,q)    

    
    
def pr_sgd_vec(X,mu,q,d,n,cs,co,gamma,T,stepsize):
    theta_vec = np.ones((n,q))
    theta_sum = theta_vec
    for t in range(T):
        z = X + gamma * X * np.random.normal(0,1,(n,d)) #(n,d)
        epsilon = mu + gamma * mu * np.random.normal(0,1,(n,q))
        D = np.sum(z,axis=1).reshape(-1,1) + epsilon #(n,q)
        gradient = - cs * np.int64(D>theta_vec) + co * np.int64(D<=theta_vec)
        theta_vec = theta_vec - stepsize * (np.log(t+2)/(t+2)) * gradient   # stepsize = logt/t
        theta_vec = np.maximum(theta_vec, 0)
        theta_sum = theta_sum + theta_vec
    theta_avg = theta_sum/(T+1) #(n,q)
    return theta_avg #(n,q)



'''Smoothing'''
def rbf_kernel(X1, X2, length_scale):
    sq_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)    
    return np.exp(- sq_dists/(length_scale**2))




