import multiprocessing as mp
from functions import *
import numpy as np
import time
from scipy.stats import norm
from scipy.stats import qmc
from tqdm import tqdm
import os
import warnings
import csv
warnings.filterwarnings('ignore', message='The balance properties of Sobol')

'''----------Settings: Please Set q and d-------------
(q,d) = (5,10), (25,10), (5,50)
'''
q = 5 #5, 25
d = 10 #10, 50

method = 'kNN'
lb = 0 * np.ones(d)
ub = 3 * np.ones(d)
inter = 0.1
gamma = 0.3
cs = 3
co = 1
ratio = cs/(cs+co)
mu = np.linspace(0, 0.4, q)

'''-----------------Allocation and Hyperparameters----------------------'''
if d==10:
    stepsize = 4
    Gamma = np.array([12000, 15000, 18000, 21000, 24000,
                        27000, 30000, 33000, 36000, 39000])
    T = np.array([35, 35, 35, 35, 40, 40, 40, 45, 45, 50])
    n = np.floor(Gamma/T)
    hypers = [6, 6, 6, 7, 7, 7, 7, 7, 8, 8]
   

elif d==50:
    stepsize = 25
    Gamma = np.array([30000, 35000, 40000, 45000,50000, 55000, 60000, 65000, 
                      70000, 75000])
    T = 100 * np.ones(len(Gamma))
    n = np.floor(Gamma/T)
    hypers = [10, 10, 10, 10, 10, 11, 11, 11, 11, 11]

Gamma_n_T_list = list(zip(Gamma.astype(int), n.astype(int), T.astype(int)))
hyper_params = {nt: hp for nt, hp in zip(Gamma_n_T_list, hypers)}


'''----------------Test Data Generation-------------------------'''
np.random.seed(2025)
n_test = 1
X_test = np.random.uniform(low = lb+inter, high =ub-inter, size = (n_test, d))
opt_solutions_test = optimal_solution_vec(X_test,mu,ratio,gamma)
opt_values_test = obj_value_vec(X_test,mu,opt_solutions_test,cs,co,gamma).squeeze()

'''-----------------Save Path---------------------------------'''
base_dir = os.path.dirname(__file__) 
save_dir = os.path.join(base_dir, str(d)+'d', str(q)+'q',method)
os.makedirs(save_dir, exist_ok=True) 

opt_save_dir = os.path.join(base_dir, str(d)+'d')
opt_value_path = os.path.join(opt_save_dir, 'opt_values_test.npy')
np.save(opt_value_path, opt_values_test)

'''------------------Computation--------------------------------'''

parallel = 4
rep = 100

X_gl = None
T_gl = None
n_gl = None
k_gl = None
w_gl = None

def _init_worker(X, w, T, n, k):
    global X_gl,w_gl, T_gl, n_gl, k_gl
    X_gl = X
    w_gl = w
    T_gl = T
    n_gl = n
    k_gl = k

def p_cscso(i):
    np.random.seed(2*i+1)

    theta_avg = pr_sgd_vec(X_gl, mu, q, d, n_gl, cs, co, gamma, T_gl, stepsize)

    pre_solutions = w_gl @ theta_avg   # (n_test, q)

    pre_opt_values = obj_value_vec(X_test, mu, pre_solutions, cs, co, gamma)
    return pre_opt_values

if __name__ == '__main__':
    gammas, ns, Ts = [], [],[]
    mean_list = []
    k_list = []

    for (Gamma, n, T) in Gamma_n_T_list:   
        k = hyper_params[(Gamma, n, T)]

        sampler = qmc.Sobol(d=d, scramble=False)
        X_unit = sampler.random(n)
        X = qmc.scale(X_unit, lb, ub)

        dist = np.sqrt(np.sum(X_test**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X_test, X.T))
        idx_knn = np.argpartition(dist, kth=k-1, axis=1)[:, :k]
        w = np.zeros((X_test.shape[0], X.shape[0]))
        rows = np.arange(X_test.shape[0])[:, None]
        w[rows, idx_knn] = 1.0 / k


        with mp.Pool(
            parallel,
            initializer=_init_worker,
            initargs=(X, w, T, n, k)
        ) as p:
            outputs = list(tqdm(p.imap(p_cscso, range(rep)), total=rep,
                                desc=f"kNN Γ={Gamma}, T={T}, n={n}, k={k}"))
            outputs = np.array(outputs)
            p.close()
            p.join()

        mean = (np.mean(outputs)-opt_values_test)/opt_values_test

        print('========kNN========')
        print('d', d)
        print('Gamma', Gamma)
        print('T', T)
        print('n', n)
        print('hyper (k):', k)
        print('Mean', mean)
        print()

        gammas.append(Gamma)
        ns.append(n)
        Ts.append(T)
        mean_list.append(float(mean))
        k_list.append(k)
        results_csv = os.path.join(save_dir, 'results.csv')  

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Gamma','n', 'T', 'mean',  'k'])
        for G, n, T, a,  k in zip(gammas, ns, Ts, mean_list,  k_list):
            writer.writerow([ G, n, T, a, k])
    print(f'[INFO] Results written to: {results_csv}')

    '''------------------Convergence Analysis-------------------------'''
    Ts_arr = np.array(Ts, dtype=float)
    gammas_arr = np.array(gammas, dtype=float)
    mean_arr = np.array(mean_list, dtype=float)

    x = np.log(gammas_arr)
    y = np.log(mean_arr)

    slope, intercept = np.polyfit(x, y, 1)

    r = np.corrcoef(x, y)[0, 1]
    r2 = float(r ** 2)

    print('===== Log-Log Regression (Mean vs Gamma) =====')
    print(f'slope          = {slope:.4f}')
    print(f'intercept (a)  = {intercept:.4f}')
    print(f'R^2            = {r2:.4f}')


