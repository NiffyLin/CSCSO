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

method = 'KRR'
lb = 0 * np.ones(d)
ub = 3 * np.ones(d)
inter = 0.1
gamma = 0.3
cs = 3
co = 1
ratio = cs/(cs+co)
mu = np.linspace(0,0.4,q)



'''-----------------Allocation and Hyperparameters----------------------'''

if d==10:
    stepsize = 4
    Gamma = np.array([12000, 15000, 18000, 21000, 24000,
                        27000, 30000, 33000, 36000, 39000])
    n = np.array([40, 41, 41, 41, 42, 42, 42, 42, 42, 43])
    T = np.floor(Gamma/n)
    hypers = [ (60, 1e-4),  (60, 1e-4), (60, 1e-4),  (60, 1e-4),  (60, 1e-4),
            (60, 1e-4), (60, 1e-4), (60, 1e-4), (60, 1e-4), (60, 1e-4)]
elif d==50:
    stepsize = 25
    Gamma = np.array([30000, 35000, 40000, 45000,50000, 55000, 60000, 65000, 
                      70000, 75000])
    n = np.array([200, 200, 200, 200, 200, 205, 205, 205, 205, 205])
    T = np.floor(Gamma/n)
    hypers = [ (200, 1e-4),  (200, 1e-4), (200, 1e-4),  (200, 1e-4),  (200, 1e-4),
            (200, 1e-4), (200, 1e-4), (200, 1e-4), (200, 1e-4), (200, 1e-4)]

Gamma_n_T_list = list(zip(Gamma.astype(int), n.astype(int), T.astype(int)))
hyper_params = {nt: hp for nt, hp in zip(Gamma_n_T_list, hypers)}


'''----------------Test Data Generation-------------------------'''
np.random.seed(2025)
n_test = 1
X_test = np.random.uniform(low = lb+inter, high =ub-inter, size = (n_test, d))
opt_solutions_test = optimal_solution_vec(X_test,mu,ratio,gamma)
opt_values_test = obj_value_vec(X_test,mu,opt_solutions_test,cs,co,gamma).squeeze() 

'''-----------------save path---------------------------------'''
base_dir = os.path.dirname(__file__) 
save_dir = os.path.join(base_dir, str(d)+'d', str(q)+'q',method)
os.makedirs(save_dir, exist_ok=True) 

opt_save_dir = os.path.join(base_dir, str(d)+'d', str(q)+'q')
opt_value_path = os.path.join(opt_save_dir, 'opt_values_test.npy')
np.save(opt_value_path, opt_values_test)

'''------------------Computation--------------------------------'''

parallel = 4
rep = 100

X_gl = None
common_gl = None
T_gl = None
n_gl = None
length_scale_gl = None
lam_gl = None

def _init_worker(X, common_krr, T, n, length_scale, lam):
    global X_gl, common_gl, T_gl, n_gl, length_scale_gl, lam_gl
    X_gl = X
    common_gl = common_krr
    T_gl = T
    n_gl = n
    length_scale_gl = length_scale
    lam_gl = lam

def p_cscso(i):
    np.random.seed(2*i+1)

    theta_avg = pr_sgd_vec(X_gl, mu, q, d, n_gl, cs, co, gamma, T_gl, stepsize)

    pre_solutions = common_gl @ theta_avg

    pre_opt_values = obj_value_vec(X_test, mu, pre_solutions, cs, co, gamma)
    return pre_opt_values

if __name__ == '__main__':
    gammas, ns, Ts = [], [],[]
    mean_list = []
    l_list, lambda_list = [], []

    for (Gamma, n, T) in Gamma_n_T_list:   

        length_scale, lam = hyper_params[(Gamma, n, T)]

        sampler = qmc.Sobol(d=d, scramble=False)
        X_unit = sampler.random(n)
        X = qmc.scale(X_unit, lb, ub)

        K_XX = rbf_kernel(X, X, length_scale=length_scale)
        K_reg = K_XX + lam * np.eye(n)
        r_test = rbf_kernel(X_test, X, length_scale=length_scale)
        common_krr = r_test @ np.linalg.inv(K_reg)

        with mp.Pool(
            parallel,
            initializer=_init_worker,
            initargs=(X, common_krr, T, n, length_scale, lam)
        ) as p:
            outputs = list(tqdm(p.imap(p_cscso, range(rep)), total=rep,
                                desc=f"KRR Γ={Gamma}, T={T}, n={n}, l={length_scale}, λ={lam}"))
            outputs = np.array(outputs)
            p.close()
            p.join()

        mean = (np.mean(outputs)-opt_values_test)/opt_values_test

        print('========KRR========')
        print('d', d)
        print('Gamma', Gamma)
        print('T', T)
        print('n', n)
        print('hyper (length_scale, lambda):', (length_scale, lam))
        print('Mean', mean)
        print()

        gammas.append(Gamma)
        ns.append(n)
        Ts.append(T)
        mean_list.append(float(mean))
        l_list.append(length_scale)
        lambda_list.append(lam)
        results_csv = os.path.join(save_dir, 'results.csv')  

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Gamma', 'n', 'T','Mean', 'length_scale', 'lambda'])
        for G, n, T, a,  l, la in zip(gammas, ns, Ts, mean_list,  l_list, lambda_list):
            writer.writerow([ G, n, T, a,  l, la])
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


