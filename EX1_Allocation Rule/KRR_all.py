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


'''--------Settings: Please Set d and rule------------'''

d = 2 # 2, 10
rule = 'opt' #'opt', 'exact', 'under', 'over'


method = 'KRR'
lb = 0 * np.ones(d)
ub = 3 * np.ones(d)
inter = 0.1
q = 5
gamma = 0.3
cs = 3
co = 1
ratio = cs/(cs+co)
mu = np.linspace(0,0.4,q)


if d==2:
    stepsize = 0.8
    Gamma = np.array([100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    hypers = [ (10, 1e-4),  (10, 1e-4), (10, 1e-4),  (10, 1e-4),  (10, 1e-4),
            (10, 1e-4), (10, 1e-4), (10, 1e-4), (10, 1e-4), (10, 1e-4)]
elif d==10:
    stepsize = 4
    Gamma = np.array([3000,  6000,  9000, 12000, 15000, 18000, 21000, 24000,
       27000, 30000])
    hypers = [ (60, 1e-4),  (60, 1e-4), (60, 1e-4),  (60, 1e-4),  (60, 1e-4),
            (60, 1e-4), (60, 1e-4), (60, 1e-4), (60, 1e-4), (60, 1e-4)]


'''-----------------Allocation------------------------'''

if rule == 'opt':
    if d==2:
        n = np.array([10, 10, 10, 10, 10, 11, 11, 11, 11, 11])
        T = np.floor(Gamma/n)
    elif d==10:
        n = np.array([40, 40, 40, 40, 41, 41, 41, 42, 42, 42])
        T = np.floor(Gamma/n)
elif rule == 'exact':
    T = 100 * np.ones(len(Gamma))
    n = np.ceil(Gamma / T)
elif rule == 'under':
    T = 50 * np.ones(len(Gamma))
    n = np.ceil(Gamma / T)
elif rule == 'over':
    T = 150 * np.ones(len(Gamma))
    n = np.ceil(Gamma / T)

Gamma_n_T_list = list(zip(Gamma.astype(int), n.astype(int), T.astype(int)))
    
hyper_params = {nt: hp for nt, hp in zip(Gamma_n_T_list, hypers)}


'''----------------Test Data Generation-------------------------'''
np.random.seed(2025)
n_test = 100
X_test = np.random.uniform(low = lb+inter, high =ub-inter, size = (n_test, d))
opt_solutions_test = optimal_solution_vec(X_test,mu,ratio,gamma)
opt_values_test = obj_value_vec(X_test,mu,opt_solutions_test,cs,co,gamma) 

'''-----------------Save Path---------------------------------'''

base_dir = os.path.dirname(__file__) 
save_dir = os.path.join(base_dir, str(d)+'d', method, rule)
os.makedirs(save_dir, exist_ok=True)   

opt_save_dir = os.path.join(base_dir, str(d)+'d')
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
    avg_list, std_list, min_list, max_list = [], [], [], []
    l_list, lambda_list = [], []

    for (Gamma, n, T) in Gamma_n_T_list:   

        length_scale, lam = hyper_params[(Gamma, n, T)]

        sampler = qmc.Sobol(d=d, scramble=False)
        X_unit = sampler.random(n)
        X = qmc.scale(X_unit, lb, ub)

        # KRR
        K_XX = rbf_kernel(X, X, length_scale=length_scale)
        K_reg = K_XX + lam * np.eye(n)
        r_test = rbf_kernel(X_test, X, length_scale=length_scale)
        common_krr = r_test @ np.linalg.inv(K_reg)

        save_path = os.path.join(save_dir, str(Gamma)+'_'+str(n)+'_'+str(T)+'.npy')

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
        np.save(save_path, outputs)
        re = (outputs-opt_values_test)/opt_values_test 
        re_avg = np.mean(re,axis=1)
        re_std = np.std(re,axis=1)
        re_min = np.min(re,axis=1)
        re_max = np.max(re,axis=1)

        print('========KRR========')
        print('d', d)
        print('Gamma', Gamma)
        print('T', T)
        print('n', n)
        print('hyper (length_scale, lambda):', (length_scale, lam))
        print('average', np.mean(re_avg))
        print('sd', np.mean(re_std))
        print('minimum', np.mean(re_min))
        print('maximum', np.mean(re_max))
        print()

        gammas.append(Gamma)
        ns.append(n)
        Ts.append(T)
        avg_list.append(np.mean(re_avg))
        std_list.append(np.mean(re_std))
        min_list.append(np.mean(re_min))
        max_list.append(np.mean(re_max))
        l_list.append(length_scale)
        lambda_list.append(lam)
        results_csv = os.path.join(save_dir, 'results.csv')  

    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Gamma','n', 'T', 'average', 'sd', 'min','max','length_scale', 'lambda'])
        for G, n, T, a, s, mi, ma, l, la in zip(gammas, ns, Ts, avg_list, std_list, min_list, max_list, l_list, lambda_list):
            writer.writerow([ G, n, T, a, s, mi, ma, l ,la])
    print(f'[INFO] Results written to: {results_csv}')



