import copy
import time
from scipy.optimize import fmin, differential_evolution
from solve_brain.brain_models import error_FC
from joblib import Parallel, delayed
from jitcdde import jitcdde
import numpy as np



# define simpler function for parallelizing 
def optimize(varlist): 
    # extract variables
    exp_PLI = varlist[0]
    y0 = varlist[1]
    threshold_exp = varlist[2]
    DE_file = varlist[3]
    control_pars = varlist[4]
    bounds = varlist[5]
    tspan = varlist[6]
    atol = varlist[7]
    rtol = varlist[8]
    cutoff = varlist[9]
    band = varlist[10]
    normalize_exp = varlist[11]
    threshold_exp = varlist[12]
    objective = varlist[13]
    popsize = varlist[14]
    opt_tol = varlist[15]
    recombination = varlist[16]
    mutation = varlist[17]
    maxiter = varlist[18]
    W = varlist[19]
    step = varlist[20]
    n = varlist[21]
    
    # compile jitcode object from file
    DE = jitcdde(module_location=DE_file, control_pars=control_pars, max_delay=0, n=n)

    # optimize
    minimize = differential_evolution(error_FC, bounds, args=(DE, W, tspan, step, \
                    atol, rtol, cutoff, band, exp_PLI, normalize_exp, threshold_exp,  \
                    False, False, 0, y0, [], objective, False), \
                    disp=True, strategy='best1bin', popsize=popsize, init='latinhypercube', tol=opt_tol, \
                    recombination=recombination, mutation=mutation, maxiter=maxiter)

    # delete compilation for space
    del DE

    # update to screen
    print(f'success: {minimize.success}')
    print(f'number of iterations: {minimize.nit}')
    print(f'minimize.x = {minimize.x}')

    # if not succesful, put optimal parameters to nan
    if minimize.success:
        return minimize
    else:
        npars = len(minimize.x)
        sols = [ np.nan for _ in range(npars) ] 
        minimize.x = sols
        return minimize


def parallell_optimize(W, DE_file, control_pars, bounds, M, n_jobs, exp_FC, thres_h, y0s, tspan=(0,11), atol=1e-6, rtol=1e-3, cutoff=1,band=[8,12],normalize_exp=True, threshold_exp=0.0,objective='pearson',popsize=30,opt_tol=1e-1, recombination=0.3, mutation=(0.5,10),maxiter=100, step=1e-1, n=1):

    print('\n\nBegin control parallelization...')
    start_time_h = time.time()
    minimizes = Parallel(n_jobs=n_jobs, prefer=None)(delayed(optimize)([exp_FC, y0s[m], thres_h, DE_file, control_pars, bounds, tspan, atol, rtol, cutoff, band, normalize_exp, threshold_exp, objective, popsize, opt_tol, recombination, mutation, maxiter, W, step, n]) for m in range(M))        
    print(f'Healthy control optimization took: {time.time() - start_time_h} seconds')
    print(f'One average optimization took: {(time.time() - start_time_h)/M} seconds')
    return minimizes
