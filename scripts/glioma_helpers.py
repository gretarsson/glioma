import copy
import time
from scipy.optimize import fmin, differential_evolution
from solve_brain.brain_models import error_FC
from joblib import Parallel, delayed



# define simpler function for parallelizing 
def optimize(varlist): 
    # extract variables
    exp_PLI = varllist[0]
    y0 = varllist[1]
    threshold_exp = varlist[2]
    DE = varlist[3]


    # optimize
    minimize = differential_evolution(error_FC, bounds, args=(DE, mean_W, tspan, step, \
                    atol, rtol, cutoff, band, exp_PLI, normalize_exp, threshold_exp,  \
                    False, False, zero_scale, y0, [], objective, freq_normal), \
                    disp=False, strategy='best1bin', popsize=popsize, init='latinhypercube', tol=tol, \
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


def parallell_optimize_hopf(DEs, M, n_jobs, exp_FC, thres_h, y0s):

    print('\n\nBegin control parallelization...')
    start_time_h = time.time()
    minimizes = Parallel(n_jobs=n_jobs, prefer=None)(delayed(optimize)([exp_FC, y0s[m], thres_h, DEs[m]]) for m in range(M))        
    print(f'Healthy control optimization took: {time.time() - start_time_h} seconds')
    print(f'One average optimization took: {(time.time() - start_time_h)/M} seconds')
    return minimizes
