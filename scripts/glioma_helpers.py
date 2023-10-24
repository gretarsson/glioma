import copy
import time
from scipy.optimize import fmin, differential_evolution
from solve_brain.brain_models import error_FC
from joblib import Parallel, delayed
from jitcdde import jitcdde
import numpy as np
import os
import networkx as nx
import itertools




def find_avg_diff(arr, interval_len): # reshape the array into intervals
    # needs to be integer
    interval_len = int(interval_len)

    # Calculate the size of the array after truncating any excess elements
    arr_size = arr.size
    truncated_size = int(arr_size // interval_len * interval_len)

    # Slice the array and reshape into intervals
    arr_intervals = arr[:truncated_size].reshape(-1, interval_len)

    # find highest values in each nth-length interval
    max_vals = np.max(arr_intervals, axis=1)

    # find lowest values in each nth-length interval
    min_vals = np.min(arr_intervals, axis=1)

    # compute average of max values
    avg_max_val = np.mean(max_vals)

    # compute average of min values
    avg_min_val = np.mean(min_vals)

    # compute absolute difference between average max and average min
    abs_diff = np.abs(avg_max_val - avg_min_val)

    return abs_diff


def amplitude_matrix(arr, interval_len):
    # intialize
    N, _ = arr.shape
    ampl_vector = np.zeros((N,))
    ampl_matrix = np.zeros((N,N))

    # store average amplitude in vector
    for i in range(N):
        xi = arr[i,:]
        ampl = find_avg_diff(xi, interval_len)
        ampl_vector[i] = ampl

    # create matrix
    ampl_cols = np.tile(ampl_vector, (N, 1))
    ampl_matrix = ampl_cols.T - ampl_cols

    # we're done
    return ampl_matrix  
        

def compute_eigenvector_centrality(G):
    try:
        centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=500)
    except nx.exception.PowerIterationFailedConvergence as e:
        # Handle the convergence failure for this node
        print(f"Convergence failed for node {e.args[0]}: {e.args[1]}")
        centrality = [np.nan for _ in range(G.number_of_nodes)]  # Set centrality to NaN for non-converged nodes
    return list(centrality.values())


# self explanatory
def create_directory(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created.")
    else:
        print(f"Directory '{folder_path}' already exists.")

def clustering_coefficient(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    clustering_coeff = nx.clustering(G, weight="weight")
    return clustering_coeff




# define simpler function for parallelizing 
def optimize(varlist): 
    # extract variables
    exp_PLI = varlist[0]
    y0 = varlist[1]
    DE_file = varlist[2]
    control_pars = varlist[3]
    bounds = varlist[4]
    tspan = varlist[5]
    atol = varlist[6]
    rtol = varlist[7]
    cutoff = varlist[8]
    band = varlist[9]
    normalize_exp = varlist[10]
    threshold_exp = varlist[11]
    objective = varlist[12]
    popsize = varlist[13]
    opt_tol = varlist[14]
    recombination = varlist[15]
    mutation = varlist[16]
    maxiter = varlist[17]
    W = varlist[18]
    step = varlist[19]
    n = varlist[20]
    inds = varlist[21]
    mean_coherence = varlist[22]
    par_coherence = varlist[23]
    
    # compile jitcode object from file
    DE = jitcdde(module_location=DE_file, control_pars=control_pars, max_delay=0, n=n)

    # optimize
    minimize = differential_evolution(error_FC, bounds, args=(DE, W, tspan, step, \
                    atol, rtol, cutoff, band, exp_PLI, normalize_exp, threshold_exp,  \
                    False, False, 0, y0, inds, objective, False, mean_coherence, par_coherence), \
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


def parallell_optimize(W, DE_file, control_pars, bounds, M, n_jobs, exp_FC, y0s, tspan=(0,11), atol=1e-6, rtol=1e-3, cutoff=1,band=[8,12],normalize_exp=True, threshold_exp=0.0,objective='pearson',popsize=30,opt_tol=1e-1, recombination=0.3, mutation=(0.5,10),maxiter=100, step=1e-1, n=1, inds=[], mean_coherence=0.0137, par_coherence=0):

    print('\n\nBegin control parallelization...')
    start_time_h = time.time()
    minimizes = Parallel(n_jobs=n_jobs, prefer=None)(delayed(optimize)([exp_FC, y0s[m], DE_file, control_pars, bounds, tspan, atol, rtol, cutoff, band, normalize_exp, threshold_exp, objective, popsize, opt_tol, recombination, mutation, maxiter, W, step, n, inds, mean_coherence, par_coherence]) for m in range(M))        
    print(f'Healthy control optimization took: {time.time() - start_time_h} seconds')
    print(f'One average optimization took: {(time.time() - start_time_h)/M} seconds')
    return minimizes



def remove_rows_and_columns(arr, indices_to_remove):
    """
    Remove rows and columns from a 2D array based on the specified indices.

    Parameters:
    arr (numpy.ndarray): Input 2D array.
    indices_to_remove (list): List of indices to remove.

    Returns:
    numpy.ndarray: 2D array with specified rows and columns removed.
    """
    # Remove rows corresponding to indices_to_remove
    arr_filtered_rows = np.delete(arr, indices_to_remove, axis=0)

    # Remove columns corresponding to indices_to_remove
    arr_filtered_rows_and_columns = np.delete(arr_filtered_rows, indices_to_remove, axis=1)

    return arr_filtered_rows_and_columns

